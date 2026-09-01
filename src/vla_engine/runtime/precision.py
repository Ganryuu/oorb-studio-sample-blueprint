"""Resolve a precision request against real hardware.

The point of this module is to turn a user's intent ("go fast") into concrete,
*justified* torch settings, and to refuse silently-wrong combinations. A
misresolved dtype on a VLA is not a crash -- it is a policy that emits subtly
bad actions, which is far worse on a real arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PrecisionConfig
from ..errors import DependencyError
from .device import DeviceInfo, probe_devices

__all__ = ["PrecisionPlan", "plan_precision", "flash_attention_available", "bitsandbytes_available"]


def flash_attention_available() -> bool:
    """True if ``flash_attn`` (v2) is importable."""
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class PrecisionPlan:
    """Concrete precision decisions plus the reasoning that produced them.

    ``notes`` is surfaced in engine logs and ``vla bench``/``vla info`` output
    so a user can see *why* they got fp16 instead of the bf16 they asked for.
    """

    dtype: str
    attention: str
    quantization: str | None = None
    tf32: bool = False
    matmul_precision: str = "high"
    notes: list[str] = field(default_factory=list)

    @property
    def is_half(self) -> bool:
        return self.dtype in ("bf16", "fp16")

    def torch_dtype(self):  # -> torch.dtype
        """Resolve to an actual ``torch.dtype`` (imports torch)."""
        import torch

        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]

    def quantization_config(self):
        """Build a ``transformers.BitsAndBytesConfig``, or ``None``.

        NF4 uses double quantization and a half-precision compute dtype: the
        4-bit weights are dequantized to ``dtype`` inside the matmul, so
        compute precision is unchanged and only the memory footprint drops.
        """
        if self.quantization not in ("nf4", "int8"):
            return None
        if not bitsandbytes_available():
            raise DependencyError(
                "bitsandbytes", f"{self.quantization} quantization", extra="vla-engine[quant]"
            )
        from transformers import BitsAndBytesConfig

        if self.quantization == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.torch_dtype(),
        )

    def apply_global_flags(self) -> None:
        """Set process-wide torch flags implied by this plan.

        TF32 is the notable one: it is off by default in modern torch, and
        turning it on is ~free throughput on Ampere for any fp32 matmul that
        survives autocast (norms, heads, the flow-matching integrator).
        """
        try:
            import torch
        except Exception:
            return
        if self.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(self.matmul_precision)
        # Inference only: autograd is pure overhead and extra memory.
        torch.backends.cudnn.benchmark = True

    def summary(self) -> str:
        parts = [f"dtype={self.dtype}", f"attn={self.attention}"]
        if self.quantization:
            parts.append(f"quant={self.quantization}")
        if self.tf32:
            parts.append("tf32")
        return " ".join(parts)


def plan_precision(
    config: PrecisionConfig,
    device: DeviceInfo | None = None,
    *,
    strict: bool = False,
) -> PrecisionPlan:
    """Resolve ``config`` against ``device`` (default: the first visible GPU).

    Args:
        config: The requested precision settings.
        device: Target accelerator. Defaults to the first probed device.
        strict: Raise instead of downgrading when a request is unsupported.

    Returns:
        A :class:`PrecisionPlan` whose ``notes`` explain every adjustment.
    """
    if device is None:
        topo = probe_devices()
        device = topo.cuda_devices[0] if topo.num_gpus else topo.devices[0]

    notes: list[str] = []
    is_cuda = device.kind == "cuda"

    # --- dtype -------------------------------------------------------------
    dtype = config.dtype
    if dtype == "auto":
        if not is_cuda:
            dtype = "fp32"
            notes.append("no CUDA device: using fp32")
        elif device.supports_bf16:
            dtype = "bf16"
            notes.append(f"{device.arch} (sm_{device.capability[0]}{device.capability[1]}) "
                         "supports bf16 tensor cores")
        else:
            dtype = "fp16"
            notes.append(f"{device.arch} predates bf16 tensor cores: using fp16")
    elif dtype == "bf16" and is_cuda and not device.supports_bf16:
        msg = f"bf16 requested but {device.name} (sm_{device.capability[0]}{device.capability[1]}) lacks bf16 tensor cores"
        if strict:
            raise DependencyError("gpu", msg)
        dtype = "fp16"
        notes.append(msg + "; downgraded to fp16")
    elif dtype in ("bf16", "fp16") and not is_cuda:
        msg = f"{dtype} on CPU is emulated and slower than fp32"
        if strict:
            raise DependencyError("gpu", msg)
        dtype = "fp32"
        notes.append(msg + "; using fp32")

    # --- attention ---------------------------------------------------------
    attention = config.attention
    half = dtype in ("bf16", "fp16")
    if attention == "auto":
        if is_cuda and half and device.supports_flash_attention_2 and flash_attention_available():
            attention = "flash_attention_2"
            notes.append("FlashAttention-2 available and supported on this GPU")
        elif is_cuda:
            attention = "sdpa"
            reason = (
                "flash_attn not installed"
                if not flash_attention_available()
                else "GPU or dtype unsuitable for FlashAttention-2"
            )
            notes.append(f"using PyTorch SDPA ({reason})")
        else:
            attention = "sdpa"
    elif attention == "flash_attention_2":
        problem = None
        if not is_cuda:
            problem = "FlashAttention-2 requires CUDA"
        elif not device.supports_flash_attention_2:
            problem = f"FlashAttention-2 needs sm_80+, this GPU is sm_{device.capability[0]}{device.capability[1]}"
        elif not half:
            problem = "FlashAttention-2 requires fp16/bf16"
        elif not flash_attention_available():
            problem = "flash_attn is not installed"
        if problem:
            if strict:
                raise DependencyError("flash-attn", problem, extra="vla-engine[flash]")
            attention = "sdpa"
            notes.append(f"{problem}; falling back to SDPA")

    # --- quantization ------------------------------------------------------
    quantization = config.quantization
    if quantization and not is_cuda:
        msg = f"{quantization} quantization requires CUDA"
        if strict:
            raise DependencyError("gpu", msg)
        quantization = None
        notes.append(msg + "; loading unquantized")
    elif quantization in ("nf4", "int8") and not bitsandbytes_available():
        msg = f"{quantization} requested but bitsandbytes is not installed"
        if strict:
            raise DependencyError("bitsandbytes", msg, extra="vla-engine[quant]")
        quantization = None
        notes.append(msg + "; loading unquantized")

    tf32 = bool(config.tf32 and device.supports_tf32)
    if config.tf32 and is_cuda and not device.supports_tf32:
        notes.append("TF32 needs Ampere or newer; ignored")

    return PrecisionPlan(
        dtype=dtype,
        attention=attention,
        quantization=quantization,
        tf32=tf32,
        matmul_precision=config.matmul_precision,
        notes=notes,
    )
