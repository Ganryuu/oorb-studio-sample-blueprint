"""Engine configuration.

Defaults here are tuned for the reference deployment target: 2x RTX 3090
(Ampere, sm_86, 24 GB each). That hardware fixes several choices:

* **bf16, not fp8.** sm_86 has no FP8 tensor cores (that starts at sm_89/Ada).
  bf16 matches fp16 throughput on Ampere without fp16's overflow babysitting.
* **TF32 on.** Free ~2x on the fp32 matmuls that survive in norm/head layers.
* **FlashAttention-2 is available** (it supports sm_80/sm_86), so it is the
  preferred attention backend, with PyTorch SDPA as the fallback.
* **Replica parallelism, not tensor parallelism.** GeForce cards have P2P
  disabled in the driver, so cross-GPU tensor parallel traffic would go over
  host memory. All three supported policies fit in 24 GB at bf16 (OpenVLA-7B
  ~15 GB, pi0 ~7 GB, SmolVLA ~1 GB), so one full replica per GPU is both
  faster and simpler. See :mod:`vla_engine.optim.replicas`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

from .errors import ConfigError

__all__ = [
    "PrecisionConfig",
    "CompileConfig",
    "DecodeConfig",
    "BatchConfig",
    "EngineConfig",
]

_DTYPES = ("bf16", "fp16", "fp32", "auto")
_QUANT = (None, "none", "nf4", "int8", "awq", "gptq")
_ATTENTION = ("auto", "flash_attention_2", "sdpa", "eager")
_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def _coerce(cls: type, data: Mapping[str, Any]) -> Any:
    """Build a dataclass from a mapping, ignoring unknown keys loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown {cls.__name__} keys: {sorted(unknown)}; valid keys: {sorted(known)}"
        )
    return cls(**dict(data))


@dataclass
class PrecisionConfig:
    """Numeric precision and weight quantization.

    Args:
        dtype: Compute dtype. ``"auto"`` picks bf16 on Ampere+ (sm_80+), fp16
            on older CUDA cards, fp32 on CPU.
        quantization: Weight-only quantization. ``"nf4"`` (bitsandbytes 4-bit)
            cuts OpenVLA-7B from ~15 GB to ~5 GB, which is what makes room for
            large batches or a second model on the same 3090. It costs latency
            at batch 1 -- dequant is not free -- so it is off by default.
        tf32: Allow TF32 for fp32 matmuls and cuDNN. Ampere-only, ~free speed.
        attention: Attention kernel. ``"auto"`` prefers FlashAttention-2 when
            importable and the dtype is half precision, else SDPA.
        matmul_precision: Passed to ``torch.set_float32_matmul_precision``.
    """

    dtype: str = "auto"
    quantization: str | None = None
    tf32: bool = True
    attention: str = "auto"
    matmul_precision: str = "high"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ConfigError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if self.quantization in ("none", ""):
            self.quantization = None
        if self.quantization not in _QUANT:
            raise ConfigError(
                f"quantization must be one of {[q for q in _QUANT if q]}, "
                f"got {self.quantization!r}"
            )
        if self.attention not in _ATTENTION:
            raise ConfigError(f"attention must be one of {_ATTENTION}, got {self.attention!r}")
        if self.quantization in ("nf4", "int8") and self.dtype == "fp32":
            raise ConfigError(
                "bitsandbytes quantization needs a half-precision compute dtype; "
                "set dtype to 'bf16' (recommended on RTX 3090) or 'fp16'"
            )


@dataclass
class CompileConfig:
    """``torch.compile`` / CUDA-graph settings.

    For a robot control loop the win is almost entirely kernel-launch overhead,
    not FLOPs: OpenVLA's 7-token action decode is ~7 sequential forward passes
    of a 7B model, each launching thousands of tiny kernels. ``reduce-overhead``
    wraps the replayed region in CUDA graphs and is typically the single
    largest latency improvement available on a 3090.

    CUDA graphs require *static* shapes and stable input addresses, which is
    why :class:`DecodeConfig` allocates a fixed-size KV cache and the image
    pipeline pads to a fixed resolution.
    """

    enabled: bool = True
    mode: str = "reduce-overhead"
    fullgraph: bool = False
    dynamic: bool = False
    # Compilation is lazy and costs 30-120 s on first call. Warmup pays that
    # cost at load time instead of on the robot's first control step.
    warmup_steps: int = 3

    def __post_init__(self) -> None:
        if self.mode not in _COMPILE_MODES:
            raise ConfigError(f"compile mode must be one of {_COMPILE_MODES}, got {self.mode!r}")
        if self.warmup_steps < 0:
            raise ConfigError("warmup_steps must be >= 0")

    @property
    def uses_cudagraphs(self) -> bool:
        return self.enabled and self.mode == "reduce-overhead"


@dataclass
class DecodeConfig:
    """Autoregressive action-token decoding (OpenVLA and other token VLAs).

    OpenVLA emits exactly ``action_dim`` action tokens greedily -- there is no
    sampling, no EOS search, and no variable length. That means the whole
    generate() machinery in ``transformers`` can be replaced by a fixed-trip-count
    loop over a preallocated static KV cache, which is both faster and
    CUDA-graph-safe. See :mod:`vla_engine.runtime.decode`.
    """

    static_kv_cache: bool = True
    # Prompt tokens + vision patches + action tokens, rounded up. OpenVLA's
    # fused DINOv2/SigLIP encoder emits 256 patches; prompts are short.
    max_seq_len: int = 512
    greedy: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.max_seq_len < 64:
            raise ConfigError("max_seq_len must be >= 64")
        if self.temperature <= 0:
            raise ConfigError("temperature must be > 0")


@dataclass
class BatchConfig:
    """Server-side continuous batching.

    Batching helps throughput when several robots or several environments share
    one GPU. It *hurts* single-robot latency, so ``max_batch_size=1`` disables
    the batcher entirely and requests take a direct path to the model.
    """

    max_batch_size: int = 1
    # How long the batcher waits for a second request before firing. Keep this
    # well under the control period; 2 ms is noise against a 100 ms forward.
    max_wait_ms: float = 2.0
    max_queue_depth: int = 64

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ConfigError("max_batch_size must be >= 1")
        if self.max_wait_ms < 0:
            raise ConfigError("max_wait_ms must be >= 0")
        if self.max_queue_depth < 1:
            raise ConfigError("max_queue_depth must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.max_batch_size > 1


@dataclass
class EngineConfig:
    """Top-level engine configuration.

    Args:
        model: Registry alias (``"openvla"``, ``"pi0"``, ``"smolvla"``) or a
            HuggingFace repo id. Repo ids are matched against known families.
        checkpoint: Overrides the registry's default weights for ``model``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit ``"cuda:1"``.
        devices: Explicit replica placement, e.g. ``["cuda:0", "cuda:1"]`` to
            run one model copy per 3090. Overrides ``device`` when set.
        unnorm_key: Dataset statistics used to un-normalize actions (OpenVLA
            calls this the "unnorm key", e.g. ``"bridge_orig"``). Wrong key ->
            plausible-looking but badly scaled actions, so it is surfaced here
            rather than buried in the adapter.
        horizon: Truncate returned chunks to this many steps. ``None`` keeps
            the policy's native horizon.
    """

    model: str
    checkpoint: str | None = None
    device: str = "auto"
    devices: list[str] = field(default_factory=list)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    unnorm_key: str | None = None
    horizon: int | None = None
    trust_remote_code: bool = True
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ConfigError("model must be a non-empty string")
        if self.horizon is not None and self.horizon < 1:
            raise ConfigError("horizon must be >= 1 when set")
        for sub, cls in (
            ("precision", PrecisionConfig),
            ("compile", CompileConfig),
            ("decode", DecodeConfig),
            ("batch", BatchConfig),
        ):
            value = getattr(self, sub)
            if isinstance(value, Mapping):
                setattr(self, sub, _coerce(cls, value))
            elif not isinstance(value, cls):
                raise ConfigError(f"{sub} must be a {cls.__name__} or a mapping")
        if self.devices:
            bad = [d for d in self.devices if not isinstance(d, str) or not d]
            if bad:
                raise ConfigError(f"devices entries must be non-empty strings, got {bad}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EngineConfig":
        return _coerce(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls, model: str | None = None, prefix: str = "VLA_") -> "EngineConfig":
        """Build a config from ``VLA_*`` environment variables.

        Recognized: ``VLA_MODEL``, ``VLA_CHECKPOINT``, ``VLA_DEVICE``,
        ``VLA_DEVICES`` (comma-separated), ``VLA_DTYPE``, ``VLA_QUANTIZATION``,
        ``VLA_ATTENTION``, ``VLA_COMPILE`` (0/1), ``VLA_COMPILE_MODE``,
        ``VLA_UNNORM_KEY``, ``VLA_HORIZON``, ``VLA_MAX_BATCH_SIZE``.
        This is how the ROS 2 node and the container entrypoint are configured.
        """

        def get(key: str) -> str | None:
            raw = os.environ.get(prefix + key)
            return raw.strip() if raw and raw.strip() else None

        resolved = model or get("MODEL")
        if not resolved:
            raise ConfigError(f"no model given and {prefix}MODEL is unset")

        precision = PrecisionConfig(
            dtype=get("DTYPE") or "auto",
            quantization=get("QUANTIZATION"),
            attention=get("ATTENTION") or "auto",
        )
        compile_flag = get("COMPILE")
        compile_cfg = CompileConfig(
            enabled=compile_flag not in ("0", "false", "no") if compile_flag else True,
            mode=get("COMPILE_MODE") or "reduce-overhead",
        )
        devices_raw = get("DEVICES")
        return cls(
            model=resolved,
            checkpoint=get("CHECKPOINT"),
            device=get("DEVICE") or "auto",
            devices=[d.strip() for d in devices_raw.split(",") if d.strip()] if devices_raw else [],
            precision=precision,
            compile=compile_cfg,
            batch=BatchConfig(max_batch_size=int(get("MAX_BATCH_SIZE") or 1)),
            unnorm_key=get("UNNORM_KEY"),
            horizon=int(get("HORIZON")) if get("HORIZON") else None,
        )
