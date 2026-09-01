"""GPU discovery, capability probing, and replica placement.

Everything here degrades gracefully without torch installed so that the ROS 2
node, the CLI, and the tests can introspect a config on a CPU-only machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from ..errors import ConfigError

__all__ = ["DeviceInfo", "GpuTopology", "probe_devices", "resolve_device", "torch_available"]

# Compute capability -> feature gates. These are architectural facts, not
# tunables: bf16 tensor cores land in sm_80 (Ampere), FP8 in sm_89 (Ada).
_SM_BF16 = (8, 0)
_SM_FP8 = (8, 9)
_SM_FLASH_ATTN2 = (8, 0)


def torch_available() -> bool:
    """True if ``torch`` can be imported. Never raises."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class DeviceInfo:
    """What one accelerator can do, as probed at runtime."""

    index: int
    name: str = "cpu"
    kind: str = "cpu"  # cpu | cuda
    total_memory_gb: float = 0.0
    capability: tuple[int, int] = (0, 0)
    multi_processor_count: int = 0

    @property
    def torch_device(self) -> str:
        return "cpu" if self.kind == "cpu" else f"cuda:{self.index}"

    @property
    def supports_bf16(self) -> bool:
        return self.kind == "cuda" and self.capability >= _SM_BF16

    @property
    def supports_fp8(self) -> bool:
        return self.kind == "cuda" and self.capability >= _SM_FP8

    @property
    def supports_flash_attention_2(self) -> bool:
        return self.kind == "cuda" and self.capability >= _SM_FLASH_ATTN2

    @property
    def supports_tf32(self) -> bool:
        return self.kind == "cuda" and self.capability >= (8, 0)

    @property
    def arch(self) -> str:
        major = self.capability[0]
        return {
            7: "Volta/Turing",
            8: "Ampere" if self.capability < _SM_FP8 else "Ada",
            9: "Hopper",
            10: "Blackwell",
        }.get(major, "unknown" if self.kind == "cuda" else "cpu")

    def summary(self) -> str:
        if self.kind == "cpu":
            return "cpu"
        sm = f"sm_{self.capability[0]}{self.capability[1]}"
        return (
            f"cuda:{self.index} {self.name} ({self.arch}, {sm}, "
            f"{self.total_memory_gb:.0f} GB)"
        )


@dataclass
class GpuTopology:
    """The set of visible accelerators and what the engine may do with them."""

    devices: list[DeviceInfo] = field(default_factory=list)

    @property
    def cuda_devices(self) -> list[DeviceInfo]:
        return [d for d in self.devices if d.kind == "cuda"]

    @property
    def num_gpus(self) -> int:
        return len(self.cuda_devices)

    @property
    def homogeneous(self) -> bool:
        """True when every GPU is the same model.

        Replica parallelism assumes this: a mixed 3090 + 1060 pair would have
        the scheduler feeding work to a card that cannot keep up.
        """
        names = {d.name for d in self.cuda_devices}
        return len(names) <= 1

    def min_memory_gb(self) -> float:
        gpus = self.cuda_devices
        return min((d.total_memory_gb for d in gpus), default=0.0)

    def fits(self, required_gb: float) -> bool:
        """Whether one replica of a model needing ``required_gb`` fits per GPU."""
        return self.num_gpus > 0 and self.min_memory_gb() >= required_gb

    def replica_devices(self, requested: list[str] | None = None) -> list[str]:
        """Resolve the device list to place model replicas on.

        With two 3090s and no explicit request this returns both, because
        GeForce cards cannot do peer-to-peer -- splitting one model across them
        would push activations through host RAM. One full replica per card and
        a least-loaded router is strictly better. See
        :mod:`vla_engine.optim.replicas`.
        """
        if requested:
            known = {d.torch_device for d in self.devices}
            unknown = [d for d in requested if d not in known and d != "cpu"]
            if unknown:
                raise ConfigError(
                    f"requested devices {unknown} are not visible; "
                    f"available: {sorted(known)}"
                )
            return list(requested)
        gpus = self.cuda_devices
        if not gpus:
            return ["cpu"]
        return [d.torch_device for d in gpus]

    def summary(self) -> str:
        if not self.cuda_devices:
            return "no CUDA devices (running on CPU)"
        return "; ".join(d.summary() for d in self.cuda_devices)


@lru_cache(maxsize=1)
def _probe_cached() -> GpuTopology:
    devices: list[DeviceInfo] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    DeviceInfo(
                        index=i,
                        name=props.name,
                        kind="cuda",
                        total_memory_gb=props.total_memory / (1024**3),
                        capability=(props.major, props.minor),
                        multi_processor_count=props.multi_processor_count,
                    )
                )
    except Exception:
        # A broken CUDA install must not take down a CPU-side import.
        devices = []
    if not devices:
        devices.append(DeviceInfo(index=0, name="cpu", kind="cpu"))
    return GpuTopology(devices=devices)


def probe_devices(refresh: bool = False) -> GpuTopology:
    """Return the visible accelerator topology (cached after the first call)."""
    if refresh:
        _probe_cached.cache_clear()
    return _probe_cached()


def resolve_device(requested: str = "auto") -> str:
    """Turn ``"auto"`` into a concrete torch device string.

    Honors ``CUDA_VISIBLE_DEVICES`` implicitly -- torch already reindexes
    around it, so ``cuda:0`` always means "the first visible GPU".
    """
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch_available():
            raise ConfigError(
                f"device {requested!r} requested but torch is not installed; "
                "install with: pip install 'vla-engine[cuda]'"
            )
        return requested
    topo = probe_devices()
    return topo.cuda_devices[0].torch_device if topo.num_gpus else "cpu"


def device_index(device: str) -> int | None:
    """Extract the integer index from ``"cuda:1"``. ``None`` for CPU."""
    if not device.startswith("cuda"):
        return None
    _, _, suffix = device.partition(":")
    if not suffix:
        return int(os.environ.get("CUDA_DEVICE", "0"))
    return int(suffix)
