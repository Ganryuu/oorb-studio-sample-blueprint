"""Hardware-facing runtime: device selection, precision, kernels, compilation."""

from .device import DeviceInfo, GpuTopology, probe_devices, resolve_device
from .precision import PrecisionPlan, plan_precision

__all__ = [
    "DeviceInfo",
    "GpuTopology",
    "probe_devices",
    "resolve_device",
    "PrecisionPlan",
    "plan_precision",
]
