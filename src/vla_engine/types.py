"""Core data contracts shared by every adapter, server, and control loop.

These types are deliberately framework-free: they hold ``numpy`` arrays and
plain Python, never ``torch`` tensors. That keeps the observation/action
boundary cheap to serialize (see :mod:`vla_engine.serve.protocol`) and lets a
ROS 2 node or a benchmark harness import them without CUDA on the machine.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ObservationError

__all__ = [
    "Observation",
    "ActionChunk",
    "InferenceStats",
    "PolicySpec",
]


def _as_uint8_rgb(name: str, image: Any) -> np.ndarray:
    """Coerce one camera frame to a contiguous ``HxWx3`` uint8 RGB array."""
    arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ObservationError(f"image {name!r} must be HxW or HxWxC, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        # Looks like CHW (a torch-style layout leaked in) -> transpose to HWC.
        arr = np.transpose(arr, (1, 2, 0))
    channels = arr.shape[2]
    if channels == 4:  # drop alpha
        arr = arr[:, :, :3]
    elif channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif channels != 3:
        raise ObservationError(f"image {name!r} must have 1, 3, or 4 channels, got {channels}")
    if arr.dtype != np.uint8:
        # Float images are assumed to be in [0, 1]; anything else in [0, 255].
        if np.issubdtype(arr.dtype, np.floating):
            peak = float(np.nanmax(arr)) if arr.size else 0.0
            scale = 255.0 if peak <= 1.0 + 1e-6 else 1.0
            arr = np.clip(arr * scale, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclass
class Observation:
    """One synchronized snapshot of what the robot sees and feels.

    Attributes:
        images: Camera name -> ``HxWx3`` uint8 RGB frame. Names are matched
            against the policy's expected camera keys by
            :meth:`PolicySpec.resolve_cameras`, so a single-camera robot can
            publish under any name.
        instruction: Natural-language task string. Ignored by non-conditioned
            policies.
        state: Optional proprioceptive vector (joint positions, gripper width,
            end-effector pose). Required by pi0 and SmolVLA, unused by OpenVLA.
        timestamp: Monotonic capture time in seconds. The control-side chunk
            scheduler uses this to age actions, so it must come from the same
            clock as the control loop (default: :func:`time.monotonic`).
        extra: Free-form passthrough for adapter-specific inputs.
    """

    images: dict[str, np.ndarray] = field(default_factory=dict)
    instruction: str = ""
    state: np.ndarray | None = None
    timestamp: float = field(default_factory=time.monotonic)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.images, Mapping):
            raise ObservationError("images must be a mapping of camera name -> array")
        self.images = {k: _as_uint8_rgb(k, v) for k, v in self.images.items()}
        if self.state is not None:
            state = np.asarray(self.state, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(state)):
                raise ObservationError("state contains NaN or Inf")
            self.state = state

    @classmethod
    def single(
        cls,
        image: Any,
        instruction: str = "",
        state: Any | None = None,
        *,
        camera: str = "primary",
        timestamp: float | None = None,
    ) -> Observation:
        """Convenience constructor for the common single-camera case."""
        return cls(
            images={camera: image},
            instruction=instruction,
            state=state,
            timestamp=time.monotonic() if timestamp is None else timestamp,
        )

    @property
    def state_dim(self) -> int:
        return 0 if self.state is None else int(self.state.shape[0])

    def nbytes(self) -> int:
        total = sum(int(v.nbytes) for v in self.images.values())
        if self.state is not None:
            total += int(self.state.nbytes)
        return total


@dataclass
class ActionChunk:
    """A horizon of actions predicted from one observation.

    Single-step policies (OpenVLA) return ``horizon == 1``; flow-matching
    policies (pi0, SmolVLA) return the full 50-step chunk so the control loop
    can run far faster than the policy. ``actions`` is always ``[T, D]`` in the
    robot's native units -- adapters un-normalize before returning.
    """

    actions: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    stats: InferenceStats | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ObservationError(f"actions must be [horizon, action_dim], got shape {arr.shape}")
        self.actions = np.ascontiguousarray(arr)

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def __len__(self) -> int:
        return self.horizon

    def __getitem__(self, index: int) -> np.ndarray:
        return self.actions[index]

    def truncate(self, steps: int) -> ActionChunk:
        """Return a copy keeping only the first ``steps`` actions."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        return ActionChunk(
            actions=self.actions[:steps],
            timestamp=self.timestamp,
            stats=self.stats,
            meta=dict(self.meta),
        )


@dataclass
class InferenceStats:
    """Per-request timing, in milliseconds, measured around a single predict.

    ``total_ms`` is wall-clock inside the engine and is the number a control
    loop should budget against; the rest attribute where that time went.
    """

    total_ms: float = 0.0
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0
    queue_ms: float = 0.0
    batch_size: int = 1
    device: str = "cpu"
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 3),
            "preprocess_ms": round(self.preprocess_ms, 3),
            "forward_ms": round(self.forward_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "queue_ms": round(self.queue_ms, 3),
            "batch_size": self.batch_size,
            "device": self.device,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class PolicySpec:
    """Static description of what a policy consumes and produces.

    Populated by the registry before load, so a caller can validate wiring
    (camera names, state dim, control rate) without downloading weights.
    """

    name: str
    action_dim: int
    horizon: int
    cameras: tuple[str, ...] = ("primary",)
    state_dim: int | None = None
    requires_state: bool = False
    language_conditioned: bool = True
    image_size: tuple[int, int] = (224, 224)
    control_hz: float = 10.0

    def resolve_cameras(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Map user-supplied camera names onto the policy's expected keys.

        Exact names win. Otherwise frames are assigned to the remaining
        expected slots in insertion order, which makes the single-camera case
        ("I published under ``/cam``, the policy wants ``primary``") just work
        while still allowing precise multi-camera wiring.
        """
        if not images:
            raise ObservationError(f"policy {self.name!r} requires at least one image")
        resolved: dict[str, np.ndarray] = {}
        leftover = dict(images)
        for cam in self.cameras:
            if cam in leftover:
                resolved[cam] = leftover.pop(cam)
        missing = [c for c in self.cameras if c not in resolved]
        spare = list(leftover.values())
        # Deliberately ragged: there may be fewer spare frames than missing
        # camera slots, and the shortfall is handled just below.
        for cam, frame in zip(missing, spare, strict=False):
            resolved[cam] = frame
        still_missing = [c for c in self.cameras if c not in resolved]
        if still_missing:
            # Policies tolerate a missing wrist view far better than a missing
            # primary view, so duplicate the primary rather than failing.
            primary = resolved.get(self.cameras[0])
            if primary is None:
                raise ObservationError(
                    f"policy {self.name!r} expects cameras {list(self.cameras)}, got {list(images)}"
                )
            for cam in still_missing:
                resolved[cam] = primary
        return {cam: resolved[cam] for cam in self.cameras}

    def validate(self, obs: Observation) -> None:
        """Raise :class:`ObservationError` if ``obs`` cannot drive this policy."""
        self.resolve_cameras(obs.images)
        if self.requires_state:
            if obs.state is None:
                raise ObservationError(
                    f"policy {self.name!r} requires a proprioceptive state vector"
                )
            if self.state_dim is not None and obs.state_dim > self.state_dim:
                raise ObservationError(
                    f"policy {self.name!r} accepts at most {self.state_dim} state dims, "
                    f"got {obs.state_dim}"
                )
        if self.language_conditioned and not obs.instruction.strip():
            raise ObservationError(
                f"policy {self.name!r} is language-conditioned but instruction is empty"
            )
