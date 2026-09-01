"""Action/state normalization statistics.

Every VLA predicts in a normalized space and relies on dataset statistics to
map back to robot units. Getting this wrong is the single most common way to
make a correctly-loaded policy behave badly: the actions look reasonable in
shape and magnitude, they are just scaled or offset wrong, and the arm moves
confidently in slightly the wrong way.

Two schemes cover the supported policies:

* **Percentile (q01/q99)** -- OpenVLA and the Open-X / RLDS ecosystem. Robust
  to outliers in teleoperated data, where a single jerk would blow up a
  min/max range.
* **Mean/std** -- LeRobot's default for pi0 and SmolVLA.

Both support a per-dimension ``mask``: dimensions that pass through untouched.
OpenVLA uses this for the gripper, which is already a 0/1 command and would be
corrupted by rescaling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ActionStats", "Normalizer"]

_EPS = 1e-8


@dataclass
class ActionStats:
    """Per-dimension statistics for one dataset ("unnorm key").

    Attributes:
        q01, q99: Percentile bounds for percentile normalization.
        mean, std: Moments for standard-score normalization.
        mask: ``True`` where a dimension should be normalized. ``False``
            dimensions pass through unchanged (typically the gripper).
        scheme: ``"percentile"`` or ``"mean_std"``.
    """

    q01: np.ndarray | None = None
    q99: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    mask: np.ndarray | None = None
    scheme: str = "percentile"

    def __post_init__(self) -> None:
        for name in ("q01", "q99", "mean", "std"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=np.float32).reshape(-1))
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool).reshape(-1)
        if self.scheme not in ("percentile", "mean_std"):
            raise ValueError(f"unknown scheme {self.scheme!r}")
        if self.scheme == "percentile" and (self.q01 is None or self.q99 is None):
            if self.mean is not None and self.std is not None:
                self.scheme = "mean_std"
            else:
                raise ValueError("percentile scheme requires both q01 and q99")
        if self.scheme == "mean_std" and (self.mean is None or self.std is None):
            raise ValueError("mean_std scheme requires both mean and std")

    @property
    def action_dim(self) -> int:
        ref = self.q01 if self.scheme == "percentile" else self.mean
        return int(ref.shape[0])  # type: ignore[union-attr]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionStats:
        """Parse the statistics dict shipped inside a policy checkpoint.

        Accepts both the OpenVLA layout (``{"action": {"q01": [...]}}``) and a
        flat ``{"q01": [...]}``, since checkpoints differ on nesting.
        """
        payload = data.get("action", data) if isinstance(data, Mapping) else data
        mask = payload.get("mask")
        has_percentile = payload.get("q01") is not None and payload.get("q99") is not None
        return cls(
            q01=payload.get("q01"),
            q99=payload.get("q99"),
            mean=payload.get("mean"),
            std=payload.get("std"),
            mask=mask,
            scheme="percentile" if has_percentile else "mean_std",
        )

    @classmethod
    def identity(cls, action_dim: int) -> ActionStats:
        """Statistics that leave actions untouched (already in robot units)."""
        return cls(
            mean=np.zeros(action_dim, np.float32),
            std=np.ones(action_dim, np.float32),
            scheme="mean_std",
        )


class Normalizer:
    """Applies and inverts an :class:`ActionStats` transform."""

    def __init__(self, stats: ActionStats) -> None:
        self.stats = stats

    def _mask_for(self, dim: int) -> np.ndarray:
        if self.stats.mask is None:
            return np.ones(dim, dtype=bool)
        mask = self.stats.mask
        if mask.shape[0] < dim:  # pad: unmasked dims are normalized
            mask = np.concatenate([mask, np.ones(dim - mask.shape[0], dtype=bool)])
        return mask[:dim]

    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        """Map normalized policy output into robot units.

        Args:
            actions: ``[..., D]`` array in normalized space.

        Returns:
            An array of the same shape in robot units.
        """
        arr = np.asarray(actions, dtype=np.float32)
        dim = arr.shape[-1]
        mask = self._mask_for(dim)
        if self.stats.scheme == "percentile":
            low = self._fit(self.stats.q01, dim)
            high = self._fit(self.stats.q99, dim)
            # Policy space is [-1, 1]; clamp first so an out-of-range token
            # cannot command a motion larger than anything in the dataset.
            clamped = np.clip(arr, -1.0, 1.0)
            scaled = 0.5 * (clamped + 1.0) * (high - low) + low
        else:
            mean = self._fit(self.stats.mean, dim)
            std = self._fit(self.stats.std, dim)
            scaled = arr * np.maximum(std, _EPS) + mean
        return np.where(mask, scaled, arr).astype(np.float32)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`unnormalize`. Used for evaluation and tests."""
        arr = np.asarray(actions, dtype=np.float32)
        dim = arr.shape[-1]
        mask = self._mask_for(dim)
        if self.stats.scheme == "percentile":
            low = self._fit(self.stats.q01, dim)
            high = self._fit(self.stats.q99, dim)
            span = np.maximum(high - low, _EPS)
            scaled = 2.0 * (arr - low) / span - 1.0
        else:
            mean = self._fit(self.stats.mean, dim)
            std = self._fit(self.stats.std, dim)
            scaled = (arr - mean) / np.maximum(std, _EPS)
        return np.where(mask, scaled, arr).astype(np.float32)

    @staticmethod
    def _fit(values: np.ndarray | None, dim: int) -> np.ndarray:
        """Broadcast stats to ``dim``, padding with neutral values.

        pi0 and SmolVLA pad the action space to 32 dims; a robot with 7 joints
        only has stats for 7. Padding keeps the extra dims a no-op instead of
        raising on a shape mismatch.
        """
        if values is None:
            return np.zeros(dim, np.float32)
        if values.shape[0] == dim:
            return values
        if values.shape[0] > dim:
            return values[:dim]
        pad = np.zeros(dim - values.shape[0], np.float32)
        return np.concatenate([values, pad])
