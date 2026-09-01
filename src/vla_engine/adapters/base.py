"""The adapter contract every policy implements.

An adapter owns exactly one job: turn :class:`~vla_engine.types.Observation`
objects into a :class:`~vla_engine.types.ActionChunk`, in robot units. The base
class handles everything that is identical across policies -- validation,
timing, horizon truncation, warmup, device bookkeeping -- so a new policy needs
only :meth:`VLAAdapter._load` and :meth:`VLAAdapter._predict_batch`.
"""

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import EngineConfig
from ..errors import NotLoadedError, ObservationError
from ..runtime.device import DeviceInfo, probe_devices, resolve_device
from ..runtime.precision import PrecisionPlan, plan_precision
from ..types import ActionChunk, InferenceStats, Observation, PolicySpec

logger = logging.getLogger(__name__)

__all__ = ["VLAAdapter"]


class VLAAdapter(abc.ABC):
    """Base class for all policy adapters.

    Subclasses must set :attr:`spec` (or override the property) and implement
    :meth:`_load` and :meth:`_predict_batch`.
    """

    #: Filled in by subclasses; describes inputs/outputs before weights load.
    spec: PolicySpec

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._device_info = self._lookup_device(self.device)
        self.precision: PrecisionPlan = plan_precision(config.precision, self._device_info)
        self.model: Any = None
        self._loaded = False
        self._load_seconds = 0.0
        self._warmup_seconds = 0.0
        self._warmup_steps = 0
        self._predict_count = 0

    # -- lifecycle -----------------------------------------------------------

    @staticmethod
    def _lookup_device(device: str) -> DeviceInfo:
        topo = probe_devices()
        for info in topo.devices:
            if info.torch_device == device:
                return info
        return topo.devices[0]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> VLAAdapter:
        """Load weights, apply optimizations, and warm up. Idempotent."""
        if self._loaded:
            return self
        started = time.perf_counter()
        self.precision.apply_global_flags()
        self._set_seed()
        logger.info(
            "loading %s on %s (%s)",
            self.spec.name,
            self._device_info.summary(),
            self.precision.summary(),
        )
        for note in self.precision.notes:
            logger.info("  precision: %s", note)
        self._load()
        self._loaded = True
        self._load_seconds = time.perf_counter() - started
        logger.info("loaded %s in %.1fs", self.spec.name, self._load_seconds)
        self._warmup_steps, self._warmup_seconds = self.warmup()
        return self

    def unload(self) -> None:
        """Release weights and free device memory."""
        self.model = None
        self._loaded = False
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _set_seed(self) -> None:
        if self.config.seed is None:
            return
        np.random.seed(self.config.seed)
        try:
            import torch

            torch.manual_seed(self.config.seed)
        except Exception:
            pass

    def warmup(self, steps: int | None = None) -> tuple[int, float]:
        """Run synthetic observations to trigger compilation and graph capture.

        Uses a blank image and a neutral state, which exercises exactly the
        shapes the real loop will use -- shape is all that compilation and CUDA
        graph capture care about.
        """
        count = self.config.compile.warmup_steps if steps is None else steps
        if count <= 0 or not self._loaded:
            return 0, 0.0
        from ..runtime.compile import warmup as _warmup

        dummy = self.dummy_observation()
        return _warmup(lambda: self._predict_batch([dummy]), count, label=self.spec.name)

    def dummy_observation(self) -> Observation:
        """A synthetic observation matching this policy's input contract."""
        height, width = self.spec.image_size
        images = {cam: np.zeros((height, width, 3), dtype=np.uint8) for cam in self.spec.cameras}
        state = (
            np.zeros(self.spec.state_dim or 7, dtype=np.float32)
            if self.spec.requires_state
            else None
        )
        return Observation(images=images, instruction="warmup", state=state)

    # -- inference -----------------------------------------------------------

    def predict(self, observation: Observation) -> ActionChunk:
        """Predict an action chunk from a single observation."""
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: Sequence[Observation]) -> list[ActionChunk]:
        """Predict for a batch of observations.

        Batching only helps when several robots or environments share a GPU;
        at batch 1 this is the direct path with no extra copies.
        """
        if not self._loaded:
            raise NotLoadedError(f"{self.spec.name} weights are not loaded; call .load() first")
        if not observations:
            return []
        for obs in observations:
            self.spec.validate(obs)

        started = time.perf_counter()
        actions = self._predict_batch(list(observations))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 2:  # [B, D] -> [B, 1, D] for single-step policies
            arr = arr[:, None, :]
        if arr.ndim != 3 or arr.shape[0] != len(observations):
            raise ObservationError(
                f"{self.spec.name} returned shape {arr.shape}; "
                f"expected [{len(observations)}, horizon, action_dim]"
            )
        if self.config.horizon is not None:
            arr = arr[:, : self.config.horizon]

        self._predict_count += len(observations)
        chunks: list[ActionChunk] = []
        for i, obs in enumerate(observations):
            stats = InferenceStats(
                total_ms=elapsed_ms,
                forward_ms=elapsed_ms,
                batch_size=len(observations),
                device=self.device,
            )
            chunks.append(
                ActionChunk(
                    actions=arr[i],
                    timestamp=obs.timestamp,
                    stats=stats,
                    meta={"model": self.spec.name, "instruction": obs.instruction},
                )
            )
        return chunks

    # -- subclass hooks ------------------------------------------------------

    @abc.abstractmethod
    def _load(self) -> None:
        """Load weights onto ``self.device`` and populate ``self.model``."""

    @abc.abstractmethod
    def _predict_batch(self, observations: list[Observation]) -> np.ndarray:
        """Run the policy. Return ``[B, horizon, action_dim]`` in robot units."""

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "model": self.spec.name,
            "loaded": self._loaded,
            "device": self.device,
            "device_name": self._device_info.name,
            "precision": self.precision.summary(),
            "precision_notes": list(self.precision.notes),
            "action_dim": self.spec.action_dim,
            "horizon": self.config.horizon or self.spec.horizon,
            "cameras": list(self.spec.cameras),
            "requires_state": self.spec.requires_state,
            "load_seconds": round(self._load_seconds, 2),
            "warmup_seconds": round(self._warmup_seconds, 2),
            "warmup_steps": self._warmup_steps,
            "predictions": self._predict_count,
        }
