"""A weightless reference policy for testing the whole pipeline.

This is not a mock in the "returns a constant" sense. It implements the full
adapter contract with deterministic, observation-dependent output and an
optional simulated forward-pass latency, which is enough to exercise every
component around the model: batching, replica routing, the HTTP and websocket
server, chunk scheduling, the ROS 2 node, and the benchmark harness.

Being able to run all of that in CI, on a laptop, and inside the OORB Studio
container -- none of which have a GPU -- is what keeps the surrounding
machinery honest. Set ``latency_ms`` in ``EngineConfig.extra`` to imitate a real
policy's timing when testing control-loop behavior under load.

Example:
    >>> cfg = EngineConfig(model="echo", extra={"latency_ms": 80.0, "horizon": 50})
    >>> engine = VLAEngine(cfg).load()
    >>> engine.predict(obs).actions.shape
    (50, 7)
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from ..config import EngineConfig
from ..types import Observation, PolicySpec
from .base import VLAAdapter

__all__ = ["EchoAdapter"]


class EchoAdapter(VLAAdapter):
    """Deterministic, weightless policy.

    Recognized ``EngineConfig.extra`` keys:
        ``latency_ms``: Simulated forward-pass time (default 0).
        ``horizon``: Action-chunk length (default 1).
        ``action_dim``: Output dimensionality (default 7).
        ``scale``: Peak absolute action magnitude (default 0.1).
    """

    def __init__(self, config: EngineConfig) -> None:
        extra = config.extra or {}
        self.latency_ms = float(extra.get("latency_ms", 0.0))
        self.scale = float(extra.get("scale", 0.1))
        horizon = int(extra.get("horizon", 1))
        action_dim = int(extra.get("action_dim", 7))
        self.spec = PolicySpec(
            name="echo",
            action_dim=action_dim,
            horizon=horizon,
            cameras=("primary",),
            requires_state=False,
            language_conditioned=False,
            image_size=(224, 224),
            control_hz=10.0,
        )
        super().__init__(config)

    def _load(self) -> None:
        self.model = "echo"  # nothing to load; satisfies the contract

    def _predict_batch(self, observations: list[Observation]) -> np.ndarray:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        out = np.empty(
            (len(observations), self.spec.horizon, self.spec.action_dim), dtype=np.float32
        )
        for i, obs in enumerate(observations):
            out[i] = self._deterministic_chunk(obs)
        return out

    def _deterministic_chunk(self, obs: Observation) -> np.ndarray:
        """Smooth, bounded, reproducible actions keyed to the observation.

        Derived from a hash of the instruction and a cheap image digest, so the
        same observation always yields the same actions (making end-to-end
        assertions possible) while different observations differ.
        """
        digest = hashlib.sha256()
        digest.update(obs.instruction.encode("utf-8"))
        for name in sorted(obs.images):
            frame = obs.images[name]
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(frame.shape, dtype=np.int64).tobytes())
            # Subsample rather than hash megabytes of pixels.
            digest.update(np.ascontiguousarray(frame[::16, ::16]).tobytes())
        if obs.state is not None:
            digest.update(obs.state.tobytes())
        seed = int.from_bytes(digest.digest()[:8], "little") % (2**32)

        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi, size=self.spec.action_dim).astype(np.float32)
        steps = np.arange(self.spec.horizon, dtype=np.float32)[:, None]
        # Sinusoids give a chunk that varies smoothly over the horizon, which
        # makes temporal-ensembling behavior visible in tests.
        chunk = self.scale * np.sin(0.1 * steps + phase[None, :])
        return chunk.astype(np.float32)
