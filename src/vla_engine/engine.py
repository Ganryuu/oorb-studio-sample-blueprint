"""The public entry point: :class:`VLAEngine`.

One class hides the differences between a 7B token-decoding VLA and a 450M
flow-matching policy, between one GPU and two, and between a local model and a
remote server. Everything below it -- adapters, replicas, batching -- is
reachable but rarely needed directly.

Example:
    >>> engine = VLAEngine.from_pretrained("openvla", unnorm_key="bridge_orig")
    >>> chunk = engine.predict(Observation.single(frame, "pick up the red block"))
    >>> chunk.actions.shape
    (1, 7)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .config import EngineConfig
from .control.chunker import ChunkExecutor, ChunkPolicy
from .errors import NotLoadedError
from .registry import get_card, resolve
from .runtime.device import probe_devices
from .types import ActionChunk, Observation, PolicySpec

logger = logging.getLogger(__name__)

__all__ = ["VLAEngine"]


class VLAEngine:
    """Loads a VLA policy and serves action predictions.

    Args:
        config: An :class:`~vla_engine.config.EngineConfig`, or a model name to
            load with default settings.

    Attributes:
        spec: The policy's input/output contract, available before loading.
    """

    def __init__(self, config: EngineConfig | str) -> None:
        if isinstance(config, str):
            config = EngineConfig(model=config)
        self.config = config
        self.card = get_card(config.model)
        self.spec: PolicySpec = self.card.spec
        self._adapter: Any = None
        self._pool: Any = None
        self._loaded = False

    # -- construction --------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> VLAEngine:
        """Build and load an engine in one call.

        Keyword arguments are forwarded to :class:`EngineConfig`, so
        ``dtype=`` and ``quantization=`` are also accepted as shorthands for
        their nested precision fields.
        """
        precision_keys = {"dtype", "quantization", "attention", "tf32"}
        precision = {k: kwargs.pop(k) for k in list(kwargs) if k in precision_keys}
        if precision:
            existing = kwargs.get("precision", {})
            if not isinstance(existing, dict):
                existing = {}
            kwargs["precision"] = {**existing, **precision}
        config = EngineConfig(model=model, **kwargs)
        return cls(config).load()

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def devices(self) -> list[str]:
        """Devices this engine will place replicas on."""
        topology = probe_devices()
        if self.config.devices:
            return topology.replica_devices(self.config.devices)
        if self.config.device != "auto":
            return [self.config.device]
        return [topology.replica_devices()[0]]

    def load(self) -> VLAEngine:
        """Load weights. Uses a replica per device when several are configured."""
        if self._loaded:
            return self
        devices = self.devices
        adapter_cls = self.card.load_adapter_class()

        if len(devices) > 1:
            from .optim.replicas import ReplicaPool

            self._check_capacity(devices)

            def factory(device: str):
                config = EngineConfig.from_dict(
                    {**self.config.to_dict(), "device": device, "devices": []}
                )
                return adapter_cls(config)

            self._pool = ReplicaPool(factory, devices).load()
            self._adapter = self._pool._replicas[0]
        else:
            config = EngineConfig.from_dict(
                {**self.config.to_dict(), "device": devices[0], "devices": []}
            )
            self._adapter = adapter_cls(config).load()

        # The adapter may refine the registry's static spec once weights are
        # loaded (a fine-tune's real action_dim, a configured chunk horizon).
        self.spec = self._adapter.spec
        self._loaded = True
        return self

    def _check_capacity(self, devices: Sequence[str]) -> None:
        """Warn early if a replica will not fit on each requested GPU.

        Loading a 7B model onto a card that cannot hold it fails minutes in,
        after the download; saying so up front is cheaper.
        """
        topology = probe_devices()
        needed = self.card.memory_gb(
            self.config.precision.dtype if self.config.precision.dtype != "auto" else "bf16",
            self.config.precision.quantization,
        )
        for device in devices:
            info = next((d for d in topology.devices if d.torch_device == device), None)
            if info and info.kind == "cuda" and info.total_memory_gb < needed:
                logger.warning(
                    "%s needs ~%.1f GB per replica but %s has %.1f GB; "
                    "consider quantization='nf4' or fewer replicas",
                    self.card.key,
                    needed,
                    device,
                    info.total_memory_gb,
                )

    def unload(self) -> None:
        """Release all weights and device memory."""
        if self._pool is not None:
            self._pool.unload()
            self._pool = None
        elif self._adapter is not None:
            self._adapter.unload()
        self._adapter = None
        self._loaded = False

    def __enter__(self) -> VLAEngine:
        return self.load()

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    # -- inference -----------------------------------------------------------

    def predict(self, observation: Observation) -> ActionChunk:
        """Predict an action chunk for one observation."""
        self._require_loaded()
        if self._pool is not None:
            return self._pool.predict(observation)
        return self._adapter.predict(observation)

    def predict_batch(self, observations: Sequence[Observation]) -> list[ActionChunk]:
        """Predict for several observations, spread across replicas if present."""
        self._require_loaded()
        if self._pool is not None:
            return self._pool.predict_batch(observations)
        return self._adapter.predict_batch(observations)

    def act(self, image: Any, instruction: str, state: Any | None = None) -> ActionChunk:
        """Shorthand for the single-camera case.

        Example:
            >>> engine.act(frame, "put the spoon in the drawer").actions[0]
        """
        return self.predict(Observation.single(image, instruction, state))

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise NotLoadedError("engine is not loaded; call .load() first")

    # -- control helpers -----------------------------------------------------

    def make_executor(
        self, policy: ChunkPolicy | None = None, action_dim: int | None = None
    ) -> ChunkExecutor:
        """Create a :class:`ChunkExecutor` matched to this policy.

        Defaults to blending overlapping chunks and treating the last dimension
        as discrete, which is the gripper for every policy here.
        """
        dim = action_dim or self.spec.action_dim
        if policy is None:
            policy = ChunkPolicy(
                strategy="temporal_ensemble",
                ensemble_decay=0.1,
                discrete_dims=(dim - 1,),
            )
        return ChunkExecutor(action_dim=dim, policy=policy)

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """Everything worth knowing about this engine's current state."""
        data: dict[str, Any] = {
            "model": self.card.key,
            "checkpoint": resolve(self.config.model, self.config.checkpoint)[1],
            "loaded": self._loaded,
            "devices": self.devices,
            "spec": {
                "action_dim": self.spec.action_dim,
                "horizon": self.config.horizon or self.spec.horizon,
                "cameras": list(self.spec.cameras),
                "requires_state": self.spec.requires_state,
                "image_size": list(self.spec.image_size),
            },
            "estimated_memory_gb": round(
                self.card.memory_gb(
                    self.config.precision.dtype
                    if self.config.precision.dtype != "auto"
                    else "bf16",
                    self.config.precision.quantization,
                ),
                2,
            ),
        }
        if self._pool is not None:
            data["pool"] = self._pool.info()
        elif self._adapter is not None:
            data["adapter"] = self._adapter.info()
        return data

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "not loaded"
        return f"<VLAEngine model={self.card.key!r} devices={self.devices} {state}>"
