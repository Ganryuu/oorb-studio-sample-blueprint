"""Shared adapter for LeRobot flow-matching policies (pi0, SmolVLA).

Both policies share an architecture and an interface: a VLM backbone produces a
conditioning embedding, and a small "action expert" turns noise into a 50-step
action chunk by integrating a flow-matching ODE for a fixed number of steps.

Two things follow from that, and they drive this implementation:

* **Latency is proportional to the denoising step count.** The action expert
  runs once per step while the VLM backbone runs once per *observation* -- its
  KV cache is reused across all denoising steps. So halving ``num_denoise_steps``
  nearly halves latency. That knob matters more than precision or attention
  backend, and it is exposed directly.
* **Output is already in robot units.** LeRobot bundles dataset statistics in
  the checkpoint and applies un-normalization inside the policy. Applying our
  own :class:`~vla_engine.control.normalize.Normalizer` on top would double the
  transform, so this adapter deliberately does not.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import EngineConfig
from ..errors import DependencyError, ObservationError
from ..registry import resolve
from ..runtime.compile import maybe_compile
from ..runtime.imageproc import resize_with_padding, stack_uint8, to_device_chw_float
from ..types import Observation, PolicySpec
from .base import VLAAdapter

logger = logging.getLogger(__name__)

__all__ = ["LeRobotFlowAdapter"]


class LeRobotFlowAdapter(VLAAdapter):
    """Base for LeRobot policies exposing ``predict_action_chunk``.

    Subclasses set :attr:`policy_import` to the ``"module:Class"`` path of the
    LeRobot policy and may override :meth:`camera_feature_key`.
    """

    #: ``"module:ClassName"`` of the LeRobot policy implementation.
    policy_import: str = ""
    #: Prefix LeRobot uses for camera features in its batch dict.
    image_key_prefix: str = "observation.images"
    state_key: str = "observation.state"
    task_key: str = "task"

    def __init__(self, config: EngineConfig) -> None:
        card, checkpoint = resolve(config.model, config.checkpoint)
        self.card = card
        self.checkpoint = checkpoint
        self.spec: PolicySpec = card.spec
        extra = config.extra or {}
        self.num_denoise_steps = int(extra.get("num_denoise_steps", 10))
        if self.num_denoise_steps < 1:
            raise ObservationError("num_denoise_steps must be >= 1")
        self._camera_keys: list[str] = []
        super().__init__(config)

    # -- loading -------------------------------------------------------------

    def _import_policy_class(self) -> type:
        module_path, _, class_name = self.policy_import.partition(":")
        try:
            import importlib

            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise DependencyError(
                "lerobot", f"{self.spec.name} (needs {module_path})", extra="vla-engine[lerobot]"
            ) from exc
        return getattr(module, class_name)

    def _load(self) -> None:
        import torch

        policy_cls = self._import_policy_class()
        policy = policy_cls.from_pretrained(self.checkpoint)
        policy = policy.to(self.device, dtype=self.precision.torch_dtype())
        policy.eval()
        for param in policy.parameters():
            param.requires_grad_(False)

        self._apply_denoise_steps(policy)
        self.model = policy
        self._camera_keys = self._discover_camera_keys(policy)
        logger.info("%s camera features: %s", self.spec.name, self._camera_keys)

        result = maybe_compile(
            policy, self.config.compile, device=self.device, label=self.spec.name
        )
        if result.compiled:
            self.model = result.module
        logger.info("%s: %s", self.spec.name, result.summary())

    def _apply_denoise_steps(self, policy) -> None:
        """Set the flow-matching step count on whichever attribute exists.

        LeRobot has spelled this ``num_steps`` and ``num_inference_steps``
        across versions, on the policy and on its config, so set every name that
        is already present and never create a new one.
        """
        targets = [policy, getattr(policy, "config", None)]
        applied = []
        for target in targets:
            if target is None:
                continue
            for attr in ("num_steps", "num_inference_steps", "num_denoise_steps"):
                if hasattr(target, attr):
                    try:
                        setattr(target, attr, self.num_denoise_steps)
                        applied.append(f"{type(target).__name__}.{attr}")
                    except Exception:
                        pass
        if applied:
            logger.info("set denoising steps=%d on %s", self.num_denoise_steps, applied)
        else:
            logger.warning(
                "could not find a denoising-step attribute on %s; using the "
                "checkpoint default",
                type(policy).__name__,
            )

    def _discover_camera_keys(self, policy) -> list[str]:
        """Read expected image feature keys from the policy config.

        The checkpoint knows which cameras it was trained with; guessing names
        would silently feed a wrist policy its scene camera.
        """
        config = getattr(policy, "config", None)
        features = getattr(config, "input_features", None) if config else None
        if isinstance(features, dict):
            keys = [k for k in features if k.startswith(self.image_key_prefix)]
            if keys:
                return sorted(keys)
        # Fall back to the registry spec's camera names.
        return [f"{self.image_key_prefix}.{cam}" for cam in self.spec.cameras]

    # -- inference -----------------------------------------------------------

    def _predict_batch(self, observations: list[Observation]) -> np.ndarray:
        import torch

        batch = self._build_batch(observations)
        with torch.inference_mode():
            chunk = self._run_policy(batch)
        actions = chunk.detach().to(torch.float32).cpu().numpy()
        if actions.ndim == 2:  # [B, D] -> single-step
            actions = actions[:, None, :]
        return actions

    def _run_policy(self, batch: dict):
        """Call the policy's chunk API, falling back across LeRobot versions."""
        predict_chunk = getattr(self.model, "predict_action_chunk", None)
        if callable(predict_chunk):
            return predict_chunk(batch)
        select_action = getattr(self.model, "select_action", None)
        if callable(select_action):
            # Older LeRobot exposes only a queue-backed single-step API. That
            # returns one action per call, so the chunk horizon collapses to 1
            # and the chunk executor simply has less to work with.
            logger.debug("%s: falling back to select_action (no chunk API)", self.spec.name)
            return select_action(batch)
        raise DependencyError(
            "lerobot",
            f"{type(self.model).__name__} exposes neither predict_action_chunk nor select_action",
            extra="vla-engine[lerobot]",
        )

    def _build_batch(self, observations: list[Observation]) -> dict:
        """Assemble LeRobot's batch dict, uploading images as uint8."""
        import torch

        batch: dict = {}
        target_size = self.spec.image_size
        dtype = self.precision.torch_dtype()

        for slot, key in enumerate(self._camera_keys):
            cam_name = self.spec.cameras[min(slot, len(self.spec.cameras) - 1)]
            frames = []
            for obs in observations:
                resolved = self.spec.resolve_cameras(obs.images)
                frame = resolved.get(cam_name, resolved[self.spec.cameras[0]])
                frames.append(resize_with_padding(frame, target_size))
            batch[key] = to_device_chw_float(stack_uint8(frames), self.device, dtype=dtype)

        states = [self._pad_state(obs) for obs in observations]
        batch[self.state_key] = torch.from_numpy(np.stack(states)).to(self.device, dtype=dtype)
        batch[self.task_key] = [obs.instruction for obs in observations]
        return batch

    def _pad_state(self, obs: Observation) -> np.ndarray:
        """Zero-pad proprioception up to the policy's padded state width.

        pi0 and SmolVLA are trained across robots with differing DoF by padding
        every state and action vector to a fixed width (32). A 7-DoF arm
        therefore supplies 7 real values and 25 zeros.
        """
        width = self.spec.state_dim or self.spec.action_dim
        state = obs.state if obs.state is not None else np.zeros(0, dtype=np.float32)
        if state.shape[0] > width:
            raise ObservationError(
                f"state has {state.shape[0]} dims but {self.spec.name} pads to {width}"
            )
        out = np.zeros(width, dtype=np.float32)
        out[: state.shape[0]] = state
        return out

    def info(self) -> dict:
        data = super().info()
        data.update(
            {
                "checkpoint": self.checkpoint,
                "num_denoise_steps": self.num_denoise_steps,
                "camera_features": list(self._camera_keys),
            }
        )
        return data
