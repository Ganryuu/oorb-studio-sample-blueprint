"""OpenVLA-7B adapter.

OpenVLA is a Prismatic VLM: a Llama-2-7B decoder over a fused DINOv2 + SigLIP
vision tower. It does not have an action head. Instead it *writes* the action
as text: seven tokens taken from the 256 least-used entries of the Llama
vocabulary, each token indexing a uniform bin over the normalized range
``[-1, 1]``. Those bins are then mapped to robot units with per-dataset q01/q99
statistics chosen by ``unnorm_key``.

That design has one consequence that dominates inference performance: producing
an action is seven sequential forward passes over a 7B model. Throughput is
irrelevant; what matters is per-pass overhead. The optimizations applied here,
in descending order of impact on a 3090:

1. **Fixed-trip greedy decode over a static KV cache**
   (:mod:`vla_engine.runtime.decode`), replacing ``generate()``. Removes the
   per-token host sync and gives CUDA graphs the static shapes they need.
2. **CUDA graphs via ``torch.compile(mode="reduce-overhead")``**, which
   collapses thousands of small kernel launches per pass into one replay.
3. **bf16 + TF32 + FlashAttention-2**, all natively supported on sm_86.
4. **NF4 weight quantization** (opt-in): ~15 GB -> ~5 GB, which is what lets a
   single 3090 host OpenVLA alongside another policy, at some latency cost.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import EngineConfig
from ..control.normalize import ActionStats, Normalizer
from ..errors import ConfigError, DependencyError
from ..registry import resolve
from ..runtime.compile import maybe_compile
from ..runtime.decode import GreedyActionDecoder
from ..types import Observation, PolicySpec
from .base import VLAAdapter

logger = logging.getLogger(__name__)

__all__ = ["OpenVLAAdapter", "decode_action_tokens", "OPENVLA_PROMPT"]

#: OpenVLA was trained with this exact prompt. Deviating from it -- different
#: capitalization, a missing "Out:" -- degrades actions without any error.
OPENVLA_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"

N_ACTION_BINS = 256


def decode_action_tokens(
    token_ids: np.ndarray, vocab_size: int, n_bins: int = N_ACTION_BINS
) -> np.ndarray:
    """Map OpenVLA action token ids to normalized actions in ``[-1, 1]``.

    Mirrors ``prismatic.vla.action_tokenizer.ActionTokenizer``: action tokens
    occupy the top of the vocabulary, so the bin index is ``vocab_size - id``,
    shifted by one and clamped into the bin-center array.

    Args:
        token_ids: ``[..., action_dim]`` integer token ids.
        vocab_size: The tokenizer's vocabulary size (32000 for Llama-2).
        n_bins: Number of discretization bins.

    Returns:
        Float array of the same shape, in normalized action space.
    """
    bins = np.linspace(-1.0, 1.0, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    indices = vocab_size - np.asarray(token_ids, dtype=np.int64)
    indices = np.clip(indices - 1, 0, bin_centers.shape[0] - 1)
    return bin_centers[indices].astype(np.float32)


class OpenVLAAdapter(VLAAdapter):
    """Adapter for ``openvla/openvla-7b`` and its fine-tunes."""

    def __init__(self, config: EngineConfig) -> None:
        card, checkpoint = resolve(config.model, config.checkpoint)
        self.checkpoint = checkpoint
        self.card = card
        self.spec: PolicySpec = card.spec
        self.unnorm_key = config.unnorm_key or card.default_unnorm_key
        self.processor = None
        self._decoder: GreedyActionDecoder | None = None
        self._normalizer: Normalizer | None = None
        self._vocab_size = 32000
        super().__init__(config)

    # -- loading -------------------------------------------------------------

    def _load(self) -> None:
        try:
            # torch is imported explicitly, not just transitively: transformers
            # installs fine without it, and the resulting failure deep inside
            # from_pretrained is far less clear than this one.
            import torch  # noqa: F401
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise DependencyError(
                exc.name or "transformers", "OpenVLA", extra="vla-engine[cuda]"
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=self.config.trust_remote_code
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            self._vocab_size = int(getattr(tokenizer, "vocab_size", self._vocab_size))

        load_kwargs: dict = {
            "torch_dtype": self.precision.torch_dtype(),
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.config.trust_remote_code,
        }
        # OpenVLA's remote code forwards attn_implementation to the LM.
        if self.precision.attention != "eager":
            load_kwargs["attn_implementation"] = self.precision.attention
        quant_config = self.precision.quantization_config()
        if quant_config is not None:
            # bitsandbytes places the model itself; an explicit .to() afterwards
            # would try to move quantized buffers and fail.
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = {"": self.device}

        model = AutoModelForVision2Seq.from_pretrained(self.checkpoint, **load_kwargs)
        if quant_config is None:
            model = model.to(self.device)
        model.eval()
        # Inference only: no graph bookkeeping, no gradient buffers.
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model
        self._resolve_norm_stats()

        if self.config.decode.static_kv_cache:
            self._decoder = GreedyActionDecoder(
                model,
                num_tokens=self.spec.action_dim,
                max_seq_len=self.config.decode.max_seq_len,
                max_batch_size=self.config.batch.max_batch_size,
            )

        result = maybe_compile(model, self.config.compile, device=self.device, label="openvla")
        if result.compiled:
            self.model = result.module
            if self._decoder is not None:
                self._decoder.model = result.module
        logger.info("openvla: %s", result.summary())

    def _resolve_norm_stats(self) -> None:
        """Pick the dataset statistics used to un-normalize actions.

        A wrong ``unnorm_key`` is the most common OpenVLA deployment mistake: it
        produces well-formed actions at the wrong scale, so the arm moves
        smoothly and incorrectly. When the key is ambiguous we refuse rather
        than guess.
        """
        norm_stats = getattr(self.model, "norm_stats", None)
        if not norm_stats:
            logger.warning(
                "checkpoint %s exposes no norm_stats; actions will be returned "
                "in normalized [-1, 1] space",
                self.checkpoint,
            )
            self._normalizer = Normalizer(ActionStats.identity(self.spec.action_dim))
            return

        available = sorted(norm_stats.keys())
        key = self.unnorm_key
        if key is None:
            if len(available) == 1:
                key = available[0]
                logger.info("using the checkpoint's only unnorm_key: %r", key)
            else:
                raise ConfigError(
                    f"checkpoint {self.checkpoint} provides {len(available)} dataset "
                    f"statistics and no unnorm_key was set. Pass one of: {available}"
                )
        elif key not in norm_stats:
            raise ConfigError(
                f"unnorm_key {key!r} not in checkpoint {self.checkpoint}. Available: {available}"
            )

        self.unnorm_key = key
        stats = ActionStats.from_dict(norm_stats[key])
        if stats.action_dim != self.spec.action_dim:
            logger.warning(
                "unnorm_key %r has %d action dims but the policy emits %d",
                key,
                stats.action_dim,
                self.spec.action_dim,
            )
        self._normalizer = Normalizer(stats)

    # -- inference -----------------------------------------------------------

    def _predict_batch(self, observations: list[Observation]) -> np.ndarray:

        inputs = self._prepare_inputs(observations)
        if self._decoder is not None:
            token_ids = self._decoder.decode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
            )
            tokens = token_ids.detach().cpu().numpy()
        else:
            tokens = self._fallback_generate(inputs)

        normalized = decode_action_tokens(tokens, self._vocab_size)
        actions = self._normalizer.unnormalize(normalized)  # type: ignore[union-attr]
        return actions[:, None, :]  # [B, 1, action_dim]: OpenVLA is single-step

    def _prepare_inputs(self, observations: list[Observation]) -> dict:
        """Tokenize prompts and preprocess images into a single batch."""
        import torch
        from PIL import Image

        prompts, frames = [], []
        for obs in observations:
            resolved = self.spec.resolve_cameras(obs.images)
            frames.append(Image.fromarray(resolved[self.spec.cameras[0]]))
            prompts.append(OPENVLA_PROMPT.format(instruction=obs.instruction.lower().strip()))

        inputs = self.processor(prompts, frames, return_tensors="pt", padding=True)
        dtype = self.precision.torch_dtype()
        prepared = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                continue
            if torch.is_floating_point(value):
                prepared[key] = value.to(self.device, dtype=dtype, non_blocking=True)
            else:
                prepared[key] = value.to(self.device, non_blocking=True)
        return prepared

    def _fallback_generate(self, inputs: dict) -> np.ndarray:
        """Path used when the static-cache decoder is disabled.

        Prefers the checkpoint's own ``predict_action`` so behavior matches the
        upstream reference implementation exactly.
        """
        import torch

        with torch.inference_mode():
            predict_action = getattr(self.model, "predict_action", None)
            if callable(predict_action) and self.unnorm_key:
                actions = predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
                arr = np.atleast_2d(np.asarray(actions, dtype=np.float32))
                # Already in robot units; invert so the shared un-normalize
                # below is a no-op round trip rather than a second application.
                return self._normalizer.normalize(arr)  # type: ignore[union-attr]
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.spec.action_dim,
                min_new_tokens=self.spec.action_dim,
                do_sample=False,
            )
        return generated[:, -self.spec.action_dim :].detach().cpu().numpy()

    def info(self) -> dict:
        data = super().info()
        data.update({"checkpoint": self.checkpoint, "unnorm_key": self.unnorm_key})
        return data
