"""Fixed-length greedy decoding for action-token VLAs.

Token-based VLAs (OpenVLA and its relatives) do not need general text
generation. They emit *exactly* ``action_dim`` tokens, greedily, with no EOS
search, no beam search, no stopping criteria, and no variable length. Running
``model.generate()`` for that pays for machinery this problem does not have:

1. **Logit processors and stopping criteria** run every step over a 32k
   vocabulary to answer a question whose answer is known in advance.
2. **A growing KV cache** reallocates and copies as the sequence extends,
   and its changing shapes force ``torch.compile`` to recompile and make CUDA
   graph capture impossible.
3. **A device-to-host sync per token** -- ``generate`` inspects the sampled id
   on the CPU to test stopping conditions. At 7 tokens that is 7 full pipeline
   stalls on a forward pass that is only a few tens of milliseconds.

This module fixes all three: a preallocated static KV cache sized once at load,
a fixed trip count, and token ids that never leave the GPU until the loop ends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DecodeSchedule", "plan_decode", "GreedyActionDecoder"]


@dataclass(frozen=True)
class DecodeSchedule:
    """Precomputed cache bookkeeping for one decode call.

    Separated from the torch code so the index arithmetic -- the part that is
    easy to get subtly wrong and produces garbage actions rather than a crash
    -- can be unit tested on any machine.
    """

    prompt_len: int
    num_tokens: int
    cache_len: int

    @property
    def prefill_positions(self) -> range:
        """Cache slots written by the prefill pass."""
        return range(0, self.prompt_len)

    @property
    def decode_positions(self) -> list[int]:
        """Cache slot each incremental decode step writes into.

        The first sampled token comes out of the prefill's last logit, so the
        decode loop runs ``num_tokens - 1`` times, starting at ``prompt_len``.
        """
        return list(range(self.prompt_len, self.prompt_len + self.num_tokens - 1))

    @property
    def num_forward_passes(self) -> int:
        return self.num_tokens  # 1 prefill + (num_tokens - 1) decode steps

    @property
    def final_len(self) -> int:
        return self.prompt_len + self.num_tokens - 1


def plan_decode(prompt_len: int, num_tokens: int, max_cache_len: int) -> DecodeSchedule:
    """Validate lengths and produce the decode schedule.

    Raises:
        ValueError: if the prompt plus generated tokens overflow the static
            cache. This is a hard error rather than a silent truncation: a
            truncated action vector would still have the right shape.
    """
    if prompt_len < 1:
        raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    needed = prompt_len + num_tokens - 1
    if needed > max_cache_len:
        raise ValueError(
            f"static KV cache too small: prompt ({prompt_len}) + generated "
            f"({num_tokens - 1}) = {needed} exceeds max_seq_len={max_cache_len}. "
            f"Raise DecodeConfig.max_seq_len to at least {needed}."
        )
    return DecodeSchedule(prompt_len=prompt_len, num_tokens=num_tokens, cache_len=max_cache_len)


class GreedyActionDecoder:
    """Decode a fixed number of action tokens with a preallocated KV cache.

    Args:
        model: A HuggingFace causal LM (or VLM) whose ``forward`` accepts
            ``past_key_values``, ``use_cache``, and ``cache_position``.
        num_tokens: Exact number of tokens to emit (OpenVLA: ``action_dim``).
        max_seq_len: Static cache capacity. Must cover the longest prompt,
            including vision patch embeddings.

    Example:
        >>> decoder = GreedyActionDecoder(vlm, num_tokens=7, max_seq_len=512)
        >>> token_ids = decoder.decode(input_ids, pixel_values=pixels)
        >>> token_ids.shape
        torch.Size([1, 7])
    """

    def __init__(
        self,
        model: Any,
        num_tokens: int,
        *,
        max_seq_len: int = 512,
        use_static_cache: bool = True,
        max_batch_size: int = 1,
    ) -> None:
        self.model = model
        self.num_tokens = int(num_tokens)
        self.max_seq_len = int(max_seq_len)
        self.use_static_cache = use_static_cache
        self.max_batch_size = int(max_batch_size)
        self._cache = None
        self._cache_batch_size = 0

    # -- cache management ---------------------------------------------------

    def _build_cache(self, batch_size: int, device, dtype):
        """Create (or reuse) a StaticCache sized for this batch.

        The cache is allocated once and reset in place between calls. Reusing
        the same tensors is what keeps input addresses stable across CUDA graph
        replays; reallocating per call would invalidate the captured graph.
        """
        if not self.use_static_cache:
            return None
        if self._cache is not None and self._cache_batch_size >= batch_size:
            self._reset_cache()
            return self._cache
        try:
            from transformers import StaticCache
        except Exception:
            logger.info("transformers.StaticCache unavailable; using the model's default cache")
            self.use_static_cache = False
            return None
        try:
            config = getattr(self.model, "config", None)
            # VLM wrappers keep the decoder config one level down.
            text_config = getattr(config, "text_config", None) or config
            self._cache = StaticCache(
                config=text_config,
                max_batch_size=batch_size,
                max_cache_len=self.max_seq_len,
                device=device,
                dtype=dtype,
            )
            self._cache_batch_size = batch_size
        except Exception as exc:  # pragma: no cover - transformers version drift
            logger.warning("failed to allocate StaticCache (%s); falling back to dynamic", exc)
            self.use_static_cache = False
            self._cache = None
        return self._cache

    def _reset_cache(self) -> None:
        if self._cache is None:
            return
        reset = getattr(self._cache, "reset", None)
        if callable(reset):
            reset()

    # -- decoding -----------------------------------------------------------

    def decode(self, input_ids, attention_mask=None, **prefill_kwargs):
        """Greedily emit ``num_tokens`` ids.

        Args:
            input_ids: ``[B, L]`` prompt token ids.
            attention_mask: ``[B, L]`` mask, or ``None`` for "attend to all".
            **prefill_kwargs: Extra model inputs consumed only by the prefill
                pass (``pixel_values`` and friends). Vision features enter the
                KV cache during prefill, so re-sending them per decode step
                would recompute the vision tower ``num_tokens`` times.

        Returns:
            ``[B, num_tokens]`` LongTensor on the model's device.
        """
        import torch

        batch_size, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        schedule = plan_decode(prompt_len, self.num_tokens, self.max_seq_len)
        device = input_ids.device
        dtype = getattr(self.model, "dtype", None) or torch.float32

        cache = self._build_cache(batch_size, device, dtype)
        out = torch.empty((batch_size, self.num_tokens), dtype=torch.long, device=device)

        with torch.inference_mode():
            cache_position = torch.arange(prompt_len, device=device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position if cache is not None else None,
                **prefill_kwargs,
            )
            past = outputs.past_key_values
            # A VLM may expand the sequence with image patches, so the true
            # post-prefill length comes from the logits, not from prompt_len.
            realized_len = int(outputs.logits.shape[1])
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            out[:, 0] = next_token

            # If the vision tower expanded the prompt, re-validate capacity
            # before stepping, so we fail loudly instead of writing past the end.
            if cache is not None and realized_len > prompt_len:
                schedule = plan_decode(realized_len, self.num_tokens, self.max_seq_len)

            position = schedule.prompt_len
            for step in range(1, self.num_tokens):
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device),
                        ],
                        dim=-1,
                    )
                outputs = self.model(
                    input_ids=next_token[:, None],
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=torch.tensor([position], device=device)
                    if cache is not None
                    else None,
                )
                past = outputs.past_key_values
                # argmax stays on device: no host sync inside the loop.
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                out[:, step] = next_token
                position += 1

        return out
