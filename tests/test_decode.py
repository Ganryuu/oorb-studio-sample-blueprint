"""Action-token decoding and KV-cache scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from vla_engine.adapters.openvla import N_ACTION_BINS, OPENVLA_PROMPT, decode_action_tokens
from vla_engine.runtime.decode import plan_decode

VOCAB = 32000


class TestDecodeSchedule:
    def test_seven_action_tokens_take_seven_forward_passes(self):
        schedule = plan_decode(prompt_len=300, num_tokens=7, max_cache_len=512)
        assert schedule.num_forward_passes == 7
        assert schedule.decode_positions == [300, 301, 302, 303, 304, 305]

    def test_single_token_needs_no_decode_steps(self):
        assert plan_decode(10, 1, 512).decode_positions == []

    def test_overflowing_the_static_cache_raises(self):
        """Silent truncation would still produce a correctly-shaped action vector."""
        with pytest.raises(ValueError, match="too small"):
            plan_decode(prompt_len=510, num_tokens=7, max_cache_len=512)

    def test_error_names_the_required_size(self):
        with pytest.raises(ValueError, match="at least 516"):
            plan_decode(prompt_len=510, num_tokens=7, max_cache_len=512)

    @pytest.mark.parametrize("prompt_len,num_tokens", [(0, 7), (10, 0)])
    def test_rejects_degenerate_lengths(self, prompt_len, num_tokens):
        with pytest.raises(ValueError):
            plan_decode(prompt_len, num_tokens, 512)


class TestActionDetokenization:
    def test_matches_reference_bin_mapping(self):
        """Action tokens occupy the top of the vocabulary, highest id = most negative."""
        assert decode_action_tokens(np.array([[VOCAB - 1]]), VOCAB)[0, 0] == pytest.approx(
            -0.9961, abs=1e-4
        )
        assert decode_action_tokens(np.array([[VOCAB - N_ACTION_BINS]]), VOCAB)[
            0, 0
        ] == pytest.approx(0.9961, abs=1e-4)

    def test_is_monotonically_decreasing_in_token_id(self):
        values = decode_action_tokens(np.arange(VOCAB - 255, VOCAB), VOCAB)
        assert np.all(np.diff(values) <= 0)

    def test_stays_within_normalized_range(self):
        values = decode_action_tokens(np.arange(VOCAB - 300, VOCAB + 10), VOCAB)
        assert values.min() >= -1.0 and values.max() <= 1.0

    def test_preserves_shape_for_a_full_action(self):
        tokens = np.array(
            [
                [
                    VOCAB - 1,
                    VOCAB - 40,
                    VOCAB - 80,
                    VOCAB - 128,
                    VOCAB - 180,
                    VOCAB - 220,
                    VOCAB - 255,
                ]
            ]
        )
        assert decode_action_tokens(tokens, VOCAB).shape == (1, 7)

    def test_out_of_range_ids_clamp_rather_than_index_error(self):
        assert np.isfinite(decode_action_tokens(np.array([[0, VOCAB + 500]]), VOCAB)).all()

    def test_refuses_already_decoded_actions(self):
        """Regression: feeding decoded floats back through detokenization used to
        collapse every action into one bin, silently and catastrophically."""
        from vla_engine.errors import ObservationError

        with pytest.raises(ObservationError, match="integer token ids"):
            decode_action_tokens(np.array([[0.5, -0.3]], np.float32), VOCAB)

    def test_prompt_format_is_exact(self):
        """OpenVLA was trained on this exact string; drift degrades actions silently."""
        assert (
            OPENVLA_PROMPT.format(instruction="pick up the red block")
            == "In: What action should the robot take to pick up the red block?\nOut:"
        )
