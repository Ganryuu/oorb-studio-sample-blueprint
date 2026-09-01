"""Chunk scheduling and action normalization."""

from __future__ import annotations

import numpy as np
import pytest

from vla_engine import ActionChunk
from vla_engine.control.chunker import ChunkExecutor, ChunkPolicy
from vla_engine.control.normalize import ActionStats, Normalizer


class TestChunkExecutor:
    def test_returns_none_before_any_chunk(self):
        assert ChunkExecutor(7).step() is None

    def test_sequential_plays_chunk_in_order(self):
        executor = ChunkExecutor(2, ChunkPolicy(strategy="sequential"))
        executor.submit(ActionChunk(np.arange(6, dtype=np.float32).reshape(3, 2)))
        assert [executor.step().tolist() for _ in range(3)] == [[0, 1], [2, 3], [4, 5]]

    def test_uniform_ensembling_averages_overlapping_chunks(self):
        executor = ChunkExecutor(2, ChunkPolicy(ensemble_decay=0.0))
        executor.submit(np.zeros((5, 2), np.float32))
        executor.step()
        executor.submit(np.ones((5, 2), np.float32))
        assert executor.step().tolist() == [0.5, 0.5]

    def test_decay_biases_toward_the_fresh_chunk(self):
        executor = ChunkExecutor(2, ChunkPolicy(ensemble_decay=5.0))
        executor.submit(np.zeros((5, 2), np.float32))
        executor.step()
        executor.submit(np.ones((5, 2), np.float32))
        assert executor.step()[0] > 0.9

    def test_discrete_dims_are_never_blended(self):
        """A gripper averaged to 0.5 is a physically meaningless command."""
        executor = ChunkExecutor(2, ChunkPolicy(ensemble_decay=0.0, discrete_dims=(1,)))
        executor.submit(np.zeros((5, 2), np.float32))
        executor.step()
        executor.submit(np.ones((5, 2), np.float32))
        action = executor.step()
        assert action[0] == pytest.approx(0.5)
        assert action[1] == 1.0

    def test_starvation_holds_last_action(self):
        executor = ChunkExecutor(2)
        executor.submit(np.full((2, 2), 7.0, np.float32))
        for _ in range(2):
            executor.step()
        assert executor.step().tolist() == [7.0, 7.0]
        assert executor.starved_steps == 1

    def test_needs_replan_at_half_horizon(self):
        executor = ChunkExecutor(2)
        executor.submit(np.zeros((10, 2), np.float32))
        for _ in range(4):
            executor.step()
        assert executor.needs_replan() is False
        executor.step()
        assert executor.needs_replan() is True

    def test_explicit_replan_after_overrides_default(self):
        executor = ChunkExecutor(2, ChunkPolicy(replan_after=2))
        executor.submit(np.zeros((10, 2), np.float32))
        executor.step()
        executor.step()
        assert executor.needs_replan() is True

    def test_clipping_bounds_actions(self):
        executor = ChunkExecutor(2, ChunkPolicy(clip=([-0.5, -0.5], [0.5, 0.5])))
        executor.submit(np.full((3, 2), 9.0, np.float32))
        assert executor.step().tolist() == [0.5, 0.5]

    def test_max_chunk_age_excludes_stale_votes(self):
        executor = ChunkExecutor(2, ChunkPolicy(ensemble_decay=0.0, max_chunk_age=0))
        executor.submit(np.zeros((5, 2), np.float32))
        executor.step()
        executor.submit(np.ones((5, 2), np.float32))
        assert executor.step().tolist() == [1.0, 1.0]

    def test_reset_clears_state(self):
        executor = ChunkExecutor(2)
        executor.submit(np.ones((5, 2), np.float32))
        executor.step()
        executor.reset()
        assert executor.pending_chunks == 0
        assert executor.step_index == 0
        assert executor.step() is None

    def test_rejects_wrong_action_dim(self):
        with pytest.raises(ValueError, match="action_dim"):
            ChunkExecutor(2).submit(np.zeros((3, 5), np.float32))

    def test_exhausted_chunks_are_evicted(self):
        executor = ChunkExecutor(2)
        executor.submit(np.zeros((2, 2), np.float32))
        for _ in range(4):
            executor.step()
        assert executor.pending_chunks == 0

    def test_slow_policy_still_yields_smooth_control(self):
        """A 5 Hz policy driving a 50 Hz loop must never starve."""
        executor = ChunkExecutor(1, ChunkPolicy(ensemble_decay=0.1))
        for inference in range(5):
            executor.submit(np.full((50, 1), float(inference), np.float32))
            for _ in range(10):  # 10 control ticks per inference
                assert executor.step() is not None
        assert executor.starved_steps == 0


class TestNormalizer:
    def test_percentile_roundtrip(self):
        stats = ActionStats(q01=[-1.0, -2.0], q99=[1.0, 2.0])
        normalizer = Normalizer(stats)
        actions = np.array([[0.3, -0.7]], np.float32)
        assert np.allclose(
            normalizer.normalize(normalizer.unnormalize(actions)), actions, atol=1e-5
        )

    def test_masked_dims_pass_through(self):
        normalizer = Normalizer(ActionStats(q01=[-1.0, 0.0], q99=[1.0, 1.0], mask=[True, False]))
        assert normalizer.unnormalize(np.array([[0.0, 0.7]], np.float32))[0, 1] == pytest.approx(
            0.7
        )

    def test_out_of_range_is_clamped_to_dataset_bounds(self):
        """An out-of-range token must not command a motion larger than any seen in training."""
        normalizer = Normalizer(ActionStats(q01=[-1.0], q99=[1.0]))
        assert normalizer.unnormalize(np.array([[50.0]], np.float32))[0, 0] == pytest.approx(1.0)

    def test_mean_std_scheme(self):
        normalizer = Normalizer(ActionStats(mean=[1.0], std=[2.0], scheme="mean_std"))
        assert normalizer.unnormalize(np.array([[1.0]], np.float32))[0, 0] == pytest.approx(3.0)

    def test_stats_pad_to_a_wider_action_space(self):
        """pi0 pads actions to 32 dims; a 7-DoF robot only has 7 dims of stats."""
        normalizer = Normalizer(ActionStats(q01=[-1.0] * 7, q99=[1.0] * 7))
        assert normalizer.unnormalize(np.zeros((1, 32), np.float32)).shape == (1, 32)

    def test_zero_std_does_not_divide_by_zero(self):
        normalizer = Normalizer(ActionStats(mean=[0.0], std=[0.0], scheme="mean_std"))
        assert np.isfinite(normalizer.normalize(np.array([[1.0]], np.float32))).all()

    def test_parses_nested_checkpoint_stats(self):
        stats = ActionStats.from_dict({"action": {"q01": [-1, -1], "q99": [1, 1]}})
        assert stats.scheme == "percentile" and stats.action_dim == 2
