"""End-to-end engine, adapter, and optimization behavior."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vla_engine import EngineConfig, Observation, VLAEngine
from vla_engine.adapters.echo import EchoAdapter
from vla_engine.config import BatchConfig
from vla_engine.errors import NotLoadedError, ObservationError
from vla_engine.optim.batching import ContinuousBatcher
from vla_engine.optim.replicas import ReplicaPool


class TestEngine:
    def test_predicts_a_chunk_of_the_declared_shape(self, engine, observation):
        chunk = engine.predict(observation)
        assert chunk.actions.shape == (50, 7)
        assert chunk.actions.dtype == np.float32

    def test_predictions_are_deterministic(self, engine, observation):
        assert np.array_equal(
            engine.predict(observation).actions, engine.predict(observation).actions
        )

    def test_different_observations_give_different_actions(self, engine, frame):
        a = engine.act(frame, "pick up the red block")
        b = engine.act(frame, "open the drawer")
        assert not np.array_equal(a.actions, b.actions)

    def test_horizon_truncation(self, frame):
        config = EngineConfig(model="echo", horizon=10, extra={"horizon": 50})
        with VLAEngine(config) as engine:
            assert engine.act(frame, "task").horizon == 10

    def test_predict_before_load_raises(self, observation):
        with pytest.raises(NotLoadedError):
            VLAEngine("echo").predict(observation)

    def test_spec_is_available_before_loading(self):
        engine = VLAEngine("openvla")
        assert engine.spec.action_dim == 7
        assert engine.info()["estimated_memory_gb"] > 0

    def test_batch_predictions_match_single(self, engine, frame):
        observations = [Observation.single(frame, f"task {i}") for i in range(4)]
        batched = engine.predict_batch(observations)
        assert len(batched) == 4
        for observation, chunk in zip(observations, batched, strict=True):
            assert np.array_equal(engine.predict(observation).actions, chunk.actions)

    def test_stats_are_attached(self, engine, observation):
        stats = engine.predict(observation).stats
        assert stats is not None and stats.total_ms >= 0

    def test_context_manager_unloads(self, frame):
        with VLAEngine("echo") as engine:
            assert engine.is_loaded
        assert engine.is_loaded is False

    def test_executor_treats_last_dim_as_gripper(self, engine):
        assert engine.make_executor().policy.discrete_dims == (6,)


class TestAdapterContract:
    def test_validates_observations_against_the_spec(self):
        adapter = EchoAdapter(EngineConfig(model="echo")).load()
        adapter.spec = adapter.spec.__class__(
            name="echo", action_dim=7, horizon=1, requires_state=True
        )
        with pytest.raises(ObservationError):
            adapter.predict(Observation.single(np.zeros((8, 8, 3), np.uint8), "task"))

    def test_warmup_runs_at_load(self):
        """Warmup must trigger compilation before the first real control step."""
        adapter = EchoAdapter(EngineConfig(model="echo", compile={"warmup_steps": 3})).load()
        assert adapter.info()["warmup_steps"] == 3

    def test_warmup_does_not_count_as_user_predictions(self):
        adapter = EchoAdapter(EngineConfig(model="echo", compile={"warmup_steps": 3})).load()
        assert adapter.info()["predictions"] == 0

    def test_empty_batch_returns_empty(self):
        adapter = EchoAdapter(EngineConfig(model="echo")).load()
        assert adapter.predict_batch([]) == []


class TestReplicaPool:
    def test_distributes_a_batch_across_replicas(self):
        pool = ReplicaPool(
            lambda device: EchoAdapter(
                EngineConfig(model="echo", device=device, extra={"horizon": 4})
            ),
            ["cpu", "cpu"],
        ).load()
        try:
            observations = [
                Observation.single(np.zeros((16, 16, 3), np.uint8), f"t{i}") for i in range(4)
            ]
            results = pool.predict_batch(observations)
            assert len(results) == 4
            assert all(chunk.actions.shape == (4, 7) for chunk in results)
        finally:
            pool.unload()

    def test_results_stay_aligned_with_inputs(self):
        """Sharding must not permute results relative to the requests."""
        pool = ReplicaPool(
            lambda device: EchoAdapter(EngineConfig(model="echo", device=device)), ["cpu", "cpu"]
        ).load()
        try:
            observations = [
                Observation.single(np.zeros((16, 16, 3), np.uint8), f"task {i}") for i in range(6)
            ]
            results = pool.predict_batch(observations)
            for observation, chunk in zip(observations, results, strict=True):
                assert chunk.meta["instruction"] == observation.instruction
        finally:
            pool.unload()

    def test_shards_evenly(self):
        assert ReplicaPool._shard(list(range(7)), 2) == [[0, 1, 2, 3], [4, 5, 6]]
        assert ReplicaPool._shard(list(range(5)), 3) == [[0, 1], [2, 3], [4]]

    def test_requires_at_least_one_device(self):
        from vla_engine.errors import BackendError

        with pytest.raises(BackendError):
            ReplicaPool(lambda d: None, [])

    def test_predict_before_load_raises(self):
        from vla_engine.errors import BackendError

        pool = ReplicaPool(lambda d: None, ["cpu"])
        with pytest.raises(BackendError, match="not loaded"):
            pool.predict(Observation.single(np.zeros((8, 8, 3), np.uint8), "x"))


class TestContinuousBatcher:
    def test_coalesces_concurrent_requests_into_one_forward(self):
        adapter = EchoAdapter(
            EngineConfig(model="echo", extra={"latency_ms": 20, "horizon": 2})
        ).load()

        async def scenario():
            batcher = ContinuousBatcher(
                adapter.predict_batch, BatchConfig(max_batch_size=8, max_wait_ms=10)
            )
            await batcher.start()
            observations = [
                Observation.single(np.zeros((16, 16, 3), np.uint8), f"t{i}") for i in range(8)
            ]
            results = await asyncio.gather(*[batcher.submit(o) for o in observations])
            await batcher.stop()
            return results, batcher.stats

        results, stats = asyncio.run(scenario())
        assert len(results) == 8
        assert stats.batches == 1, "8 concurrent requests should form a single batch"
        assert stats.mean_batch_size == 8.0

    def test_rejects_when_the_queue_is_saturated(self):
        adapter = EchoAdapter(EngineConfig(model="echo", extra={"latency_ms": 50})).load()

        async def scenario():
            from vla_engine.errors import BackendError

            batcher = ContinuousBatcher(
                adapter.predict_batch,
                BatchConfig(max_batch_size=2, max_wait_ms=1, max_queue_depth=2),
            )
            await batcher.start()
            observations = [
                Observation.single(np.zeros((8, 8, 3), np.uint8), f"t{i}") for i in range(40)
            ]
            outcomes = await asyncio.gather(
                *[batcher.submit(o) for o in observations], return_exceptions=True
            )
            await batcher.stop()
            return sum(isinstance(o, BackendError) for o in outcomes)

        assert asyncio.run(scenario()) > 0, "an overloaded GPU must shed load, not queue forever"

    def test_submit_before_start_raises(self):
        from vla_engine.errors import BackendError

        async def scenario():
            batcher = ContinuousBatcher(lambda obs: [])
            with pytest.raises(BackendError):
                await batcher.submit(Observation.single(np.zeros((8, 8, 3), np.uint8), "x"))

        asyncio.run(scenario())
