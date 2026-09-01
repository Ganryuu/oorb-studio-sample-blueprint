"""Benchmark harness correctness."""

from __future__ import annotations

from vla_engine import EngineConfig, VLAEngine
from vla_engine.bench.latency import BenchmarkResult, benchmark, make_observation


class TestBenchmarkResult:
    def test_percentiles_are_exact_for_a_known_distribution(self):
        result = BenchmarkResult("t", latencies_ms=[float(i) for i in range(1, 101)])
        assert (result.p50_ms, result.p95_ms, result.p99_ms) == (50.0, 95.0, 99.0)

    def test_throughput_scales_with_batch_size(self):
        single = BenchmarkResult("a", latencies_ms=[10.0] * 10, batch_size=1)
        batched = BenchmarkResult("b", latencies_ms=[10.0] * 10, batch_size=4)
        assert single.throughput_hz == 100.0
        assert batched.throughput_hz == 400.0

    def test_empty_result_does_not_divide_by_zero(self):
        empty = BenchmarkResult("e")
        assert empty.p50_ms == 0.0 and empty.throughput_hz == 0.0

    def test_summary_is_single_line(self):
        result = BenchmarkResult("x", latencies_ms=[1.0, 2.0])
        assert "\n" not in result.summary()


class TestBenchmark:
    def test_measures_a_known_latency(self):
        engine = VLAEngine(EngineConfig(model="echo", extra={"latency_ms": 15})).load()
        try:
            observations = make_observation(engine.spec, batch=1)
            result = benchmark(
                engine.predict_batch, observations, iterations=5, warmup=1, label="echo"
            )
            assert result.iterations == 5
            assert 14.0 < result.p50_ms < 30.0
        finally:
            engine.unload()

    def test_synthetic_observations_match_the_policy_spec(self):
        engine = VLAEngine("pi0")  # not loaded: spec comes from the registry
        observation = make_observation(engine.spec, batch=1)[0]
        assert set(observation.images) == set(engine.spec.cameras)
        assert observation.state is not None
        assert observation.images["primary"].shape == (*engine.spec.image_size, 3)

    def test_synthetic_images_are_not_blank(self):
        """Blank inputs can take a different kernel path and flatter the result."""
        engine = VLAEngine("openvla")
        frame = make_observation(engine.spec, batch=1)[0].images["primary"]
        assert frame.std() > 10
