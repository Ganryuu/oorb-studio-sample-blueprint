"""Benchmark harness.

Optimization claims are worthless unmeasured, and VLA inference is easy to
measure badly. Three mistakes this harness avoids:

1. **Reporting a mean.** A control loop is hurt by its worst steps, not its
   average one. A policy averaging 40 ms with a p99 of 250 ms will visibly
   stutter. p50/p95/p99 are reported; the mean is included only for reference.
2. **Measuring compilation.** The first call after ``torch.compile`` includes
   graph construction and can be 100x slower than steady state. Warmup
   iterations are run and discarded.
3. **Measuring CUDA asynchronously.** Kernel launches return immediately, so
   without a synchronize you time the launch, not the work. Every iteration is
   synchronized before the clock stops.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..types import ActionChunk, Observation

logger = logging.getLogger(__name__)

__all__ = ["BenchmarkResult", "benchmark", "sweep", "make_observation"]


def _sync() -> None:
    """Block until queued CUDA work has actually finished."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass
    return 0.0


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@dataclass
class BenchmarkResult:
    """Timing distribution for one configuration."""

    label: str
    latencies_ms: list[float] = field(default_factory=list)
    batch_size: int = 1
    warmup_iters: int = 0
    peak_memory_gb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def _pct(self, q: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        # Nearest-rank percentile: with 20-100 samples this is more honest
        # than interpolating between two measurements that were never taken.
        index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * len(ordered))) - 1))
        return ordered[index]

    @property
    def iterations(self) -> int:
        return len(self.latencies_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    @property
    def p50_ms(self) -> float:
        return self._pct(50)

    @property
    def p95_ms(self) -> float:
        return self._pct(95)

    @property
    def p99_ms(self) -> float:
        return self._pct(99)

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def throughput_hz(self) -> float:
        """Observations served per second, using median latency."""
        return (self.batch_size * 1000.0 / self.p50_ms) if self.p50_ms > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "iterations": self.iterations,
            "batch_size": self.batch_size,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "stdev_ms": round(self.stdev_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "throughput_hz": round(self.throughput_hz, 2),
            "peak_memory_gb": round(self.peak_memory_gb, 2),
            **self.metadata,
        }

    def summary(self) -> str:
        return (
            f"{self.label:<28} p50 {self.p50_ms:7.2f} ms  p95 {self.p95_ms:7.2f} ms  "
            f"p99 {self.p99_ms:7.2f} ms  {self.throughput_hz:6.1f} Hz"
            + (f"  {self.peak_memory_gb:5.2f} GB" if self.peak_memory_gb else "")
        )


def make_observation(
    spec, *, batch: int = 1, seed: int = 0, instruction: str = "pick up the red block"
) -> list[Observation]:
    """Build synthetic observations matching a policy's input contract.

    Content is random rather than blank: a zero image can take a different path
    through sparsity-aware kernels than real data, which would flatter the
    measurement.
    """
    rng = np.random.default_rng(seed)
    height, width = spec.image_size
    observations = []
    for _ in range(batch):
        images = {
            cam: rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for cam in spec.cameras
        }
        state = (
            rng.standard_normal(spec.state_dim or 7).astype(np.float32)
            if spec.requires_state
            else None
        )
        observations.append(Observation(images=images, instruction=instruction, state=state))
    return observations


def benchmark(
    predict: Callable[[Sequence[Observation]], list[ActionChunk]],
    observations: Sequence[Observation],
    *,
    label: str = "benchmark",
    iterations: int = 50,
    warmup: int = 5,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Time ``predict`` over ``observations``.

    Args:
        predict: Batched inference callable.
        observations: The batch to run each iteration.
        iterations: Timed iterations after warmup.
        warmup: Discarded iterations, to absorb compilation and graph capture.

    Returns:
        A :class:`BenchmarkResult` with the latency distribution.
    """
    batch = list(observations)
    for _ in range(warmup):
        predict(batch)
    _sync()

    # Keep a collection pause from landing mid-measurement.
    gc.collect()
    gc.disable()
    _reset_peak_memory()
    latencies: list[float] = []
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            predict(batch)
            _sync()
            latencies.append((time.perf_counter() - started) * 1000.0)
    finally:
        gc.enable()

    return BenchmarkResult(
        label=label,
        latencies_ms=latencies,
        batch_size=len(batch),
        warmup_iters=warmup,
        peak_memory_gb=_peak_memory_gb(),
        metadata=metadata or {},
    )


def sweep(
    engine_factory: Callable[[dict], Any],
    configurations: Sequence[dict],
    *,
    iterations: int = 30,
    warmup: int = 5,
    batch_sizes: Sequence[int] = (1,),
) -> list[BenchmarkResult]:
    """Benchmark several configurations, loading and unloading each in turn.

    Used by ``vla bench`` to answer the questions that actually come up: what
    does NF4 cost in latency, how much does compilation help, where does batch
    scaling stop paying.
    """
    results: list[BenchmarkResult] = []
    for config in configurations:
        label = config.pop("_label", None) or ",".join(f"{k}={v}" for k, v in config.items())
        engine = None
        try:
            engine = engine_factory(config)
            for batch_size in batch_sizes:
                observations = make_observation(engine.spec, batch=batch_size)
                suffix = f" b{batch_size}" if len(batch_sizes) > 1 else ""
                results.append(
                    benchmark(
                        engine.predict_batch,
                        observations,
                        label=f"{label}{suffix}",
                        iterations=iterations,
                        warmup=warmup,
                        metadata={"config": dict(config)},
                    )
                )
                logger.info("%s", results[-1].summary())
        except Exception as exc:
            logger.error("configuration %s failed: %s", label, exc)
        finally:
            if engine is not None and hasattr(engine, "unload"):
                engine.unload()
    return results
