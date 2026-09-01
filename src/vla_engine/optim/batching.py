"""Continuous batching for the inference server.

Batching is a throughput optimization that *costs* latency, so it is off by
default. It earns its place in one situation: several robots, several parallel
environments, or a fleet of MuJoCo instances sharing one GPU. There, a VLA
forward pass at batch 1 leaves an RTX 3090 badly underutilized -- the vision
tower and the action expert are both far from saturating the SMs -- and folding
four requests into one pass costs a few percent more time than one request.

The batcher collects requests for at most ``max_wait_ms`` (default 2 ms, noise
against a 30-100 ms forward pass) or until ``max_batch_size`` is reached,
whichever comes first, then issues a single ``predict_batch``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..config import BatchConfig
from ..errors import BackendError
from ..types import ActionChunk, Observation

logger = logging.getLogger(__name__)

__all__ = ["ContinuousBatcher", "BatcherStats"]


@dataclass
class BatcherStats:
    """Aggregate counters, exposed on the server's ``/metrics`` endpoint."""

    requests: int = 0
    batches: int = 0
    rejected: int = 0
    total_queue_ms: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch_size(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def mean_queue_ms(self) -> float:
        return self.total_queue_ms / self.requests if self.requests else 0.0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "batches": self.batches,
            "rejected": self.rejected,
            "mean_batch_size": round(self.mean_batch_size, 2),
            "mean_queue_ms": round(self.mean_queue_ms, 3),
        }


@dataclass
class _Request:
    observation: Observation
    future: asyncio.Future[ActionChunk]
    enqueued_at: float


class ContinuousBatcher:
    """Coalesces concurrent single-observation requests into batched forwards.

    Args:
        predict_batch: The underlying batched inference callable. It is run in
            a worker thread, so a blocking torch call does not stall the event
            loop serving other requests.
        config: Batching parameters.

    Example:
        >>> batcher = ContinuousBatcher(engine.predict_batch, BatchConfig(max_batch_size=8))
        >>> await batcher.start()
        >>> chunk = await batcher.submit(obs)
    """

    def __init__(
        self,
        predict_batch: Callable[[Sequence[Observation]], list[ActionChunk]],
        config: BatchConfig | None = None,
    ) -> None:
        self._predict_batch = predict_batch
        self.config = config or BatchConfig()
        self._queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running = False
        self.stats = BatcherStats()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="vla-batcher")
        logger.info(
            "continuous batching on: max_batch_size=%d max_wait_ms=%.1f",
            self.config.max_batch_size,
            self.config.max_wait_ms,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Fail anything still queued rather than leaving callers hanging.
        while not self._queue.empty():
            request = self._queue.get_nowait()
            if not request.future.done():
                request.future.set_exception(BackendError("batcher stopped"))

    async def submit(self, observation: Observation) -> ActionChunk:
        """Enqueue one observation and await its action chunk."""
        if not self._running:
            raise BackendError("batcher is not running; call .start() first")
        if self._queue.qsize() >= self.config.max_queue_depth:
            self.stats.rejected += 1
            raise BackendError(
                f"inference queue is full ({self.config.max_queue_depth}); "
                "the GPU cannot keep up with the request rate"
            )
        loop = asyncio.get_running_loop()
        request = _Request(observation, loop.create_future(), time.perf_counter())
        await self._queue.put(request)
        return await request.future

    async def _run(self) -> None:
        while self._running:
            try:
                batch = await self._collect()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("batcher collection failed")
                continue
            if batch:
                await self._execute(batch)

    async def _collect(self) -> list[_Request]:
        """Wait for a first request, then briefly for companions."""
        first = await self._queue.get()
        batch = [first]
        if self.config.max_batch_size <= 1:
            return batch

        deadline = time.perf_counter() + self.config.max_wait_ms / 1000.0
        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _execute(self, batch: list[_Request]) -> None:
        now = time.perf_counter()
        self.stats.requests += len(batch)
        self.stats.batches += 1
        self.stats.batch_sizes.append(len(batch))
        if len(self.stats.batch_sizes) > 1000:  # bounded history
            del self.stats.batch_sizes[:-1000]
        for request in batch:
            self.stats.total_queue_ms += (now - request.enqueued_at) * 1000.0

        observations = [r.observation for r in batch]
        try:
            # to_thread keeps the blocking CUDA call off the event loop.
            chunks = await asyncio.to_thread(self._predict_batch, observations)
        except Exception as exc:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        if len(chunks) != len(batch):
            error = BackendError(
                f"batched predict returned {len(chunks)} chunks for {len(batch)} requests"
            )
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return

        # Lengths were verified equal above; strict=True guards that check.
        for request, chunk in zip(batch, chunks, strict=True):
            if chunk.stats is not None:
                chunk.stats.queue_ms = (now - request.enqueued_at) * 1000.0
            if not request.future.done():
                request.future.set_result(chunk)
