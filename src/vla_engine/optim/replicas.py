"""Multi-GPU replica parallelism.

On a dual-RTX-3090 box the instinct is to split one model across both cards.
That is the wrong move here, for a specific reason: GeForce drivers disable
peer-to-peer DMA, so every cross-device tensor in a tensor-parallel or
pipeline-parallel setup round-trips through host memory over PCIe. For a
latency-critical control loop that is strictly worse than not splitting at all.

All three supported policies fit comfortably in 24 GB at bf16 (OpenVLA-7B is
the largest at ~15 GB), so the right arrangement is one complete replica per
GPU with requests routed to whichever is idle. That gives near-linear
throughput scaling and leaves single-request latency untouched -- each request
is served entirely by one card, with no cross-device traffic at all.

Replicas are held in a queue rather than round-robined: with variable-length
work, handing a request to the *idle* GPU beats handing it to the *next* one.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from ..errors import BackendError
from ..types import ActionChunk, Observation

logger = logging.getLogger(__name__)

__all__ = ["ReplicaPool"]


class ReplicaPool:
    """A pool of identical adapters, one per device.

    Args:
        factory: Builds an (unloaded) adapter bound to a given device string.
        devices: Device strings, e.g. ``["cuda:0", "cuda:1"]``.

    Example:
        >>> pool = ReplicaPool(lambda dev: OpenVLAAdapter(cfg_for(dev)),
        ...                    ["cuda:0", "cuda:1"]).load()
        >>> chunk = pool.predict(obs)
    """

    def __init__(self, factory: Callable[[str], object], devices: Sequence[str]) -> None:
        if not devices:
            raise BackendError("ReplicaPool requires at least one device")
        self._factory = factory
        self.devices = list(devices)
        self._replicas: list = []
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._loaded = False
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._dispatch_counts: dict[str, int] = {d: 0 for d in self.devices}

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> ReplicaPool:
        """Build and load one replica per device.

        Loading is sequential on purpose: two 7B checkpoints materializing at
        once can spike host RAM past what a workstation has, and the disk read
        is shared anyway once the OS page cache is warm.
        """
        with self._lock:
            if self._loaded:
                return self
            for device in self.devices:
                logger.info("loading replica on %s", device)
                replica = self._factory(device)
                replica.load()
                self._replicas.append(replica)
                self._idle.put(replica)
            self._pool = ThreadPoolExecutor(
                max_workers=len(self._replicas), thread_name_prefix="vla-replica"
            )
            self._loaded = True
        logger.info("replica pool ready across %s", self.devices)
        return self

    def unload(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
            for replica in self._replicas:
                replica.unload()
            self._replicas.clear()
            while not self._idle.empty():
                try:
                    self._idle.get_nowait()
                except queue.Empty:
                    break
            self._loaded = False

    @property
    def size(self) -> int:
        return len(self._replicas)

    @property
    def replicas(self) -> list:
        """The loaded replicas, in device order."""
        return list(self._replicas)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- routing -------------------------------------------------------------

    @contextmanager
    def acquire(self, timeout: float | None = None) -> Iterator[object]:
        """Check out an idle replica, returning it on exit.

        LIFO ordering means a recently-used replica is preferred, which keeps
        its CUDA context and compiled graphs hot rather than spreading work
        thinly across cards that then all go cold.
        """
        if not self._loaded:
            raise BackendError("replica pool is not loaded; call .load() first")
        try:
            replica = self._idle.get(timeout=timeout)
        except queue.Empty as exc:
            raise BackendError(
                f"no replica became available within {timeout}s ({self.size} replicas, all busy)"
            ) from exc
        device = getattr(replica, "device", "?")
        self._dispatch_counts[device] = self._dispatch_counts.get(device, 0) + 1
        try:
            yield replica
        finally:
            self._idle.put(replica)

    # -- inference -----------------------------------------------------------

    def predict(self, observation: Observation, timeout: float | None = None) -> ActionChunk:
        """Serve one observation on whichever replica is free."""
        with self.acquire(timeout=timeout) as replica:
            return replica.predict(observation)

    def predict_batch(
        self, observations: Sequence[Observation], timeout: float | None = None
    ) -> list[ActionChunk]:
        """Split a batch across replicas and run the shards concurrently.

        Threads are the right tool despite the GIL: the work happens in CUDA
        kernels and the torch call releases the GIL for their duration, so two
        replicas genuinely overlap.
        """
        if not observations:
            return []
        if self.size == 1 or len(observations) == 1:
            with self.acquire(timeout=timeout) as replica:
                return replica.predict_batch(list(observations))

        shards = self._shard(list(observations), self.size)
        assert self._pool is not None

        def run(shard: list[Observation]) -> list[ActionChunk]:
            with self.acquire(timeout=timeout) as replica:
                return replica.predict_batch(shard)

        futures = [self._pool.submit(run, shard) for shard in shards if shard]
        results: list[ActionChunk] = []
        for future in futures:
            results.extend(future.result())
        return results

    @staticmethod
    def _shard(items: list, parts: int) -> list[list]:
        """Split as evenly as possible; earlier shards take the remainder."""
        if parts <= 1:
            return [items]
        size, remainder = divmod(len(items), parts)
        shards, start = [], 0
        for i in range(parts):
            length = size + (1 if i < remainder else 0)
            shards.append(items[start : start + length])
            start += length
        return shards

    # -- introspection -------------------------------------------------------

    def info(self) -> dict:
        return {
            "replicas": self.size,
            "devices": list(self.devices),
            "loaded": self._loaded,
            "idle": self._idle.qsize(),
            "dispatch_counts": dict(self._dispatch_counts),
            "replica_info": [r.info() for r in self._replicas],
        }
