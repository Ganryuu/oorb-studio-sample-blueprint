"""``torch.compile`` integration with safe fallback and explicit warmup.

Why this matters for VLAs specifically: a control-loop forward pass is
launch-bound, not compute-bound. OpenVLA decodes 7 action tokens as 7
sequential passes over a 7B model; each pass launches on the order of a
thousand tiny kernels whose individual runtime is smaller than the ~5-10 us it
costs to launch them. ``mode="reduce-overhead"`` captures the region into a
CUDA graph and replays it as a single launch, which is typically the largest
single latency win available on a 3090.

CUDA graph capture has hard requirements -- static shapes, stable input
pointers, no CPU-GPU syncs inside the region -- so compilation is treated as an
*optimization*, never a correctness dependency: any failure logs and falls back
to eager.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import CompileConfig

logger = logging.getLogger(__name__)

__all__ = ["CompileResult", "maybe_compile", "warmup", "reset_compile_cache"]


@dataclass
class CompileResult:
    """Outcome of a compile attempt, including why it may have been skipped."""

    module: Any
    compiled: bool
    mode: str | None = None
    reason: str = ""
    compile_seconds: float = 0.0

    def summary(self) -> str:
        if self.compiled:
            return f"torch.compile(mode={self.mode}) in {self.compile_seconds:.1f}s"
        return f"eager ({self.reason})"


def maybe_compile(
    module: Any,
    config: CompileConfig,
    *,
    device: str = "cuda",
    label: str = "model",
) -> CompileResult:
    """Compile ``module`` if requested and supported, else return it untouched.

    Never raises on compilation failure -- a policy that runs slowly is far
    better than one that does not run.
    """
    if not config.enabled:
        return CompileResult(module, False, reason="disabled in config")

    try:
        import torch
    except Exception:
        return CompileResult(module, False, reason="torch not installed")

    if not hasattr(torch, "compile"):
        return CompileResult(
            module, False, reason=f"torch {torch.__version__} has no torch.compile"
        )

    if not device.startswith("cuda"):
        # Inductor supports CPU, but reduce-overhead/CUDA graphs do not, and
        # the win on CPU is small relative to the compile cost.
        return CompileResult(module, False, reason="compilation only enabled for CUDA devices")

    mode = config.mode
    started = time.perf_counter()
    try:
        compiled = torch.compile(
            module,
            mode=mode,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
        )
    except Exception as exc:  # pragma: no cover - depends on torch internals
        logger.warning("torch.compile failed for %s, using eager: %s", label, exc)
        return CompileResult(module, False, reason=f"compile raised {type(exc).__name__}")

    elapsed = time.perf_counter() - started
    logger.info("compiled %s with mode=%s (graph build deferred to first call)", label, mode)
    return CompileResult(compiled, True, mode=mode, compile_seconds=elapsed)


def warmup(
    fn: Callable[[], Any],
    steps: int,
    *,
    label: str = "model",
    synchronize: bool = True,
) -> tuple[int, float]:
    """Run ``fn`` ``steps`` times to trigger compilation and CUDA-graph capture.

    Torch compiles lazily on first call and re-compiles on shape changes, and
    ``reduce-overhead`` needs a couple of extra iterations to record and then
    replay the graph. Paying that at load time keeps the robot's first control
    step from taking 60 seconds.

    Returns:
        ``(completed_steps, seconds)``. A completed count below ``steps`` means
        warmup aborted early, so compilation may not have been triggered and
        the first real inference will pay for it.
    """
    if steps <= 0:
        return 0, 0.0
    started = time.perf_counter()
    completed = 0
    for i in range(steps):
        try:
            fn()
            completed += 1
        except Exception as exc:  # pragma: no cover - model specific
            logger.warning("warmup step %d/%d for %s failed: %s", i + 1, steps, label, exc)
            break
    if synchronize:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    elapsed = time.perf_counter() - started
    logger.info("warmed up %s: %d/%d steps in %.1fs", label, completed, steps, elapsed)
    return completed, elapsed


def reset_compile_cache() -> None:
    """Drop compiled artifacts. Used between benchmark configurations."""
    try:
        import torch

        torch._dynamo.reset()
    except Exception:
        pass
