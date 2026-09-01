"""Decouple policy inference rate from robot control rate.

A VLA is slow. OpenVLA-7B runs at roughly 5-15 Hz on a 3090 depending on
precision; a manipulator wants commands at 30-100 Hz. Bridging that gap is not
optional, and how you bridge it determines whether motion is smooth or jerky.

Two strategies are provided:

``sequential``
    Play the chunk out one step at a time and request a new one when it runs
    low. Simple and exactly reproduces the policy's intent, but every replan
    introduces a discontinuity: the new chunk's first action need not agree
    with the old chunk's next action.

``temporal_ensemble``
    The ACT strategy (Zhao et al., 2023). Overlapping chunks each vote on the
    current timestep, weighted by age as ``exp(-m * age)``. Because a fresh
    chunk only gradually takes over from its predecessors, commanded motion is
    continuous across replans. This is the default.

Both run on plain numpy so a control loop can use them without importing torch.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..types import ActionChunk

__all__ = ["ChunkPolicy", "ChunkExecutor"]


@dataclass
class ChunkPolicy:
    """How a chunk executor turns overlapping predictions into one command.

    Args:
        strategy: ``"temporal_ensemble"`` or ``"sequential"``.
        ensemble_decay: The ``m`` in ``exp(-m * age)``. Larger values trust the
            newest chunk more. ACT uses 0.01 (nearly uniform averaging); 0.1-0.5
            tracks fresh predictions more aggressively, which suits VLAs whose
            chunks are predicted further apart.
        replan_after: Emit a "needs new inference" signal once this many steps
            of the newest chunk have been consumed. Defaults to half the
            horizon, so a replacement is in flight before the current chunk is
            exhausted.
        max_chunk_age: Steps after which a chunk stops voting.
        discrete_dims: Action dimensions that must not be averaged -- a gripper
            command of 0.5 between "open" (0) and "closed" (1) is a physically
            meaningless instruction. These take the newest chunk's value.
        clip: Optional ``(low, high)`` per-dimension clamp applied last.
    """

    strategy: str = "temporal_ensemble"
    ensemble_decay: float = 0.1
    replan_after: int | None = None
    max_chunk_age: int | None = None
    discrete_dims: tuple[int, ...] = ()
    clip: tuple[Sequence[float], Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("temporal_ensemble", "sequential"):
            raise ValueError(
                f"strategy must be 'temporal_ensemble' or 'sequential', got {self.strategy!r}"
            )
        if self.ensemble_decay < 0:
            raise ValueError("ensemble_decay must be >= 0")
        if self.replan_after is not None and self.replan_after < 1:
            raise ValueError("replan_after must be >= 1 when set")


@dataclass
class _PendingChunk:
    """One chunk in flight, tracked by the control step it was issued at."""

    actions: np.ndarray
    issued_at_step: int
    issued_at_time: float

    def action_for(self, step: int) -> np.ndarray | None:
        index = step - self.issued_at_step
        if 0 <= index < self.actions.shape[0]:
            return self.actions[index]
        return None

    def age_at(self, step: int) -> int:
        return step - self.issued_at_step


class ChunkExecutor:
    """Buffers action chunks and serves one action per control tick.

    Example:
        >>> ex = ChunkExecutor(action_dim=7)
        >>> ex.submit(ActionChunk(np.zeros((50, 7), np.float32)))
        >>> action = ex.step()          # called at control rate
        >>> ex.needs_replan()           # True once the chunk runs low
        False
    """

    def __init__(
        self,
        action_dim: int,
        policy: ChunkPolicy | None = None,
        *,
        max_pending: int = 8,
    ) -> None:
        self.action_dim = int(action_dim)
        self.policy = policy or ChunkPolicy()
        self._chunks: deque[_PendingChunk] = deque(maxlen=max_pending)
        self._step = 0
        self._last_action: np.ndarray | None = None
        self._starved_steps = 0
        self._total_steps = 0

    # -- state ---------------------------------------------------------------

    @property
    def step_index(self) -> int:
        """Number of control ticks served so far."""
        return self._step

    @property
    def starved_steps(self) -> int:
        """Ticks served by holding the last action because no chunk covered them.

        A nonzero and growing value means inference is not keeping up with the
        control loop; the ROS 2 node surfaces it as a diagnostic.
        """
        return self._starved_steps

    @property
    def pending_chunks(self) -> int:
        return len(self._chunks)

    def remaining(self) -> int:
        """Control steps still covered by the newest chunk."""
        if not self._chunks:
            return 0
        newest = self._chunks[-1]
        return max(0, newest.actions.shape[0] - newest.age_at(self._step))

    def needs_replan(self) -> bool:
        """Whether a new inference should be dispatched now."""
        if not self._chunks:
            return True
        newest = self._chunks[-1]
        horizon = newest.actions.shape[0]
        threshold = self.policy.replan_after or max(1, horizon // 2)
        return newest.age_at(self._step) >= threshold

    # -- ingest --------------------------------------------------------------

    def submit(self, chunk: ActionChunk | np.ndarray) -> None:
        """Add a freshly predicted chunk, valid from the current control step.

        The chunk is anchored at the *current* step rather than at the
        observation timestamp: the actions describe what to do starting now, and
        anchoring them in the past would skip the leading actions.
        """
        actions = chunk.actions if isinstance(chunk, ActionChunk) else np.asarray(chunk)
        actions = np.atleast_2d(np.asarray(actions, dtype=np.float32))
        if actions.shape[1] != self.action_dim:
            raise ValueError(
                f"chunk has action_dim {actions.shape[1]}, executor expects {self.action_dim}"
            )
        self._chunks.append(
            _PendingChunk(
                actions=actions,
                issued_at_step=self._step,
                issued_at_time=time.monotonic(),
            )
        )

    def reset(self) -> None:
        """Drop all buffered chunks. Call on task change or E-stop recovery."""
        self._chunks.clear()
        self._step = 0
        self._last_action = None
        self._starved_steps = 0
        self._total_steps = 0

    # -- serve ---------------------------------------------------------------

    def step(self) -> np.ndarray | None:
        """Return the action for this control tick and advance time.

        Returns ``None`` only before the first chunk arrives; afterwards a
        starved executor holds its last command, which is the safe behavior for
        a position-controlled arm.
        """
        action = self._resolve(self._step)
        self._step += 1
        self._total_steps += 1
        if action is None:
            self._starved_steps += 1
            return self._last_action
        self._last_action = action
        self._evict(self._step)
        return action

    def peek(self) -> np.ndarray | None:
        """The action for the current tick without advancing."""
        return self._resolve(self._step) or self._last_action

    def _resolve(self, step: int) -> np.ndarray | None:
        candidates: list[tuple[_PendingChunk, np.ndarray]] = []
        for chunk in self._chunks:
            action = chunk.action_for(step)
            if action is None:
                continue
            if (
                self.policy.max_chunk_age is not None
                and chunk.age_at(step) > self.policy.max_chunk_age
            ):
                continue
            candidates.append((chunk, action))
        if not candidates:
            return None
        if self.policy.strategy == "sequential" or len(candidates) == 1:
            # Newest chunk wins outright.
            merged = np.array(candidates[-1][1], dtype=np.float32, copy=True)
        else:
            merged = self._ensemble(candidates, step)
        return self._finalize(merged, candidates)

    def _ensemble(
        self, candidates: list[tuple[_PendingChunk, np.ndarray]], step: int
    ) -> np.ndarray:
        """Age-weighted average, ``w_i = exp(-m * age_i)``, normalized to sum 1."""
        ages = np.array([c.age_at(step) for c, _ in candidates], dtype=np.float32)
        weights = np.exp(-self.policy.ensemble_decay * ages)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            # Degenerate decay: fall back to the freshest prediction.
            return np.array(candidates[-1][1], dtype=np.float32, copy=True)
        weights /= total
        stacked = np.stack([a for _, a in candidates]).astype(np.float32)
        return (stacked * weights[:, None]).sum(axis=0)

    def _finalize(
        self, action: np.ndarray, candidates: list[tuple[_PendingChunk, np.ndarray]]
    ) -> np.ndarray:
        # Discrete dims (gripper) take the newest vote, never a blend.
        if self.policy.discrete_dims:
            newest = candidates[-1][1]
            for dim in self.policy.discrete_dims:
                if 0 <= dim < action.shape[0]:
                    action[dim] = newest[dim]
        if self.policy.clip is not None:
            low, high = self.policy.clip
            action = np.clip(action, np.asarray(low, np.float32), np.asarray(high, np.float32))
        return action.astype(np.float32, copy=False)

    def _evict(self, step: int) -> None:
        """Drop chunks that can no longer contribute to any future step."""
        while self._chunks and self._chunks[0].action_for(step) is None:
            if self._chunks[0].issued_at_step + self._chunks[0].actions.shape[0] <= step:
                self._chunks.popleft()
            else:
                break

    def stats(self) -> dict[str, float | int]:
        return {
            "steps": self._total_steps,
            "starved_steps": self._starved_steps,
            "starvation_rate": (
                self._starved_steps / self._total_steps if self._total_steps else 0.0
            ),
            "pending_chunks": len(self._chunks),
            "remaining": self.remaining(),
        }
