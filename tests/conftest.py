"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from vla_engine import EngineConfig, Observation, VLAEngine


@pytest.fixture
def frame() -> np.ndarray:
    """A deterministic 224x224 RGB frame."""
    from vla_engine.ros.demo_scene import render_scene

    return render_scene(224, 224, frame=7)


@pytest.fixture
def observation(frame: np.ndarray) -> Observation:
    return Observation.single(frame, "pick up the red block", np.zeros(7, dtype=np.float32))


@pytest.fixture
def engine() -> VLAEngine:
    """A loaded weightless engine producing 50-step, 7-dim chunks."""
    engine = VLAEngine(EngineConfig(model="echo", extra={"horizon": 50, "action_dim": 7})).load()
    yield engine
    engine.unload()
