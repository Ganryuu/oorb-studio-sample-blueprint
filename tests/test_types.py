"""Observation/action contracts."""

from __future__ import annotations

import numpy as np
import pytest

from vla_engine import ActionChunk, Observation, PolicySpec
from vla_engine.errors import ObservationError


class TestObservation:
    def test_normalizes_grayscale_to_rgb(self):
        obs = Observation(images={"cam": np.zeros((8, 8), np.uint8)}, instruction="x")
        assert obs.images["cam"].shape == (8, 8, 3)

    def test_transposes_chw_to_hwc(self):
        obs = Observation(images={"cam": np.zeros((3, 16, 20), np.uint8)}, instruction="x")
        assert obs.images["cam"].shape == (16, 20, 3)

    def test_drops_alpha_channel(self):
        obs = Observation(images={"cam": np.zeros((8, 8, 4), np.uint8)}, instruction="x")
        assert obs.images["cam"].shape == (8, 8, 3)

    def test_scales_unit_float_images(self):
        obs = Observation(images={"cam": np.ones((4, 4, 3), np.float32)}, instruction="x")
        assert obs.images["cam"].dtype == np.uint8
        assert obs.images["cam"].max() == 255

    def test_float_images_already_in_0_255_are_not_rescaled(self):
        source = np.full((4, 4, 3), 200.0, np.float32)
        obs = Observation(images={"cam": source}, instruction="x")
        assert obs.images["cam"].max() == 200

    def test_rejects_nonfinite_state(self):
        with pytest.raises(ObservationError, match="NaN"):
            Observation(
                images={"cam": np.zeros((4, 4, 3), np.uint8)},
                instruction="x",
                state=np.array([1.0, np.nan]),
            )

    def test_rejects_bad_channel_count(self):
        with pytest.raises(ObservationError, match="channels"):
            Observation(images={"cam": np.zeros((8, 8, 5), np.uint8)}, instruction="x")


class TestActionChunk:
    def test_promotes_single_action_to_chunk(self):
        chunk = ActionChunk(np.zeros(7, np.float32))
        assert chunk.horizon == 1 and chunk.action_dim == 7

    def test_truncate_preserves_metadata(self):
        chunk = ActionChunk(np.zeros((50, 7), np.float32), meta={"model": "x"})
        short = chunk.truncate(5)
        assert short.horizon == 5 and short.meta == {"model": "x"}

    def test_rejects_3d_actions(self):
        with pytest.raises(ObservationError):
            ActionChunk(np.zeros((2, 3, 4), np.float32))


class TestPolicySpec:
    def test_maps_unknown_camera_name_onto_expected_slot(self):
        spec = PolicySpec("p", 7, 1, cameras=("primary",))
        resolved = spec.resolve_cameras({"my_webcam": np.zeros((4, 4, 3), np.uint8)})
        assert list(resolved) == ["primary"]

    def test_exact_names_take_precedence_over_positional(self):
        spec = PolicySpec("p", 7, 1, cameras=("primary", "wrist"))
        wrist = np.ones((4, 4, 3), np.uint8)
        resolved = spec.resolve_cameras({"wrist": wrist, "other": np.zeros((4, 4, 3), np.uint8)})
        assert np.array_equal(resolved["wrist"], wrist)

    def test_duplicates_primary_when_a_camera_is_missing(self):
        spec = PolicySpec("p", 7, 1, cameras=("primary", "wrist"))
        resolved = spec.resolve_cameras({"primary": np.zeros((4, 4, 3), np.uint8)})
        assert np.array_equal(resolved["wrist"], resolved["primary"])

    def test_requires_state_when_declared(self):
        spec = PolicySpec("p", 7, 1, requires_state=True)
        obs = Observation.single(np.zeros((4, 4, 3), np.uint8), "task")
        with pytest.raises(ObservationError, match="state"):
            spec.validate(obs)

    def test_requires_nonempty_instruction_when_language_conditioned(self):
        spec = PolicySpec("p", 7, 1, language_conditioned=True)
        obs = Observation.single(np.zeros((4, 4, 3), np.uint8), "   ")
        with pytest.raises(ObservationError, match="instruction"):
            spec.validate(obs)
