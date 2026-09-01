"""ROS message conversions. These run without ROS installed, which is the point."""

from __future__ import annotations

import numpy as np
import pytest

from vla_engine.ros.conversions import image_msg_to_array, joint_state_to_array
from vla_engine.ros.demo_scene import render_scene


class FakeMessage:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def image(array: np.ndarray, encoding: str, step: int | None = None, bigendian: int = 0):
    row = step if step is not None else array.shape[1] * (array.shape[2] if array.ndim == 3 else 1)
    return FakeMessage(
        height=array.shape[0],
        width=array.shape[1],
        encoding=encoding,
        data=array.tobytes(),
        step=row,
        is_bigendian=bigendian,
    )


RGB = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


class TestImageConversion:
    def test_rgb8_passthrough(self):
        assert np.array_equal(image_msg_to_array(image(RGB, "rgb8")), RGB)

    def test_bgr8_channel_order_is_corrected(self):
        """Feeding a policy BGR is a silent failure: it just acts slightly wrong."""
        bgr = np.ascontiguousarray(RGB[:, :, ::-1])
        assert np.array_equal(image_msg_to_array(image(bgr, "bgr8")), RGB)

    def test_rgba8_drops_alpha(self):
        rgba = np.dstack([RGB, np.full((2, 3, 1), 255, np.uint8)])
        assert np.array_equal(image_msg_to_array(image(rgba, "rgba8", step=12)), RGB)

    def test_mono8_expands_to_three_channels(self):
        mono = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
        result = image_msg_to_array(image(mono, "mono8", step=3))
        assert result.shape == (2, 3, 3)
        assert np.array_equal(result[:, :, 0], mono)

    def test_padded_row_stride_is_handled(self):
        """Publishers may pad rows for alignment; a wrong stride shears the image."""
        padded = np.zeros((2, 16), np.uint8)
        padded[:, :9] = RGB.reshape(2, 9)
        message = FakeMessage(
            height=2, width=3, encoding="rgb8", data=padded.tobytes(), step=16, is_bigendian=0
        )
        assert np.array_equal(image_msg_to_array(message), RGB)

    def test_16bit_depth_is_scaled_not_wrapped(self):
        depth = np.array([[256, 512], [1024, 2048]], np.uint16)
        message = FakeMessage(
            height=2, width=2, encoding="16uc1", data=depth.tobytes(), step=4, is_bigendian=0
        )
        assert image_msg_to_array(message)[:, :, 0].tolist() == [[1, 2], [4, 8]]

    def test_output_is_contiguous(self):
        bgr = np.ascontiguousarray(RGB[:, :, ::-1])
        assert image_msg_to_array(image(bgr, "bgr8")).flags["C_CONTIGUOUS"]

    def test_unsupported_encoding_raises(self):
        with pytest.raises(ValueError, match="unsupported image encoding"):
            image_msg_to_array(FakeMessage(height=1, width=1, encoding="yuv422", data=b"", step=0))

    def test_truncated_payload_raises(self):
        message = FakeMessage(height=2, width=3, encoding="rgb8", data=b"abc", step=0)
        with pytest.raises(ValueError, match="needs 18"):
            image_msg_to_array(message)


class TestJointState:
    def test_reorders_to_the_requested_joint_order(self):
        """ROS does not guarantee joint ordering; the wrong order fails silently."""
        message = FakeMessage(name=["b", "a", "c"], position=[2.0, 1.0, 3.0])
        assert joint_state_to_array(message, ["a", "b", "c"]).tolist() == [1.0, 2.0, 3.0]

    def test_without_names_preserves_publisher_order(self):
        message = FakeMessage(name=["b", "a"], position=[2.0, 1.0])
        assert joint_state_to_array(message).tolist() == [2.0, 1.0]

    def test_missing_joint_raises(self):
        message = FakeMessage(name=["a"], position=[1.0])
        with pytest.raises(ValueError, match="missing required joints"):
            joint_state_to_array(message, ["a", "z"])


class TestDemoScene:
    def test_renders_a_valid_frame(self):
        frame = render_scene(224, 224, 0)
        assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8

    def test_block_moves_across_frames(self):
        assert not np.array_equal(render_scene(224, 224, 0), render_scene(224, 224, 40))

    def test_contains_a_red_block(self):
        frame = render_scene(224, 224, 0)
        assert ((frame[:, :, 0] > 180) & (frame[:, :, 1] < 60)).sum() > 100
