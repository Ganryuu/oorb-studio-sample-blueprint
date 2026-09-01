"""Convert ROS message payloads to the arrays a policy expects.

Written against the message *fields* rather than the message classes, so this
module imports and tests without ROS installed -- which matters because it
holds the byte-layout logic most likely to be subtly wrong (channel order,
stride, endianness), and that logic deserves tests that run in CI.

``cv_bridge`` is deliberately not used. It pulls in OpenCV, which is a heavy
dependency for a robot container, and the conversions a VLA needs amount to a
reshape and a channel flip.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["image_msg_to_array", "joint_state_to_array", "SUPPORTED_ENCODINGS"]

SUPPORTED_ENCODINGS = (
    "rgb8",
    "bgr8",
    "rgba8",
    "bgra8",
    "mono8",
    "mono16",
    "16uc1",
)


def image_msg_to_array(msg: Any) -> np.ndarray:
    """Convert a ``sensor_msgs/Image``-shaped object to ``HxWx3`` uint8 RGB.

    Args:
        msg: Anything with ``height``, ``width``, ``encoding``, ``data``, and
            optionally ``step`` and ``is_bigendian``.

    Returns:
        A contiguous ``HxWx3`` uint8 RGB array.

    Raises:
        ValueError: On an unsupported encoding or a payload whose length does
            not match the declared geometry. Both are silent-corruption bugs
            otherwise -- a wrong stride yields a sheared image that a policy
            will happily act on.
    """
    height, width = int(msg.height), int(msg.width)
    encoding = str(msg.encoding).lower()
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; supported: {SUPPORTED_ENCODINGS}"
        )

    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding, 1)
    itemsize = 2 if encoding in ("mono16", "16uc1") else 1
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    expected = height * width * channels * itemsize
    step = int(getattr(msg, "step", 0) or 0)
    row_bytes = width * channels * itemsize

    if step and step != row_bytes:
        # Padded rows: reshape by the real stride, then trim.
        if raw.size < height * step:
            raise ValueError(
                f"image payload is {raw.size} bytes, too small for {height} rows of step {step}"
            )
        raw = raw[: height * step].reshape(height, step)[:, :row_bytes].reshape(-1)
    elif raw.size != expected:
        raise ValueError(
            f"image payload is {raw.size} bytes but {encoding} {width}x{height} needs {expected}"
        )

    if itemsize == 2:
        dtype = ">u2" if getattr(msg, "is_bigendian", 0) else "<u2"
        depth = raw.view(dtype).reshape(height, width)
        # Depth is not an RGB image; compress the high byte so the frame is at
        # least visually meaningful rather than wrapping every 256 mm.
        scaled = (depth >> 8).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(scaled[:, :, None], 3, axis=2))

    frame = raw.reshape(height, width, channels)
    if encoding in ("bgr8", "bgra8"):
        frame = frame[:, :, :3][:, :, ::-1]
    elif encoding in ("rgba8",):
        frame = frame[:, :, :3]
    elif encoding == "mono8":
        frame = np.repeat(frame, 3, axis=2)
    return np.ascontiguousarray(frame)


def joint_state_to_array(msg: Any, names: list[str] | None = None) -> np.ndarray:
    """Extract joint positions from a ``sensor_msgs/JointState``.

    Args:
        msg: Object with ``name`` and ``position``.
        names: Desired joint order. ROS does not guarantee a stable ordering
            across publishers, and feeding a policy its joints in the wrong
            order is a silent, confusing failure -- so when the caller knows the
            expected order, reorder to match it.

    Returns:
        A float32 vector of joint positions.
    """
    positions = np.asarray(list(msg.position), dtype=np.float32)
    if names is None:
        return positions
    lookup = {name: i for i, name in enumerate(list(msg.name))}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"joint state is missing required joints: {missing}")
    return np.asarray([positions[lookup[n]] for n in names], dtype=np.float32)
