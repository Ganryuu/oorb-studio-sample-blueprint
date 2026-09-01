"""A synthetic scene renderer for smoke-testing a policy pipeline.

Deliberately not random noise: a policy fed noise produces meaningless actions,
which makes it impossible to tell a working pipeline from a broken one. This
renders a recognizable tabletop -- a gradient background, a horizon, and a red
block that moves -- so downstream actions should vary smoothly with the block's
position, which is something you can actually eyeball on ``/vla/action``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["render_scene"]


def render_scene(height: int = 224, width: int = 224, frame: int = 0) -> np.ndarray:
    """Render one ``HxWx3`` uint8 RGB frame of a moving red block on a table.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        frame: Monotonic frame counter; drives the block's motion.

    Returns:
        A contiguous uint8 RGB array.
    """
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Wall above, table below.
    horizon = int(height * 0.45)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[..., 0] = 0.35 + 0.15 * cols
    image[..., 1] = 0.40 + 0.10 * rows
    image[..., 2] = 0.55 - 0.15 * rows
    image[horizon:, 0] = 0.55
    image[horizon:, 1] = 0.45
    image[horizon:, 2] = 0.32

    # A red block tracing a slow ellipse on the table.
    size = max(8, height // 10)
    center_x = int(width * (0.5 + 0.28 * math.sin(frame * 0.03)))
    center_y = int(height * (0.68 + 0.08 * math.cos(frame * 0.045)))
    top, bottom = max(0, center_y - size), min(height, center_y + size)
    left, right = max(0, center_x - size), min(width, center_x + size)
    image[top:bottom, left:right] = (0.80, 0.12, 0.12)

    # A static blue marker, so absolute position stays observable.
    marker = max(4, height // 24)
    image[horizon + 4 : horizon + 4 + marker, 6 : 6 + marker] = (0.15, 0.30, 0.85)

    return np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
