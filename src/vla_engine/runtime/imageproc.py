"""Image preprocessing on the GPU.

The naive path -- convert uint8 HWC to float32 CHW in numpy, then upload --
sends four times the bytes over PCIe and burns CPU time that a control loop
does not have. For a 3-camera policy at 512x512 that is 9 MB per observation
instead of 2.3 MB, on every single inference.

This module uploads the raw uint8 frames and does the permute, cast, and scale
on the GPU, where they are effectively free next to the model forward.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["stack_uint8", "to_device_chw_float", "resize_with_padding"]


def stack_uint8(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-observation frames into one contiguous ``[B, H, W, 3]`` uint8 array.

    Contiguity matters: it lets the subsequent host-to-device copy be a single
    DMA transfer rather than a scatter of row copies.
    """
    if not frames:
        raise ValueError("no frames to stack")
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"frames must share a shape for batching, got {sorted(shapes)}")
    return np.ascontiguousarray(np.stack(frames, axis=0))


def to_device_chw_float(batch_hwc_uint8: np.ndarray, device: str, dtype=None):
    """Upload ``[B, H, W, 3]`` uint8 and return ``[B, 3, H, W]`` float in ``[0, 1]``.

    The cast and permute happen after the transfer, so only one byte per channel
    crosses the bus. ``pin_memory`` plus ``non_blocking`` lets the copy overlap
    with whatever the CPU does next.

    Args:
        batch_hwc_uint8: Contiguous uint8 batch, channels last.
        device: Target torch device string.
        dtype: Output dtype (default float32).

    Returns:
        A torch tensor of shape ``[B, 3, H, W]`` scaled to ``[0, 1]``.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    host = torch.from_numpy(batch_hwc_uint8)
    if device.startswith("cuda") and not host.is_pinned():
        try:
            host = host.pin_memory()
        except RuntimeError:
            pass  # pinning can fail under memory pressure; the copy still works
    gpu = host.to(device, non_blocking=True)
    return gpu.permute(0, 3, 1, 2).to(dtype).div_(255.0)


def resize_with_padding(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Resize preserving aspect ratio, padding the remainder with zeros.

    SmolVLA's SmolVLM-2 backbone expects square inputs and was trained with
    aspect-preserving letterboxing; stretching a 640x480 camera frame to square
    distorts the geometry the policy uses to localize objects.

    Uses PIL when available (higher quality resampling) and falls back to
    nearest-neighbour indexing so this stays importable without Pillow.
    """
    target_h, target_w = target
    height, width = image.shape[:2]
    if (height, width) == (target_h, target_w):
        return image

    scale = min(target_h / height, target_w / width)
    new_h, new_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))

    try:
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8
        )
    except Exception:
        rows = (np.arange(new_h) * (height / new_h)).astype(np.int64).clip(0, height - 1)
        cols = (np.arange(new_w) * (width / new_w)).astype(np.int64).clip(0, width - 1)
        resized = image[rows[:, None], cols[None, :]]

    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas
