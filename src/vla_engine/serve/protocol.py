"""Wire format for observations and action chunks.

The workstation with the 3090s is usually not the machine holding the robot, so
observations cross a network on every control step. What that costs is decided
almost entirely here.

A 224x224 RGB frame is 150 KB raw, and base64 inflates it to ~196 KB. Two
cameras at 10 Hz is then 32 Mbit/s of pure pixel traffic -- enough to saturate a
shared wifi link and add tens of milliseconds of jitter to a control loop.

Measured on this implementation, JPEG at quality 90 brings the same two-camera
stream to 0.88 Mbit/s: 36x smaller on a synthetic scene (mean absolute pixel
error 0.76/255) and 3.3x on pure random noise, which is JPEG's pathological
worst case and nothing like a camera frame. Real scenes land nearer the former.
The encode costs well under a millisecond, and the loss is invisible to a vision
encoder trained on JPEG-compressed web and robot data.

So JPEG is the default, with raw available for lossless debugging.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Mapping

import numpy as np

from ..errors import ObservationError
from ..types import ActionChunk, InferenceStats, Observation

__all__ = [
    "encode_observation",
    "decode_observation",
    "encode_chunk",
    "decode_chunk",
    "encode_image",
    "decode_image",
]

PROTOCOL_VERSION = 1


def encode_image(image: np.ndarray, image_format: str = "jpeg", quality: int = 90) -> dict[str, Any]:
    """Encode one ``HxWx3`` uint8 frame for transport."""
    if image_format == "raw":
        return {
            "format": "raw",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "data": base64.b64encode(np.ascontiguousarray(image).tobytes()).decode("ascii"),
        }
    try:
        from PIL import Image
    except ImportError:
        # Degrade to raw rather than fail: correctness beats bandwidth.
        return encode_image(image, "raw")

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return {
        "format": "jpeg",
        "shape": list(image.shape),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def decode_image(payload: Mapping[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_image`."""
    fmt = payload.get("format", "raw")
    raw = base64.b64decode(payload["data"])
    if fmt == "raw":
        shape = tuple(payload["shape"])
        dtype = np.dtype(payload.get("dtype", "uint8"))
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if fmt == "jpeg":
        try:
            from PIL import Image
        except ImportError as exc:
            raise ObservationError(
                "received a JPEG-encoded image but Pillow is not installed"
            ) from exc
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    raise ObservationError(f"unknown image format {fmt!r}")


def encode_observation(
    observation: Observation, *, image_format: str = "jpeg", quality: int = 90
) -> dict[str, Any]:
    """Serialize an observation to a JSON-safe dict."""
    return {
        "v": PROTOCOL_VERSION,
        "images": {
            name: encode_image(frame, image_format, quality)
            for name, frame in observation.images.items()
        },
        "instruction": observation.instruction,
        "state": observation.state.tolist() if observation.state is not None else None,
        "timestamp": observation.timestamp,
    }


def decode_observation(payload: Mapping[str, Any]) -> Observation:
    """Inverse of :func:`encode_observation`."""
    if not isinstance(payload, Mapping):
        raise ObservationError("observation payload must be a mapping")
    images = payload.get("images")
    if not images:
        raise ObservationError("observation payload has no images")
    state = payload.get("state")
    return Observation(
        images={name: decode_image(data) for name, data in images.items()},
        instruction=payload.get("instruction", "") or "",
        state=np.asarray(state, dtype=np.float32) if state is not None else None,
        timestamp=float(payload.get("timestamp", 0.0)) or 0.0,
    )


def encode_chunk(chunk: ActionChunk) -> dict[str, Any]:
    """Serialize an action chunk.

    Actions stay as nested lists rather than base64: a 50x32 chunk is ~6 KB of
    JSON, small enough that readability in logs and `curl` output is worth more
    than the bytes saved.
    """
    return {
        "v": PROTOCOL_VERSION,
        "actions": chunk.actions.tolist(),
        "horizon": chunk.horizon,
        "action_dim": chunk.action_dim,
        "timestamp": chunk.timestamp,
        "stats": chunk.stats.as_dict() if chunk.stats else None,
        "meta": chunk.meta,
    }


def decode_chunk(payload: Mapping[str, Any]) -> ActionChunk:
    """Inverse of :func:`encode_chunk`."""
    stats_payload = payload.get("stats")
    stats = None
    if stats_payload:
        known = {
            k: v
            for k, v in stats_payload.items()
            if k in InferenceStats.__dataclass_fields__
        }
        stats = InferenceStats(**known)
    return ActionChunk(
        actions=np.asarray(payload["actions"], dtype=np.float32),
        timestamp=float(payload.get("timestamp", 0.0)),
        stats=stats,
        meta=dict(payload.get("meta") or {}),
    )
