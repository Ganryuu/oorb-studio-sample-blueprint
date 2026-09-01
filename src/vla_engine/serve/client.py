"""Client for a remote :mod:`vla_engine.serve.server`.

Exposes the same ``predict``/``act`` surface as :class:`~vla_engine.VLAEngine`,
so code can move between local and remote inference by swapping the object it
was handed.

The websocket path is the one to use in a control loop -- see the note in
:mod:`vla_engine.serve.server` on handshake cost.

Example:
    >>> client = VLAClient("ws://workstation:8000")
    >>> chunk = client.act(frame, "pick up the red block")
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

from ..errors import BackendError, DependencyError
from ..types import ActionChunk, Observation
from .protocol import decode_chunk, encode_observation

logger = logging.getLogger(__name__)

__all__ = ["VLAClient"]


class VLAClient:
    """Talks to a remote inference server over HTTP or websocket.

    Args:
        url: Server base URL. ``ws://`` or ``wss://`` selects the persistent
            websocket transport; ``http://`` uses request/response.
        timeout: Per-request timeout in seconds.
        image_format: ``"jpeg"`` (default) or ``"raw"`` for lossless transport.
        quality: JPEG quality when applicable.
    """

    def __init__(
        self,
        url: str = "http://localhost:8000",
        *,
        timeout: float = 10.0,
        image_format: str = "jpeg",
        quality: int = 90,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.image_format = image_format
        self.quality = quality
        self._ws: Any = None
        self.use_websocket = self.url.startswith(("ws://", "wss://"))

    # -- transport -----------------------------------------------------------

    def _http_url(self, path: str) -> str:
        base = self.url
        if base.startswith("ws://"):
            base = "http://" + base[len("ws://") :]
        elif base.startswith("wss://"):
            base = "https://" + base[len("wss://") :]
        return base + path

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            self._http_url(path), data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError(f"server returned {exc.code}: {detail}") from exc
        except urlerror.URLError as exc:
            raise BackendError(f"cannot reach {self.url}: {exc.reason}") from exc

    def _get(self, path: str) -> dict:
        try:
            with urlrequest.urlopen(self._http_url(path), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise BackendError(f"cannot reach {self.url}: {exc.reason}") from exc

    def connect(self) -> "VLAClient":
        """Open the websocket. Called lazily by :meth:`predict` if needed."""
        if not self.use_websocket or self._ws is not None:
            return self
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise DependencyError(
                "websockets", "the websocket client", extra="vla-engine[client]"
            ) from exc
        self._ws = connect(f"{self.url}/stream", open_timeout=self.timeout)
        logger.info("connected to %s/stream", self.url)
        return self

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def __enter__(self) -> "VLAClient":
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- inference -----------------------------------------------------------

    def predict(self, observation: Observation) -> ActionChunk:
        """Request an action chunk for one observation."""
        payload = encode_observation(
            observation, image_format=self.image_format, quality=self.quality
        )
        if self.use_websocket:
            self.connect()
            assert self._ws is not None
            self._ws.send(json.dumps(payload))
            response = json.loads(self._ws.recv(timeout=self.timeout))
        else:
            response = self._post("/predict", payload)
        if "error" in response:
            raise BackendError(f"server error: {response['error']}")
        return decode_chunk(response)

    def act(self, image: Any, instruction: str, state: Any | None = None) -> ActionChunk:
        """Shorthand for the single-camera case."""
        return self.predict(Observation.single(image, instruction, state))

    # -- introspection -------------------------------------------------------

    def info(self) -> dict:
        return self._get("/info")

    def health(self) -> dict:
        return self._get("/healthz")

    def metrics(self) -> dict:
        return self._get("/metrics")
