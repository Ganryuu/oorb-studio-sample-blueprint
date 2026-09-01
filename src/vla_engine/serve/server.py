"""FastAPI inference server.

Runs on the machine with the GPUs; robots, MuJoCo instances, and the ROS 2 node
talk to it over the network. Two transports:

``POST /predict``
    Request/response. Simple, works from ``curl``, right for episodic use.

``WS /stream``
    A persistent websocket. This is what a control loop should use: it avoids
    per-request TCP and TLS handshakes, which on a LAN is worth more than the
    inference savings this engine works so hard for. A 5 ms handshake on a 25 ms
    forward pass is a 20% latency tax paid on every step for nothing.

Concurrency: inference runs in a worker thread via the batcher (or directly via
``asyncio.to_thread``), so a slow forward pass never blocks the event loop from
accepting other connections or serving ``/healthz``.
"""

# NOTE: deliberately no `from __future__ import annotations` in this module.
# FastAPI resolves endpoint annotations with typing.get_type_hints() against
# module globals. Because fastapi is an optional dependency it is imported
# inside create_app(), so `WebSocket` is a local name. With postponed
# evaluation the annotation stays the string "WebSocket", FastAPI fails to
# resolve it, silently treats the parameter as a query field, and every
# websocket connection is rejected before the handler runs.

import asyncio
import contextlib
import logging
import time
from typing import Any

from ..config import EngineConfig
from ..engine import VLAEngine
from ..errors import DependencyError, VLAEngineError
from .protocol import decode_observation, encode_chunk

logger = logging.getLogger(__name__)

__all__ = ["create_app", "serve"]


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        raise DependencyError("fastapi", "the inference server", extra="vla-engine[serve]") from exc


def create_app(config: EngineConfig, *, engine: VLAEngine | None = None):
    """Build the FastAPI application.

    Args:
        config: Engine configuration. ``config.batch.max_batch_size > 1``
            enables continuous batching.
        engine: A preloaded engine, mainly for tests. Otherwise one is built
            from ``config`` and loaded during application startup.

    Returns:
        A configured ``fastapi.FastAPI`` instance.
    """
    _require_fastapi()
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

    from ..optim.batching import ContinuousBatcher

    state: dict[str, Any] = {"engine": engine, "batcher": None, "started_at": time.time()}

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        """Load weights before accepting traffic; release them on shutdown.

        Loading here rather than at import time means ``/healthz`` reports
        "loading" instead of the process refusing connections while a 7B
        checkpoint is read off disk.
        """
        if state["engine"] is None:
            logger.info("loading %s ...", config.model)
            state["engine"] = await asyncio.to_thread(lambda: VLAEngine(config).load())
        if config.batch.enabled:
            batcher = ContinuousBatcher(state["engine"].predict_batch, config.batch)
            await batcher.start()
            state["batcher"] = batcher
        logger.info("server ready on %s", state["engine"].devices)
        try:
            yield
        finally:
            if state["batcher"] is not None:
                await state["batcher"].stop()
            if state["engine"] is not None:
                state["engine"].unload()

    app = FastAPI(
        title="vla-engine",
        description="Inference server for OpenVLA, pi0, and SmolVLA",
        version="0.1.0",
        lifespan=lifespan,
    )

    async def _predict(payload: dict) -> dict:
        """Shared path for both transports."""
        observation = decode_observation(payload)
        batcher = state["batcher"]
        if batcher is not None:
            chunk = await batcher.submit(observation)
        else:
            # Still off the event loop: a 100 ms forward would otherwise stall
            # every other connection, including health checks.
            chunk = await asyncio.to_thread(state["engine"].predict, observation)
        return encode_chunk(chunk)

    @app.get("/healthz")
    async def healthz() -> dict:
        """Liveness probe. Cheap and never touches the GPU."""
        engine = state["engine"]
        return {
            "status": "ok" if engine is not None and engine.is_loaded else "loading",
            "uptime_s": round(time.time() - state["started_at"], 1),
        }

    @app.get("/info")
    async def info() -> dict:
        """Full engine state: model, devices, precision, and its reasoning."""
        engine = state["engine"]
        if engine is None:
            raise HTTPException(status_code=503, detail="engine is still loading")
        return engine.info()

    @app.get("/metrics")
    async def metrics() -> dict:
        """Batching counters, for spotting a GPU that cannot keep up."""
        batcher = state["batcher"]
        return {
            "batching": batcher.stats.as_dict() if batcher else {"enabled": False},
            "uptime_s": round(time.time() - state["started_at"], 1),
        }

    @app.post("/predict")
    async def predict(payload: dict) -> dict:
        """Predict an action chunk for one observation."""
        try:
            return await _predict(payload)
        except VLAEngineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        """Persistent control-loop transport: one observation in, one chunk out."""
        await websocket.accept()
        peer = websocket.client
        logger.info("stream connected: %s", peer)
        try:
            while True:
                payload = await websocket.receive_json()
                try:
                    await websocket.send_json(await _predict(payload))
                except VLAEngineError as exc:
                    # Report and keep the socket open: one malformed frame
                    # should not cost the robot its connection.
                    await websocket.send_json({"error": str(exc)})
        except WebSocketDisconnect:
            logger.info("stream disconnected: %s", peer)
        except Exception:
            logger.exception("stream failed for %s", peer)
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

    return app


def serve(
    config: EngineConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    log_level: str = "info",
) -> None:
    """Run the server with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise DependencyError("uvicorn", "the inference server", extra="vla-engine[serve]") from exc
    uvicorn.run(create_app(config), host=host, port=port, log_level=log_level)
