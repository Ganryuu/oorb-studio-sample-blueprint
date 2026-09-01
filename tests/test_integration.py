"""End-to-end: a real uvicorn server driven by the real client, then a control loop.

Everything else in the suite exercises components in isolation or through
FastAPI's in-process TestClient. This module starts an actual HTTP server on a
socket and talks to it with :class:`~vla_engine.serve.client.VLAClient`, which
is the only way to catch transport-level mistakes -- URL scheme handling,
websocket framing, JSON round-tripping over the wire.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from vla_engine import EngineConfig, Observation
from vla_engine.control.chunker import ChunkPolicy

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import uvicorn  # noqa: E402

from vla_engine.serve.client import VLAClient  # noqa: E402
from vla_engine.serve.server import create_app  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server():
    """A real uvicorn server on a real socket, torn down after the module."""
    port = _free_port()
    config = EngineConfig(model="echo", extra={"horizon": 50, "action_dim": 7})
    server = uvicorn.Server(
        uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while time.time() < deadline:
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    else:  # pragma: no cover
        pytest.fail("server did not start within 30s")

    yield f"127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


class TestHttpClient:
    def test_health_and_info(self, live_server):
        client = VLAClient(f"http://{live_server}")
        assert client.health()["status"] == "ok"
        assert client.info()["model"] == "echo"

    def test_act_returns_a_chunk(self, live_server, frame):
        client = VLAClient(f"http://{live_server}")
        chunk = client.act(frame, "pick up the red block")
        assert chunk.actions.shape == (50, 7)
        assert chunk.actions.dtype == np.float32

    def test_matches_local_inference(self, live_server, frame):
        """Remote inference must agree with local inference on the same input."""
        from vla_engine import VLAEngine

        local = VLAEngine(EngineConfig(model="echo", extra={"horizon": 50, "action_dim": 7})).load()
        try:
            remote = VLAClient(f"http://{live_server}", image_format="raw")
            observation = Observation.single(frame, "pick up the red block")
            assert np.allclose(
                local.predict(observation).actions,
                remote.predict(observation).actions,
                atol=1e-5,
            )
        finally:
            local.unload()

    def test_unreachable_server_raises_a_clear_error(self):
        from vla_engine.errors import BackendError

        client = VLAClient(f"http://127.0.0.1:{_free_port()}", timeout=2.0)
        with pytest.raises(BackendError, match="cannot reach"):
            client.health()

    def test_server_side_error_surfaces_to_the_client(self, live_server):
        from vla_engine.errors import BackendError

        client = VLAClient(f"http://{live_server}")
        with pytest.raises(BackendError):
            client._post("/predict", {"images": {}})


class TestWebsocketClient:
    def test_streams_over_a_persistent_connection(self, live_server, frame):
        pytest.importorskip("websockets")
        observation = Observation.single(frame, "pick up the red block")
        with VLAClient(f"ws://{live_server}") as client:
            first = client.predict(observation)
            for _ in range(4):
                chunk = client.predict(observation)
                assert np.array_equal(chunk.actions, first.actions)

    def test_info_works_over_the_ws_url_scheme(self, live_server):
        """A ws:// client must still reach the HTTP introspection endpoints."""
        pytest.importorskip("websockets")
        assert VLAClient(f"ws://{live_server}").info()["model"] == "echo"


class TestRemoteControlLoop:
    def test_slow_remote_policy_drives_a_fast_control_loop(self, live_server, frame):
        """The point of chunking: a 5 Hz policy must not stutter a 50 Hz loop."""
        pytest.importorskip("websockets")
        from vla_engine.control.chunker import ChunkExecutor

        executor = ChunkExecutor(7, ChunkPolicy(ensemble_decay=0.1, discrete_dims=(6,)))
        observation = Observation.single(frame, "pick up the red block")

        with VLAClient(f"ws://{live_server}") as client:
            commands = []
            for _ in range(5):  # 5 inferences
                executor.submit(client.predict(observation))
                for _ in range(10):  # 10 control ticks each
                    action = executor.step()
                    assert action is not None
                    commands.append(action)

        assert len(commands) == 50
        assert executor.starved_steps == 0
        # Blended commands must stay finite and bounded.
        stacked = np.stack(commands)
        assert np.all(np.isfinite(stacked))
        assert np.abs(stacked).max() <= 1.0
