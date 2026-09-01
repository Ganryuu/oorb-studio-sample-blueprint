"""Wire protocol, HTTP/websocket server, and the client."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vla_engine import EngineConfig
from vla_engine.errors import ObservationError
from vla_engine.serve.protocol import (
    decode_chunk,
    decode_image,
    decode_observation,
    encode_chunk,
    encode_image,
    encode_observation,
)
from vla_engine.types import ActionChunk, InferenceStats

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vla_engine.serve.server import create_app  # noqa: E402


class TestProtocol:
    def test_raw_encoding_is_lossless(self, frame):
        assert np.array_equal(decode_image(encode_image(frame, "raw")), frame)

    def test_jpeg_is_much_smaller_than_raw(self, frame):
        """Bandwidth decides control-loop jitter over a network."""
        jpeg = len(json.dumps(encode_image(frame, "jpeg")))
        raw = len(json.dumps(encode_image(frame, "raw")))
        assert raw / jpeg > 5

    def test_jpeg_is_visually_faithful(self, frame):
        decoded = decode_image(encode_image(frame, "jpeg", quality=90))
        assert decoded.shape == frame.shape
        assert np.abs(decoded.astype(int) - frame.astype(int)).mean() < 5.0

    def test_observation_roundtrip(self, observation):
        decoded = decode_observation(encode_observation(observation, image_format="raw"))
        assert decoded.instruction == observation.instruction
        assert np.array_equal(decoded.state, observation.state)
        assert set(decoded.images) == set(observation.images)

    def test_chunk_roundtrip_preserves_stats(self):
        chunk = ActionChunk(
            np.arange(20, dtype=np.float32).reshape(10, 2),
            stats=InferenceStats(total_ms=12.5, device="cuda:0"),
        )
        decoded = decode_chunk(encode_chunk(chunk))
        assert np.allclose(decoded.actions, chunk.actions)
        assert decoded.stats.total_ms == 12.5
        assert decoded.stats.device == "cuda:0"

    def test_payloads_are_json_serializable(self, observation):
        json.dumps(encode_observation(observation))

    def test_rejects_payload_without_images(self):
        with pytest.raises(ObservationError, match="no images"):
            decode_observation({"images": {}})

    def test_unknown_image_format_raises(self):
        with pytest.raises(ObservationError, match="unknown image format"):
            decode_image({"format": "webp", "data": "", "shape": [1, 1, 3]})


@pytest.fixture
def client():
    config = EngineConfig(model="echo", extra={"horizon": 50, "action_dim": 7})
    with TestClient(create_app(config)) as test_client:
        yield test_client


class TestServer:
    def test_healthz(self, client):
        assert client.get("/healthz").json()["status"] == "ok"

    def test_info_reports_the_loaded_model(self, client):
        assert client.get("/info").json()["model"] == "echo"

    def test_predict_returns_a_chunk(self, client, observation):
        response = client.post("/predict", json=encode_observation(observation))
        assert response.status_code == 200
        chunk = decode_chunk(response.json())
        assert chunk.actions.shape == (50, 7)

    def test_malformed_payload_is_a_400_not_a_500(self, client):
        assert client.post("/predict", json={"images": {}}).status_code == 400

    def test_websocket_streams_predictions(self, client, observation):
        payload = encode_observation(observation)
        with client.websocket_connect("/stream") as socket:
            for _ in range(3):
                socket.send_json(payload)
                assert socket.receive_json()["horizon"] == 50

    def test_websocket_survives_a_bad_frame(self, client, observation):
        """One malformed message must not cost the robot its connection."""
        with client.websocket_connect("/stream") as socket:
            socket.send_json({"images": {}})
            assert "error" in socket.receive_json()
            socket.send_json(encode_observation(observation))
            assert socket.receive_json()["horizon"] == 50

    def test_metrics_reports_batching_state(self, client):
        assert client.get("/metrics").json()["batching"] == {"enabled": False}


class TestBatchedServer:
    def test_batching_metrics_are_reported(self, observation):
        config = EngineConfig(
            model="echo", batch={"max_batch_size": 4, "max_wait_ms": 5}, extra={"horizon": 2}
        )
        with TestClient(create_app(config)) as client:
            for _ in range(3):
                assert (
                    client.post("/predict", json=encode_observation(observation)).status_code == 200
                )
            metrics = client.get("/metrics").json()["batching"]
            assert metrics["requests"] == 3
