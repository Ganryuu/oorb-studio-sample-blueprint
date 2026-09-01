"""vla-engine: an optimized inference engine for popular vision-language-action models.

Supports OpenVLA, pi0, and SmolVLA behind one interface, with a Python API, an
HTTP/websocket server, and a ROS 2 node.

Example:
    >>> from vla_engine import VLAEngine, Observation
    >>> engine = VLAEngine.from_pretrained("openvla", unnorm_key="bridge_orig")
    >>> chunk = engine.act(frame, "pick up the red block")
"""

from .config import BatchConfig, CompileConfig, DecodeConfig, EngineConfig, PrecisionConfig
from .control import ActionStats, ChunkExecutor, ChunkPolicy, Normalizer
from .engine import VLAEngine
from .errors import (
    BackendError,
    ConfigError,
    DependencyError,
    ModelNotFoundError,
    NotLoadedError,
    ObservationError,
    VLAEngineError,
)
from .registry import ModelCard, get_card, list_models, register
from .types import ActionChunk, InferenceStats, Observation, PolicySpec

__version__ = "0.1.0"

__all__ = [
    "VLAEngine",
    "EngineConfig",
    "PrecisionConfig",
    "CompileConfig",
    "DecodeConfig",
    "BatchConfig",
    "Observation",
    "ActionChunk",
    "InferenceStats",
    "PolicySpec",
    "ChunkExecutor",
    "ChunkPolicy",
    "ActionStats",
    "Normalizer",
    "ModelCard",
    "get_card",
    "list_models",
    "register",
    "VLAEngineError",
    "ConfigError",
    "ModelNotFoundError",
    "DependencyError",
    "BackendError",
    "ObservationError",
    "NotLoadedError",
    "__version__",
]
