"""Exception hierarchy for the VLA inference engine."""

from __future__ import annotations


class VLAEngineError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(VLAEngineError):
    """A configuration value is missing, malformed, or mutually inconsistent."""


class ModelNotFoundError(VLAEngineError):
    """The requested model id is not present in the registry."""


class DependencyError(VLAEngineError):
    """An optional third-party dependency is required but not installed."""

    def __init__(self, package: str, reason: str, extra: str | None = None) -> None:
        hint = f"pip install '{extra}'" if extra else f"pip install {package}"
        super().__init__(f"{reason} requires `{package}`, which is not installed. Try: {hint}")
        self.package = package
        self.extra = extra


class BackendError(VLAEngineError):
    """A runtime backend (CUDA, compile, quantization) failed in a non-recoverable way."""


class ObservationError(VLAEngineError):
    """An observation does not satisfy the loaded policy's input contract."""


class NotLoadedError(VLAEngineError):
    """An operation needs model weights that have not been loaded yet."""
