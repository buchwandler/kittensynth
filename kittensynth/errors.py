class KittenSynthError(Exception):
    """Base error for kittensynth."""


class EmptyTextError(KittenSynthError, ValueError):
    """Raised when a synthesis request is empty."""


class InvalidSpeedError(KittenSynthError, ValueError):
    """Raised when speed is invalid."""


class InvalidSynthesisConfigError(KittenSynthError, ValueError):
    """Raised when a synthesis configuration value is invalid."""


class InvalidVoiceError(KittenSynthError, ValueError):
    """Raised when a voice is not in the active voice archive."""


class UnsupportedModelError(KittenSynthError, RuntimeError):
    """Raised when an installation does not satisfy the Kitten contract."""


class OnnxVoiceContractError(KittenSynthError, RuntimeError):
    """Raised when the installed OnnxVoice lacks Kitten support."""


class CatalogDiscoveryError(KittenSynthError, RuntimeError):
    """Raised when the Kitten model catalog cannot be listed."""


class CatalogUnavailableError(CatalogDiscoveryError):
    """Raised when the Kitten model catalog cannot be accessed or listed."""


class ModelInferenceError(KittenSynthError, RuntimeError):
    """Raised when the Kitten runtime fails while generating audio."""
