"""KittenTTS synthesis frontend backed by kitteng2p + OnnxVoice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .api_contract import REQUEST_API_VERSION, request_api_contract
from .config import SynthesisConfig
from .discovery import DescribedVoice, DiscoveredModel, discover_models
from .errors import (
    CatalogDiscoveryError,
    CatalogUnavailableError,
    EmptyTextError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVoiceError,
    KittenSynthError,
    ModelInferenceError,
    OnnxVoiceContractError,
    UnsupportedModelError,
)
from .identity import runtime_identity
from .types import SynthesisResult, VoiceInfo
from .voice import KittenVoice
from .voice_bank import DEFAULT_VOICE_ALIASES, VoiceBank
from .voice_level import (
    CalibrationDataError,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    VoiceLevelMode,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

try:
    __version__ = _distribution_version("kittensynth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "REQUEST_API_VERSION",
    "DEFAULT_VOICE_ALIASES",
    "CatalogDiscoveryError",
    "CatalogUnavailableError",
    "DescribedVoice",
    "DiscoveredModel",
    "EmptyTextError",
    "InvalidSpeedError",
    "InvalidSynthesisConfigError",
    "InvalidVoiceError",
    "KittenSynthError",
    "ModelInferenceError",
    "KittenVoice",
    "OnnxVoiceContractError",
    "UnsupportedModelError",
    "SynthesisConfig",
    "SynthesisResult",
    "VoiceBank",
    "VoiceInfo",
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "discover_models",
    "request_api_contract",
    "load_voice_calibration",
    "runtime_identity",
]
