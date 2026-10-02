"""KittenTTS synthesis frontend backed by kitteng2p + OnnxVoice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .config import SynthesisConfig
from .discovery import (
    DescribedVoice,
    DiscoveredModel,
    discover_models,
    runtime_identity,
)
from .errors import (
    CatalogDiscoveryError,
    EmptyTextError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVoiceError,
    KittenSynthError,
    OnnxVoiceContractError,
    UnsupportedModelError,
)
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
    "DEFAULT_VOICE_ALIASES",
    "CatalogDiscoveryError",
    "DescribedVoice",
    "DiscoveredModel",
    "EmptyTextError",
    "InvalidSpeedError",
    "InvalidSynthesisConfigError",
    "InvalidVoiceError",
    "KittenSynthError",
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
    "load_voice_calibration",
    "runtime_identity",
]
