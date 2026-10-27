"""KittenTTS synthesis frontend backed by kitteng2p + OnnxVoice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .api_contract import REQUEST_API_VERSION, request_api_contract

try:
    from kitteng2p import (
        G2LexFileInspection,
        G2LexFileRef,
        KittenG2P,
        KittenG2PConfig,
        LexiconDiscoveryResult,
        LinguisticToken,
        NamedLexiconInfo,
        PhonemizeResult,
        PronunciationDiagnostic,
        PronunciationOverride,
        PronunciationTrace,
        discover_lexicons,
        inspect_g2lex_file,
    )
except ImportError as exc:
    raise ImportError(
        "KittenSynth requires kitteng2p>=0.1.2 for the public G2Lex request API"
    ) from exc
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
    ManagedModelDependencyError,
    ModelDownloadError,
    ModelInferenceError,
    OnnxVoiceContractError,
    SynthesisInputTooLongError,
    UnsupportedModelError,
)
from .identity import runtime_identity
from .types import SynthesisMeasure, SynthesisResult, VoiceInfo
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
    "G2LexFileInspection",
    "G2LexFileRef",
    "KittenG2P",
    "KittenG2PConfig",
    "LexiconDiscoveryResult",
    "LinguisticToken",
    "NamedLexiconInfo",
    "PhonemizeResult",
    "PronunciationDiagnostic",
    "PronunciationOverride",
    "PronunciationTrace",
    "discover_lexicons",
    "inspect_g2lex_file",
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
    "ManagedModelDependencyError",
    "ModelDownloadError",
    "ModelInferenceError",
    "SynthesisInputTooLongError",
    "KittenVoice",
    "OnnxVoiceContractError",
    "UnsupportedModelError",
    "SynthesisConfig",
    "SynthesisMeasure",
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
