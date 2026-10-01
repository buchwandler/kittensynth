"""KittenTTS synthesis frontend backed by kitteng2p + OnnxVoice."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .config import SynthesisConfig
from .types import SynthesisResult, VoiceInfo
from .voice import KittenVoice
from .voice_bank import DEFAULT_VOICE_ALIASES, VoiceBank

try:
    __version__ = _distribution_version("kittensynth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "DEFAULT_VOICE_ALIASES",
    "KittenVoice",
    "SynthesisConfig",
    "SynthesisResult",
    "VoiceBank",
    "VoiceInfo",
]
