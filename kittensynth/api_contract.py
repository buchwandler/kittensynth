"""Declared capabilities of the current KittenSynth request API."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

REQUEST_API_VERSION = 1

_REQUEST_API_CONTRACT: Mapping[str, str | bool] = MappingProxyType(
    {
        "entrypoint": "KittenVoice.synthesize_prepared",
        "supports_linguistic_tokens": False,
        "supports_pronunciation_overrides": False,
        "supports_whole_request_phonemes": False,
        "supports_speakers": False,
        "supports_word_timings": False,
        "supports_voice_level": False,
        "caller_owns_text_boundaries": True,
    }
)


def request_api_contract() -> Mapping[str, str | bool]:
    """Return an immutable declaration of the currently supported request API."""
    return _REQUEST_API_CONTRACT


__all__ = ["REQUEST_API_VERSION", "request_api_contract"]
