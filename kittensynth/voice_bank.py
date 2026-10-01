from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .errors import InvalidSpeedError, InvalidVoiceError

DEFAULT_VOICE_ALIASES: dict[str, str] = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


class VoiceBank:
    def __init__(
        self,
        path: str | Path,
        *,
        voice_aliases: Mapping[str, str] | None = None,
        speed_priors: Mapping[str, float] | None = None,
    ) -> None:
        self.path = Path(path)
        aliases = dict(DEFAULT_VOICE_ALIASES if voice_aliases is None else voice_aliases)
        self.voice_aliases = {str(key): str(value) for key, value in aliases.items()}
        self.speed_priors = {str(key): float(value) for key, value in (speed_priors or {}).items()}
        with np.load(self.path, allow_pickle=False) as archive:
            self._styles = {str(name): np.asarray(archive[name]) for name in archive.files}
        if not self._styles:
            raise ValueError("voices.npz is empty")

    @property
    def available_voices(self) -> tuple[str, ...]:
        aliases = [
            alias for alias, internal in self.voice_aliases.items() if internal in self._styles
        ]
        return tuple(aliases)

    def resolve(self, voice: str) -> str:
        internal = self.voice_aliases.get(voice, voice)
        if internal not in self._styles:
            choices = ", ".join(self.available_voices)
            raise InvalidVoiceError(f"Voice {voice!r} is unavailable; choose from {choices}")
        return internal

    def style_for(self, voice: str, *, text_length: int) -> np.ndarray:
        internal = self.resolve(voice)
        styles = np.asarray(self._styles[internal])
        if styles.ndim < 2 or styles.shape[0] < 1:
            raise ValueError(f"invalid style shape for {internal}: {styles.shape}")
        index = min(max(int(text_length), 0), styles.shape[0] - 1)
        return np.asarray(styles[index : index + 1])

    def effective_speed(self, voice: str, requested: float) -> float:
        speed = float(requested)
        if not math.isfinite(speed) or speed <= 0:
            raise InvalidSpeedError("speed must be a finite positive number")
        internal = self.resolve(voice)
        effective = speed * self.speed_priors.get(internal, 1.0)
        if not math.isfinite(effective) or effective <= 0:
            raise InvalidSpeedError("effective speed must be finite and positive")
        return effective
