from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import InvalidSpeedError, InvalidSynthesisConfigError
from .voice_level import VoiceLevelConfig


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    speed: float = 1.0
    voice_level: VoiceLevelConfig = field(default_factory=VoiceLevelConfig)

    def validated(self) -> SynthesisConfig:
        value = float(self.speed)
        if not math.isfinite(value) or value <= 0:
            raise InvalidSpeedError("speed must be a finite positive number")
        if not isinstance(self.voice_level, VoiceLevelConfig):
            raise InvalidSynthesisConfigError("voice_level must be a VoiceLevelConfig")
        return SynthesisConfig(speed=value, voice_level=self.voice_level)
