from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidSpeedError


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    speed: float = 1.0

    def validated(self) -> SynthesisConfig:
        value = float(self.speed)
        if not math.isfinite(value) or value <= 0:
            raise InvalidSpeedError("speed must be a finite positive number")
        return self
