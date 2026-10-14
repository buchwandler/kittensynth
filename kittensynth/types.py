from __future__ import annotations

import math
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    name: str
    internal_id: str


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    voice: str
    model_ref: str | None
    speed: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def duration_seconds(self) -> float:
        return float(self.audio.size) / float(self.sample_rate)

    def save_wav(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(self.audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm.tobytes())


@dataclass(frozen=True, slots=True)
class SynthesisMeasure:
    """Exact model-input measurement for a prepared synthesis request."""

    token_count: int
    max_tokens: int | None
    fits: bool | None
    model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")

        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None")

        if self.fits is not None and not isinstance(self.fits, bool):
            raise ValueError("fits must be a bool or None")
        expected_fits = None if self.max_tokens is None else self.token_count <= self.max_tokens
        if self.fits is not expected_fits:
            raise ValueError("fits must match token_count and max_tokens")

        if self.model_id is not None and not isinstance(self.model_id, str):
            raise ValueError("model_id must be a string or None")

        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        metadata = dict(self.metadata)
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("metadata values must be JSON-friendly scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("metadata float values must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
