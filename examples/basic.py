#!/usr/bin/env python3
"""Synthesize one prepared English request with the default managed Kitten model."""

from __future__ import annotations

import os

from kittensynth import KittenVoice, SynthesisConfig, VoiceLevelConfig

try:
    from examples._output import artefact_path
except ModuleNotFoundError:
    from _output import artefact_path

MODEL = os.environ.get("KITTENSYNTH_EXAMPLE_MODEL", "nano-0.8-int8")
VOICE = os.environ.get("KITTENSYNTH_EXAMPLE_VOICE", "Jasper")
TEXT = "Hello from KittenSynth. This is prepared, speakable text."

with KittenVoice.from_pretrained(MODEL) as model:
    result = model.synthesize_prepared(
        TEXT,
        voice=VOICE,
        config=SynthesisConfig(
            speed=1.0,
            voice_level=VoiceLevelConfig(mode="calibrated"),
        ),
    )

wav_path = artefact_path("basic.wav")
result.save_wav(wav_path)
print(
    f"Model: {MODEL}\n"
    f"Voice: {VOICE}\n"
    f"Duration: {result.duration_seconds:.2f}s\n"
    f"Sample rate: {result.sample_rate} Hz\n"
    f"WAV: {wav_path}"
)
