from __future__ import annotations

import math

from kittensynth.voice_level import VoiceCalibrationKey, default_voice_calibration

EXPECTED_INTERNAL_VOICES = {
    "expr-voice-2-f",
    "expr-voice-2-m",
    "expr-voice-3-f",
    "expr-voice-3-m",
    "expr-voice-4-f",
    "expr-voice-4-m",
    "expr-voice-5-f",
    "expr-voice-5-m",
}


def test_packaged_voice_calibration_catalog_loads():
    catalog = default_voice_calibration()

    assert catalog.schema == 1
    assert catalog.method == "bs1770"
    assert catalog.corpus == "kittensynth-prepared-speech-v1"
    assert catalog.reference_lufs == -24.0
    assert len(catalog.revision or "") == 64
    expected_keys = {
        VoiceCalibrationKey("kitten", "nano-0.8-int8", voice) for voice in EXPECTED_INTERNAL_VOICES
    }
    assert set(catalog.voices) == expected_keys

    for record in catalog.voices.values():
        assert math.isfinite(record.gain_db)
        assert -12.0 < record.gain_db < 8.0
        assert record.measured_lufs is not None and math.isfinite(record.measured_lufs)
        assert record.reference_lufs == -24.0
        assert record.mad_lu is not None and math.isfinite(record.mad_lu)
        assert record.mad_lu <= 0.75
        assert record.samples == 9
        assert record.method == "bs1770"
        assert record.corpus_version == catalog.corpus
