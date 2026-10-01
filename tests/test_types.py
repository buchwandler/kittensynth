import wave

import numpy as np

from kittensynth.types import SynthesisResult


def test_wav_output(tmp_path):
    result = SynthesisResult(
        audio=np.array([0.0, 0.5, -0.5], dtype=np.float32),
        sample_rate=24000,
        voice="Jasper",
        model_ref="kitten:test",
        speed=1.0,
    )
    path = tmp_path / "out.wav"
    result.save_wav(path)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 24000
        assert handle.getnframes() == 3
