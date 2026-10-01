from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kitteng2p.types import PhonemizeResult

from kittensynth import KittenVoice


@dataclass
class FakeInference:
    audio: np.ndarray
    sample_rate: int = 24000


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, token_ids, **kwargs):
        self.calls.append((tuple(token_ids), kwargs))
        return FakeInference(np.array([[0.0, 0.25, -0.25]], dtype=np.float32))

    def close(self):
        self.closed = True


class FakeG2P:
    def __init__(self):
        self.closed = False

    def phonemize_prepared(self, text):
        return PhonemizeResult(
            text=text,
            raw_phonemes="həˈloʊ",
            phonemes="həˈloʊ",
            token_ids=(0, 1, 2, 10, 0),
            language="en-us",
        )

    def close(self):
        self.closed = True


def make_voices(path: Path):
    np.savez(
        path,
        **{"expr-voice-2-m": np.arange(40, dtype=np.float32).reshape(10, 4)},
    )


def test_synthesis_consumes_kitteng2p_ids_only(tmp_path):
    voices = tmp_path / "voices.npz"
    make_voices(voices)
    runtime = FakeRuntime()
    g2p = FakeG2P()
    model = KittenVoice(
        runtime=runtime,
        voices_path=voices,
        metadata={
            "voice_aliases": {"Jasper": "expr-voice-2-m"},
            "speed_priors": {"expr-voice-2-m": 0.8},
        },
        model_ref="kitten:test",
        g2p=g2p,
    )

    result = model.synthesize_prepared("hello", voice="Jasper", speed=1.25)

    assert result.audio.dtype == np.float32
    ids, kwargs = runtime.calls[0]
    assert ids == (0, 1, 2, 10, 0)
    assert kwargs["style"].shape == (1, 4)
    assert kwargs["speed"] == 1.0
    assert result.metadata["phonemes"] == "həˈloʊ"

    model.close()
    assert runtime.closed
    assert g2p.closed is False  # injected g2p is caller-owned
