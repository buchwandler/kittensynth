from pathlib import Path

import numpy as np
import pytest

from kittensynth.voice_bank import VoiceBank


def test_upstream_style_index_and_speed_prior(tmp_path: Path):
    path = tmp_path / "voices.npz"
    np.savez(path, **{"expr-voice-2-f": np.arange(24, dtype=np.float32).reshape(6, 4)})
    bank = VoiceBank(
        path,
        voice_aliases={"Bella": "expr-voice-2-f"},
        speed_priors={"expr-voice-2-f": 0.8},
    )
    assert bank.style_for("Bella", text_length=2).tolist() == [[8.0, 9.0, 10.0, 11.0]]
    assert bank.style_for("Bella", text_length=999).tolist() == [[20.0, 21.0, 22.0, 23.0]]
    assert bank.effective_speed("Bella", 1.25) == pytest.approx(1.0)
