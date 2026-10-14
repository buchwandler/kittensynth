import wave

import numpy as np
import pytest

from kittensynth.types import SynthesisMeasure, SynthesisResult


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_count": True, "max_tokens": None, "fits": None},
        {"token_count": -1, "max_tokens": None, "fits": None},
        {"token_count": None, "max_tokens": None, "fits": None},
        {"token_count": 1.5, "max_tokens": None, "fits": None},
        {"token_count": 1, "max_tokens": True, "fits": True},
        {"token_count": 1, "max_tokens": 0, "fits": True},
        {"token_count": 1, "max_tokens": -1, "fits": False},
        {"token_count": 1, "max_tokens": 1.5, "fits": True},
        {"token_count": 1, "max_tokens": 2, "fits": 1},
        {"token_count": 2, "max_tokens": 1, "fits": True},
        {"token_count": 1, "max_tokens": None, "fits": False},
        {"token_count": 1, "max_tokens": 2, "fits": None},
    ],
)
def test_synthesis_measure_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        SynthesisMeasure(**kwargs)


def test_synthesis_measure_freezes_fields_and_copies_metadata():
    metadata = {"frontend": "kitteng2p", "token_count": 3, "exact": True}
    measure = SynthesisMeasure(token_count=3, max_tokens=None, fits=None, metadata=metadata)

    metadata["extra"] = "not copied"
    assert dict(measure.metadata) == {
        "frontend": "kitteng2p",
        "token_count": 3,
        "exact": True,
    }
    with pytest.raises(TypeError):
        measure.metadata["changed"] = True
    with pytest.raises(AttributeError):
        measure.token_count = 4


def test_synthesis_measure_rejects_invalid_metadata_and_model_id():
    for metadata in (
        {1: "invalid key"},
        {"nested": {"not": "scalar"}},
        {"infinite": float("inf")},
        {"nan": float("nan")},
        [],
    ):
        with pytest.raises(ValueError):
            SynthesisMeasure(token_count=0, max_tokens=None, fits=None, metadata=metadata)

    with pytest.raises(ValueError, match="model_id"):
        SynthesisMeasure(
            token_count=0,
            max_tokens=None,
            fits=None,
            model_id=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("token_count", "expected_fits"), [(2, True), (3, False)])
def test_synthesis_measure_reports_known_maximum_fit(token_count, expected_fits):
    measure = SynthesisMeasure(
        token_count=token_count,
        max_tokens=2,
        fits=expected_fits,
    )
    assert measure.fits is expected_fits
