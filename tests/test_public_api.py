import subprocess
import sys
from typing import Any, cast

import kittensynth
from kittensynth import ModelDownloadError


def test_readio_expected_public_api_and_typed_errors_are_exported():
    required = (
        "KittenVoice",
        "SynthesisConfig",
        "SynthesisResult",
        "SynthesisMeasure",
        "discover_models",
        "runtime_identity",
    )
    required_errors = (
        "KittenSynthError",
        "EmptyTextError",
        "InvalidSpeedError",
        "InvalidVoiceError",
        "UnsupportedModelError",
        "OnnxVoiceContractError",
        "ModelDownloadError",
        "SynthesisInputTooLongError",
    )

    assert all(hasattr(kittensynth, name) for name in (*required, *required_errors))
    assert all(name in kittensynth.__all__ for name in (*required, *required_errors))
    assert kittensynth.ModelDownloadError is ModelDownloadError
    assert callable(kittensynth.KittenVoice.synthesize_prepared)
    assert callable(kittensynth.KittenVoice.measure_prepared)
    assert callable(kittensynth.KittenVoice.from_pretrained)
    assert callable(kittensynth.KittenVoice.from_local)
    assert callable(kittensynth.KittenVoice.close)
    assert callable(kittensynth.discover_models)
    assert callable(kittensynth.runtime_identity)


def test_request_api_contract_is_truthful_and_public():
    expected = {
        "entrypoint": "KittenVoice.synthesize_prepared",
        "supports_linguistic_tokens": False,
        "supports_pronunciation_overrides": False,
        "supports_whole_request_phonemes": False,
        "supports_speakers": False,
        "supports_word_timings": False,
        "supports_voice_level": False,
        "caller_owns_text_boundaries": True,
        "supports_request_measurement": True,
        "measurement_entrypoint": "KittenVoice.measure_prepared",
        "measurement_metric": "model_tokens",
    }

    assert kittensynth.REQUEST_API_VERSION == 1
    assert "REQUEST_API_VERSION" in kittensynth.__all__
    assert "request_api_contract" in kittensynth.__all__
    assert dict(kittensynth.request_api_contract()) == expected
    try:
        cast(Any, kittensynth.request_api_contract())["supports_speakers"] = True
    except TypeError:
        pass
    else:
        raise AssertionError("request API contract must be immutable")


def test_measurement_public_api_import_does_not_load_onnx_runtime():
    script = """
import sys
import kittensynth
assert kittensynth.SynthesisMeasure
assert callable(kittensynth.KittenVoice.measure_prepared)
assert kittensynth.request_api_contract()["measurement_metric"] == "model_tokens"
assert "onnxvoice" not in sys.modules
assert "onnxruntime" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
