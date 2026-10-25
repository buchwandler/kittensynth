import kittensynth
from kittensynth import ModelDownloadError


def test_public_errors_are_typed_and_discovery_runtime_errors_are_distinct():
    required = (
        "KittenSynthError",
        "EmptyTextError",
        "InvalidSpeedError",
        "InvalidVoiceError",
        "UnsupportedModelError",
        "OnnxVoiceContractError",
        "CatalogDiscoveryError",
        "CatalogUnavailableError",
        "ManagedModelDependencyError",
        "ModelInferenceError",
        "ModelDownloadError",
        "SynthesisInputTooLongError",
    )
    for name in required:
        error_type = getattr(kittensynth, name)
        assert isinstance(error_type, type)
        assert issubclass(error_type, kittensynth.KittenSynthError)
        assert name in kittensynth.__all__

    assert kittensynth.ModelDownloadError is ModelDownloadError
    assert issubclass(kittensynth.CatalogUnavailableError, kittensynth.CatalogDiscoveryError)
    assert not issubclass(kittensynth.ModelInferenceError, ValueError)
    assert not issubclass(kittensynth.CatalogUnavailableError, ValueError)


def test_public_overflow_error_exposes_capacity_fields():
    error = kittensynth.SynthesisInputTooLongError(513, 512)

    assert error.amount == 513
    assert error.maximum == 512
    assert error.unit == "model_tokens"
    assert error.token_count == 513
    assert error.max_tokens == 512
