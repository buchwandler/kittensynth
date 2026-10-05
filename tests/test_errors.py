import kittensynth


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
        "ModelInferenceError",
    )
    for name in required:
        error_type = getattr(kittensynth, name)
        assert isinstance(error_type, type)
        assert issubclass(error_type, kittensynth.KittenSynthError)
        assert name in kittensynth.__all__

    assert issubclass(kittensynth.CatalogUnavailableError, kittensynth.CatalogDiscoveryError)
    assert not issubclass(kittensynth.ModelInferenceError, ValueError)
    assert not issubclass(kittensynth.CatalogUnavailableError, ValueError)
