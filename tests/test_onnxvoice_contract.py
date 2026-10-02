from collections.abc import Sequence
from typing import Any, get_type_hints

from onnxvoice import OnnxVoice

from kittensynth import _onnxvoice
from kittensynth.voice import KittenVoice

EXPECTED_PROVIDER_OPTIONS = Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None


def test_provider_options_match_onnxvoice_02_contract():
    functions = (
        OnnxVoice.open,
        OnnxVoice.open_local,
        KittenVoice.from_pretrained,
        KittenVoice.from_local,
        _onnxvoice.open_installed_model,
        _onnxvoice.open_local_model,
    )

    for function in functions:
        assert get_type_hints(function)["provider_options"] == EXPECTED_PROVIDER_OPTIONS
