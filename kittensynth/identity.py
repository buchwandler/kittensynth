"""Stable, dependency-light identity for KittenSynth synthesis."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from types import MappingProxyType

from .api_contract import REQUEST_API_VERSION


def _distribution_version_safe(distribution: str) -> str | None:
    try:
        return _distribution_version(distribution)
    except PackageNotFoundError:
        return None


def runtime_identity() -> Mapping[str, str | None]:
    """Return stable software-version inputs that can affect generated PCM.

    This function reads package metadata only. It does not inspect catalog or
    model files, initialize a runtime, or include machine-specific state.
    """
    return MappingProxyType(
        {
            "engine": "kitten",
            "engine_version": _distribution_version_safe("kittensynth"),
            "g2p_revision": _distribution_version_safe("kitteng2p"),
            "runtime_revision": _distribution_version_safe("onnxvoice"),
            "request_api_version": str(REQUEST_API_VERSION),
        }
    )


__all__ = ["runtime_identity"]
