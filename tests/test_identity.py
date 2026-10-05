from pathlib import Path
from typing import Any, cast

from kittensynth import identity


def test_runtime_identity_is_exact_deterministic_and_path_free(monkeypatch):
    versions = {
        "kittensynth": "0.1.2",
        "kitteng2p": "0.1.1",
        "onnxvoice": "0.2.3",
    }
    monkeypatch.setattr(identity, "_distribution_version_safe", versions.__getitem__)

    identity_map = identity.runtime_identity()

    assert dict(identity_map) == {
        "engine": "kitten",
        "engine_version": "0.1.2",
        "g2p_revision": "0.1.1",
        "runtime_revision": "0.2.3",
        "request_api_version": "1",
    }
    assert all(not isinstance(value, Path) for value in identity_map.values())
    assert all("/" not in value for value in identity_map.values() if value is not None)
    try:
        cast(Any, identity_map)["engine"] = "changed"
    except TypeError:
        pass
    else:
        raise AssertionError("runtime identity must be immutable")


def test_runtime_identity_uses_none_for_missing_distribution(monkeypatch):
    monkeypatch.setattr(
        identity,
        "_distribution_version_safe",
        lambda distribution: None if distribution == "onnxvoice" else "1",
    )

    assert identity.runtime_identity()["runtime_revision"] is None
