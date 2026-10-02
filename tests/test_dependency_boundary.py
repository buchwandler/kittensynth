from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_g2p_is_a_separate_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "kitteng2p" in deps
    assert "onnxvoice" in deps
    assert "phonemizer" not in deps
    assert "espeakng-runtime" not in deps


def test_kitteng2p_dependency_floors_match_the_verified_release():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    base = next(dep for dep in data["project"]["dependencies"] if dep.startswith("kitteng2p"))
    bundled = data["project"]["optional-dependencies"]["bundled-g2p"][0]

    assert base == "kitteng2p>=0.1.1,<0.2"
    assert bundled == "kitteng2p[bundled]>=0.1.1,<0.2"
