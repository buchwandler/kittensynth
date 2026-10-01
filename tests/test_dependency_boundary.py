from pathlib import Path
import tomllib


def test_g2p_is_a_separate_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "kitteng2p" in deps
    assert "onnxvoice" in deps
    assert "phonemizer" not in deps
    assert "espeakng-runtime" not in deps
