from __future__ import annotations

from examples.run_all import example_paths


def test_expected_examples_are_discoverable():
    assert {path.name for path in example_paths()} == {"basic.py", "all_voices.py"}
