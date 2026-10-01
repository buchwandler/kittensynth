from importlib.metadata import PackageNotFoundError, version

import kittensynth


def test_public_version_matches_distribution_metadata():
    try:
        expected = version("kittensynth")
    except PackageNotFoundError:
        expected = "0+unknown"
    assert kittensynth.__version__ == expected
