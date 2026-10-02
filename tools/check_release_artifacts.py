"""Validate wheel/sdist contents, metadata, version, and runtime dependency boundaries."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

PACKAGE = "kittensynth"
FORBIDDEN = frozenset({"espeakng-loader", "phonemizer", "espeakng-runtime"})
BASE_REQUIREMENTS = {
    "numpy": (">=1.23", ()),
    "kitteng2p": (">=0.1.1,<0.2", ()),
    "onnxvoice": (">=0.2,<0.3", ()),
    "audiosig": (">=0.1.4,<0.2", ()),
}
OPTIONAL_REQUIREMENTS = (
    ("bundled-g2p", "kitteng2p", ">=0.1.1,<0.2", ("bundled",)),
    ("cpu", "onnxvoice", ">=0.2,<0.3", ("cpu",)),
    ("gpu", "onnxvoice", ">=0.2,<0.3", ("gpu",)),
    ("docs", "sphinx", ">=7.0.0", ()),
    ("docs", "myst-parser", ">=2.0.0", ()),
    ("docs", "sphinx-rtd-theme", ">=2.0.0", ()),
)


def _normalize_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise SystemExit("wheel must contain exactly one dist-info/METADATA file")
        return archive.read(metadata_files[0]).decode("utf-8")


def _wheel_version(path: Path) -> str:
    metadata = Parser().parsestr(_wheel_metadata(path))
    version = metadata.get("Version")
    if not version:
        raise SystemExit("wheel metadata has no Version")
    return version


def _requirements(path: Path) -> list[Requirement]:
    metadata = Parser().parsestr(_wheel_metadata(path))
    return [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]


def _sdist_version(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(members) != 1:
            raise SystemExit("sdist must contain exactly one PKG-INFO file")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise SystemExit("could not read sdist PKG-INFO")
        metadata = Parser().parsestr(handle.read().decode("utf-8"))
    version = metadata.get("Version")
    if not version:
        raise SystemExit("sdist PKG-INFO has no Version")
    return version


def _matches_extra(requirement: Requirement, extra: str | None) -> bool:
    if requirement.marker is None:
        return extra is None
    return requirement.marker.evaluate({"extra": extra or ""})


def _check_requirement(
    requirements: list[Requirement],
    *,
    name: str,
    specifier: str,
    extras: tuple[str, ...] = (),
    extra: str | None = None,
) -> None:
    candidates = [
        requirement
        for requirement in requirements
        if _normalize_name(requirement.name) == name and _matches_extra(requirement, extra)
    ]
    description = f"{name}{f'[{extra}]' if extra else ''}{specifier}"
    if not candidates:
        raise SystemExit(f"wheel is missing dependency {description}")
    expected_extras = set(extras)
    if not any(
        requirement.specifier == SpecifierSet(specifier) and requirement.extras == expected_extras
        for requirement in candidates
    ):
        raise SystemExit(f"wheel has an incorrect dependency range or extra for {description}")


def _check_wheel(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if not any(name.startswith(PACKAGE + "/") and name.endswith(".py") for name in names):
        raise SystemExit(f"wheel does not contain {PACKAGE} Python files")
    required_files = (
        f"{PACKAGE}/py.typed",
        f"{PACKAGE}/data/voice_level_calibration.json",
    )
    missing = [name for name in required_files if name not in names]
    if missing:
        raise SystemExit("wheel is missing required package files: " + ", ".join(missing))
    if any(name.startswith(("tests/", "docs/", ".github/")) for name in names):
        raise SystemExit("wheel contains development-only directories")

    requirements = _requirements(path)
    installed = {_normalize_name(requirement.name) for requirement in requirements}
    bad = sorted(installed & FORBIDDEN)
    if bad:
        raise SystemExit("forbidden dependencies: " + ", ".join(bad))
    for name, (specifier, extras) in BASE_REQUIREMENTS.items():
        _check_requirement(
            requirements,
            name=name,
            specifier=specifier,
            extras=extras,
        )
    for extra, name, specifier, extras in OPTIONAL_REQUIREMENTS:
        _check_requirement(
            requirements,
            name=name,
            specifier=specifier,
            extras=extras,
            extra=extra,
        )
    return _wheel_version(path)


def _check_sdist(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    generated = ("benchmarks/output/", "docs/_build/", "example-artefacts/")
    if any("/.ledger/" in name or name.endswith("/.ledger") for name in names):
        raise SystemExit("sdist contains project-local ledger state")
    if any(fragment in name for fragment in generated for name in names):
        raise SystemExit("sdist contains generated build, benchmark, or example artifacts")
    required = (
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "NOTICE",
        f"{PACKAGE}/__init__.py",
        f"{PACKAGE}/py.typed",
        f"{PACKAGE}/data/voice_level_calibration.json",
        "docs/index.md",
        "docs/conf.py",
        "docs/make.py",
        "docs/architecture.md",
        "docs/onnxvoice-contract.md",
        "docs/requirements.txt",
        "examples/basic.py",
        "examples/all_voices.py",
        "examples/run_all.py",
        "benchmarks/voice_level_benchmark.py",
        "benchmarks/promote_voice_calibration.py",
        "benchmarks/verify_voice_calibration.py",
        "benchmarks/data/voice_level_policy.json",
        "benchmarks/data/voice_level_stimuli.json",
        "benchmarks/data/voice_calibration_verification_policy.json",
    )
    missing = [value for value in required if not any(name.endswith(value) for name in names)]
    if missing:
        raise SystemExit("sdist is missing required project files: " + ", ".join(missing))
    return _sdist_version(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    parser.add_argument("--expected-version")
    args = parser.parse_args()

    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("dist must contain exactly one wheel and one sdist")

    wheel_version = _check_wheel(wheels[0])
    sdist_version = _check_sdist(sdists[0])
    if wheel_version != sdist_version:
        raise SystemExit(f"artifact version mismatch: {wheel_version} != {sdist_version}")
    if args.expected_version is not None and wheel_version != args.expected_version:
        raise SystemExit(
            f"artifact version {wheel_version!r} does not match {args.expected_version!r}"
        )
    print(f"checked {wheels[0].name} and {sdists[0].name} (version {wheel_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
