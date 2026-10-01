"""Validate wheel/sdist shape, version, and dependency boundaries."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement

PACKAGE = "kittensynth"
DISTRIBUTION = "kittensynth"
FORBIDDEN = frozenset({"espeakng-loader", "phonemizer", "espeakng-runtime"})


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(value for value in archive.namelist() if value.endswith(".dist-info/METADATA"))
        return archive.read(name).decode("utf-8")


def _wheel_version(path: Path) -> str:
    for row in _wheel_metadata(path).splitlines():
        if row.startswith("Version: "):
            return row.removeprefix("Version: ")
    raise SystemExit("wheel metadata has no Version")


def _requirements(path: Path) -> list[Requirement]:
    metadata = Parser().parsestr(_wheel_metadata(path))
    return [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]


def _sdist_version(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        member = next(value for value in archive.getmembers() if value.name.endswith("/PKG-INFO"))
        handle = archive.extractfile(member)
        if handle is None:
            raise SystemExit("could not read sdist PKG-INFO")
        rows = handle.read().decode("utf-8").splitlines()
    for row in rows:
        if row.startswith("Version: "):
            return row.removeprefix("Version: ")
    raise SystemExit("sdist metadata has no Version")


def _check_wheel(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if not any(name.startswith(PACKAGE + "/") and name.endswith(".py") for name in names):
        raise SystemExit(f"wheel does not contain {PACKAGE} Python files")
    if any(name.startswith(("tests/", "docs/", ".github/")) for name in names):
        raise SystemExit("wheel contains development-only directories")
    installed = {req.name.casefold().replace("_", "-") for req in _requirements(path)}
    bad = sorted(installed & FORBIDDEN)
    if bad:
        raise SystemExit("forbidden dependencies: " + ", ".join(bad))
    return _wheel_version(path)


def _check_sdist(path: Path) -> str:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    required = ("pyproject.toml", "README.md", "LICENSE", f"{PACKAGE}/__init__.py")
    if not all(any(name.endswith(value) for name in names) for value in required):
        raise SystemExit("sdist is missing required project files")
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
