"""Build a slim 'detector pack' archive for the STAMM model registry.

The registry's ``POST /api/v1/detector_packs/register`` endpoint expects a
.zip that contains the ``drift_detectors/`` package and nothing else: each
detector is a folder with ``detector.py`` + ``metadata.yaml``. The endpoint
reads the version from ``pyproject.toml`` (``[project].version``) or
``drift_detectors/__init__.py`` (``__version__``), parses every
``metadata.yaml`` into a read-only catalogue, and stores a slim copy. Data,
use cases, tests and benchmarks are explicitly not wanted.

This script produces exactly that archive: the package (re-rooted so paths
start with ``drift_detectors/``) plus ``pyproject.toml`` for the version.

Usage (from the repo root):
    python tools/build_detector_pack.py
    # -> dist/drift_detectors_pack-<version>.zip
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "drift_detectors"
PYPROJECT = ROOT / "pyproject.toml"
EXCLUDE_SUFFIX = {".pyc"}
EXCLUDE_PARTS = {"__pycache__"}


def read_version() -> str:
    m = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']',
                  PYPROJECT.read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def main() -> None:
    if not PKG.is_dir():
        sys.exit(f"package not found at {PKG} (run from the repo root)")
    ver = read_version()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"drift_detectors_pack-{ver}.zip"

    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_dir() or p.suffix in EXCLUDE_SUFFIX or EXCLUDE_PARTS & set(p.parts):
                continue
            zf.write(p, Path("drift_detectors") / p.relative_to(PKG))
            n_files += 1
        zf.write(PYPROJECT, "pyproject.toml")  # registry reads the version here

    detectors = sorted({
        p.parent.name for p in PKG.rglob("detector.py")
        if (p.parent / "metadata.yaml").exists()
    })
    print(f"wrote {out}")
    print(f"  version   : {ver}")
    print(f"  pkg files : {n_files}")
    print(f"  detectors : {len(detectors)} -> {', '.join(detectors)}")


if __name__ == "__main__":
    main()
