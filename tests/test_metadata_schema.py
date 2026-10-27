"""Schema of every detector's metadata.yaml (keys platforms rely on)."""
from pathlib import Path

import pytest
import yaml

PKG = Path(__file__).resolve().parent.parent / "src" / "drift_detectors"
FILES = sorted(PKG.glob("*/*/metadata.yaml"))

TYPES = {"univariate", "multivariate", "model_based"}
MODES = {"batch", "streaming"}
INPUT_KINDS = {"signal", "error", "predictions"}


def _load(path):
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def test_metadata_files_found():
    assert len(FILES) >= 30


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.parent.name)
def test_required_keys(path):
    meta = _load(path)
    for key in ("name", "type", "mode", "input_kind", "description"):
        assert meta.get(key), f"{path.parent.name}: missing {key}"
    assert meta["type"] in TYPES
    assert meta["mode"] in MODES
    assert meta["input_kind"] in INPUT_KINDS


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.parent.name)
def test_input_kind_consistent_with_inputs(path):
    meta = _load(path)
    inputs = set((meta.get("inputs") or {}).keys())
    if meta["input_kind"] == "predictions":
        assert inputs == {"predictions"}
    else:
        assert "test_data" in inputs
    if "reference_data" in inputs:
        assert meta["input_kind"] == "signal"


# ---------------------------------------------------------------- 0.5.3
CALLER_SUPPLIED = {"reference_data", "online", "models"}


def test_every_constructor_parameter_is_documented():
    """A parameter of ``__init__`` that changes the result must be in metadata.yaml
    (``reference_data``, ``online`` and ``models`` are supplied by the caller)."""
    import ast
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "drift_detectors"
    missing = {}
    for meta in sorted(root.rglob("metadata.yaml")):
        det = meta.parent / "detector.py"
        if not det.exists():
            continue
        names = set()
        for cls in [n for n in ast.parse(det.read_text()).body if isinstance(n, ast.ClassDef)]:
            for f in cls.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    names |= {a.arg for a in f.args.args[1:]} | {a.arg for a in f.args.kwonlyargs}
        documented = set((yaml.safe_load(meta.read_text()).get("parameters") or {}).keys())
        gap = names - documented - CALLER_SUPPLIED
        if gap:
            missing[str(meta.parent.relative_to(root))] = sorted(gap)
    assert not missing, f"constructor parameters missing in metadata.yaml: {missing}"
