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
