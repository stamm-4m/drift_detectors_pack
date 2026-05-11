# Contributing to drift_detectors_pack

Thank you for your interest in improving this project. This guide explains
how to set up a development environment, the conventions we follow, and how
to add a new drift detector to the catalogue.

## Development setup

```bash
git clone https://gitlab.com/stamm-4m/drift_detectors_pack.git
cd drift_detectors_pack
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,benchmark]"
pytest -ra
```

## Code style

We use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting and
[`mypy`](https://mypy-lang.org/) for static typing. CI enforces both:

```bash
ruff check drift_detectors
ruff format drift_detectors
mypy --ignore-missing-imports drift_detectors
```

Public functions and classes must carry NumPy-style docstrings.

## Tests

All detectors must ship at least four tests:

1. **`test_<name>_drift_detected`** — synthetic stream with a known shift.
2. **`test_<name>_no_drift`** — stationary stream returns `drift=False`.
3. **`test_<name>_metadata_loaded`** — confirms `metadata.yaml` is reachable.
4. **`test_<name>_output_fields`** — confirms result `details` schema.

Run the full suite in parallel with `pytest -n auto`.

## Adding a new detector

1. Create a folder under `drift_detectors/univariate/` or
   `drift_detectors/multivariate/` named after your method (snake_case).
2. Add three files:
   - `detector.py` — class inheriting from
     `drift_detectors.drift_detector.DriftDetector`. Implement
     `calculate()` returning a result dataclass from
     `drift_detectors.utility.drift_detection_output`.
   - `metadata.yaml` — human-readable description, parameters, references.
   - `usage.md` — minimal worked example, mock data only.
3. Re-export the class from `drift_detectors/__init__.py`.
4. Add unit tests in `tests/test_<name>.py`.
5. (Optional) Add a use case under `use_cases/` exercising your detector
   against a real dataset.

## Reporting bugs

Please file issues on GitLab with:

- A minimal reproducer (script + data shape).
- Versions of Python, NumPy, SciPy, scikit-learn, river.
- Expected vs. observed behaviour.

## Releases

Releases are cut from `main` after CI is green:

```bash
# bump version in pyproject.toml and drift_detectors/__init__.py
git tag v0.2.0
git push --tags
```

The CI pipeline will build wheels and (on a tag) publish to PyPI.
