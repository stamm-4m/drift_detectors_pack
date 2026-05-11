# Benchmarks

Reproducible benchmarking utilities for `drift_detectors_pack`.

## Quick start

Run the full IndPenSim benchmark across all univariate detectors:

```bash
python -m benchmarks.run_indpensim --output results/
```

Run only the multivariate detectors against the default variable bundle:

```bash
python -m benchmarks.run_indpensim --multivariate --output results/
```

Pick a subset:

```bash
python -m benchmarks.run_indpensim --detectors PSI KS HDDM-A
```

## Output

Each run writes a CSV with one row per `(detector, case_id)` pair containing:

| column           | description                                              |
| ---------------- | -------------------------------------------------------- |
| `detector`       | detector name                                            |
| `case_id`        | identifier of the (reference, test) pair                 |
| `expected_drift` | 0/1 ground-truth label, blank if unknown                 |
| `detected_drift` | 0/1 detector decision                                    |
| `outcome`        | `TP` / `FP` / `TN` / `FN` / `unlabelled`                 |
| `score`          | numeric drift score (when applicable)                    |
| `runtime_ms`     | wall-clock time of the `calculate()` call                |
| `peak_memory_kb` | peak memory delta of the `calculate()` call (tracemalloc)|

## Programmatic API

```python
from benchmarks.runner import BenchmarkRunner, discover_default_detectors
from benchmarks.indpensim import build_univariate_cases

cases = build_univariate_cases("penicillin_concentration")
runner = BenchmarkRunner(discover_default_detectors(), cases)
runner.run()
runner.to_csv("results/my_run.csv")
print(runner.summary_by_detector())
```

## Adding a new dataset

A "benchmark case" is just a `BenchmarkCase` with a reference array, a test
array, and an optional ground-truth `expected_drift` flag. Implement a
loader in `benchmarks/<dataset>.py` that returns a list of cases and you're
done.
