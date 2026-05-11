"""
benchmarks — reproducible benchmarking utilities for drift_detectors_pack.

The two main entry points are :class:`benchmarks.runner.BenchmarkRunner`
(programmatic API) and the ``run_indpensim`` CLI script.
"""

from benchmarks.runner import BenchmarkResult, BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkResult"]
