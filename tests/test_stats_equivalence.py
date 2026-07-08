"""Equivalence tests: the NumPy-only primitives in drift_detectors.utility._stats
(and the ADWIN port) must match the reference libraries they replaced.

These run only when the reference libraries are installed (they are declared
in the ``dev`` / ``compat`` extras, NOT as runtime dependencies):

    pip install -e ".[dev]"
    pytest tests/test_stats_equivalence.py
"""
from __future__ import annotations

import numpy as np
import pytest

from drift_detectors.utility import _stats

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ scipy/sklearn
scipy_stats = pytest.importorskip("scipy.stats", reason="scipy not installed")
skl_pw = pytest.importorskip("sklearn.metrics.pairwise", reason="scikit-learn not installed")
skl_dec = pytest.importorskip("sklearn.decomposition", reason="scikit-learn not installed")


def test_rbf_kernel_matches_sklearn():
    X = rng.normal(size=(30, 4)); Y = rng.normal(size=(20, 4))
    for gamma in (0.1, 1.0, 3.7):
        assert np.allclose(_stats.rbf_kernel(X, Y, gamma),
                           skl_pw.rbf_kernel(X, Y, gamma=gamma), atol=1e-12)


def test_kl_entropy_matches_scipy():
    for _ in range(20):
        p = np.abs(rng.normal(size=5)) + 1e-3
        q = np.abs(rng.normal(size=5)) + 1e-3
        assert np.isclose(_stats.kl_entropy(p, q), scipy_stats.entropy(p, q), atol=1e-12)


def test_ks_statistic_and_pvalue():
    """Statistic is exact vs scipy; our p-value is the asymptotic Kolmogorov
    survival function (scipy.stats.kstwobign.sf). scipy's ``method="asymp"``
    switched to a finite-sample refinement (kstwo) years ago, so its p-value
    differs slightly -- but the drift *decision* is unchanged, which is what
    the detector uses."""
    kstwobign = scipy_stats.kstwobign
    for _ in range(50):
        a = rng.normal(0, 1, rng.integers(50, 400))
        b = rng.normal(rng.uniform(0, 1), 1, rng.integers(50, 400))
        d, p = _stats.ks_2samp(a, b)
        assert np.isclose(d, scipy_stats.ks_2samp(a, b).statistic, atol=1e-12)
        n1, n2 = a.size, b.size
        en = n1 * n2 / (n1 + n2)
        assert np.isclose(p, kstwobign.sf(np.sqrt(en) * d), atol=1e-10)

    # operational equivalence: same drift decision on clearly separated cases
    same = rng.normal(0, 1, 300)
    _, p_same = _stats.ks_2samp(same, rng.normal(0, 1, 300))
    _, p_shift = _stats.ks_2samp(same, rng.normal(1.0, 1, 300))
    assert p_same > 0.05 and (scipy_stats.ks_2samp(same, rng.normal(0, 1, 300)).pvalue > 0.05)
    assert p_shift < 0.05 and (scipy_stats.ks_2samp(same, rng.normal(1.0, 1, 300)).pvalue < 0.05)


def test_fisher_matches_scipy():
    for _ in range(30):
        pv = rng.uniform(1e-4, 1.0, size=rng.integers(2, 8))
        stat, p = _stats.fisher_combine(pv)
        ref = scipy_stats.combine_pvalues(pv, method="fisher")
        assert np.isclose(stat, ref.statistic, atol=1e-9)
        assert np.isclose(p, ref.pvalue, rtol=1e-6, atol=1e-9)


def test_knn_indices_match_kdtree_neighbour_sets():
    spatial = pytest.importorskip("scipy.spatial")
    data = rng.normal(size=(120, 3)); q = data[:10]
    k = 15
    ours = _stats.knn_indices(data, q, k)
    tree = spatial.KDTree(data).query(q, k=k)[1]
    for a, b in zip(ours, tree):
        assert set(a.tolist()) == set(b.tolist())   # same neighbours (tie order may differ)


# ------------------------------------------------------------------ river / ADWIN
def test_adwin_matches_river_behaviourally():
    river_drift = pytest.importorskip("river.drift", reason="river not installed")
    from drift_detectors.utility._adwin import ADWIN as OursADWIN

    def run(det, stream, is_river):
        pts = []
        for i, v in enumerate(stream):
            det.update(v)
            if det.drift_detected:
                pts.append(i)
        return pts

    # (1) stationary constant stream: neither should fire
    const = np.zeros(1500)
    assert run(OursADWIN(0.002), const, False) == []
    assert run(river_drift.ADWIN(0.002), const, True) == []

    # (2) abrupt shift at 1000: both fire, first detection shortly after
    shift = np.concatenate([rng.normal(0, 0.1, 1000), rng.normal(1.0, 0.1, 1000)])
    ours = run(OursADWIN(0.002), shift, False)
    theirs = run(river_drift.ADWIN(0.002), shift, True)
    first_ours = next((x for x in ours if x >= 1000), None)
    first_theirs = next((x for x in theirs if x >= 1000), None)
    assert first_ours is not None and first_theirs is not None
    assert abs(first_ours - first_theirs) <= 50   # detect the change at comparable latency
