"""Dependency-light statistical primitives (NumPy only).

These replace the handful of scipy / scikit-learn calls the detectors used
to make, so that the runtime footprint of drift_detectors is just NumPy.
Each function mirrors the reference implementation it replaces closely
enough that the detectors' outputs are numerically equivalent; the
``tests/test_stats_equivalence.py`` suite checks this against scipy /
scikit-learn when those libraries are installed (dev only).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rbf_kernel", "pca_project", "kl_entropy",
    "ks_2samp", "fisher_combine", "knn_indices",
    "hist_pq", "wasserstein_1d",
]


# --------------------------------------------------------------------- MMD
def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix, K[i,j] = exp(-gamma * ||x_i - y_j||^2).

    Equivalent to sklearn.metrics.pairwise.rbf_kernel.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    d2 = xx + yy - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)      # guard tiny negatives from round-off
    return np.exp(-gamma * d2)


# --------------------------------------------------------------------- PCA
def pca_project(ref: np.ndarray, test: np.ndarray, n_components: int):
    """PCA fit on ``ref`` (centered), project both. Returns (ref_proj, test_proj).

    Equivalent to sklearn.decomposition.PCA(n_components).fit(ref) then
    transform, up to the arbitrary sign of each component -- which does not
    affect PCA-CD's score (it uses squared mean shifts and variances, both
    sign-invariant).
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    mean = ref.mean(axis=0)
    ref_c = ref - mean
    # full_matrices=False -> Vt has shape (min(n,d), d)
    _, _, vt = np.linalg.svd(ref_c, full_matrices=False)
    comps = vt[:n_components]
    return ref_c @ comps.T, (test - mean) @ comps.T


# --------------------------------------------------------------------- KL
def kl_entropy(pk: np.ndarray, qk: np.ndarray) -> float:
    """Kullback-Leibler divergence sum(pk * log(pk / qk)) after normalising
    pk and qk to probability vectors. Equivalent to scipy.stats.entropy(pk, qk).
    """
    pk = np.asarray(pk, dtype=float)
    qk = np.asarray(qk, dtype=float)
    pk = pk / pk.sum()
    qk = qk / qk.sum()
    return float(np.sum(pk * np.log(pk / qk)))


# --------------------------------------------------------------------- KS
def _kolmogorov_sf(x: float) -> float:
    """Survival function of the Kolmogorov distribution,
    Q(x) = 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 x^2). == scipy kstwobign.sf(x).
    """
    if x <= 0:
        return 1.0
    if x > 8:      # underflow region: Q(x) ~ 0
        return 0.0
    k = np.arange(1, 101)
    terms = ((-1.0) ** (k - 1)) * np.exp(-2.0 * (k * x) ** 2)
    return float(min(max(2.0 * terms.sum(), 0.0), 1.0))


def _ecdf_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov: returns (statistic, p_value).

    ``statistic`` is exact (max |F_a - F_b|). ``p_value`` uses the classic
    asymptotic Kolmogorov distribution (== scipy.stats.kstwobign.sf). Note
    that modern scipy.stats.ks_2samp(method='asymp') applies a finite-sample
    refinement (kstwo), so its p-value differs slightly in magnitude; the two
    agree on the drift decision, which is what the detectors act on.
    """
    n1 = np.asarray(a).size
    n2 = np.asarray(b).size
    d = _ecdf_diff(a, b)
    en = n1 * n2 / (n1 + n2)
    p = _kolmogorov_sf(np.sqrt(en) * d)
    return d, p


# --------------------------------------------------------------- Fisher
def fisher_combine(pvalues) -> tuple[float, float]:
    """Fisher's method for combining p-values. Returns (statistic, p_value),
    matching scipy.stats.combine_pvalues(pvalues, method='fisher').

    statistic = -2 * sum(log p_i);  p = chi2.sf(statistic, 2k), which for the
    even d.o.f. 2k has the closed form exp(-s/2) * sum_{i<k} (s/2)^i / i!.
    """
    p = np.clip(np.asarray(pvalues, dtype=float).ravel(), 1e-300, 1.0)
    k = p.size
    stat = float(-2.0 * np.sum(np.log(p)))
    # chi2 survival at 2k dof (even) -- closed form via Poisson tail
    half = stat / 2.0
    term = 1.0
    total = 1.0
    for i in range(1, k):
        term *= half / i
        total += term
    pval = float(min(max(np.exp(-half) * total, 0.0), 1.0))
    return stat, pval


# --------------------------------------------------------------- kNN
def knn_indices(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """For each query row, the indices of its ``k`` nearest rows in ``data``
    (Euclidean), ordered nearest-first. Equivalent to
    scipy.spatial.KDTree(data).query(queries, k)[1].
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    queries = np.atleast_2d(np.asarray(queries, dtype=float))
    dd = np.einsum("ij,ij->i", data, data)[None, :]
    qq = np.einsum("ij,ij->i", queries, queries)[:, None]
    d2 = qq + dd - 2.0 * (queries @ data.T)     # (n_queries, n_data)
    np.maximum(d2, 0.0, out=d2)
    part = np.argpartition(d2, kth=min(k, data.shape[0]) - 1, axis=1)[:, :k]
    # order the k picked neighbours by true distance (nearest first)
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(d2[rows, part], axis=1)
    return part[rows, order]


# ------------------------------------------------------- distributional (1-D)
def hist_pq(ref, test, bins: int = 20, epsilon: float = 1e-8):
    """Bin ``ref`` and ``test`` on common edges; return smoothed probability
    vectors (p, q) that each sum to 1. Shared by the histogram-based
    univariate distributional detectors (JS, Hellinger, TV, chi-square, ...).
    """
    ref = np.asarray(ref, dtype=float).ravel()
    test = np.asarray(test, dtype=float).ravel()
    lo = min(ref.min(), test.min())
    hi = max(ref.max(), test.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(ref, bins=edges)[0].astype(float)
    q = np.histogram(test, bins=edges)[0].astype(float)
    p = np.clip(p / p.sum(), epsilon, None)
    q = np.clip(q / q.sum(), epsilon, None)
    return p / p.sum(), q / q.sum()


def wasserstein_1d(a, b) -> float:
    """Exact 1-D Wasserstein-1 (earth-mover) distance = integral |F_a - F_b|.
    Equivalent to scipy.stats.wasserstein_distance.
    """
    a = np.sort(np.asarray(a, dtype=float).ravel())
    b = np.sort(np.asarray(b, dtype=float).ravel())
    allv = np.sort(np.concatenate([a, b]))
    deltas = np.diff(allv)
    cdf_a = np.searchsorted(a, allv[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, allv[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))
