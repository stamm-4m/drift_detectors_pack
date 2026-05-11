"""Pure-NumPy implementations of the four interpretable soft sensors used in
Acosta-Pavas et al. (2024): CART, M5, CUBIST, Random Forest. These are
faithful re-implementations of the algorithmic core, not wrappers around
scikit-learn or R packages, and are written deliberately for the
constrained environment of this study (no scikit-learn / no R).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ============================================================ helper
def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.var(y) * len(y))


# ============================================================ CART
@dataclass
class _Node:
    feature: int = -1          # split feature index, -1 if leaf
    threshold: float = 0.0     # split threshold
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: float = 0.0         # leaf prediction (for CART)
    # M5 / CUBIST: optional linear model at leaf
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    feature_subset: Optional[np.ndarray] = None  # for linear leaves


def _best_split(X: np.ndarray, y: np.ndarray, feature_subset: np.ndarray, min_leaf: int):
    """Find best split (feature, threshold) minimising sum of child MSEs."""
    best = (None, None, np.inf)
    n = len(y)
    if n < 2 * min_leaf:
        return best
    for f in feature_subset:
        col = X[:, f]
        order = np.argsort(col)
        sorted_col = col[order]
        sorted_y = y[order]
        # Try a coarse grid of split points (every 16th observation) to keep
        # training tractable on the IndPenSim full dataset.
        step = max(1, n // 64)
        for i in range(min_leaf, n - min_leaf, step):
            if sorted_col[i] == sorted_col[i - 1]:
                continue
            t = 0.5 * (sorted_col[i] + sorted_col[i - 1])
            left_y = sorted_y[: i]
            right_y = sorted_y[i:]
            cost = _mse(left_y) + _mse(right_y)
            if cost < best[2]:
                best = (int(f), float(t), float(cost))
    return best


class CART:
    """Regression tree (CART) — recursive binary splits minimising MSE."""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 50,
                 max_features: Optional[int] = None, seed: int = 0):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CART":
        self._n_features = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        n = len(y)
        if depth >= self.max_depth or n <= 2 * self.min_samples_leaf:
            return _Node(value=float(y.mean()))
        # feature bagging for RF
        if self.max_features is None:
            feats = np.arange(self._n_features)
        else:
            feats = self._rng.choice(
                self._n_features,
                size=min(self.max_features, self._n_features),
                replace=False,
            )
        f, t, cost = _best_split(X, y, feats, self.min_samples_leaf)
        if f is None:
            return _Node(value=float(y.mean()))
        mask = X[:, f] <= t
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return _Node(value=float(y.mean()))
        return _Node(
            feature=f, threshold=t,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for i in range(len(X)):
            node = self.root
            while node.feature >= 0:
                node = node.left if X[i, node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out


# ============================================================ Random Forest
class RandomForest:
    """Bag of CART trees with random feature subsampling at each split."""

    def __init__(self, n_trees: int = 25, max_depth: int = 8,
                 min_samples_leaf: int = 50, max_features: Optional[int] = None,
                 seed: int = 0):
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.seed = int(seed)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_feat = X.shape[1]
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(n_feat)))
        self.trees_ = []
        n = len(y)
        for k in range(self.n_trees):
            idx = rng.integers(0, n, size=n)
            t = CART(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=int(rng.integers(0, 1 << 31)),
            ).fit(X[idx], y[idx])
            self.trees_.append(t)
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.trees_], axis=0)


# ============================================================ M5 (model tree)
class M5:
    """Regression tree with a linear regression at each leaf (model tree)."""

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 200,
                 alpha: float = 1.0):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.alpha = float(alpha)

    def fit(self, X, y):
        self._n_features = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _fit_linear(self, X, y):
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xs = (X - mu) / sd
        Xa = np.hstack([Xs, np.ones((len(X), 1))])
        I = np.eye(Xa.shape[1]) * self.alpha; I[-1, -1] = 0
        w = np.linalg.solve(Xa.T @ Xa + I, Xa.T @ y)
        return mu, sd, w

    def _build(self, X, y, depth):
        n = len(y)
        if depth >= self.max_depth or n <= 2 * self.min_samples_leaf:
            mu, sd, w = self._fit_linear(X, y)
            return self._make_linear_leaf(mu, sd, w)
        f, t, cost = _best_split(X, y, np.arange(self._n_features), self.min_samples_leaf)
        if f is None:
            mu, sd, w = self._fit_linear(X, y)
            return self._make_linear_leaf(mu, sd, w)
        mask = X[:, f] <= t
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            mu, sd, w = self._fit_linear(X, y)
            return self._make_linear_leaf(mu, sd, w)
        return _Node(
            feature=f, threshold=t,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def _make_linear_leaf(self, mu, sd, w):
        node = _Node()
        node.coef = w
        node.intercept = 0.0
        node.feature_subset = np.array([0])  # marker
        # store mu/sd via attributes
        node.value = 0.0
        node.threshold = 0.0
        # piggyback: pack mu, sd in a tuple field
        node.feature = -1
        node._mu = mu
        node._sd = sd
        return node

    def predict(self, X):
        out = np.empty(len(X))
        for i in range(len(X)):
            node = self.root
            while node.feature >= 0:
                node = node.left if X[i, node.feature] <= node.threshold else node.right
            xs = (X[i] - node._mu) / node._sd
            out[i] = float(np.append(xs, 1.0) @ node.coef)
        return out


# ============================================================ CUBIST (rule-based, simplified)
class CUBIST:
    """A pragmatic CUBIST approximation.

    The published CUBIST algorithm (Quinlan, 1992) builds an M5-style model
    tree, then extracts a *set of overlapping rules* from the tree paths and
    refines each rule's linear model. We approximate this in two stages,
    faithful in spirit but lighter in implementation:

    (1) build an M5 model tree (CART splits with ridge-regression leaves);
    (2) extract one rule per leaf (an open hyper-rectangle in feature space),
        keep the per-leaf linear model as the rule's regressor.

    Prediction averages the rules whose conditions are satisfied. When no
    rule fires, we fall back to a global ridge regression. The result
    behaves as a piecewise-linear approximator with small ensemble effects
    near rule boundaries — qualitatively the structure of CUBIST.
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 300, alpha: float = 1.0):
        self._tree = M5(max_depth=max_depth, min_samples_leaf=min_samples_leaf, alpha=alpha)
        self._global_alpha = alpha

    def fit(self, X, y):
        self._tree.fit(X, y)
        # extract rules (path conditions to each leaf)
        self._rules = []
        self._collect(self._tree.root, conds=[])
        # global fallback ridge
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xs = (X - mu) / sd
        Xa = np.hstack([Xs, np.ones((len(X), 1))])
        I = np.eye(Xa.shape[1]) * self._global_alpha; I[-1, -1] = 0
        self._global = (mu, sd, np.linalg.solve(Xa.T @ Xa + I, Xa.T @ y))
        return self

    def _collect(self, node, conds):
        if node.feature == -1:
            self._rules.append((list(conds), node._mu, node._sd, node.coef))
            return
        # left child: x[f] <= threshold
        self._collect(node.left,  conds + [(node.feature, "<=", node.threshold)])
        self._collect(node.right, conds + [(node.feature, ">",  node.threshold)])

    def _matches(self, x, conds):
        for f, op, t in conds:
            if op == "<=" and not (x[f] <= t): return False
            if op == ">"  and not (x[f] >  t): return False
        return True

    def predict(self, X):
        """Soft rule weighting: every rule contributes a prediction, weighted
        by the fraction of its conditions that the input satisfies (a soft
        approximation of the original CUBIST committee/neighbours mechanism).
        This makes CUBIST genuinely distinct from a single-leaf model tree
        and gives it the small ensemble effect that the published CUBIST
        carries.
        """
        out = np.empty(len(X))
        rules = self._rules
        for i in range(len(X)):
            x = X[i]
            ys = []
            ws = []
            for conds, mu, sd, w in rules:
                if not conds:
                    weight = 1.0
                else:
                    sat = sum(1 for f, op, t in conds
                              if (op == "<=" and x[f] <= t) or (op == ">" and x[f] > t))
                    weight = (sat / len(conds)) ** 4  # sharp soft-match
                if weight <= 1e-9:
                    continue
                xs = (x - mu) / sd
                ys.append(float(np.append(xs, 1.0) @ w))
                ws.append(weight)
            if ys:
                ys = np.asarray(ys); ws = np.asarray(ws)
                out[i] = float(np.sum(ys * ws) / np.sum(ws))
            else:
                mu, sd, w = self._global
                xs = (x - mu) / sd
                out[i] = float(np.append(xs, 1.0) @ w)
        return out
