"""BIC scores for one variable given a set of parents ("local scores").

The BIC of a whole DAG is the sum of the local scores of its variables, so a search only rescores the variables whose
parents change. Higher is better:

    BIC_j = (log-likelihood of X_j given its parents) - (free parameters / 2) * log(n)

Two models are available:

* ``GaussianBIC``: continuous data, X_j a linear function of its parents plus Gaussian noise. It only needs the
  covariance matrix, so it costs nothing per sample.
* ``CategoricalBIC``: categorical data coded 0..k-1, one probability table per parent configuration.

``penalty`` multiplies the complexity term (Tetrad calls it "penalty discount"). Values above 1 give sparser graphs.
"""
from __future__ import annotations

import math

import numpy as np


class GaussianBIC:
    def __init__(self, cov: np.ndarray, n: int, penalty: float = 1.0) -> None:
        self.cov = np.asarray(cov, dtype=np.float64)
        self.n = int(n)
        self.pen = 0.5 * penalty * math.log(self.n)
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def residual_variance(self, j: int, parents) -> float:
        p = list(parents)
        v = self.cov[j, j]
        if p:
            c = self.cov[np.ix_(p, [j])][:, 0]
            try:
                v = v - c @ np.linalg.solve(self.cov[np.ix_(p, p)], c)
            except np.linalg.LinAlgError:
                v = v - c @ np.linalg.lstsq(self.cov[np.ix_(p, p)], c, rcond=None)[0]
        return max(float(v), 1e-12)

    def local(self, j: int, parents) -> float:
        key = (j, tuple(sorted(parents)))
        hit = self._cache.get(key)
        if hit is None:
            ll = -0.5 * self.n * (math.log(2 * math.pi * self.residual_variance(j, key[1])) + 1.0)
            hit = ll - self.pen * (len(key[1]) + 1)       # one coefficient per parent, plus the noise variance
            self._cache[key] = hit
        return hit


class CategoricalBIC:
    def __init__(self, codes: np.ndarray, penalty: float = 1.0, max_parent_cells: int = 1_000_000) -> None:
        self.X = np.ascontiguousarray(codes, dtype=np.int64)
        self.n, self.d = self.X.shape
        self.cards = self.X.max(axis=0) + 1
        self.pen = 0.5 * penalty * math.log(self.n)
        self.max_cells = max_parent_cells
        self._cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def loglik(self, j: int, parents) -> tuple[float, int]:
        """Maximized log-likelihood of X_j given its parents, and the number of free parameters."""
        p = list(parents)
        r = int(self.cards[j])
        if p:
            q = int(np.prod([int(self.cards[i]) for i in p], dtype=np.float64))
            cfg = np.zeros(self.n, dtype=np.int64)
            for i in p:
                cfg = cfg * self.cards[i] + self.X[:, i]
            if q * r > 50_000_000:                       # too many cells for a dense count: count observed ones only
                _, cfg = np.unique(cfg, return_inverse=True)
        else:
            q, cfg = 1, np.zeros(self.n, dtype=np.int64)
        counts = np.bincount(cfg * r + self.X[:, j])
        counts = counts[counts > 0].astype(np.float64)
        parent_counts = np.bincount(cfg).astype(np.float64)
        parent_counts = parent_counts[parent_counts > 0]
        ll = float(np.sum(counts * np.log(counts)) - np.sum(parent_counts * np.log(parent_counts)))
        return ll, q * (r - 1)

    def local(self, j: int, parents) -> float:
        key = (j, tuple(sorted(parents)))
        hit = self._cache.get(key)
        if hit is None:
            if np.prod([float(self.cards[i]) for i in key[1]]) > self.max_cells:
                hit = -math.inf
            else:
                ll, k = self.loglik(j, key[1])
                hit = ll - self.pen * k
            self._cache[key] = hit
        return hit
