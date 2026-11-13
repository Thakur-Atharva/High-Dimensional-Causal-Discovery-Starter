"""Synthetic data with a known true graph, for checking that everything works before using real data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_dag(d: int, edges_per_node: float = 2.0, seed: int = 0) -> np.ndarray:
    """Erdos-Renyi DAG: each pair gets an edge with the same probability; directions follow a random variable order."""
    rng = np.random.default_rng(seed)
    p = min(1.0, 2 * edges_per_node / max(d - 1, 1))
    upper = np.triu(rng.random((d, d)) < p, k=1)
    perm = rng.permutation(d)
    return upper[np.ix_(perm, perm)].astype(np.int8)


def topological_order(A: np.ndarray) -> list[int]:
    A = np.asarray(A) != 0
    indeg = A.sum(axis=0).astype(int)
    order, stack = [], [j for j in range(A.shape[0]) if indeg[j] == 0]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in np.nonzero(A[u])[0]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    if len(order) != A.shape[0]:
        raise ValueError("graph has a cycle")
    return order


def linear_gaussian(A: np.ndarray, n: int, seed: int = 0, weight_range=(0.5, 2.0)) -> tuple[np.ndarray, np.ndarray]:
    """X_j = sum_i W[i, j] X_i + N(0, 1), with |W| uniform in ``weight_range`` and random signs."""
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    W = rng.uniform(*weight_range, size=(d, d)) * rng.choice([-1.0, 1.0], size=(d, d)) * (A != 0)
    X = np.zeros((n, d))
    for j in topological_order(A):
        X[:, j] = X @ W[:, j] + rng.standard_normal(n)
    return X, W


def categorical(A: np.ndarray, n: int, seed: int = 0, max_card: int = 4, concentration: float = 0.5) -> np.ndarray:
    """Each variable has 2..max_card categories and a random probability table per parent configuration."""
    rng = np.random.default_rng(seed)
    d = A.shape[0]
    cards = rng.integers(2, max_card + 1, size=d)
    X = np.zeros((n, d), dtype=np.int64)
    for j in topological_order(A):
        P = np.nonzero(A[:, j])[0]
        q = int(np.prod(cards[P])) if len(P) else 1
        table = rng.dirichlet(np.full(cards[j], concentration), size=q)
        cfg = np.zeros(n, dtype=np.int64)
        for i in P:
            cfg = cfg * cards[i] + X[:, i]
        cum = table.cumsum(axis=1)[cfg]
        X[:, j] = (rng.random((n, 1)) > cum).sum(axis=1).clip(max=cards[j] - 1)
    return X


def make_dataset(d: int, n: int, kind: str = "continuous", edges_per_node: float = 2.0, seed: int = 0):
    """Return (DataFrame with columns x0..x{d-1}, true adjacency matrix)."""
    A = random_dag(d, edges_per_node, seed)
    X = linear_gaussian(A, n, seed + 1)[0] if kind == "continuous" else categorical(A, n, seed + 1)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(d)]), A
