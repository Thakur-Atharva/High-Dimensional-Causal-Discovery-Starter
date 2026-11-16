"""Masked DAGMA and the masked lasso solver (CPU, a few seconds)."""
import time

import numpy as np
import pytest
import torch

from causal_discovery.dagma_torch import DagmaLinear, is_acyclic_mask


def ordered_problem(d: int = 12, n: int = 4000, seed: int = 0):
    """Linear SEM whose true DAG follows a random order; the mask allows every edge consistent with that order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)                       # order[i] = position of variable i in time
    allowed = order[:, None] < order[None, :]
    A = allowed & (rng.random((d, d)) < 0.3)
    W = A * rng.uniform(0.5, 2.0, (d, d)) * rng.choice([-1.0, 1.0], (d, d))
    X = np.zeros((n, d))
    for j in np.argsort(order):                      # parents first
        X[:, j] = X @ W[:, j] + rng.normal(size=n)
    X = (X - X.mean(0)) / X.std(0)
    return X.T @ X / n, allowed, A.astype(np.int8)


def test_is_acyclic_mask():
    assert is_acyclic_mask(np.triu(np.ones((5, 5), bool), 1))
    m = np.zeros((3, 3), bool)
    m[0, 1] = m[1, 0] = True
    assert not is_acyclic_mask(m)
    perm = np.random.default_rng(0).permutation(4000)
    big = np.triu(np.ones((4000, 4000), bool), 1)[np.ix_(perm, perm)]   # a DAG, but not upper-triangular
    t0 = time.perf_counter()
    assert is_acyclic_mask(big)
    assert time.perf_counter() - t0 < 10


def test_masked_dagma_skip_equals_full():
    cov, mask, _ = ordered_problem()
    kw = dict(cov=cov, mask=mask, warm_iter=5000, max_iter=10000)
    a = DagmaLinear(device="cpu", dtype=torch.float64)
    a.fit(skip_acyclicity=True, **kw)
    b = DagmaLinear(device="cpu", dtype=torch.float64)
    # Rounding leaves entries of about -1.6e-16 in inv(sI - W o W) where the exact value is 0, which the original's
    # zero-tolerance float64 domain check treats as leaving the domain (it then retries with larger s and smaller lr).
    # Allow for rounding so the comparison isolates what the skip removes: h and its gradient.
    b._neg_tol = 1e-12
    b.fit(skip_acyclicity=False, **kw)
    assert a.skipped_acyclicity and not b.skipped_acyclicity
    np.testing.assert_allclose(a.W_raw, b.W_raw, atol=1e-6)
    assert ((a.W_est != 0) == (b.W_est != 0)).all()
    assert not a.W_raw[~mask].any() and abs(a.h_final) < 1e-8


def test_masked_dagma_cyclic_mask():
    cov, _, _ = ordered_problem()
    full = ~np.eye(12, dtype=bool)
    m = DagmaLinear(device="cpu", dtype=torch.float64)
    m.fit(cov=cov, mask=full, warm_iter=500, max_iter=1000)
    assert not m.skipped_acyclicity                  # "auto" falls back to the full path
    with pytest.raises(ValueError):
        DagmaLinear(device="cpu", dtype=torch.float64).fit(cov=cov, mask=full, skip_acyclicity=True)


def test_masked_lasso_matches_dagma():
    from causal_discovery.masked_lasso import MaskedLasso
    cov, mask, _ = ordered_problem()
    dag = DagmaLinear(device="cpu", dtype=torch.float64)
    dag.fit(cov=cov, mask=mask)
    las = MaskedLasso(device="cpu", dtype=torch.float64)
    las.fit(cov, mask)
    assert las.converged
    assert ((dag.W_est != 0) == (las.W_est != 0)).all()
    # DAGMA's fixed-size Adam steps and its relative-change stopping rule leave it ~1e-3 from the exact minimiser
    np.testing.assert_allclose(las.W_raw, dag.W_raw, atol=5e-3)

    def objective(W):
        D = np.eye(len(W)) - W
        return 0.5 * np.trace(D.T @ cov @ D) + 0.03 * np.abs(W).sum()
    assert objective(las.W_raw) <= objective(dag.W_raw) + 1e-9


def test_masked_lasso_kkt():
    from causal_discovery.masked_lasso import MaskedLasso
    cov, mask, _ = ordered_problem(seed=3)
    las = MaskedLasso(device="cpu", dtype=torch.float64)
    las.fit(cov, mask, lambda1=0.03)
    W = las.W_raw
    G = cov @ W - cov
    tol = 1e-3
    act = mask & (W != 0)
    ina = mask & (W == 0)
    assert np.abs(G[act] + 0.03 * np.sign(W[act])).max() < tol
    assert np.abs(G[ina]).max() < 0.03 + tol and not W[~mask].any()


def test_masked_lasso_rejects_cyclic_mask():
    from causal_discovery.masked_lasso import MaskedLasso
    cov, _, _ = ordered_problem()
    with pytest.raises(ValueError):
        MaskedLasso(device="cpu").fit(cov, ~np.eye(12, dtype=bool))
