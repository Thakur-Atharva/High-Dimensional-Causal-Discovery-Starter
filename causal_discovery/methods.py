"""The methods, behind one interface: each takes preprocessed data and returns a 0/1 adjacency matrix A
(A[i, j] = 1 means i -> j) plus a dict of details."""
from __future__ import annotations

import logging
import time

import numpy as np

from .dagma_torch import DagmaLinear
from .hill_climb import candidate_pairs, hill_climb, partial_correlations
from .scores import CategoricalBIC, GaussianBIC, resolve_penalty

log = logging.getLogger(__name__)


def _as_float(prep) -> np.ndarray:
    """DAGMA needs numbers. Continuous data are already standardized; categorical codes are standardized here."""
    X = prep.X.astype(np.float64)
    if prep.data_type == "categorical":
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return X


def fit_dagma(prep, lambda1: float = 0.03, w_threshold: float = 0.3, device=None, dtype=None,
              verbose: bool = True, **fit_kwargs) -> DagmaLinear:
    """Fit DAGMA once. The fitted object holds the thresholded graph (W_est) and the raw weights (W_raw)."""
    X = _as_float(prep)
    cov = X.T @ X / X.shape[0]
    model = DagmaLinear(device=device, dtype=dtype, verbose=verbose)
    log.info("DAGMA: %d variables on %s (%s)", cov.shape[0], model.device, str(model.dtype).replace("torch.", ""))
    model.fit(cov=cov, lambda1=lambda1, w_threshold=w_threshold, **fit_kwargs)
    return model


def dagma(prep, model: DagmaLinear | None = None, **kwargs):
    """DAGMA as it is normally used: linear least-squares fit, edges with |weight| >= 0.3 (the library defaults)."""
    model = model or fit_dagma(prep, **kwargs)
    A = (model.W_est != 0).astype(np.int8)
    return A, {"seconds": model.seconds, "h_final": model.h_final, "acyclic": model.h_final < 1e-6}


def bic_score(prep, penalty: float = 1.0):
    if prep.data_type == "continuous":
        X = prep.X
        Xc = X - X.mean(axis=0)
        return GaussianBIC(Xc.T @ Xc / X.shape[0], X.shape[0], penalty=penalty)
    return CategoricalBIC(prep.X, penalty=penalty)


def dagma_hc(prep, model: DagmaLinear | None = None, k: int = 20, penalty: float = 1.0,
             min_partial_corr: float | None = None, max_parents: int | None = None, restarts: int = 5,
             candidates: str = "dagma+pcorr", tabu: int = 0, seed: int = 0, verbose: bool = True, **dagma_kwargs):
    """DAGMA proposes the k strongest neighbours of each variable; BIC hill climbing picks edges and directions.

    candidates="dagma+pcorr" (default) adds each variable's k largest partial correlations to DAGMA's proposals:
    DAGMA's list alone misses 3-13% of true neighbours in our simulations, which caps what hill climbing can find.
    candidates="dagma" uses DAGMA's proposals only.
    tabu > 0 runs tabu search instead of plain hill climbing (see hill_climb.HillClimber.run).

    Greedy hill climbing depends on how exact ties are broken (adding i -> j and j -> i often score the same), so it
    is run ``restarts`` times: first with ties going to DAGMA's stronger direction, then with random tie orders.
    The graph with the highest BIC is returned; no ground truth is involved.
    """
    model = model or fit_dagma(prep, verbose=verbose, **dagma_kwargs)
    pairs = set(candidate_pairs(model.W_raw, k=k))
    if candidates == "dagma+pcorr":
        pairs |= set(candidate_pairs(partial_correlations(_as_float(prep)), k=k))
    elif candidates != "dagma":
        raise ValueError("candidates must be 'dagma' or 'dagma+pcorr'")
    pairs = sorted(pairs)
    d = prep.X.shape[1]
    log.info("DAGMA + hill climbing: %d candidate pairs (%s, k = %d per variable), %d restart(s)", len(pairs),
             candidates, k, restarts)
    penalty = resolve_penalty(penalty, min_partial_corr, prep.X.shape[0])
    if min_partial_corr is not None:
        log.info("min partial correlation %.3g -> BIC penalty %.2f", min_partial_corr, penalty)
    score = bic_score(prep, penalty)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best = None
    for r in range(max(1, restarts)):
        prefer = np.abs(model.W_raw) if r == 0 else rng.random((d, d))
        A = hill_climb(score, d, pairs, max_parents=max_parents, prefer=prefer, tabu=tabu, verbose=verbose)
        hc = hill_climb.last
        bic = float(hc.local.sum())
        log.info("  restart %d: %d edges, BIC %.1f (%.0f s)", r + 1, int(A.sum()), bic, hc.seconds)
        if best is None or bic > best[1]:
            best = (A, bic, hc)
    A, bic, hc = best
    hc_seconds = time.perf_counter() - t0
    return A, {"seconds": model.seconds + hc_seconds, "dagma_seconds": model.seconds, "hc_seconds": hc_seconds,
               "candidate_pairs": len(pairs), "candidates": candidates, "restarts": restarts, "penalty": penalty,
               "steps": hc.steps, "moves": hc.moves, "bic": bic}


def fges(prep, **kwargs):
    """fGES from Tetrad (needs Java 21+ and py-tetrad); see fges.py."""
    from .fges import fges as _fges
    return _fges(prep, **kwargs)


METHODS = {"dagma": dagma, "dagma_hc": dagma_hc, "fges": fges}
