"""Judging a learned graph.

Without a known true graph, the honest test is prediction on held-out rows: fit each variable's model given its
learned parents on the training rows, then measure how probable the test rows are (the held-out log-likelihood).

* A graph that misses real dependencies predicts worse.
* A graph with spurious edges overfits a little, but with 80,000 training rows that penalty is tiny. So held-out
  likelihood mainly detects *missing* structure; compare edge counts too.
* Graphs that differ only in the direction of some edges can have exactly the same likelihood. Held-out data cannot
  tell you which way an arrow points.

With a known true graph (simulated data), ``structure_metrics`` also reports edge-level accuracy.
"""
from __future__ import annotations

import math

import numpy as np


def heldout_loglik(A: np.ndarray, train, test, alpha: float = 1.0) -> np.ndarray:
    """Per-test-row log-likelihood (summed over variables) of the DAG ``A`` fitted on ``train``.

    ``train`` / ``test`` are ``Prepared`` objects (same preprocessing). Continuous: linear-Gaussian model per variable.
    Categorical: probability tables with add-``alpha`` smoothing, so unseen combinations get non-zero probability.
    """
    A = np.asarray(A) != 0
    d = A.shape[0]
    out = np.zeros(test.X.shape[0])
    if train.data_type == "continuous":
        Xtr, Xte = train.X, test.X
        cov = Xtr.T @ Xtr / Xtr.shape[0]            # standardized with training statistics, so mean ~ 0
        mtr = Xtr.mean(axis=0)
        for j in range(d):
            P = np.nonzero(A[:, j])[0]
            if len(P):
                C = cov[np.ix_(P, P)] - np.outer(mtr[P], mtr[P])
                c = cov[P, j] - mtr[P] * mtr[j]
                beta = np.linalg.lstsq(C, c, rcond=None)[0]
                pred_tr = mtr[j] + (Xtr[:, P] - mtr[P]) @ beta
                pred_te = mtr[j] + (Xte[:, P] - mtr[P]) @ beta
            else:
                pred_tr = np.full(Xtr.shape[0], mtr[j]); pred_te = np.full(Xte.shape[0], mtr[j])
            var = max(float(np.mean((Xtr[:, j] - pred_tr) ** 2)), 1e-12)
            out += -0.5 * (math.log(2 * math.pi * var) + (Xte[:, j] - pred_te) ** 2 / var)
        # report on the original units' scale: standardizing x by s shifts each density by -log s
        out -= np.sum(np.log(train.std))
        return out
    Xtr, Xte = train.X, test.X
    cards = np.array([len(c) for c in train.categories])
    for j in range(d):
        P = np.nonzero(A[:, j])[0]
        r = int(cards[j])
        cfg_tr = np.zeros(Xtr.shape[0], dtype=np.int64)
        cfg_te = np.zeros(Xte.shape[0], dtype=np.int64)
        for i in P:
            cfg_tr = cfg_tr * cards[i] + Xtr[:, i]
            cfg_te = cfg_te * cards[i] + Xte[:, i]
        keys, inv = np.unique(cfg_tr, return_inverse=True)
        counts = np.bincount(inv * r + Xtr[:, j], minlength=len(keys) * r).reshape(len(keys), r).astype(np.float64)
        pos = np.searchsorted(keys, cfg_te)
        pos = np.clip(pos, 0, len(keys) - 1)
        seen = keys[pos] == cfg_te
        num = np.where(seen, counts[pos, Xte[:, j]], 0.0) + alpha
        den = np.where(seen, counts[pos].sum(axis=1), 0.0) + alpha * r
        out += np.log(num / den)
    return out


def edges_of(A: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(*np.nonzero(np.asarray(A) != 0)))


def structure_metrics(A_est: np.ndarray, A_true: np.ndarray) -> dict:
    """Edge accuracy against a known true DAG.

    shd: structural Hamming distance (missing + extra + reversed edges; lower is better).
    directed_*: an edge counts as correct only with the right direction.
    skeleton_*: direction ignored.
    """
    est, true = edges_of(A_est), edges_of(A_true)
    skel = lambda E: {tuple(sorted(e)) for e in E}
    s_est, s_true = skel(est), skel(true)
    tp_dir = len(est & true)
    reversed_ = len({e for e in est if (e[1], e[0]) in true})
    tp_sk = len(s_est & s_true)
    f1 = lambda tp, p, t: 2 * tp / (p + t) if p + t else 1.0
    return {
        "true_edges": len(true), "est_edges": len(est),
        "directed_precision": tp_dir / len(est) if est else 0.0,
        "directed_recall": tp_dir / len(true) if true else 0.0,
        "directed_f1": f1(tp_dir, len(est), len(true)),
        "skeleton_precision": tp_sk / len(s_est) if s_est else 0.0,
        "skeleton_recall": tp_sk / len(s_true) if s_true else 0.0,
        "skeleton_f1": f1(tp_sk, len(s_est), len(s_true)),
        "reversed": reversed_,
        "shd": len(s_est - s_true) + len(s_true - s_est) + reversed_,
    }


def skeleton_jaccard(A: np.ndarray, B: np.ndarray) -> float:
    a = {tuple(sorted(e)) for e in edges_of(A)}
    b = {tuple(sorted(e)) for e in edges_of(B)}
    return len(a & b) / len(a | b) if a | b else 1.0
