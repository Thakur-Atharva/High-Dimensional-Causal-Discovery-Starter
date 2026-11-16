"""Held-out scoring of a column-level graph on categorical data.

Each column's category is predicted from its parents' categories by a multinomial logistic regression fitted on the
training lots (L2-regularised, ``C``). The score is the log-probability of the test lots' actual categories.

Why not one probability table per combination of parent values (the package's categorical score)? With more than a
handful of parents almost every test lot's combination is unseen in training and gets a flat probability, so dense
graphs would score below the empty graph for reasons unrelated to their edges.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _codes(train: pd.Series, test: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer codes from the training categories; test categories unseen in training get -1."""
    cats = np.array(sorted(train.astype(str).unique()))
    tr = np.searchsorted(cats, train.astype(str).to_numpy())
    te_raw = test.astype(str).to_numpy()
    te = np.searchsorted(cats, te_raw).clip(0, len(cats) - 1)
    te = np.where(cats[te] == te_raw, te, -1)
    return tr, te, len(cats)


def _one_hot(codes: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """All levels of every parent; a code of -1 (unseen) gives an all-zero block."""
    n = len(codes[0][0])
    F = np.zeros((n, sum(r for _, r in codes)), dtype=np.float64)
    offset = 0
    for c, r in codes:
        rows = np.flatnonzero(c >= 0)
        F[rows, offset + c[rows]] = 1.0
        offset += r
    return F


def _score_column(y_tr, y_te, r, par_tr, par_te, C):
    n_tr = len(y_tr)
    out = np.empty(len(y_te))
    unseen = y_te < 0
    if not par_tr or r < 2:
        p = (np.bincount(y_tr, minlength=r) + 1.0) / (n_tr + r)       # add-one smoothed frequencies
        out[:] = np.log(p[y_te.clip(0)])
    else:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = LogisticRegression(C=C, max_iter=1000).fit(_one_hot(par_tr), y_tr)
        P = model.predict_proba(_one_hot(par_te))                     # classes are 0..r-1 (all seen in training)
        out[:] = np.log(np.clip(P[np.arange(len(y_te)), y_te.clip(0)], 1e-300, None))
    out[unseen] = math.log(1.0 / (n_tr + r + 1))
    return out


def heldout_logprob(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], parents: dict[str, list[str]],
                    C: float = 1.0, jobs: int = -1) -> np.ndarray:
    """Log-probability of each test lot's category, per column: shape (len(test), len(columns))."""
    needed = set(columns) | {p for c in columns for p in parents.get(c, [])}
    codes = {c: _codes(train[c], test[c]) for c in needed}
    tasks = []
    for c in columns:
        y_tr, y_te, r = codes[c]
        pars = parents.get(c, [])
        tasks.append((y_tr, y_te, r, [(codes[p][0], codes[p][2]) for p in pars],
                      [(codes[p][1], codes[p][2]) for p in pars], C))
    if jobs == 1:
        cols = [_score_column(*t) for t in tasks]
    else:
        cols = Parallel(n_jobs=jobs)(delayed(_score_column)(*t) for t in tasks)
    return np.column_stack(cols) if cols else np.zeros((len(test), 0))


def compare(lp: np.ndarray, lp_ref: np.ndarray) -> dict:
    """Mean per-lot gain in total log-probability over a reference graph, with a 95% interval half-width."""
    diff = lp.sum(axis=1) - lp_ref.sum(axis=1)
    return {"gain": float(diff.mean()), "ci95": float(1.96 * diff.std(ddof=1) / math.sqrt(len(diff)))}
