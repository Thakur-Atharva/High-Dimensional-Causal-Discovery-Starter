"""Train/test benchmark of the methods on one table.

1. Split the rows: a random 20% (independent samples) or the last 20% (time-ordered data).
2. Preprocess using training rows only, so no test information leaks in.
3. Learn a graph with each method on the training rows.
4. For each graph, fit every variable's model given its parents on the training rows, then compute the
   log-likelihood of the test rows (``evaluate.heldout_loglik``). Higher is better.
5. Report it next to the empty graph (every variable independent) and compare methods row by row, with a 95%
   interval for the difference.

If a true graph is supplied (simulated data), edge-level accuracy is reported as well.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import methods as M
from .data import Preprocessor, split_rows
from .evaluate import heldout_loglik, skeleton_jaccard, structure_metrics
from .simulate import topological_order

log = logging.getLogger(__name__)


def is_dag(A: np.ndarray) -> bool:
    try:
        topological_order(A)
        return True
    except ValueError:
        return False


def save_graph(A: np.ndarray, names: list[str], path: Path, weights: np.ndarray | None = None) -> None:
    src, dst = np.nonzero(A)
    df = pd.DataFrame({"source": [names[i] for i in src], "target": [names[j] for j in dst]})
    if weights is not None:
        df["weight"] = weights[src, dst]
    df.to_csv(path, index=False)


def load_truth(path: str | Path, names: list[str]) -> np.ndarray:
    """True graph as an edge list CSV (columns source,target, using the data's column names) or an n x n .npy."""
    path = Path(path)
    if path.suffix == ".npy":
        return (np.load(path) != 0).astype(np.int8)
    e = pd.read_csv(path)
    idx = {n: i for i, n in enumerate(names)}
    A = np.zeros((len(names), len(names)), dtype=np.int8)
    for s, t in zip(e["source"].astype(str), e["target"].astype(str)):
        A[idx[s], idx[t]] = 1
    return A


def run_method(name: str, train, options: dict, shared: dict):
    """Run one method on the training data. DAGMA is fitted once and shared by 'dagma' and 'dagma_hc'."""
    opts = options.get(name, {})
    if name in ("dagma", "dagma_hc"):
        if "dagma_model" not in shared:
            shared["dagma_model"] = M.fit_dagma(train, **options.get("dagma_fit", {}))
        if name == "dagma":
            return M.dagma(train, model=shared["dagma_model"])
        return M.dagma_hc(train, model=shared["dagma_model"], **opts)
    if name == "fges":
        from .fges import fges
        return fges(train, **opts)
    raise ValueError(f"unknown method {name!r}; choose from dagma, dagma_hc, fges")


def run_benchmark(df: pd.DataFrame, methods=("dagma", "dagma_hc", "fges"), split: str = "random",
                  test_frac: float = 0.2, seed: int = 0, data_type: str = "auto", truth: np.ndarray | None = None,
                  out_dir: str | Path | None = None, options: dict | None = None) -> dict:
    options = options or {}
    tr_idx, te_idx = split_rows(len(df), split, test_frac, seed)
    pre = Preprocessor(data_type).fit(df.iloc[tr_idx])
    train, test = pre.transform(df.iloc[tr_idx]), pre.transform(df.iloc[te_idx])
    log.info("data: %d rows x %d variables (%s); split=%s -> %d train / %d test rows", len(df), df.shape[1],
             train.data_type, split, len(tr_idx), len(te_idx))
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    d = df.shape[1]
    graphs = {"empty graph": np.zeros((d, d), dtype=np.int8)}
    info = {"empty graph": {"seconds": 0.0}}
    shared: dict = {}
    for name in methods:
        log.info("=== %s", name)
        t0 = time.perf_counter()
        A, details = run_method(name, train, options, shared)
        details = {k: v for k, v in details.items() if k != "cpdag"}
        details.setdefault("seconds", time.perf_counter() - t0)
        graphs[name], info[name] = np.asarray(A, dtype=np.int8), details
        if out:
            W = shared["dagma_model"].W_raw if name == "dagma" else None
            save_graph(graphs[name], train.names, out / f"graph_{name}.csv", W)
    ll = {name: heldout_loglik(A, train, test) for name, A in graphs.items()}
    rows = []
    for name, A in graphs.items():
        diff = ll[name] - ll["empty graph"]
        row = {"method": name, "edges": int(A.sum()), "acyclic": is_dag(A),
               "seconds": round(float(info[name].get("seconds", 0.0)), 1),
               "heldout_loglik_per_row": float(ll[name].mean()),
               "per_variable": float(ll[name].mean() / d),
               "gain_vs_empty_per_row": float(diff.mean()),
               "gain_vs_empty_95ci": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))}
        if truth is not None:
            row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in structure_metrics(A, truth).items()
                        if k != "est_edges"})
        rows.append(row)
    table = pd.DataFrame(rows)
    names = [m for m in graphs if m != "empty graph"]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = ll[a] - ll[b]
            pairs.append({"comparison": f"{a} - {b}", "heldout_loglik_diff_per_row": float(diff.mean()),
                          "95ci": float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))),
                          "skeleton_jaccard": round(skeleton_jaccard(graphs[a], graphs[b]), 3)})
    pair_table = pd.DataFrame(pairs)
    result = {"split": split, "test_frac": test_frac, "seed": seed, "data_type": train.data_type,
              "n_train": len(tr_idx), "n_test": len(te_idx), "n_variables": d,
              "methods": table.to_dict(orient="records"), "pairwise": pair_table.to_dict(orient="records"),
              "details": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool, dict))}
                          for k, v in info.items()}}
    if out:
        (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
        md = [f"# Benchmark: {len(df)} rows x {d} variables ({train.data_type}), split = {split}, "
              f"test = {len(te_idx)} rows", "",
              "Held-out log-likelihood: average over test rows of the log-probability of the whole row under the "
              "graph's model (higher is better). 'gain_vs_empty' compares with treating every variable as "
              "independent; '95ci' is the half-width of its 95% interval.", "",
              table.to_markdown(index=False, floatfmt=".4g"), "", "## Method vs method", "",
              pair_table.to_markdown(index=False, floatfmt=".4g") if len(pair_table) else "(one method only)", ""]
        if split == "last":
            md += ["Note: with time-ordered rows, neighbouring rows are correlated, so the intervals above are "
                   "too narrow.", ""]
        (out / "results.md").write_text("\n".join(md))
    return result
