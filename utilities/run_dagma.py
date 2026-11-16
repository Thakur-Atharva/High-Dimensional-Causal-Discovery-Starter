"""Benchmark time-ordered DAGMA on held-out lots, or learn the final graph on all lots.

    python -m utilities.run_dagma benchmark --prepared prepared/ --out results/bench [--solver lasso]
    python -m utilities.run_dagma fit       --prepared prepared/ --out results/final [--solver lasso]

``benchmark`` fits on a random 80% of lots and scores the other 20%: each column's category is predicted from its
parents (logistic regression), and the log-probability of the held-out lots is compared with two baselines, the
empty graph and the "within-step only" graph (each step's machine column -> its presence column).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_discovery.data import split_rows

from .encoding import check_time_order, collapse_edges
from .fitting import column_parents, learn_graph, within_step_parents
from .scoring import compare, score_tasks

log = logging.getLogger(__name__)


def _load(prepared: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(prepared / "data.parquet")
    columns = pd.read_csv(prepared / "columns.csv")
    return data, columns[columns["kept"]].reset_index(drop=True)


def _fit(args, rows: pd.DataFrame, kept: pd.DataFrame):
    return learn_graph(rows, kept, solver=args.solver, lambda1=args.lambda1, w_threshold=args.w_threshold,
                       time_order=not args.no_time_order, device=args.device, warm_iter=args.warm_iter,
                       max_iter=args.max_iter)


def _write_edges(out: Path, fit, kept: pd.DataFrame, threshold: float, time_order: bool) -> pd.DataFrame:
    edges, ind = collapse_edges(fit.W_raw, fit.enc, threshold)
    if time_order:
        check_time_order(edges, kept)
    edges.to_csv(out / "edges.csv", index=False)
    ind.to_csv(out / "edges_indicators.csv", index=False)
    return edges


def benchmark(args) -> int:
    data, kept = _load(Path(args.prepared))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te = split_rows(len(data), "random", args.test_frac, args.seed)
    train, test = data.iloc[tr].reset_index(drop=True), data.iloc[te].reset_index(drop=True)
    log.info("benchmark: %d training lots, %d held-out lots", len(train), len(test))
    fit = _fit(args, train, kept)
    saved = _write_edges(out, fit, kept, args.w_threshold, not args.no_time_order)

    columns = kept["name"].tolist()
    graphs = {"empty graph": ({}, pd.DataFrame(columns=saved.columns)),
              "within-step only": (within_step_parents(kept, fit.enc.identical), None)}
    thresholds = sorted(set(args.thresholds) | {args.w_threshold})
    for t in thresholds:
        e = collapse_edges(fit.W_raw, fit.enc, t)[0]
        graphs[f"dagma (threshold {t:g})"] = (column_parents(e), e)
    t0 = time.perf_counter()
    tasks = {name: [(c, tuple(p.get(c, []))) for c in columns] for name, (p, _) in graphs.items()}
    flat = [t for ts in tasks.values() for t in ts]
    lp_all = score_tasks(train, test, flat, args.C, args.jobs)
    lp, k = {}, 0
    for name, ts in tasks.items():
        lp[name], k = lp_all[:, k:k + len(ts)], k + len(ts)
    log.info("scored %d graphs on %d held-out lots in %.0f s", len(graphs), len(test), time.perf_counter() - t0)

    rows = []
    for name, (parents, e) in graphs.items():
        ge, gw = compare(lp[name], lp["empty graph"]), compare(lp[name], lp["within-step only"])
        if e is None:                                  # within-step baseline: machine -> presence, plus identical
            pairs = {(par, t) for t, ps in parents.items() for par in ps}
            ident = set(zip(fit.enc.identical["rep_column"], fit.enc.identical["column"]))
            learned, identical = len(pairs - ident), len(ident)
        else:
            learned, identical = int((e["kind"] == "learned").sum()), int((e["kind"] == "identical").sum())
        if name not in [f"dagma (threshold {t:g})" for t in args.thresholds] + ["empty graph", "within-step only"]:
            continue                                   # the saved threshold, if not requested: used for per_column only
        rows.append({"graph": name, "learned_edges": learned, "identical_edges": identical,
                     "heldout_logprob_per_lot": float(lp[name].sum(axis=1).mean()),
                     "gain_vs_empty": ge["gain"], "gain_vs_empty_ci95": ge["ci95"],
                     "gain_vs_within_step": gw["gain"], "gain_vs_within_step_ci95": gw["ci95"]})
    saved_name = f"dagma (threshold {args.w_threshold:g})"
    saved_parents = graphs[saved_name][0]
    pd.DataFrame({"column": columns, "n_parents": [len(saved_parents.get(c, [])) for c in columns],
                  "gain_vs_empty": (lp[saved_name] - lp["empty graph"]).mean(axis=0),
                  "gain_vs_within_step": (lp[saved_name] - lp["within-step only"]).mean(axis=0)}
                 ).sort_values("gain_vs_empty", ascending=False).to_csv(out / "per_column.csv", index=False)
    table = pd.DataFrame(rows)
    options = {k: v for k, v in vars(args).items() if k != "func"}
    result = {"graphs": rows, "n_train": len(train), "n_test": len(test), "options": options, "fit": fit.info}
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    (out / "info.json").write_text(json.dumps({**fit.info, "options": options}, indent=2, default=float))
    md = [f"# DAGMA benchmark: {len(data)} lots ({len(train)} train / {len(test)} held out), "
          f"{len(columns)} columns, {fit.info['d']} DAGMA columns, solver {args.solver}", "",
          "Held-out log-probability per lot: each column's category predicted from its parents (logistic regression "
          "fitted on the training lots), summed over columns, averaged over held-out lots. Higher is better. "
          "`gain_vs_empty` compares with every column independent; `gain_vs_within_step` with only each step's "
          "machine -> presence edges, i.e. what the edges *between* steps add. `_ci95` is the half-width of a 95% "
          "interval.", "", table.to_markdown(index=False, floatfmt=".4g"), "",
          f"Fit: {fit.info['seconds']} s, {'acyclicity term skipped (time order is strict)' if fit.info['skipped_acyclicity'] else 'full DAGMA'}."
          f" Saved graph: threshold {args.w_threshold:g} -> edges.csv; per-column gains -> per_column.csv.", ""]
    (out / "results.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


def fit_final(args) -> int:
    data, kept = _load(Path(args.prepared))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fit = _fit(args, data, kept)
    edges = _write_edges(out, fit, kept, args.w_threshold, not args.no_time_order)
    columns = kept["name"].tolist()
    index = {c: i for i, c in enumerate(columns)}
    A = np.zeros((len(columns), len(columns)), dtype=np.int8)
    for s, t in zip(edges["source"], edges["target"]):
        A[index[s], index[t]] = 1
    np.save(out / "adjacency.npy", A)
    info = {**fit.info, "edges_learned": int((edges["kind"] == "learned").sum()),
            "edges_identical": int((edges["kind"] == "identical").sum()), "columns": columns,
            "options": {k: v for k, v in vars(args).items() if k != "func"}}
    (out / "info.json").write_text(json.dumps(info, indent=2, default=float))
    log.info("%d learned + %d identical edges -> %s", info["edges_learned"], info["edges_identical"],
             out / "edges.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m utilities.run_dagma", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func, help_ in [("benchmark", benchmark, "fit on 80% of lots, score the held-out 20%"),
                              ("fit", fit_final, "learn the final graph on all lots")]:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=func)
        p.add_argument("--prepared", required=True, help="folder written by utilities.preprocess")
        p.add_argument("--out", required=True, help="output folder")
        p.add_argument("--solver", default="dagma", choices=["dagma", "lasso"],
                       help="dagma: DAGMA's own optimizer (default); lasso: same objective, much faster solver")
        p.add_argument("--lambda1", type=float, default=0.03, help="DAGMA's L1 weight (library default 0.03)")
        p.add_argument("--w-threshold", type=float, default=0.3, help="edge cut-off on |weight| (library default 0.3)")
        p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
        p.add_argument("--no-time-order", action="store_true",
                       help="allow edges in any direction (slow; full DAGMA; for comparison only)")
        p.add_argument("--warm-iter", type=int, default=30_000, help=argparse.SUPPRESS)
        p.add_argument("--max-iter", type=int, default=60_000, help=argparse.SUPPRESS)
        if name == "benchmark":
            p.add_argument("--test-frac", type=float, default=0.2)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5],
                           help="edge cut-offs to score (all from one fit)")
            p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the scoring regressions")
            p.add_argument("--jobs", type=int, default=-1, help="parallel scoring jobs (-1: all cores)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
