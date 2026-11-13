"""Command line: python -m causal_discovery {simulate,run,benchmark} --help"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="table with one column per variable (.csv, .parquet or .feather)")
    p.add_argument("--drop-columns", nargs="*", default=[], help="columns to ignore, e.g. an id or timestamp column")
    p.add_argument("--data-type", default="auto", choices=["auto", "continuous", "categorical"])
    p.add_argument("--penalty", type=float, default=1.0,
                   help="BIC penalty multiplier for dagma_hc and fges (1 = standard BIC; larger = sparser)")
    p.add_argument("--min-partial-corr", type=float, default=None,
                   help="alternative to --penalty for continuous data: only keep edges whose partial correlation "
                        "is at least this large (e.g. 0.03); overrides --penalty")
    p.add_argument("--lambda1", type=float, default=0.03, help="DAGMA sparsity weight (library default 0.03)")
    p.add_argument("--w-threshold", type=float, default=0.3, help="DAGMA edge threshold (library default 0.3)")
    p.add_argument("--device", default=None, help="cuda or cpu for DAGMA (default: cuda if available)")
    p.add_argument("--k", type=int, default=20, help="dagma_hc: candidate neighbours per variable from DAGMA")
    p.add_argument("--restarts", type=int, default=5,
                   help="dagma_hc: hill-climbing runs with different tie orders; the highest-BIC graph is kept")
    p.add_argument("--candidates", default="dagma+pcorr", choices=["dagma+pcorr", "dagma"],
                   help="dagma_hc: where candidate neighbours come from (default: DAGMA plus partial correlations)")
    p.add_argument("--tabu", type=int, default=0,
                   help="dagma_hc: >0 uses tabu search (allows that many non-improving moves to escape local optima; "
                        "10 is a good value)")
    p.add_argument("--max-parents", type=int, default=None, help="dagma_hc: cap on parents per variable")
    p.add_argument("--threads", type=int, default=None, help="fges: CPU threads (default: all cores)")
    p.add_argument("--max-degree", type=int, default=-1, help="fges: cap on neighbours per variable (-1: none)")
    p.add_argument("--java-heap", default="16g", help="fges: Java memory limit, e.g. 16g")


def _options(args, n_rows: int) -> dict:
    penalty = args.penalty
    if args.min_partial_corr is not None:
        from .fges import penalty_for_min_partial_corr
        penalty = penalty_for_min_partial_corr(args.min_partial_corr, n_rows)
        logging.info("min partial correlation %.3g -> BIC penalty %.2f", args.min_partial_corr, penalty)
    fit = {"lambda1": args.lambda1, "w_threshold": args.w_threshold, "device": args.device}
    return {"dagma_fit": fit,
            "dagma_hc": {"k": args.k, "penalty": penalty, "max_parents": args.max_parents, "restarts": args.restarts,
                         "candidates": args.candidates, "tabu": args.tabu},
            "fges": {"penalty": penalty, "threads": args.threads, "max_degree": args.max_degree,
                     "max_heap": args.java_heap}}


def _load(args):
    from .data import load_table
    df = load_table(args.data)
    if args.drop_columns:
        df = df.drop(columns=args.drop_columns)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m causal_discovery")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a synthetic dataset with a known true graph")
    s.add_argument("--nodes", type=int, default=50)
    s.add_argument("--samples", type=int, default=10_000)
    s.add_argument("--type", default="continuous", choices=["continuous", "categorical"])
    s.add_argument("--edges-per-node", type=float, default=2.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True, help="output data file (.csv or .parquet); the true graph is written "
                                                "next to it as <name>_truth.csv")

    r = sub.add_parser("run", help="learn a graph from the whole table with one method")
    _common(r)
    r.add_argument("--method", required=True, choices=["dagma", "dagma_hc", "fges"])
    r.add_argument("--out", required=True, help="output folder")

    b = sub.add_parser("benchmark", help="compare methods on held-out rows")
    _common(b)
    b.add_argument("--methods", nargs="+", default=["dagma", "dagma_hc", "fges"])
    b.add_argument("--split", default="random", choices=["random", "last"],
                   help="random: shuffle and hold out rows; last: hold out the final rows (time-ordered data)")
    b.add_argument("--test-frac", type=float, default=0.2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--truth", default=None, help="optional true graph (edge list CSV or .npy) for simulated data")
    b.add_argument("--out", required=True, help="output folder")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "simulate":
        from .simulate import make_dataset
        df, A = make_dataset(args.nodes, args.samples, args.type, args.edges_per_node, args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False)
        src, dst = np.nonzero(A)
        truth = out.with_name(out.stem + "_truth.csv")
        import pandas as pd
        pd.DataFrame({"source": df.columns[src], "target": df.columns[dst]}).to_csv(truth, index=False)
        logging.info("wrote %s (%d x %d) and %s (%d edges)", out, *df.shape, truth, int(A.sum()))
        return 0

    df = _load(args)
    opts = _options(args, len(df))
    if args.cmd == "run":
        from .benchmark import run_method, save_graph
        from .data import Preprocessor
        prep = Preprocessor(args.data_type).fit(df).transform(df)
        shared: dict = {}
        A, info = run_method(args.method, prep, opts, shared)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        W = shared["dagma_model"].W_raw if args.method == "dagma" else None
        save_graph(A, prep.names, out / "edges.csv", W)
        np.save(out / "adjacency.npy", A)
        info = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool, dict))}
        (out / "info.json").write_text(json.dumps({"method": args.method, "edges": int(A.sum()), **info}, indent=2,
                                                  default=float))
        logging.info("%s: %d edges -> %s", args.method, int(A.sum()), out / "edges.csv")
        return 0

    from .benchmark import load_truth, run_benchmark
    truth = load_truth(args.truth, list(map(str, df.columns))) if args.truth else None
    res = run_benchmark(df, args.methods, args.split, args.test_frac, args.seed, args.data_type, truth, args.out, opts)
    print((Path(args.out) / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
