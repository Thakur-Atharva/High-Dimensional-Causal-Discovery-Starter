"""Learning the graph: encode, standardize, mask, and fit DAGMA (or the masked lasso) on the GPU."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from causal_discovery.dagma_torch import DagmaLinear, default_device, is_acyclic_mask
from causal_discovery.masked_lasso import MaskedLasso

from .encoding import Encoding, build_mask, estimate_gpu_gb, fit_encoding, indicator_matrix
from .presence import parse_columns, find_tied_steps

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    W_raw: np.ndarray        # indicator x indicator weights, unthresholded
    enc: Encoding
    info: dict


def check_gpu_memory(d: int, solver: str, device) -> None:
    """Stop before fitting if the estimated memory for d indicators exceeds what is free on the GPU."""
    device = torch.device(device) if device is not None else default_device()
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info(device)
    need = estimate_gpu_gb(d, solver) * 1e9
    if need > free:
        raise SystemExit(
            f"--solver {solver} needs about {need / 1e9:.1f} GB of GPU memory for {d} DAGMA columns, but only "
            f"{free / 1e9:.1f} GB is free on {device}. Re-run utilities.preprocess with a larger --min-lots (e.g. 200) "
            f"to drop more steps, use --solver lasso (less memory), or use a GPU with more memory.")


def prepare_problem(df: pd.DataFrame, columns: pd.DataFrame,
                    time_order: bool = True) -> tuple[Encoding, np.ndarray, np.ndarray]:
    """Indicators, their covariance after standardizing (what DAGMA fits), and the allowed-edge mask."""
    enc = fit_encoding(df, columns)
    X = indicator_matrix(df, enc)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    cov = X.T @ X / X.shape[0]
    del X
    return enc, cov, build_mask(enc, time_order)


def learn_graph(df: pd.DataFrame, columns: pd.DataFrame, *, solver: str = "dagma", lambda1: float = 0.03,
                w_threshold: float = 0.3, time_order: bool = True, device=None, warm_iter: int = 30_000,
                max_iter: int = 60_000, verbose: bool = True) -> FitResult:
    """Fit on the rows of ``df`` (the kept columns of a prepared folder)."""
    t0 = time.perf_counter()
    enc, cov, mask = prepare_problem(df, columns, time_order)
    n, d = len(df), len(enc.names)
    acyclic = is_acyclic_mask(mask)
    log.info("%d lots, %d DAGMA columns; mask %s (%.0f s to encode)", n, d,
             "acyclic" if acyclic else "has cycles", time.perf_counter() - t0)
    if solver == "lasso" and not acyclic:
        if time_order:
            infos, _ = parse_columns(columns["name"].tolist())
            tied = find_tied_steps(infos)
            n, keys = next(iter(tied.items()), ("?", []))
            raise SystemExit(f"Some different steps share a step number ({len(tied)} number(s), e.g. {n}: {keys}), "
                             "so edges between those steps may go both ways and --solver lasso cannot be used. "
                             "Use --solver dagma, or give the steps distinct numbers.")
        raise SystemExit("--no-time-order allows cycles, so --solver lasso cannot be used; use --solver dagma")
    device = torch.device(device) if device is not None else default_device()
    check_gpu_memory(d, solver, device)
    on_gpu = device.type == "cuda" and torch.cuda.is_available()
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    if solver == "lasso":
        model = MaskedLasso(device=device, verbose=verbose)
        model.fit(cov, mask, lambda1=lambda1, w_threshold=w_threshold)
        skipped, iterations = True, model.iterations
    elif solver == "dagma":
        model = DagmaLinear(device=device, verbose=verbose)
        model.fit(cov=cov, mask=mask, lambda1=lambda1, w_threshold=w_threshold, warm_iter=warm_iter, max_iter=max_iter)
        skipped, iterations = model.skipped_acyclicity, None
    else:
        raise ValueError("solver must be 'dagma' or 'lasso'")
    info = {"d": d, "n_rows": n, "solver": solver, "skipped_acyclicity": bool(skipped),
            "seconds": round(float(model.seconds), 1), "h_final": float(model.h_final), "iterations": iterations,
            "time_order": time_order, "lambda1": lambda1, "w_threshold": w_threshold, "device": str(device),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if on_gpu else None}
    return FitResult(model.W_raw, enc, info)


def column_parents(edges: pd.DataFrame) -> dict[str, list[str]]:
    """target column -> its parent columns (sorted), from an edges table."""
    out: dict[str, set[str]] = {}
    for s, t in zip(edges["source"], edges["target"]):
        out.setdefault(t, set()).add(s)
    return {t: sorted(p) for t, p in out.items()}


def within_step_parents(columns: pd.DataFrame, identical: pd.DataFrame) -> dict[str, list[str]]:
    """Baseline graph: each presence column's parent is its own step's machine column; plus the identical edges."""
    kept = columns[columns["kept"]] if "kept" in columns else columns
    machine = {r.step_key: r.name for r in kept.itertuples(index=False) if r.kind == "machine"}
    out: dict[str, set[str]] = {}
    for r in kept.itertuples(index=False):
        if r.kind == "presence" and r.step_key in machine:
            out.setdefault(r.name, set()).add(machine[r.step_key])
    for rep, col in zip(identical["rep_column"], identical["column"]):
        out.setdefault(col, set()).add(rep)
    return {t: sorted(p) for t, p in out.items()}
