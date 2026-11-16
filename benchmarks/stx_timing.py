"""Timing and solver agreement for the step-presence pipeline (used by benchmarks/stx_scale.sh).

    # seconds per iteration of each solver path, at the prepared folder's size
    python benchmarks/stx_timing.py time --prepared data/stx_scale/prepared --iters 200 --out timing.json
    # full fits with --solver dagma and --solver lasso, and how far their edges agree
    python benchmarks/stx_timing.py agree --prepared data/stx_scale/mid/prepared --out agreement.json
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_discovery.dagma_torch import DagmaLinear            # noqa: E402
from causal_discovery.masked_lasso import MaskedLasso            # noqa: E402
from utilities.encoding import collapse_edges                    # noqa: E402
from utilities.fitting import prepare_problem                    # noqa: E402

log = logging.getLogger("stx_timing")


def _problem(prepared: Path):
    data = pd.read_parquet(prepared / "data.parquet")
    cols = pd.read_csv(prepared / "columns.csv")
    t0 = time.perf_counter()
    enc, cov, mask = prepare_problem(data, cols[cols["kept"]])
    log.info("d = %d indicators (%.0f s to encode)", len(enc.names), time.perf_counter() - t0)
    return enc, cov, mask


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_paths(args) -> dict:
    enc, cov, mask = _problem(Path(args.prepared))
    out = {"d": len(enc.names), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for label, skip, iters in (("dagma_skip", True, args.iters), ("dagma_full", False, args.full_iters)):
        if iters <= 0:
            continue
        m = DagmaLinear(device=args.device)
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        m.fit(cov=cov, mask=mask, skip_acyclicity=skip, T=1, warm_iter=iters, max_iter=iters, checkpoint=10 ** 9)
        _sync()
        out[label] = {"iterations": iters, "seconds": round(m.seconds, 1),
                      "sec_per_iter": m.seconds / iters,
                      "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
        log.info("%s: %.4f s per iteration", label, m.seconds / iters)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    m = MaskedLasso(device=args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    m.fit(cov, mask, max_iter=args.iters)
    out["lasso"] = {"iterations": m.iterations, "seconds": round(m.seconds, 1), "sec_per_iter": m.seconds / m.iterations,
                    "converged": m.converged,
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
    if "dagma_full" in out:
        out["skip_speedup"] = out["dagma_full"]["sec_per_iter"] / out["dagma_skip"]["sec_per_iter"]
    return out


def agree(args) -> dict:
    enc, cov, mask = _problem(Path(args.prepared))
    fits = {}
    for name, model in (("lasso", MaskedLasso(device=args.device)), ("dagma", DagmaLinear(device=args.device))):
        if name == "lasso":
            model.fit(cov, mask)
        else:
            model.fit(cov=cov, mask=mask)
        fits[name] = model
        log.info("%s: %.0f s", name, model.seconds)
    Wd, Wl = fits["dagma"].W_raw, fits["lasso"].W_raw
    ed, el = np.abs(Wd) >= 0.3, np.abs(Wl) >= 0.3
    differ = ed ^ el
    col = {k: set(map(tuple, collapse_edges(f.W_raw, enc, 0.3)[0].query("kind == 'learned'")[["source", "target"]]
                       .itertuples(index=False))) for k, f in fits.items()}
    return {"d": len(enc.names),
            "seconds": {k: round(f.seconds, 1) for k, f in fits.items()},
            "lasso_iterations": fits["lasso"].iterations, "lasso_converged": fits["lasso"].converged,
            "indicator_edges": {"dagma": int(ed.sum()), "lasso": int(el.sum()), "both": int((ed & el).sum()),
                                "only_dagma": int((ed & ~el).sum()), "only_lasso": int((el & ~ed).sum())},
            "disagreeing_weights_max_abs": {"dagma": float(np.abs(Wd[differ]).max()) if differ.any() else None,
                                            "lasso": float(np.abs(Wl[differ]).max()) if differ.any() else None},
            "max_abs_weight_difference": float(np.abs(Wd - Wl).max()),
            "column_edges": {"dagma": len(col["dagma"]), "lasso": len(col["lasso"]),
                             "both": len(col["dagma"] & col["lasso"])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["time", "agree"])
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=200, help="time: iterations for the skip path and the lasso")
    ap.add_argument("--full-iters", type=int, default=20, help="time: iterations for DAGMA's full path (0: skip)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    res = time_paths(args) if args.mode == "time" else agree(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
