"""Turn presence_matrix.csv into a prepared folder for run_dagma.

    python -m utilities.preprocess --input presence_matrix.csv --out prepared/ [--min-lots 100]

Writes:
  data.parquet   the id column and the kept columns, in time order (presence as int8, machine as text)
  columns.csv    every step column: step number, stage, step, kind, lots visited, categories, kept, why dropped
  summary.json   counts, the number of DAGMA columns (one-hot indicators) and GPU memory / runtime estimates
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .encoding import estimate_gpu_gb, fit_encoding, indicator_matrix
from .fitting import count_near_duplicates
from .presence import filter_and_pool, find_tied_steps, load_presence_matrix, parse_columns

log = logging.getLogger(__name__)

# Seconds per iteration measured at REF_D indicators (float32 GPU) and typical iteration counts. Each iteration is a
# d x d matrix product for both solvers, so time scales with (d / REF_D)^3. Placeholders until the full-size run.
REF_D = 10_000
REF_SEC_PER_ITER = {"dagma": 0.035, "lasso": 0.035}
TYPICAL_ITERS = {"dagma": 88_000, "lasso": 2_000}
REF_GPU = "RTX 4090-class GPU (estimate)"


def runtime_note(d: int) -> str:
    parts = []
    for solver in ("dagma", "lasso"):
        minutes = REF_SEC_PER_ITER[solver] * TYPICAL_ITERS[solver] * (d / REF_D) ** 3 / 60
        parts.append(f"--solver {solver}: ~{minutes:,.0f} min per fit")
    return f"{'; '.join(parts)} on an {REF_GPU}; time grows with d^3"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m utilities.preprocess", description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="presence_matrix.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--min-lots", type=int, default=100, help="drop steps visited (presence >= 1) by fewer lots")
    ap.add_argument("--min-machine-lots", type=int, default=None,
                    help="pool machines used on fewer lots into 'other' (default: --min-lots)")
    ap.add_argument("--id-column", default="lot_id")
    ap.add_argument("--ignore-unparsed", action="store_true",
                    help="skip columns whose names don't match 'N_stage:step[::mach]' instead of stopping")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df, infos = load_presence_matrix(args.input, args.id_column, args.ignore_unparsed)
    log.info("read %d lots x %d step columns", len(df), len(infos))
    unparsed = df.attrs.get("unparsed", [])
    kept, table, summary = filter_and_pool(df, infos, args.id_column, args.min_lots, args.min_machine_lots)
    if unparsed:
        table = pd.concat([table, pd.DataFrame({"name": unparsed, "kept": False, "drop_reason": "unparsed"})],
                          ignore_index=True)
    kept_infos, _ = parse_columns(list(kept.columns), args.id_column)
    tied = find_tied_steps(kept_infos)
    enc = fit_encoding(kept, table)
    d = len(enc.names)
    X = indicator_matrix(kept, enc)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    near = count_near_duplicates(X, enc.column)
    del X
    summary.update({"min_lots": args.min_lots,
                    "min_machine_lots": args.min_machine_lots if args.min_machine_lots is not None else args.min_lots,
                    "tied_step_numbers": {str(k): v for k, v in tied.items()},
                    "columns_unparsed": len(unparsed),
                    "encoded_columns": d, "identical_indicators_merged": int(sum(map(len, enc.aliases.values()))),
                    "near_duplicate_indicator_pairs": near,
                    "gpu_gb_dagma": float(f"{estimate_gpu_gb(d, 'dagma'):.3g}"),
                    "gpu_gb_lasso": float(f"{estimate_gpu_gb(d, 'lasso'):.3g}"),
                    "runtime_note": runtime_note(d)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out / "data.parquet", index=False)
    table.to_csv(out / "columns.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['rows']} lots; {summary['original_columns']} step columns in {summary['steps_total']} steps")
    print(f"  dropped {summary['steps_dropped_rare']} steps visited by fewer than {args.min_lots} lots, "
          f"and {summary['columns_dropped_constant']} constant columns")
    print(f"  pooled {summary['machines_pooled']} rarely used machines into 'other'")
    print(f"  kept {summary['kept_columns']} columns -> {d} one-hot DAGMA columns "
          f"({summary['identical_indicators_merged']} identical indicators merged)")
    if unparsed:
        print(f"  skipped {len(unparsed)} columns whose names don't match the pattern (see columns.csv)")
    if near:
        print(f"  {near} pairs of indicators from different columns are near-duplicates (|correlation| >= 0.95): "
              "DAGMA may split an effect between them, so also read the 0.1-0.2 threshold rows of the benchmark")
    if tied:
        print(f"  WARNING: {len(tied)} step numbers are shared by different steps (e.g. {next(iter(tied.items()))}); "
              "edges between them are allowed both ways, so --solver lasso cannot be used and DAGMA runs its "
              "full (slow) path")
    print(f"  GPU memory needed: ~{summary['gpu_gb_dagma']} GB (--solver dagma), "
          f"~{summary['gpu_gb_lasso']} GB (--solver lasso)")
    print(f"  runtime: {summary['runtime_note']}")
    print(f"wrote {out / 'data.parquet'}, {out / 'columns.csv'}, {out / 'summary.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
