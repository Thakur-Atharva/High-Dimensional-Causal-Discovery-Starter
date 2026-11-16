#!/bin/bash
# Full-size check of the step-presence pipeline on simulated data shaped like the real table.
#
#   sbatch -p <partition> --gres=gpu:1 -c 16 --mem=128G -t 24:00:00 benchmarks/stx_scale.sh full
#   sbatch -p <partition> --gres=gpu:1 -c 16 --mem=64G  -t 12:00:00 benchmarks/stx_scale.sh agree
#
# full : 9,000 lots x 5,000 steps (10,000 columns, 1-11 machines per step, mean ~4): preprocess, benchmark and fit
#        with --solver lasso, the viewer, and per-iteration timings of every solver path.
# agree: 9,000 lots x 1,500 steps: full fits with --solver dagma and --solver lasso, and how far their edges agree.
# Run from the repository root. ENV points at the Python environment (default: the one used for development).
set -euo pipefail
MODE=${1:-full}
ENV=${ENV:-/.gavea/store/thakur27/envs/cd_starter312}
source "$ENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
OUT=${OUT:-data/stx_scale}
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; date

if [ "$MODE" = full ]; then
  mkdir -p "$OUT"
  python -m utilities.simulate_presence --lots 9000 --steps 5000 --effects 200 --seed 0 --out "$OUT/presence_matrix.csv"
  /usr/bin/time -v python -m utilities.preprocess --input "$OUT/presence_matrix.csv" --out "$OUT/prepared" 2>&1 \
    | grep -vE "^\s+(Average|Exit|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page|Percent)"
  /usr/bin/time -v python -m utilities.run_dagma benchmark --prepared "$OUT/prepared" --out "$OUT/bench" --solver lasso 2>&1 \
    | grep -vE "^\s+(Average|Exit|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page|Percent)"
  /usr/bin/time -v python -m utilities.run_dagma fit --prepared "$OUT/prepared" --out "$OUT/final" --solver lasso 2>&1 \
    | grep -vE "^\s+(Average|Exit|Major|Minor|Voluntary|Involuntary|Swaps|File|Socket|Signals|Page|Percent)"
  python -m utilities.plot_edges --edges "$OUT/final/edges.csv" --columns "$OUT/prepared/columns.csv" \
    --out "$OUT/final/graph.html"
  python benchmarks/stx_timing.py time --prepared "$OUT/prepared" --iters 200 --full-iters 20 --out "$OUT/timing.json"
else
  mkdir -p "$OUT/mid"
  python -m utilities.simulate_presence --lots 9000 --steps 1500 --effects 60 --seed 1 --out "$OUT/mid/presence_matrix.csv"
  python -m utilities.preprocess --input "$OUT/mid/presence_matrix.csv" --out "$OUT/mid/prepared"
  python benchmarks/stx_timing.py agree --prepared "$OUT/mid/prepared" --out "$OUT/mid/agreement.json"
  python benchmarks/stx_timing.py time --prepared "$OUT/mid/prepared" --iters 200 --full-iters 200 \
    --out "$OUT/mid/timing.json"
fi
date
