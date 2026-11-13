#!/bin/bash
# Reproduce the simulated-data results in README.md (section "Results on simulated data").
# Run from the repository root with the virtual environment active. Needs a GPU for reasonable DAGMA times
# and Java for fGES. Outputs go to benchmarks/<name>/ (results.md, results.json, graph_<method>.csv).
set -euo pipefail
OUT=benchmarks
mkdir -p data
sim() { [ -f "data/$1.csv" ] || python -m causal_discovery simulate --nodes "$2" --samples "$3" --type "$4" --seed 1 --out "data/$1.csv"; }

sim cont100 100 20000 continuous
python -m causal_discovery benchmark --data data/cont100.csv --truth data/cont100_truth.csv --out $OUT/cont100
python -m causal_discovery benchmark --data data/cont100.csv --truth data/cont100_truth.csv --methods dagma_hc --tabu 10 --out $OUT/cont100_tabu
python -m causal_discovery benchmark --data data/cont100.csv --truth data/cont100_truth.csv --methods fges --fges-bound 20 --out $OUT/cont100_fges_bound

sim cont500 500 50000 continuous
python -m causal_discovery benchmark --data data/cont500.csv --truth data/cont500_truth.csv --out $OUT/cont500
python -m causal_discovery benchmark --data data/cont500.csv --truth data/cont500_truth.csv --methods dagma_hc --tabu 10 --out $OUT/cont500_tabu
python -m causal_discovery benchmark --data data/cont500.csv --truth data/cont500_truth.csv --methods fges --fges-bound 0 --out $OUT/cont500_fges_unbounded

sim cat50 50 20000 categorical
python -m causal_discovery benchmark --data data/cat50.csv --truth data/cat50_truth.csv --methods dagma dagma_hc --out $OUT/cat50

# 2,500 variables x 100,000 rows (about 4 GB as CSV; parquet is smaller and faster). fGES is bounded by default at
# this size; unbounded, it had not finished after 2.5 hours (and ran out of memory with a 16 GB Java heap).
[ -f data/sim2500.parquet ] || python -m causal_discovery simulate --nodes 2500 --samples 100000 --out data/sim2500.parquet
python -m causal_discovery benchmark --data data/sim2500.parquet --truth data/sim2500_truth.csv --java-heap 64g --out $OUT/cont2500
