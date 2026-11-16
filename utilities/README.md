# Step-presence data: DAGMA with a time order

These utilities take a lot × step table (`presence_matrix.csv`) and learn which steps influence which later steps,
with DAGMA. Edges are only allowed forward in time.

Expected input:

| Column | Meaning |
|---|---|
| `lot_id` | one row per lot |
| `N_stage:step` | presence at that step: `0`, `1` or `2`, treated as categories |
| `N_stage:step::mach` | the machine that processed the lot there; an empty cell means *absent* (also a category) |

`N` is the step's position in time.

Run every command from the repository root.

## 0. Install

In the environment where `python -m pytest -q` already passes:

```bash
pip install -e ".[stx]"          # adds plotly and scikit-learn
python -m pytest -q              # should now run the utilities' tests too
```

A GPU is needed at full size (see section 6).

## 1. Preprocess

```bash
python -m utilities.preprocess --input presence_matrix.csv --out prepared/ --min-lots 100
```

**What it does:**
- Drops every step visited (presence ≥ 1) by fewer than `--min-lots` lots, both columns of it.
- Pools machines used on fewer than `--min-machine-lots` lots (default: `--min-lots`) into `other`.
- Drops columns that are constant.

**What it prints:** what was dropped and the number of **DAGMA columns** (`d`, explained in section 5), followed by
the GPU memory and runtime to expect. Check these before going on. If `d` is too large for your GPU, re-run with
`--min-lots 200`.

**It stops, with the offending columns listed,** if:
- a column name doesn't match `N_stage:step` or `N_stage:step::mach` (pass `--ignore-unparsed` to skip such columns);
- a presence value is missing or isn't 0/1/2.

Machine names are kept exactly as written: `NA`, `None` and `007` are machines, not missing values.

**Output in `prepared/`:**
- `data.parquet`: the kept columns, in time order.
- `columns.csv`: every step column, with its step number, stage, step, lots visited, number of categories, and why it
  was dropped, if it was.
- `summary.json`: the printed summary.

## 2. Benchmark on held-out lots

```bash
python -m utilities.run_dagma benchmark --prepared prepared/ --out results/bench --solver lasso
```

**What it does:** learns the graph on a random 80% of lots and scores it on the other 20% (`--test-frac`, `--seed`).

**What the score is:** for every column, a logistic regression predicts the lot's category from the column's parents
in the graph. It is fitted on the training lots. The score is how probable the held-out lots' actual categories are,
as a log-probability summed over columns and averaged over lots. Higher is better.

**`results.md`** has one row per graph:

| Row | What it is |
|---|---|
| `empty graph` | every column independent |
| `within-step only` | each step's machine column → its presence column, and nothing else |
| `dagma (threshold t)` | the learned graph, keeping edges with \|weight\| ≥ t; one row per t in `--thresholds` (default 0.1 … 0.5), all from one fit |

**How to read it:**
- **`gain_vs_within_step`** is what the edges *between* steps add. It is the number to watch, because the
  machine → presence edges inside a step are nearly certain and dominate `gain_vs_empty`.
- **`_ci95`** is the half-width of a 95% interval. If an interval includes 0, the difference isn't clear.
- **Choosing a threshold:** when two thresholds score about the same, prefer the higher one, because it gives fewer
  edges. Extra edges barely hurt prediction.
- **`per_column.csv`** shows which columns the graph predicts well, with their number of parents.

## 3. Learn the final graph

```bash
python -m utilities.run_dagma fit --prepared prepared/ --out results/final --solver lasso [--w-threshold 0.3]
```

This fits on all lots. **`edges.csv`** has one row per pair of columns:

| Column | Meaning |
|---|---|
| `source`, `target` | original column names. With the time order, `source` is always earlier |
| `weight` | the strongest indicator-level weight behind the edge, signed |
| `n_indicator_edges` | how many category pairs connect the two columns |
| `kind` | `learned` (found by DAGMA) or `identical` (see below) |
| `detail` | up to 5 of those category pairs, e.g. `T0012B→absent (-0.41)`: lots run on `T0012B` at the earlier step are *less* often absent at the later step. `[≡ ...]` lists columns that were identical to an indicator and merged into it |

Also written:
- `edges_indicators.csv`: every category-level edge, with its weight.
- `adjacency.npy`: column level, in the order of `info.json["columns"]`.
- `info.json`: settings, `d`, time, and peak GPU memory.

**`identical` edges** connect two columns that are the same up to relabelling on these lots, so DAGMA cannot tell
them apart. Examples:
- a step with a single machine and no revisits, where "ran on M" is the same as "visited";
- two steps that are always visited together.

They are reported rather than learned.

## 4. Look at it

```bash
python -m utilities.plot_edges --edges results/final/edges.csv --columns prepared/columns.csv \
    --out results/final/graph.html
```

This writes one HTML file that works offline: copy it to your laptop and open it in a browser. `--cdn` makes it about
4 MB smaller but then needs internet.

**Layout:**
- Time runs left to right.
- Machine columns are on the top lane and presence columns on the bottom lane.
- Edges are arcs; every edge points to the right.
- Shaded bands are stages.

**Using it:**
- Hover a column, or the top of an arc, for details.
- **Click a column** to highlight its parents and children.
- Type part of a name in the search box and press Enter to jump to it.
- Esc or **Clear** resets the highlight.
- Drag to zoom along time; double-click to zoom back out.
- The legend toggles learned and identical edges.

`--min-weight 0.4` hides weaker edges. `--max-edges` caps how many are drawn (default 30,000), keeping the strongest.
The viewer also works on `edges.csv` from `python -m causal_discovery run`.

## 5. What happens inside

**Indicators.** DAGMA is a linear model, so each category becomes its own 0/1 column ("one-hot"):
- **Machine columns keep every category, `absent` included.** The `absent` indicator then carries "was the lot
  processed at this step?" on its own.
  - In simulations, dropping `absent` instead split an effect on *visiting* a step across all of its machines, and
    hid about half of the planted edges.
  - Presence level 1 usually equals "not absent", so it merges into that indicator.
- **Presence columns drop level `0`.**

`d`, the number of DAGMA columns, is the number of indicators. Expect about 5–6 per kept step.

**Identical indicators are merged.** When two indicators are equal, or exact opposites, on the lots being fitted, only
the earliest one in time is kept. Otherwise DAGMA's L1 penalty splits an effect's weight between the copies, and both
halves can fall below the 0.3 cut-off: a 0.49 edge became 0.24 + 0.24 in a test.

*Near*-duplicates, such as columns that differ on a handful of lots, are not merged, so they can still share a weight.
If an edge you expect is missing, check whether its column has a near-twin.

**Time-order mask.** An indicator may only have parents from earlier steps. Within a step, the machine may be a
parent of the presence, not the reverse. Edges between two different steps that share a step number are allowed both
ways.

**Two solvers for the same objective:**
- `--solver dagma` runs DAGMA's own optimizer (Adam, the library's schedule, 0.3 cut-off).
  - Because the mask already rules out cycles, DAGMA's acyclicity term is exactly zero for every allowed graph, and
    its gradient is too. So it is skipped, with an identical result: it was a d × d matrix inversion per iteration.
- `--solver lasso` solves the same objective directly (FISTA).
  - With the mask, DAGMA's objective is a separate L1 regression for each indicator on its earlier indicators.
  - It reaches the exact minimum in far fewer iterations.
  - It found the same edges as `--solver dagma` (section 6).
  - It cannot be used if steps share a number, because the mask then has cycles.

**Reading the edges:**
- **An edge into a step's *machine* column often means "affects whether lots go through this step".** The `absent`
  indicator lives in the machine column. The `detail` string says which categories.
- **Directions come from the time order, not from the data.** An edge says the earlier column predicts the later one,
  given everything else that came before it. An unmeasured cause of both can create an edge. Treat edges as
  hypotheses to check.

## 6. Runtime and memory (measured)

{{MEASURED}}

**Scaling:** both solvers do one d × d matrix product per iteration, so time grows with d³. Doubling `d` makes a fit
about 8× slower and needs 4× the memory. `preprocess` prints an estimate for your `d`.

## 7. Troubleshooting

| Message | What to do |
|---|---|
| `column name(s) don't match ...` | rename those columns, or pass `--ignore-unparsed` to skip them |
| `presence column(s) have values other than 0, 1 or 2` | fix or drop those cells; the message lists examples |
| `Some different steps share a step number` | use `--solver dagma` (slower: it falls back to full DAGMA), or give the steps distinct numbers |
| `needs about X GB of GPU memory ... only Y GB is free` | re-run `preprocess` with `--min-lots 200`, or use a larger GPU |
| a fit is too slow | use `--solver lasso`; raise `--min-lots` |

## 8. Dry run without the real data

```bash
python -m utilities.simulate_presence --lots 9000 --steps 500 --out sim/presence_matrix.csv   # + sim/presence_matrix_truth.csv
python -m utilities.preprocess --input sim/presence_matrix.csv --out sim/prepared
python -m utilities.run_dagma benchmark --prepared sim/prepared --out sim/bench --solver lasso
```

The simulator writes the real format, with planted machine → later-step effects listed in `*_truth.csv`.
`benchmarks/stx_scale.sh` runs this at full size on Slurm.
