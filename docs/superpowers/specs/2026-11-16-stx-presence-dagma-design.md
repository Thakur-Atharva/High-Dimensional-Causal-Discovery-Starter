# Design: DAGMA on the step-presence matrix, with a time-ordered mask and an interactive edge viewer

Date: 2026-11-16. Branch: `atharva-stx-testing`.

## 1. Intent

The user has a manufacturing routing table, `presence_matrix.csv`, of about 9,000 rows (lots) and 10,000 columns, and wants to
learn a causal graph over it with DAGMA. They want to:

1. preprocess the table into something DAGMA can use,
2. benchmark the learned graph on a random 20% of held-out lots (there is no known true graph), then
3. learn the final graph on all lots, and
4. explore the resulting `edges.csv` in an interactive plotly view.

They will run all of this themselves, on a separate Linux server that has the data. This repository is already
installed there and its 7 tests pass. The data cannot be seen from here, so everything has to work on data we
can't inspect, and it has to fail with clear messages when the data doesn't match the expected format.

### Input format

- `lot_id`: unique identifier per row.
- For each process step, two columns:
  - `N_stage:step`: the **presence** column, with values 0, 1 or 2, treated as **categories**, not numbers.
  - `N_stage:step::mach`: the **machine** column, holding categorical machine names. An empty cell means "absent", which
    is itself a category.
- `N` (the integer before the first `_`) is the step's position in time. Stage and step names are real names
  and may contain any characters.
- **Machines per step:** between 1 and 11, with a long right tail; the average is 4 (from the user).

### Expected size

- Each kept step becomes about 5–6 one-hot columns: about 4 machine indicators plus 1–2 presence indicators.
- With most of the ~5,000 steps kept, the encoded column count is **d ≈ 20,000–30,000**, before the step filter and
  the duplicate merging (§5.3) reduce it.
- Even with the acyclicity term skipped, DAGMA's optimizer does one d × d matrix multiply per iteration. The
  2,500-column run needed about 88,000 iterations (log: `cd_scale_test/big_221172.log`).
- On a 4090-class GPU that is about **45 minutes per fit at d = 10,000 and about a day at d = 25,000**, and the
  workflow needs two fits (benchmark and final).
- GPU memory is about 12 · d² · 4 bytes in float32: 19 GB at 20,000, 30 GB at 25,000.
- **The user's GPU model is not known,** so the runner must check memory before starting (§5.5).

## 2. Decisions the user made

- **DAGMA only.** fGES is out of scope: the package's fGES supports only continuous data, and this data is
  categorical. DAGMA + hill climbing (`dagma_hc`) is not requested either.
- **Benchmark on a random 80/20 split of lots,** judged by held-out prediction. No ground truth exists.
- **Drop steps visited by fewer than `--min-lots` lots** (default 100; the user may raise it to 200). "Visited" means
  presence ≥ 1. Both of the step's columns are dropped.
- **Enforce time order.** Allowed edges:
  - any column of step *a* → any column of step *b*, when `N_a < N_b`;
  - within one step, the machine column → its presence column only. The user expects this edge to be strong; it is
    wanted, not an artefact to suppress.
  - Everything else is forbidden: later → earlier, presence → own machine, and self-loops.
- **One-hot encode** presence and machine values for DAGMA, because DAGMA is a linear model and the values are categories.
- **Pool rare machines:** a machine seen on fewer than `--min-machine-lots` lots (default: the same as `--min-lots`) is relabelled
  `other` within that step's machine column.
- **Collapse edges back to column level:** `edges.csv` has one row per (source column, target column), keeping the
  indicator-level detail (which machines, which presence levels) for the viewer.

## 3. Approaches considered for the time order (chosen: A)

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Allowed-edge mask inside DAGMA, and **skip DAGMA's acyclicity term when the mask alone guarantees a DAG** | Identical result to B (see below), at a fraction of the cost per iteration. Falls back to B automatically when the mask has cycles (for example, two different steps share a number) |
| B | Mask inside DAGMA, acyclicity term computed as usual | Smallest change; each iteration still inverts a d × d matrix, so hours at ~10,000 columns |
| C | Unconstrained DAGMA, then delete backward edges | No code change, but wrong: the fit used the backward edges to explain real dependencies, and deleting them leaves those dependencies with no edge rather than a forward one |

**Why A gives the same answer as B.** If every allowed edge goes forward in a total order, then `W ∘ W` is
nilpotent for every feasible `W`.
- So `det(sI − W∘W) = s^d`, and DAGMA's `h(W) = −log det(sI − W∘W) + d log s` is exactly 0.
- Its gradient, `2 W ∘ (sI − W∘W)^{−T}`, is exactly 0 on every allowed entry, because the inverse is triangular in the
  same order.
- The M-matrix domain check can never fail.

So DAGMA's iterates are the same with or without computing `h`, up to rounding. Skipping `h` replaces a d × d
inversion (plus a log-determinant at checkpoints) with one d × d matrix multiply per iteration. The DAGMA
optimizer (Adam, the `mu`/`s` schedule, stopping rule and 0.3 threshold) stays as it is, so results remain "DAGMA as
published, with forbidden edges".

**A faster solver for the same objective (`--solver lasso`, opt-in).**
- With `h ≡ 0`, the `mu` schedule only rescales the objective. DAGMA then solves the masked problem
  `min_W ½ tr((I−W)ᵀ Σ (I−W)) + λ₁‖W‖₁`, which is d separate lasso regressions: each indicator on its allowed
  earlier indicators.
- A solver built for that problem reaches the same minimiser in far fewer iterations than Adam, for example an
  accelerated proximal gradient (FISTA) run on the GPU for all columns at once. It then applies the same
  `w_threshold`.
- **Default: `--solver dagma`** (DAGMA's own optimizer), so "DAGMA" in the user's results means DAGMA's code path.
- `--solver lasso` is allowed only when the mask is acyclic, and is offered as the fast path at the real size.
- **The Slurm run must show that the two agree on the thresholded edge set** before the README recommends the fast
  solver. Agreement means near-identical edges, and every disagreement within a small margin of the threshold.

## 4. Goals, non-goals, anti-goals

**Goals**
- One command turns `presence_matrix.csv` into a clean prepared folder and prints what was dropped, and why.
- One command benchmarks time-ordered DAGMA on held-out lots; one command learns the final graph on all lots.
- A self-contained HTML viewer that stays responsive with ~10,000 nodes and tens of thousands of edges.
- Clear `utilities/README.md` instructions the user can follow on the other server without help.

**Non-goals**
- fGES, `dagma_hc`, or any other method.
- Changing the package's existing CLI behaviour (`python -m causal_discovery ...`) or its outputs.
- Simulated-truth accuracy metrics (F1/SHD) on the real data.
- Lagged or time-series models. Each lot is one independent row.

**Anti-goals** (failure even if it technically runs)
- **A backward edge in the output.** No edge may go from a later step to an earlier one, or from presence to its
  own machine column. This must be verified by a test and asserted at runtime before `edges.csv` is written.
- **Silently wrong parsing.** Examples: a machine literally named `NA` or `None` turned into "absent" by pandas'
  default NA handling, or a column that doesn't match the naming pattern quietly ignored. Unparseable columns are
  listed and stop the run, unless `--ignore-unparsed` is given.
- **A held-out score that punishes the graph for the scoring model's shortcomings.** Probability tables over many
  parents leave most test combinations unseen, which makes dense graphs score worse than the empty one. See §5.4.
- **Running out of GPU memory after hours.** Preprocessing estimates memory up front, and the runner checks it before
  starting.

## 5. The design

### 5.1 Layout

```
utilities/
  __init__.py
  README.md             how to run everything, in order, on the real data
  presence.py           shared: column-name parsing, step table, one-hot encoding, mask construction, edge collapse
  preprocess.py         CLI: presence_matrix.csv -> prepared folder
  run_dagma.py          CLI: `benchmark` and `fit` subcommands
  plot_edges.py         CLI: edges.csv -> interactive HTML
  simulate_presence.py  CLI: a synthetic presence matrix with planted edges (tests, timing, dry run)
tests/test_utilities.py
```

Run from the repository root as `python -m utilities.<name> ...`. `utilities` is not added to the installed package.
`pyproject.toml` gets a new extra `stx = ["plotly>=5.20", "scikit-learn>=1.4"]`, and `dev` gains the same two so
the tests can run.

### 5.2 Preprocessing (`utilities.preprocess`)

```
python -m utilities.preprocess --input presence_matrix.csv --out prepared/ \
    [--min-lots 100] [--min-machine-lots <min-lots>] [--id-column lot_id] [--ignore-unparsed]
```

1. **Read the header first,** then read the file with explicit dtypes:
   - presence columns as small ints;
   - machine columns as strings, with `keep_default_na=False` so that only truly empty cells are missing.
2. **Parse** each column name with `^(\d+)_(.+?)(::mach)?$`:
   - step number = group 1;
   - step key = the name without `::mach`;
   - stage and step = group 2 split on the first `:` (display only).
   Pair presence and machine columns by step key. A step with only one of the two columns is kept with a warning.
3. **Validate:**
   - presence values must be in {0, 1, 2}, with no missing cells (otherwise stop, listing up to 10 offending
     columns with example values);
   - `lot_id` must be unique (warn if not).
4. **Fill** empty machine cells with `absent`.
5. **Filter steps:** drop a step if fewer than `--min-lots` lots have presence ≥ 1.
6. **Pool rare machines:** within each kept machine column, machines on fewer than `--min-machine-lots` lots become
   `other`. `absent` is never pooled.
7. **Drop constant columns** (a single category after steps 5–6).
8. **Detect tied step numbers:** different step keys with the same `N`. Report them; they make the mask cyclic
   (§5.3) and DAGMA then falls back to the slow path.
9. **Write:**
   - `prepared/data.parquet`: `lot_id` plus kept columns in time order, presence as int8, machine as string.
   - `prepared/columns.csv`: one row per *original* column, with name, step_key, step_number, stage, step, kind
     (`presence`/`machine`), lots_visited, n_categories, kept, drop_reason (`rare_step`, `constant`, `unparsed`).
   - `prepared/summary.json` and a printed summary:
     - rows, original and kept columns, steps dropped per reason, machines pooled;
     - tied step numbers;
     - **encoded column count `d`** (the number of one-hot indicators, §5.3);
     - estimated DAGMA GPU memory (~12 · d² · 4 bytes in float32) and an indicative runtime band.

### 5.3 Encoding and mask (`utilities.presence`)

- **One-hot, dropping one reference level per column:**
  - machine columns drop `absent`, so the indicators read "lot ran on machine M";
  - presence columns drop `0`, so the indicators read "visited once" and "visited twice".
  - If the reference level doesn't occur, drop the most common level.
  - Categories are taken from the training rows; indicators that are constant on the training rows are dropped.
- **Merge identical indicators.** This is Claude's call; the user can override it. Two indicators are identical if they
  are equal or exact complements on the rows being fitted, i.e. |correlation| = 1.
  - **When it happens:** a single-machine step with no revisits, where "ran on M" equals "visited"; or two steps
    always visited together.
  - **Why it matters:** the L1 penalty splits an effect's weight evenly across the copies. The builder check
    reproduced this with the repo's `DagmaLinear`: a 0.49 edge came out as 0.24 + 0.24, both under the 0.3
    threshold, so the edge was lost.
  - **How:** find the groups by hashing each indicator's 0/1 vector and its complement. Keep the **earliest member**
    in the time order as the representative (within a step, machine comes before presence). Its step number and
    kind decide the mask, and the other members are removed from the fit.
  - **What gets reported:**
    - edges into and out of the representative are reported on the representative's column, with the aliases
      named in `detail`;
    - each representative → member pair is written as an edge of kind `identical`. These edges always point forward
      in time, so they respect the time order.
  - Near-duplicates, which differ on a few lots, are not merged; the README notes that they can still split weights.
- **Standardize** each indicator with the training mean and standard deviation, as the package does for DAGMA input.
- **Mask** (bool, d × d, built vectorized from each indicator's (step_number, kind, source column)):
  - allowed if `step_number[i] < step_number[j]`, or
  - allowed if the step numbers are equal, the step keys are the same, `kind[i] == machine` and `kind[j] == presence`;
  - allowed in both directions if the step numbers are equal but the step keys differ (the tie case);
  - never allowed within the same original column, and never on the diagonal.

### 5.4 DAGMA with a mask (package change: `causal_discovery/dagma_torch.py`)

- **New `fit(..., mask=None, skip_acyclicity="auto")`:**
  - `mask` is a d × d 0/1 array of allowed edges.
  - It is combined with the existing `exclude_edges`, which keeps working as before.
  - `W` is multiplied by the mask after every update, as `mask_exc` is today.
- **`skip_acyclicity`:**
  - `"auto"` skips `h` when the mask is acyclic. Check this with a topological sort of the mask; it must handle
    d ≈ 20,000 in seconds (vectorized Kahn's algorithm, not a Python loop over `np.nonzero` rows).
  - `True` forces skipping and raises an error if the mask is cyclic.
  - `False` never skips (used by the equivalence test).
  - When skipping, `_minimize` computes neither the inverse nor the M-matrix check, and `_func` treats `h` as 0.
  - `h_final` is still computed once at the end as a sanity value; it should be ≈ 0.
- **Log** which path ran.
- **No behaviour change** when `mask` is None: all existing tests and the package CLI stay identical.
- **Keep the masked path lean on memory,** to stay usable at d ≈ 25,000 on a 24–48 GB GPU:
  - store the mask as bool, or as a single float tensor that replaces `mask_exc`, not both;
  - don't materialise a dense identity in the skip path;
  - update in place where possible.
- **The `--solver lasso` fast path (§3) lives next to DAGMA in the package,** with the same inputs (covariance,
  mask, `lambda1`, `w_threshold`) and outputs (`W_raw`, `W_est`, `seconds`). The algorithm and stopping rule are
  the builder's choice, within the agreement requirement in §3.

### 5.5 Benchmark and final fit (`utilities.run_dagma`)

```
python -m utilities.run_dagma benchmark --prepared prepared/ --out results/bench \
    [--test-frac 0.2] [--seed 0] [--lambda1 0.03] [--w-threshold 0.3] [--thresholds 0.1 0.2 0.3 0.4 0.5] \
    [--solver dagma|lasso] [--device cuda] [--no-time-order]
python -m utilities.run_dagma fit --prepared prepared/ --out results/final [same DAGMA options]
```

`--thresholds` (benchmark only) scores several edge cut-offs from **one** fit's raw weights, at no extra fitting cost.
`--w-threshold` is the cut-off used for the saved graph.

**`benchmark`:**
1. Make a random split of lots (the package's `split_rows`).
2. One-hot encode and standardize on the training lots.
3. Compute the covariance and fit masked DAGMA (`--no-time-order` runs without the mask, as a comparison; expect it
   to be slow).
4. Collapse to column level.
5. Score held-out lots, then write the results.

**Held-out score.** For every kept original column, a multinomial logistic regression predicts its category from
its parents' one-hot indicators.
- It is fitted on the training lots, with scikit-learn's default L2 regularisation, `C` exposed as `--C`.
- The score is the **log-probability of the test lots' actual categories**, summed over columns and averaged over
  test lots.
- A column with no parents uses training frequencies with add-one smoothing; this is also the **empty graph**
  baseline.
- **Second baseline, "within-step only":** each presence column's parents are just its own step's machine column,
  plus the `identical` edges. These near-certain edges would otherwise dominate the gain over the empty graph. The
  report therefore shows DAGMA's gain over this baseline too, which is what the edges *between* steps add.
- Every scored graph includes the `identical` edges (§5.3), because they are real, deterministic relationships.
- A test category never seen in training gets the smoothed floor probability.
- This replaces the package's probability-table score for this data, because that score breaks down once a column
  has more than a handful of parents (§4).
- Fits are run in parallel over columns (`joblib`, `--jobs`, default all cores).

Report, in `results.md` / `results.json`, in the package's style:
- edges, time, encoded column count, and the path DAGMA took (skip vs full);
- held-out log-likelihood per lot, and gain over the empty graph with a 95% interval;
- `per_column.csv`: each column's held-out gain, so the user can see which steps the graph predicts well.

The graph learned on the training lots is also saved (`edges.csv`, `edges_indicators.csv`).

**`fit`:** the same encoding and DAGMA fit on all lots. It writes:
- `edges.csv`: one row per column-level edge, with columns `source, target, weight, n_indicator_edges, kind, detail`.
  - `kind` is `learned` or `identical` (§5.3).
  - `weight` is the indicator weight with the largest absolute value, with its sign.
  - `detail` lists up to 5 indicator pairs, as `"M17→1 (0.52); M03→1 (−0.41)"`.
- `edges_indicators.csv`: every indicator-level edge, with columns `source, source_category, target, target_category,
  weight`.
- `adjacency.npy` (column level) and `info.json` (options, d, seconds, path taken, h_final, edge count).

**Safety:**
- Before writing, assert that no edge violates the time order (§4).
- Before fitting, compare the memory estimate against `torch.cuda.mem_get_info()` and stop with a clear message if
  it won't fit.

### 5.6 Viewer (`utilities.plot_edges`)

```
python -m utilities.plot_edges --edges results/final/edges.csv [--columns prepared/columns.csv] \
    --out results/final/graph.html [--min-weight 0] [--max-edges 30000]
```

- **Inputs:**
  - Works on any `edges.csv` with `source,target` columns, including the package's own `run` output. Extra
    columns (`weight`, `detail`) are used when present.
  - Node metadata comes from `columns.csv` if given; otherwise it is parsed from the names with the same regex.
  - Nodes without edges are not drawn.
- **Layout: a timeline.** The x-axis is the step number and there are two lanes, machine on top and presence below:

  ```
  machine   ●   ●      ●    ●        ●      ●
            │    ╲___________╱‾‾‾‾‾‾‾‾╲      │
  presence  ●   ●      ●    ●        ●      ●
           ├── stage A ──┤├──── stage B ────┤├─ ...
           1             120                 310        step number →
  ```

  Edges are drawn as arcs whose height grows with the distance they span. Because every edge points forward in
  time, direction reads left to right without arrowheads; the within-step machine → presence edge is a short
  vertical segment. Stages are shown as alternating shaded bands, with the stage name on hover.
- **Rendering:**
  - Nodes, edges and edge-hover markers are one WebGL (`Scattergl`) trace each, with edges as polyline segments
    separated by `None`, so the page stays responsive at ~10,000 nodes and ~30,000 edges.
  - Node size scales with degree; colour shows kind.
  - `identical` edges are drawn in a muted style, distinct from learned edges, and are off by default via a legend
    toggle.
- **Interaction, with no server:** a self-contained HTML file (plotly.js embedded, so it works offline; `--cdn` makes
  it smaller):
  - zoom and pan;
  - hovering a node shows its name, stage, step, lots visited and in/out degree;
  - hovering an edge's midpoint shows source → target, weight and `detail`;
  - **clicking a node** highlights its parents and children and fades everything else; clicking the background
    clears it. This is done with a small script injected via `write_html(post_script=...)`;
  - a **search box** jumps to and highlights a node by name substring.
- **CLI filters:**
  - `--min-weight` drops weak edges;
  - `--max-edges` keeps the strongest edges if there are too many, and prints a warning when it does.

### 5.7 Synthetic data (`utilities.simulate_presence`)

Writes a CSV in the real format, with configurable lots, steps, and machines per step. The default number of
machines per step follows the user's distribution: 1–11, long right tail, mean about 4. Include some single-machine,
no-revisit steps, so the identical-indicator merge is exercised.
- Some steps are rarely visited, to exercise the filter.
- Some machine names are `NA`/`None`, to exercise the parsing.
- Planted forward effects: a given machine at step *a* changes the probability of visiting step *b*, or of revisiting it
  (presence 2).
- The planted column-level graph is written as `<name>_truth.csv`.

It is used by the tests, by the Slurm timing run at the real size, and by the user as a dry run before the real data.

### 5.8 Verification

**Tests** (`tests/test_utilities.py`, CPU, together under ~1 minute):
- **Parsing:** names containing `:` and `_`, a `NA` machine kept as a machine, empty → `absent`, bad presence values
  stop the run.
- **Filtering:** `--min-lots` drops both columns of rare steps; machine pooling; constant columns dropped.
- **Mask:** no forbidden entries are allowed; the tie case is detected as cyclic.
- **DAGMA equivalence:** on a small masked problem, `skip_acyclicity=True` and `False` give `W_raw` equal within
  float64 tolerance, and no forbidden edge is non-zero.
- **Solver agreement:** on a small masked problem, `--solver lasso` and `--solver dagma` give the same thresholded
  edges.
- **Identical-indicator merge:** a single-machine step and two always-together steps are merged. An effect planted
  on a merged column is recovered at full strength, not split.
- **Baselines:** the within-step-only baseline scores above the empty graph on data with machine → presence effects.
- **End to end:** simulate → preprocess → `benchmark` → `fit` → `plot_edges`.
  - Every edge goes forward in time.
  - The planted machine → presence effects are recovered (high recall on a small, strong simulation).
  - The gain over the empty graph is positive.
  - The HTML file exists and contains the plot.

**Slurm run** (this repo's cluster; the GPU on the current machine is busy): simulate at the real size (9,000 lots ×
~5,000 steps, 1–11 machines per step, mean about 4), preprocess, benchmark and fit on a free GPU node. Record:
- time and peak GPU memory for each solver;
- `d`;
- the skip vs full speed-up at a size where full is still feasible;
- the HTML file size.

Also run `--solver lasso` against `--solver dagma` on one mid-size problem (where the dagma solver finishes in under
an hour) and record their edge agreement.

Put the measured numbers in `utilities/README.md`. At the real size, the dagma solver may need longer than a test job
allows. If so, record its per-iteration time and iteration count and extrapolate, saying so.

## 6. Calls made by Claude (not explicitly decided by the user)

- Merging identical or complementary indicators (§5.3).
- The `--solver lasso` fast path, opt-in, with DAGMA's own optimizer as the default (§3). The user was offered this
  and did not object.
- The within-step-only baseline and the `--thresholds` sweep in the benchmark (§5.5).
- Per-column logistic regression as the held-out score, instead of probability tables (§5.5).
- Rare *presence* levels are not pooled; only machines are.
- Viewer: a timeline layout, a self-contained HTML file, and click-to-highlight implemented in injected JavaScript
  rather than a Dash server (§5.6).
- Hard stops for unparseable columns and bad presence values (§4).

## 7. Left to the builder

- Internal function names and module boundaries within `utilities/`, beyond the CLIs above.
- The exact arc shape, colours and band styling of the viewer, within the constraints in §5.6.
- How the simulator's planted effects are parametrised.
- Whether `--jobs` uses processes or threads for the logistic fits.
