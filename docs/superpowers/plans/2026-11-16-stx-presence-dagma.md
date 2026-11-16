# STX presence-matrix DAGMA utilities: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `utilities/` toolkit that does four things:
- preprocesses `presence_matrix.csv`;
- benchmarks time-ordered DAGMA on held-out lots;
- learns the final graph;
- renders `edges.csv` as an interactive plotly page.

It also adds a time-order mask and a fast masked solver to the package's DAGMA.

**Architecture:**
- **Package changes.** `causal_discovery/dagma_torch.py` gains an allowed-edge `mask`, and skips the acyclicity
  term when the mask is itself acyclic. A new `causal_discovery/masked_lasso.py` solves the same masked objective
  faster.
- **The `utilities/` modules:** `presence.py` parses, loads and filters; `encoding.py` handles one-hot encoding,
  merging identical indicators, the mask, and collapsing edges back to columns; `scoring.py` does held-out logistic
  scoring; `fitting.py` glues these together.
- **Thin CLIs on top:** `preprocess.py`, `run_dagma.py`, `plot_edges.py` and `simulate_presence.py`.

**Tech Stack:** Python 3.12, numpy, pandas, pyarrow, PyTorch, scikit-learn (with joblib), plotly, and pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-stx-presence-dagma-design.md`. Read it before starting any task. The
section numbers below (§) refer to it.

## Global Constraints

- **Branch:** `atharva-stx-testing`.
- **Commit messages** end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Test environment:** `source /.gavea/store/thakur27/envs/cd_starter312/bin/activate`, then run from the repo root.
  The full suite is `python -m pytest -q`. New tests run on CPU and together take about a minute.
- **No change to package behaviour when `mask` is None.** The existing 7 tests and `python -m causal_discovery ...`
  must behave exactly as before.
- **Utilities are run as `python -m utilities.<name>` from the repo root.** `utilities` is *not* added to the
  installed package (`[tool.setuptools.packages.find]` is unchanged).
- **`pyproject.toml`:** new extra `stx = ["plotly>=5.20", "scikit-learn>=1.4"]`; the `dev` extra gains the same two.
- **Column-name regex:** `^(\d+)_(.+?)(::mach)?$`. Category literals are `"absent"` (empty machine cell) and
  `"other"` (pooled rare machines).
- **Defaults:**
  - preprocessing: `--min-lots 100`, `--min-machine-lots` (= `--min-lots` when omitted), `--id-column lot_id`;
  - fitting: `--test-frac 0.2`, `--seed 0`, `--lambda1 0.03`, `--w-threshold 0.3`,
    `--thresholds 0.1 0.2 0.3 0.4 0.5`, `--solver dagma`, `--C 1.0`, `--jobs -1`;
  - viewer: `--min-weight 0`, `--max-edges 30000`.
- **No output may contain an edge from a later step to an earlier one, or from presence to its own step's machine
  column.** Call `check_time_order` before writing any `edges.csv`.
- **GPU memory estimate:** `12 · d² · 4` bytes for `--solver dagma` and `8 · d² · 4` bytes for `--solver lasso`.

## Review Focus

1. **Machine cells pandas would misread:**
   - literal `NA` and `None`;
   - numeric-looking names such as `007`;
   - whitespace-only cells;
   - a UTF-8 BOM before `lot_id`.

   Expected: names kept verbatim, whitespace-only cells treated as `absent`, and the id column found (Task 4 test).
2. **Presence written as floats** (`1.0`, which pandas writes when a column once held NaN). Expected: accepted as
   0/1/2 (Task 4 test).
3. **A step that has only its presence column, or only its machine column.** Expected: kept with a warning, and
   the mask and edge collapse work without its partner (Task 4 and Task 5 tests).
4. **An `edges.csv` from the package's own `run` command** (`source,target` only, with names like `X1` that don't
   match the pattern). Expected: the viewer still renders every node (Task 9 test).
5. **A GPU too small for d.** Expected: a clear stop *before* fitting that names `--min-lots` as the fix (Task 8
   test).

---

### Task 1: Masked DAGMA, with the acyclicity term skipped when it's redundant

**Files:**
- Modify: `causal_discovery/dagma_torch.py` (`_minimize`, `_func`, `fit`, and a new module-level function)
- Test: `tests/test_masked.py` (new)

**Interfaces:**
- Produces:
  - `is_acyclic_mask(mask: np.ndarray) -> bool` in `causal_discovery/dagma_torch.py`.
  - `DagmaLinear.fit(X=None, cov=None, ..., exclude_edges=None, mask: np.ndarray | None = None, skip_acyclicity: bool | str = "auto") -> np.ndarray`.
  - After `fit`, these attributes are set: `W_raw`, `W_est`, `seconds`, `h_final`, and the new
    `skipped_acyclicity: bool`.

- [ ] **Step 1: Write the failing tests** in `tests/test_masked.py`.

  The tests share a helper, `ordered_problem(d=12, n=4000, seed=0) -> (cov, mask, A_true)`. It builds a linear SEM
  whose true DAG follows a random permutation `order`, with |weights| in [0.5, 2]. The data are standardized and
  `cov = X.T @ X / n`, and `mask[i, j] = order[i] < order[j]`.

```python
def test_is_acyclic_mask():
    assert is_acyclic_mask(np.triu(np.ones((5, 5), bool), 1))
    m = np.zeros((3, 3), bool); m[0, 1] = m[1, 0] = True
    assert not is_acyclic_mask(m)
    perm = np.random.default_rng(0).permutation(4000)
    big = np.triu(np.ones((4000, 4000), bool), 1)[np.ix_(perm, perm)]   # a DAG, but not upper-triangular
    t0 = time.perf_counter(); assert is_acyclic_mask(big); assert time.perf_counter() - t0 < 10

def test_masked_dagma_skip_equals_full():
    cov, mask, _ = ordered_problem()
    kw = dict(cov=cov, mask=mask, warm_iter=5000, max_iter=10000)
    a = DagmaLinear(device="cpu", dtype=torch.float64); a.fit(skip_acyclicity=True, **kw)
    b = DagmaLinear(device="cpu", dtype=torch.float64); b.fit(skip_acyclicity=False, **kw)
    assert a.skipped_acyclicity and not b.skipped_acyclicity
    np.testing.assert_allclose(a.W_raw, b.W_raw, atol=1e-6)
    assert ((a.W_est != 0) == (b.W_est != 0)).all()
    assert not a.W_raw[~mask].any() and abs(a.h_final) < 1e-8

def test_masked_dagma_cyclic_mask():
    cov, _, _ = ordered_problem()
    full = ~np.eye(12, dtype=bool)
    m = DagmaLinear(device="cpu", dtype=torch.float64); m.fit(cov=cov, mask=full, warm_iter=500, max_iter=1000)
    assert not m.skipped_acyclicity                     # "auto" falls back to the full path
    with pytest.raises(ValueError):
        DagmaLinear(device="cpu", dtype=torch.float64).fit(cov=cov, mask=full, skip_acyclicity=True)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_masked.py -q`. Expected: FAIL, with an ImportError for
  `is_acyclic_mask`.

- [ ] **Step 3: Implement.**
  - **`is_acyclic_mask`:** a vectorized Kahn's algorithm. Repeatedly remove every node whose in-degree is 0, and
    update the in-degrees with `mask[removed].sum(0)`. The mask is cyclic when no node can be removed but some
    remain.
  - **The mask in `fit`:** combine `mask` with `exclude_edges` into the existing single float tensor `mask_exc`, and
    set its diagonal to 0.
  - **Choosing the path:** `skip = is_acyclic_mask(mask)` for `"auto"`. With `True` on a cyclic mask, raise
    `ValueError`. Without a mask, never skip.
  - **When skipping:**
    - `_minimize` computes neither `torch.linalg.inv` nor the M-matrix check, and uses
      `Gobj = G_score + mu * lambda1 * sign(W)`.
    - `_func` returns `h = 0` without calling `slogdet`.
    - Write `G_score` as `-mu * (cov - cov @ W)`, so no dense identity is needed.
  - Still compute `h_final` once at the end.
  - Log which path ran, via `log.info`.

- [ ] **Step 4: Run** `python -m pytest -q`. Expected: all pass, including the 7 existing tests.

- [ ] **Step 5: Commit:** `git add causal_discovery/dagma_torch.py tests/test_masked.py && git commit -m "DAGMA: allowed-edge mask; skip the acyclicity term when the mask is acyclic"`

### Task 2: Fast masked solver (`MaskedLasso`)

**Files:**
- Create: `causal_discovery/masked_lasso.py`
- Test: `tests/test_masked.py`

**Interfaces:**
- Consumes: `is_acyclic_mask`, plus the `ordered_problem` test helper (both from Task 1).
- Produces: `MaskedLasso(device=None, dtype=None, verbose=False)`, with
  `.fit(cov, mask, lambda1=0.03, w_threshold=0.3, max_iter=20_000, check_every=100, kkt_tol=None) -> np.ndarray`.
  After `fit`, it sets `W_raw`, `W_est`, `seconds`, `iterations`, `converged: bool` and `h_final = 0.0`, the same
  names as `DagmaLinear`. The device and dtype defaults also match `DagmaLinear`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_masked_lasso_matches_dagma():
    cov, mask, _ = ordered_problem()
    dag = DagmaLinear(device="cpu", dtype=torch.float64); dag.fit(cov=cov, mask=mask)
    las = MaskedLasso(device="cpu", dtype=torch.float64); las.fit(cov, mask)
    assert las.converged
    assert ((dag.W_est != 0) == (las.W_est != 0)).all()
    np.testing.assert_allclose(las.W_raw, dag.W_raw, atol=1e-3)   # Adam's step size (3e-4) limits DAGMA's precision

def test_masked_lasso_kkt():
    cov, mask, _ = ordered_problem(seed=3)
    las = MaskedLasso(device="cpu", dtype=torch.float64); las.fit(cov, mask, lambda1=0.03)
    W = las.W_raw; G = cov @ W - cov; tol = 1e-3
    act = mask & (W != 0); ina = mask & (W == 0)
    assert np.abs(G[act] + 0.03 * np.sign(W[act])).max() < tol
    assert np.abs(G[ina]).max() < 0.03 + tol and not W[~mask].any()

def test_masked_lasso_rejects_cyclic_mask():
    cov, _, _ = ordered_problem()
    with pytest.raises(ValueError):
        MaskedLasso(device="cpu").fit(cov, ~np.eye(12, dtype=bool))
```

- [ ] **Step 2: Run** `python -m pytest tests/test_masked.py -q`. Expected: the 3 new tests FAIL with an
  ImportError.

- [ ] **Step 3: Implement `MaskedLasso.fit`.** It minimises `½ tr((I−W)ᵀ Σ (I−W)) + λ₁‖W‖₁` over masked `W`. This
  is the same objective as DAGMA's `_score` plus its L1 term (§3).

```
L   = largest eigenvalue of cov (power iteration, ~100 steps, times 1.01)
W = Y = 0;  t = 1;  kkt_tol = kkt_tol or 1e-2 * lambda1
loop k = 1..max_iter:
    G     = cov @ Y - cov                                  # gradient
    W_new = soft_threshold(Y - G / L, lambda1 / L) * mask
    if objective(W_new) > objective(W) (checked every check_every): t = 1, Y = W  # adaptive restart
    t_new = (1 + sqrt(1 + 4 t^2)) / 2;  Y = W_new + ((t - 1) / t_new) (W_new - W);  W, t = W_new, t_new
    every check_every: compute the KKT violation at W (as in the test); stop when it is <= kkt_tol (converged = True)
```

  Raise `ValueError` if `not is_acyclic_mask(mask)`. Keep at most about 8 d × d tensors alive at once.

- [ ] **Step 4: Run** `python -m pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit:** `git add causal_discovery/masked_lasso.py tests/test_masked.py && git commit -m "Add MaskedLasso: DAGMA's masked objective solved by FISTA"`

### Task 3: Synthetic presence matrix

**Files:**
- Create: `utilities/__init__.py` (empty) and `utilities/simulate_presence.py`
- Test: `tests/test_utilities.py` (new)

**Interfaces:**
- Produces:
  - `simulate_presence(n_lots=2000, n_steps=60, mean_machines=4.0, max_machines=11, rare_steps=3, n_effects=10, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]`.
    It returns `(table, truth)`:
    - `table`: `lot_id`, then for each step k the columns `f"{k}_{stage}:{step}"` (int 0/1/2) and
      `f"{k}_{stage}:{step}::mach"` (str, with `""` meaning absent).
    - `truth`: columns `source, target`, holding the planted cross-step effects at column level (machine column of
      step a → presence column of step b, with a < b).
  - CLI: `python -m utilities.simulate_presence --lots N --steps S --seed 0 --out path.csv`. It writes the CSV and
    `<stem>_truth.csv`.
- Required features (§5.7):
  - stage and step names contain both `_` and `:`, e.g. stage `STG_3`, step `step:7`;
  - machines per step range from 1 to 11, with a long right tail and a mean of about 4;
  - at least one step uses a machine literally named `NA`, and at least one uses `None`;
  - `rare_steps` steps are visited by fewer than 2% of lots;
  - at least 2 steps have a single machine and no revisits;
  - one pair of steps is always visited together;
  - each planted effect raises step b's visit probability from about 0.2 to about 0.9 for lots that ran on a chosen
    machine at step a;
  - presence 2 means a revisit;
  - the machine cell is empty exactly when presence is 0.

- [ ] **Step 1: Write the failing tests.**

```python
def test_simulate_presence_format():
    df, truth = simulate_presence(n_lots=500, n_steps=40, seed=1)
    assert df.columns[0] == "lot_id" and df["lot_id"].is_unique
    mach = [c for c in df.columns[1:] if c.endswith("::mach")]; pres = [c for c in df.columns[1:] if c not in mach]
    assert len(pres) == len(mach) == 40
    assert set(np.unique(df[pres].to_numpy())) <= {0, 1, 2}
    assert all(((df[p] == 0) == (df[p + "::mach"] == "")).all() for p in pres)
    assert {"NA", "None"} <= set(df[mach].to_numpy().ravel())
    num = lambda c: int(c.split("_", 1)[0])
    assert len(truth) > 0 and all(num(s) < num(t) for s, t in truth.itertuples(index=False))

def test_simulate_machine_distribution():
    df, _ = simulate_presence(n_lots=3000, n_steps=500, rare_steps=0, seed=2)
    k = np.array([df[c].replace("", np.nan).nunique() for c in df.columns if c.endswith("::mach")])
    assert k.min() >= 1 and k.max() <= 11 and 3 <= k.mean() <= 5
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py -q`. Expected: FAIL with an ImportError.
- [ ] **Step 3: Implement** `simulate_presence` and its CLI `main(argv=None) -> int` in `utilities/simulate_presence.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_utilities.py -q`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add utilities/__init__.py utilities/simulate_presence.py tests/test_utilities.py && git commit -m "utilities: synthetic presence matrix with planted edges"`

### Task 4: Parse, load and filter (`utilities/presence.py`)

**Files:**
- Create: `utilities/presence.py`
- Test: `tests/test_utilities.py`

**Interfaces:**
- Consumes: `simulate_presence` (Task 3), in tests.
- Produces:
  - **Constants:** `COLUMN_RE`, `ABSENT = "absent"`, `OTHER = "other"`.
  - **`ColumnInfo`:** `@dataclass(frozen=True) class ColumnInfo: name: str; step_key: str; step_number: int; stage: str; step: str; kind: str  # "presence" | "machine"`.
    - `step_key` is the name without `::mach`.
    - `stage` and `step` are group 2 of the regex, split on the first `:`. If there is no `:`, `step` is `""`.
  - **`parse_columns(names: Sequence[str], id_column: str = "lot_id") -> tuple[list[ColumnInfo], list[str]]`**
    returns `(parsed, unparsed)`; the id column is in neither list.
  - **`load_presence_matrix(path, id_column="lot_id", ignore_unparsed=False) -> tuple[pd.DataFrame, list[ColumnInfo]]`**
    returns the id column plus every parsed column, with presence as int8 and machines as str (empty or
    whitespace-only cells become `ABSENT`).
    - Read the header with `encoding="utf-8-sig"`.
    - Read presence columns as float and validate them: no missing values, every value in {0, 1, 2}. Otherwise
      raise `ValueError` naming up to 10 columns, each with an example value.
    - Read machine columns with `dtype=str, na_filter=False` and strip them.
    - If any names are unparsed and `ignore_unparsed` is False, raise `ValueError` listing them.
    - Warn (via `logging`) if the id values aren't unique.
  - **`filter_and_pool(df, infos, id_column="lot_id", min_lots=100, min_machine_lots=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]`**
    returns `(kept_df, columns_table, summary)`.
    - `kept_df` holds the id column and the kept columns in time order: by `(step_number, step_key)`, with the
      machine column before the presence column.
    - `columns_table` has one row per parsed column, with the columns `name, step_key, step_number, stage, step,
      kind, lots_visited, n_categories, kept, drop_reason`. `drop_reason` is one of `"" | "rare_step" | "constant"`.
    - The order of operations is §5.2 steps 5–7, and `ABSENT` is never pooled.
    - A step that has only one of its two columns is kept, with a `logging` warning.
  - **`find_tied_steps(infos) -> dict[int, list[str]]`** returns `step_number → step_keys`, only for numbers shared
    by more than one step key.

- [ ] **Step 1: Write the failing tests.** Each test writes a small CSV to `tmp_path`.

```python
def test_parse_columns():
    parsed, unparsed = parse_columns(["lot_id", "1_ST_A:step_x", "1_ST_A:step_x::mach", "12_B:c:d", "bad", "3_Q"])
    by = {p.name: p for p in parsed}
    assert by["1_ST_A:step_x::mach"] == ColumnInfo("1_ST_A:step_x::mach", "1_ST_A:step_x", 1, "ST_A", "step_x", "machine")
    assert (by["12_B:c:d"].stage, by["12_B:c:d"].step, by["3_Q"].step) == ("B", "c:d", "")
    assert unparsed == ["bad"]

def test_load_machine_names_verbatim(tmp_path):        # Review Focus 1
    p = tmp_path / "m.csv"
    p.write_text("﻿lot_id,1_S:a,1_S:a::mach\nL1,1,NA\nL2,1,None\nL3,1,007\nL4,0, \nL5,0,\n", encoding="utf-8")
    df, _ = load_presence_matrix(p)
    assert df["1_S:a::mach"].tolist() == ["NA", "None", "007", "absent", "absent"] and "lot_id" in df

def test_load_float_presence(tmp_path):                # Review Focus 2
    p = tmp_path / "f.csv"; p.write_text("lot_id,1_S:a,1_S:a::mach\nL1,1.0,M\nL2,0.0,\nL3,2.0,M\n")
    df, _ = load_presence_matrix(p)
    assert df["1_S:a"].tolist() == [1, 0, 2] and df["1_S:a"].dtype == np.int8

@pytest.mark.parametrize("cell", ["3", ""])
def test_load_rejects_bad_presence(tmp_path, cell):
    p = tmp_path / "b.csv"; p.write_text(f"lot_id,1_S:a,1_S:a::mach\nL1,{cell},M\nL2,1,M\n")
    with pytest.raises(ValueError, match="1_S:a"):
        load_presence_matrix(p)

def test_load_rejects_unparsed(tmp_path):
    p = tmp_path / "u.csv"; p.write_text("lot_id,weird,1_S:a,1_S:a::mach\nL1,x,1,M\n")
    with pytest.raises(ValueError, match="weird"):
        load_presence_matrix(p)
    df, _ = load_presence_matrix(p, ignore_unparsed=True); assert "weird" not in df

def test_filter_rare_steps_pool_and_constant():
    # Build, as a DataFrame with 1000 lots:
    #   step 1: visited by 150 lots on machines A (120) and B (30)
    #   step 2: visited by 50 lots
    #   step 3: visited by all 1000 lots on machine C only
    df, infos = ...
    kept, table, summary = filter_and_pool(df, infos, min_lots=100)
    t = table.set_index("name")
    assert t.loc[["2_S:b", "2_S:b::mach"], "drop_reason"].eq("rare_step").all()
    assert t.loc[["3_S:c", "3_S:c::mach"], "drop_reason"].eq("constant").all()
    assert set(kept["1_S:a::mach"]) == {"A", "other", "absent"}
    assert list(kept.columns) == ["lot_id", "1_S:a::mach", "1_S:a"]
    assert filter_and_pool(df, infos, min_lots=40)[1].set_index("name").loc["2_S:b", "kept"]

def test_step_with_one_column_kept(caplog):            # Review Focus 3
    # Step 4 has a presence column but no machine column, and is visited by 500 of 1000 lots.
    kept, table, _ = filter_and_pool(df, infos, min_lots=100)
    assert "4_S:d" in kept and "4_S:d" in caplog.text

def test_find_tied_steps():
    parsed, _ = parse_columns(["5_A:x", "5_A:x::mach", "5_B:y", "6_C:z"])
    assert find_tied_steps(parsed) == {5: ["5_A:x", "5_B:y"]}
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the functions above in `utilities/presence.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_utilities.py -q`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add utilities/presence.py tests/test_utilities.py && git commit -m "utilities: parse, load and filter the presence matrix"`

### Task 5: Encoding, identical-indicator merge, mask, edge collapse (`utilities/encoding.py`)

**Files:**
- Create: `utilities/encoding.py`
- Test: `tests/test_utilities.py`

**Interfaces:**
- Consumes: `ABSENT`, the `columns_table` format (Task 4), `MaskedLasso` and `is_acyclic_mask` (Tasks 1–2).
- Produces:
  - **`Encoding`** (dataclass), one entry per *kept representative* indicator, in time order:
    - `names: list[str]` (`f"{column}={category}"`), `column: list[str]`, `category: list[str]`;
    - `step_number: np.ndarray` (int), `step_key: list[str]`, `kind: list[str]`;
    - `levels: dict[str, list[str]]` (per original column, its non-reference categories);
    - `aliases: dict[int, list[str]]` (representative index → merged indicator names);
    - `identical: pd.DataFrame`, with columns `rep_column, rep_category, column, category, complement`.
  - **`fit_encoding(df: pd.DataFrame, columns: pd.DataFrame) -> Encoding`.** `columns` holds the kept rows of
    `columns_table`. Rules (§5.3):
    - reference level: `ABSENT` for machine columns and `"0"` for presence columns; if that level is missing, the
      most common one;
    - categories as `str`;
    - sort order: `(step_number, kind: machine < presence, step_key, column, category)`;
    - identical and complement indicators are merged into the earliest member, using a hash of
      `np.packbits(v if v[0] == 0 else 1 - v)`;
    - a member from the same column as its representative is dropped without an `identical` row.
  - **`indicator_matrix(df, enc) -> np.ndarray`:** n × d float64 0/1, for the representative indicators.
  - **`build_mask(enc, time_order: bool = True, block: int = 2048) -> np.ndarray`:** d × d bool, built in row
    blocks.
    - With `time_order`: the §5.3 rules.
    - Without: everything allowed except the same column and the diagonal.
  - **`collapse_edges(W: np.ndarray, enc: Encoding, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]`** returns
    `(edges, indicator_edges)`.
    - `edges` has columns `source, target, weight, n_indicator_edges, kind, detail`.
      - Learned rows group indicator edges with `|W| >= threshold` by column pair. `weight` is the signed value
        with the largest |W|. `detail` holds up to 5 pairs as `f"{ci}→{cj} ({w:+.2f})"`, followed by `" [≡ " +
        aliases + "]"` when the source or target indicator has aliases.
      - Identical rows come from `enc.identical`, one per column pair, with `weight` set to `-1.0` for a complement
        and `1.0` otherwise, and `detail` as `f"{rep_category}≡{category}"`.
    - `indicator_edges` has columns `source, source_category, target, target_category, weight`.
  - **`check_time_order(edges: pd.DataFrame, columns: pd.DataFrame) -> None`:** applies the §5.3 rule at column level
    and raises `ValueError` listing up to 10 violations.
  - **`estimate_gpu_gb(d: int, solver: str) -> float`:** `12·d²·4/1e9` for `"dagma"` and `8·d²·4/1e9` for `"lasso"`.

- [ ] **Step 1: Write the failing tests.** Small hand-built DataFrames plus a `columns` table built from them with
  `filter_and_pool(..., min_lots=1)`.

```python
def test_reference_levels():
    # machine column {absent, A, B}; presence {0, 1, 2}; a machine column with no absent: {C (60%), D}
    enc = fit_encoding(df, cols)
    assert set(enc.levels["1_S:a::mach"]) == {"A", "B"} and set(enc.levels["1_S:a"]) == {"1", "2"}
    assert enc.levels["2_S:b::mach"] == ["D"]

def test_merge_identical_and_complement():
    # Step 1 has a single machine M and no revisits, so "1_S:a::mach=M" equals "1_S:a=1".
    # Step 3's presence is the complement of step 2's.
    enc = fit_encoding(df, cols)
    assert "1_S:a=1" not in enc.names and "1_S:a::mach=M" in enc.names
    row = enc.identical.set_index("column").loc["1_S:a"]
    assert row.rep_column == "1_S:a::mach" and not row.complement
    assert enc.identical.set_index("column").loc["3_S:c"].complement

def test_merge_keeps_full_weight():
    # Step 1 duplicates (single machine, no revisits); step 2's visit probability depends on visiting step 1.
    # Simulate 5000 lots with P(visit 2) = 0.9 if visited 1 else 0.2.
    enc = fit_encoding(df, cols); X = indicator_matrix(df, enc); X = (X - X.mean(0)) / X.std(0)
    las = MaskedLasso(device="cpu", dtype=torch.float64); las.fit(X.T @ X / len(X), build_mask(enc))
    edges, _ = collapse_edges(las.W_raw, enc, 0.3)
    assert ((edges.source == "1_S:a::mach") & (edges.target == "2_S:b")).any()   # not split below 0.3

def test_build_mask_rules():
    # Steps: 1 (machine + presence), 2 (machine + presence), and a tie "2_T:x" (presence only)
    m = build_mask(enc); i = {n: k for k, n in enumerate(enc.names)}
    assert m[i["1_S:a::mach=A"], i["1_S:a=1"]] and not m[i["1_S:a=1"], i["1_S:a::mach=A"]]
    assert m[i["1_S:a=1"], i["2_S:b=1"]] and not m[i["2_S:b=1"], i["1_S:a=1"]]
    assert m[i["2_S:b=1"], i["2_T:x=1"]] and m[i["2_T:x=1"], i["2_S:b=1"]]
    assert not m[i["1_S:a::mach=A"], i["1_S:a::mach=B"]] and not m.diagonal().any()
    assert not is_acyclic_mask(m) and is_acyclic_mask(build_mask(enc_without_tie))
    assert build_mask(enc, time_order=False)[i["2_S:b=1"], i["1_S:a=1"]]

def test_collapse_edges_and_identical():
    # W: two indicator edges from 1_S:a::mach to 2_S:b (+0.5, -0.7) and one at 0.1; enc has one identical row
    edges, ind = collapse_edges(W, enc, 0.3)
    r = edges[(edges.kind == "learned")].iloc[0]
    assert (r.source, r.target, r.weight, r.n_indicator_edges) == ("1_S:a::mach", "2_S:b", -0.7, 2)
    assert len(ind) == 2 and (edges.kind == "identical").sum() == 1

def test_column_fully_merged_and_partnerless():          # Review Focus 3
    # Step 5 has presence only, always visited together with step 4, so it merges entirely into step 4.
    enc = fit_encoding(df, cols); edges, _ = collapse_edges(np.zeros((len(enc.names),) * 2), enc, 0.3)
    assert not any(n.startswith("5_S:e=") for n in enc.names)
    assert ((edges.target == "5_S:e") & (edges.kind == "identical")).any()

def test_check_time_order():
    good = pd.DataFrame({"source": ["1_S:a::mach"], "target": ["1_S:a"]})
    check_time_order(good, cols)
    with pytest.raises(ValueError, match="2_S:b"):
        check_time_order(pd.DataFrame({"source": ["2_S:b"], "target": ["1_S:a"]}), cols)
    with pytest.raises(ValueError):
        check_time_order(pd.DataFrame({"source": ["1_S:a"], "target": ["1_S:a::mach"]}), cols)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the functions above in `utilities/encoding.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_utilities.py -q`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add utilities/encoding.py tests/test_utilities.py && git commit -m "utilities: one-hot encoding, identical-indicator merge, time-order mask, edge collapse"`

### Task 6: Preprocessing CLI

**Files:**
- Create: `utilities/preprocess.py`
- Test: `tests/test_utilities.py`

**Interfaces:**
- Consumes: `load_presence_matrix`, `filter_and_pool` and `find_tied_steps` (Task 4); `fit_encoding` and
  `estimate_gpu_gb` (Task 5); `simulate_presence` (Task 3), in tests.
- Produces:
  - `main(argv=None) -> int`, with the options `--input`, `--out`, `--min-lots`, `--min-machine-lots`,
    `--id-column` and `--ignore-unparsed`.
  - Files written to `--out`:
    - `data.parquet`: the id column plus kept columns, in time order;
    - `columns.csv`: the `columns_table`;
    - `summary.json`, with the keys `rows, original_columns, kept_columns, steps_total, steps_dropped_rare,
      columns_dropped_constant, machines_pooled, tied_step_numbers, encoded_columns, gpu_gb_dagma, gpu_gb_lasso,
      runtime_note`.
  - The summary is also printed in readable form. `encoded_columns = len(fit_encoding(all kept rows).names)`.
  - `runtime_note` is an indicative runtime per fit for each solver. It scales measured seconds-per-iteration at a
    reference d by `(d / d_ref)³`, using module constants `REF_D` and `REF_SEC_PER_ITER` (one per solver) and the
    iteration counts. Use placeholder values in this task; Task 10 replaces them with the measured ones.

- [ ] **Step 1: Write the failing test.**

```python
def test_preprocess_cli(tmp_path):
    df, _ = simulate_presence(n_lots=800, n_steps=30, rare_steps=3, seed=4)
    df.to_csv(tmp_path / "p.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / "p.csv"), "--out", str(tmp_path / "prep")]) == 0
    s = json.loads((tmp_path / "prep" / "summary.json").read_text())
    cols = pd.read_csv(tmp_path / "prep" / "columns.csv"); data = pd.read_parquet(tmp_path / "prep" / "data.parquet")
    assert s["steps_dropped_rare"] == 3 and (cols.drop_reason == "rare_step").sum() == 6
    kept = cols[cols.kept].assign(k=cols.kind.eq("presence"))
    assert list(data.columns[1:]) == kept.sort_values(["step_number", "step_key", "k"]).name.tolist()
    assert s["encoded_columns"] > 0 and s["gpu_gb_dagma"] > s["gpu_gb_lasso"]
    preprocess_main([... , "--min-lots", "200", "--out", str(tmp_path / "prep2")])
    assert json.loads((tmp_path / "prep2" / "summary.json").read_text())["kept_columns"] <= s["kept_columns"]
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py::test_preprocess_cli -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `utilities/preprocess.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_utilities.py -q`. Expected: PASS.
- [ ] **Step 5: Commit:** `git add utilities/preprocess.py tests/test_utilities.py && git commit -m "utilities: preprocessing CLI"`

### Task 7: Held-out scoring (`utilities/scoring.py`) and dependencies

**Files:**
- Create: `utilities/scoring.py`
- Modify: `pyproject.toml` (the `stx` extra, and `dev` gains plotly and scikit-learn)
- Test: `tests/test_utilities.py`

**Interfaces:**
- Produces:
  - **`heldout_logprob(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], parents: dict[str, list[str]], C: float = 1.0, jobs: int = -1) -> np.ndarray`**
    returns an array of shape `(len(test), len(columns))` holding the log-probability of each test lot's actual
    category.
    - **A column with no parents, or one class in training:** add-one frequencies,
      `(count + 1) / (n_train + r)`.
    - **A column with parents:** `sklearn.linear_model.LogisticRegression(C=C, max_iter=1000)`, fitted on a one-hot
      of the parents' training categories. Use all levels; a parent category unseen in training gets all-zero
      features.
    - **A test category unseen in training:** `log(1 / (n_train + r + 1))`.
    - Run the columns in parallel with `joblib.Parallel(n_jobs=jobs)`.
  - **`compare(lp: np.ndarray, lp_ref: np.ndarray) -> dict`** returns `{"gain": mean over lots of (row sums
    difference), "ci95": 1.96 · sd / √n}`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_frequency_model_exact():
    tr = pd.DataFrame({"a": ["x", "x", "y"]}); te = pd.DataFrame({"a": ["x", "z"]})
    lp = heldout_logprob(tr, te, ["a"], {}, jobs=1)
    np.testing.assert_allclose(lp[:, 0], np.log([3 / 5, 1 / 6]))

def test_parent_improves_heldout():
    # 2000 lots: x uniform over 4 categories, y = x with 5% noise
    lp0 = heldout_logprob(tr, te, ["x", "y"], {}, jobs=1)
    lp1 = heldout_logprob(tr, te, ["x", "y"], {"y": ["x"]}, jobs=1)
    c = compare(lp1, lp0); assert c["gain"] > 0.3 and c["gain"] - c["ci95"] > 0

def test_many_parents_still_gain():          # the anti-goal: a table model would score below the empty graph here
    # y depends on p0; p1..p11 are noise with 5 categories each; 3000 train / 750 test lots
    c = compare(heldout_logprob(tr, te, ["y"], {"y": [f"p{i}" for i in range(12)]}, jobs=1),
                heldout_logprob(tr, te, ["y"], {}, jobs=1))
    assert c["gain"] > 0

def test_unseen_parent_category_is_finite():
    # The test set has a parent category never seen in training
    assert np.isfinite(heldout_logprob(tr, te, ["y"], {"y": ["x"]}, jobs=1)).all()
```

- [ ] **Step 2: Install the dependencies and add the extras.** Edit `pyproject.toml` per Global Constraints, then run
  `pip install "plotly>=5.20" "scikit-learn>=1.4"` in the test environment. Expected:
  `python -c "import sklearn, plotly, joblib"` succeeds.
- [ ] **Step 3: Run** `python -m pytest tests/test_utilities.py -q`. Expected: the new tests FAIL with an
  ImportError.
- [ ] **Step 4: Implement** `utilities/scoring.py`.
- [ ] **Step 5: Run** `python -m pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit:** `git add pyproject.toml utilities/scoring.py tests/test_utilities.py && git commit -m "utilities: held-out logistic scoring; stx extra"`

### Task 8: Fitting and the `run_dagma` CLI

**Files:**
- Create: `utilities/fitting.py` and `utilities/run_dagma.py`
- Test: `tests/test_utilities.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5 and 7; `split_rows` from `causal_discovery.data`.
- Produces:
  - **`FitResult`:** `@dataclass class FitResult: W_raw: np.ndarray; enc: Encoding; info: dict`. The `info` dict
    holds `d, solver, skipped_acyclicity, seconds, h_final, iterations, time_order`.
  - **`learn_graph(df, columns, *, solver="dagma", lambda1=0.03, w_threshold=0.3, time_order=True, device=None, warm_iter=30_000, max_iter=60_000, verbose=True) -> FitResult`.** Steps:
    1. `fit_encoding` → `indicator_matrix`, then standardize;
    2. `cov = X.T @ X / n` (the X here is the standardized matrix);
    3. `build_mask`, then `check_gpu_memory`;
    4. fit with `DagmaLinear(...).fit(cov=..., mask=...)` or `MaskedLasso(...).fit(...)`.

    With `solver == "lasso"` and a cyclic mask, raise `SystemExit` with a message containing `"share a step number"`
    and `"--solver dagma"`.
  - **`check_gpu_memory(d: int, solver: str, device) -> None`:** only when the device is CUDA. It compares
    `estimate_gpu_gb` with the free memory from `torch.cuda.mem_get_info()`. If the estimate doesn't fit, it raises
    `SystemExit` with a message naming d, both numbers and `--min-lots`.
  - **`column_parents(edges: pd.DataFrame) -> dict[str, list[str]]`.**
  - **`within_step_parents(columns: pd.DataFrame, identical: pd.DataFrame) -> dict[str, list[str]]`:** each kept
    presence column's parent is its own step's kept machine column; the identical edges are added on top.
  - **`run_dagma.main(argv=None) -> int`**, with subcommands `benchmark` and `fit` and the options in spec §5.5. It
    also exposes `--warm-iter` and `--max-iter`, passed through to DAGMA.
    - **`benchmark`** writes `results.md`, `results.json`, `per_column.csv`, `edges.csv`, `edges_indicators.csv` and
      `info.json`.
      - The scored graphs are `"empty graph"`, `"within-step only"`, and `f"dagma (threshold {t})"` for each `t`
        in `--thresholds`.
      - The table columns are `graph, learned_edges, identical_edges, heldout_logprob_per_lot, gain_vs_empty,
        gain_vs_empty_ci95, gain_vs_within_step, gain_vs_within_step_ci95`.
      - `results.json` has the keys `graphs` (a list of those rows, as dicts), `n_train`, `n_test`, `options` and
        `fit` (the `FitResult.info`).
      - `per_column.csv` covers the `--w-threshold` graph, with the columns `column, n_parents, gain_vs_empty,
        gain_vs_within_step`.
    - **`fit`** writes `edges.csv`, `edges_indicators.csv`, `adjacency.npy` (column level, in the kept-column order
      of `columns.csv`) and `info.json` (which includes that column order).
    - Both call `check_time_order` before writing edges when the time order is on.

- [ ] **Step 1: Write the failing tests.** The shared fixture `prepared(tmp_path)` runs `simulate_presence(n_lots=1500, n_steps=25, n_effects=6, seed=5)` and then `preprocess_main`.

```python
def test_benchmark_and_fit_e2e(prepared, tmp_path):
    assert run_main(["benchmark", "--prepared", str(prepared), "--out", str(tmp_path / "b"),
                     "--solver", "lasso", "--device", "cpu", "--jobs", "1"]) == 0
    res = json.loads((tmp_path / "b" / "results.json").read_text()); g = {r["graph"]: r for r in res["graphs"]}
    assert g["within-step only"]["gain_vs_empty"] > 0
    assert g["dagma (threshold 0.3)"]["gain_vs_within_step"] > 0
    assert run_main(["fit", "--prepared", str(prepared), "--out", str(tmp_path / "f"),
                     "--solver", "lasso", "--device", "cpu"]) == 0
    edges = pd.read_csv(tmp_path / "f" / "edges.csv"); check_time_order(edges, cols)
    step = lambda c: int(c.split("_", 1)[0])
    found = {(step(s), step(t)) for s, t in zip(edges.source, edges.target)}
    recall = np.mean([(step(s), step(t)) in found for s, t in truth.itertuples(index=False)])
    assert recall >= 0.8

def test_fit_dagma_solver_runs(prepared, tmp_path):
    assert run_main(["fit", "--prepared", str(prepared), "--out", str(tmp_path / "d"), "--device", "cpu",
                     "--warm-iter", "2000", "--max-iter", "4000"]) == 0
    assert json.loads((tmp_path / "d" / "info.json").read_text())["skipped_acyclicity"] is True

def test_lasso_with_tied_steps_exits(tmp_path):
    # A prepared folder whose CSV has two step keys with the same number
    with pytest.raises(SystemExit, match="share a step number"):
        run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "x"), "--solver", "lasso", "--device", "cpu"])

def test_gpu_memory_check(monkeypatch):                 # Review Focus 5
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 * 1024**3, 24 * 1024**3))
    with pytest.raises(SystemExit, match="--min-lots"):
        check_gpu_memory(20_000, "dagma", "cuda")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `utilities/fitting.py` and `utilities/run_dagma.py`.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: all pass, with the new tests under about a minute together.
- [ ] **Step 5: Commit:** `git add utilities/fitting.py utilities/run_dagma.py tests/test_utilities.py && git commit -m "utilities: run_dagma benchmark and fit"`

### Task 9: Interactive viewer (`utilities/plot_edges.py`)

**Files:**
- Create: `utilities/plot_edges.py`
- Test: `tests/test_utilities.py`

**Interfaces:**
- Consumes: `parse_columns` (Task 4); the `edges.csv` and `columns.csv` formats (Tasks 6 and 8).
- Produces:
  - **`load_nodes(edges: pd.DataFrame, columns_csv: str | None) -> pd.DataFrame`** returns one row per node that has
    at least one edge, with the columns `name, step_number, stage, step, kind, lots_visited, in_degree, out_degree`.
    For a name that doesn't parse, `step_number` is NaN and `kind` is `"other"`.
  - **`build_figure(edges: pd.DataFrame, nodes: pd.DataFrame) -> plotly.graph_objects.Figure`**, laid out per §5.6.
    - Lanes: `x = step_number` and `y = 1` for the machine lane, `0` for the presence lane.
    - Steps that share a number are spread slightly along x.
    - Unparseable nodes go in a third lane at `y = 0.5`, with `x` equal to their rank of first appearance.
    - Traces, all `Scattergl`: nodes; learned-edge arcs (polyline points separated by `None`); identical-edge arcs
      (`visible="legendonly"`); edge-midpoint hover markers; an empty highlight trace.
    - Stage bands are layout shapes.
  - **`write_viewer(fig, nodes, edges, path, cdn: bool = False) -> None`.** It calls
    `fig.write_html(path, include_plotlyjs="cdn" if cdn else True, post_script=JS)`.
    - `JS` embeds a JSON payload (node names, x, y, and parent and child index lists). It defines
      `stxHighlight(i)`, which restyles node opacity and fills the highlight trace with the incident arcs; clicking a
      node calls it, and clicking empty space clears it.
    - It also injects an `<input id="stx-search">` that highlights and zooms to the first node whose name contains
      the text, ignoring case.
  - **`main(argv=None) -> int`**, with the options `--edges`, `--columns`, `--out`, `--min-weight`, `--max-edges`
    and `--cdn`. Above `--max-edges`, it keeps the edges with the largest `|weight|`, or the first ones when there
    is no weight column, and prints a warning.

- [ ] **Step 1: Write the failing tests.**

```python
def test_viewer_pipeline_edges(tmp_path, edges_and_columns):   # learned + identical edges, plus columns.csv
    out = tmp_path / "g.html"
    assert plot_main(["--edges", e_path, "--columns", c_path, "--out", str(out)]) == 0
    html = out.read_text()
    assert "stxHighlight" in html and 'id="stx-search"' in html and "1_S:a::mach" in html

def test_viewer_package_edges(tmp_path):                       # Review Focus 4
    pd.DataFrame({"source": ["X1", "X2"], "target": ["X2", "X3"]}).to_csv(tmp_path / "e.csv", index=False)
    edges = pd.read_csv(tmp_path / "e.csv"); nodes = load_nodes(edges, None)
    assert set(nodes.name) == {"X1", "X2", "X3"}
    assert len(build_figure(edges, nodes).data[0].x) == 3

def test_viewer_max_edges_and_isolated(tmp_path, capsys):
    # 100 weighted edges; columns.csv also lists nodes that have no edges
    plot_main(["--edges", e100_path, "--columns", c_path, "--max-edges", "10", "--out", str(tmp_path / "m.html")])
    assert "10" in capsys.readouterr().out    # the warning names the number of edges kept
    nodes = load_nodes(top10, c_path); assert nodes.in_degree.add(nodes.out_degree).gt(0).all()

def test_viewer_js_parses(tmp_path):
    html = ...   # generate as in test_viewer_pipeline_edges, then read the file
    js = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)[-1]   # the injected post_script
    if shutil.which("node") is None: pytest.skip("node not installed")
    (tmp_path / "s.js").write_text(js); subprocess.run(["node", "--check", str(tmp_path / "s.js")], check=True)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_utilities.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `utilities/plot_edges.py`.
- [ ] **Step 4: Run** `python -m pytest tests/test_utilities.py -q`. Expected: PASS.
- [ ] **Step 5: Open the HTML generated from Task 8's e2e output in a browser** (or a headless one, if available).
  Check that clicking a node highlights its parents and children, and that search jumps to a node. Record what was
  checked in the commit message.
- [ ] **Step 6: Commit:** `git add utilities/plot_edges.py tests/test_utilities.py && git commit -m "utilities: interactive timeline viewer for edges.csv"`

### Task 10: Full-size run on Slurm, and documentation

**Files:**
- Create:
  - `benchmarks/stx_scale.sh`: an sbatch script. It activates the env, simulates 9,000 lots × 5,000 steps,
    preprocesses, runs benchmark and fit with `--solver lasso`, runs `plot_edges`, and logs `/usr/bin/time -v`
    plus `nvidia-smi` peak memory.
  - `benchmarks/stx_timing.py`: times N iterations of DAGMA's skip path, DAGMA's full path and MaskedLasso at a
    given d. It also runs the dagma-vs-lasso agreement at a mid size (about 9,000 lots × 1,500 steps), reporting
    the share of edges that agree and the |W| of the edges that disagree.
  - `utilities/README.md`: how to install (`pip install -e ".[stx]"`) and run, in order, with every option and
    output, and how to read `results.md`. It includes the **measured** runtimes and memory, the agreement numbers,
    the d³ scaling rule for estimating your own d, the near-duplicate caveat (§5.3), and what to do when memory runs
    out (`--min-lots 200`).
- Modify: `README.md`, adding a short "Step-presence data (utilities/)" pointer section after section 8.

**Interfaces:**
- Consumes: every CLI from Tasks 3, 6, 8 and 9.

- [ ] **Step 1: Write the two scripts.**
- [ ] **Step 2: Submit to a free GPU partition.** Pick one from `sinfo` that is idle or mixed with a free GPU. Use
  `--gres=gpu:1 -c 16 --mem=128G -t 24:00:00`. Prefer a GPU with ≥ 48 GB (birck-gpu L40S, or the RTX PRO 6000 nodes);
  record which one ran. Expected: the job reaches the `R` state in `squeue -u $USER`.
- [ ] **Step 3: Wait for completion.** Expected: exit code 0, `results.md` shows positive gains, and
  `check_time_order` passed. Record d, time and peak GPU memory per step, the per-iteration times from
  `stx_timing.py` (extrapolated to about 88,000 iterations for the dagma solver, marked as an extrapolation), the
  agreement, and the HTML size.
- [ ] **Step 4: Write `utilities/README.md` and the `README.md` pointer,** using the recorded numbers. Replace the
  placeholder `REF_D` and `REF_SEC_PER_ITER` constants from Task 6 with the measured values.
- [ ] **Step 5: Run** `python -m pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit:** `git add benchmarks/stx_scale.sh benchmarks/stx_timing.py utilities/README.md README.md && git commit -m "utilities: full-size Slurm run, measured runtimes, README"`
