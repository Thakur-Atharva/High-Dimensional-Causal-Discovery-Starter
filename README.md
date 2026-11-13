# Causal discovery starter

Learn a causal graph from a large table (thousands of columns, ~100,000 rows) with three methods, and compare them
on held-out rows.

| Method | What it is | Needs |
|---|---|---|
| `dagma` | **DAGMA** (Bello et al., NeurIPS 2022), the way it is usually run: linear model, library defaults | Python; a GPU strongly recommended above ~500 columns |
| `dagma_hc` | **DAGMA + BIC hill climbing**: DAGMA (plus partial correlations) proposes likely neighbours for each column, then a greedy search scored by the BIC decides which edges to keep and their direction | Python (+ GPU for the DAGMA part) |
| `fges` | **fGES** (Fast Greedy Equivalence Search) from CMU's Tetrad, the most widely used score-based method that scales to thousands of variables | Java 21+ and the `py-tetrad` package |

<!-- RESULTS -->

---

## 1. Causal discovery in five minutes

**The output is a graph.**
- Each column of your table is a variable, and the output is a directed graph over them.
- An arrow `A -> B` means "A directly influences B, with the other variables held fixed".
- The graph is a **DAG** (directed acyclic graph): following arrows never brings you back to where you started.

**The input is just your rows.** There are no experiments or interventions. The methods use the pattern of
(conditional) dependences in the data. If A and C are dependent but become independent once you know B, that
suggests a chain or fork through B. If A and C are independent but both depend on B, that suggests `A -> B <- C`.

**What these methods assume.** Read these before trusting any arrow:
1. **No hidden common causes.** If something unmeasured drives two columns, the methods may draw an edge between
   them that isn't causal.
2. **No feedback loops,** because the output is acyclic.
3. **Rows are independent draws from one process.** Time series break this (see section 5).
4. **Relationships are roughly linear** for continuous data: all three methods here use linear models. Strong
   nonlinearity can hide edges or create spurious ones.

**Directions are only partly identifiable.**
- From observational data alone, some arrows cannot be oriented. `A -> B` and `B -> A` explain the data equally
  well unless a third variable breaks the tie, such as `A -> B <- C`.
- fGES reports which edges it could orient; the other two methods pick a direction anyway.
- Treat directions as the weakest part of the result.

**The output is a set of hypotheses, not facts.** Use it to prioritize: which relationships to check with domain
experts, and which experiments to run.

## 2. Install

You need Python 3.12+, and for `fges`, Java 21+.

```bash
git clone <this repo> && cd causal-discovery-starter
python3.12 -m venv .venv && source .venv/bin/activate
```

**1. PyTorch.** Install a build that matches your GPU driver. Run `nvidia-smi`; the driver version is in the top
line. The CUDA 12.8 build works with drivers 525 and newer, including the 550 series:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # no GPU
```

**2. This package**, with fGES support:

```bash
pip install -e ".[fges,dev]"
```

**3. Java,** for `fges` only. Check with `java -version` (21 or newer). To install it:
- Ubuntu: `sudo apt install openjdk-21-jdk`
- macOS: `brew install openjdk@21`
- No admin rights: `conda install -c conda-forge openjdk=21`

**4. Check everything works,** in about a minute on a CPU:

```bash
python -m pytest -q
```

## 3. Quick start on simulated data, with a known answer

```bash
# 100 variables, 20,000 rows, a random true graph (written to data/sim_truth.csv)
python -m causal_discovery simulate --nodes 100 --samples 20000 --out data/sim.csv

# learn graphs on 80% of the rows, score them on the other 20%, and compare with the true graph
python -m causal_discovery benchmark --data data/sim.csv --truth data/sim_truth.csv --out results/sim
```

The table is printed and saved in `results/sim/results.md`. Each method's graph is saved as
`results/sim/graph_<method>.csv`, one `source,target` row per edge. Section 4 explains the columns.

## 4. Your data

**Format.**
- One row per sample and one column per variable, in `.csv`, `.parquet` (recommended at this size) or `.feather`.
- No missing values: impute or drop them first.
- Drop identifier and timestamp columns with `--drop-columns id timestamp`.
- Constant columns are rejected.

**Benchmark the three methods:**

```bash
# rows are independent samples: hold out a random 20%
python -m causal_discovery benchmark --data mydata.parquet --split random --out results/bench

# rows are in time order: train on the first 80%, test on the last 20%
python -m causal_discovery benchmark --data mydata.parquet --split last --out results/bench_time
```

**Learn the final graph** with your chosen method, on all rows:

```bash
python -m causal_discovery run --data mydata.parquet --method fges --out results/final
```

This writes `edges.csv`, `adjacency.npy` (`A[i, j] = 1` means column i -> column j) and `info.json`.

**Reading the benchmark table.**

| Column | Meaning |
|---|---|
| `edges` | Number of arrows |
| `heldout_loglik_per_row` | Each method fits a model to the training rows: every variable predicted from its parents (linear regression; a probability table for categorical data). This column is how probable the *test* rows are under that model, averaged over rows. Higher is better |
| `gain_vs_empty_per_row` | Improvement over the "empty graph" where every variable is independent, with a 95% interval (`gain_vs_empty_95ci`) |
| Method-vs-method table | Row-by-row difference between two methods, with a 95% interval. If the interval includes 0, the two fit equally well. `skeleton_jaccard` is the share of their connections, ignoring direction, that the two methods agree on |
| `skeleton_f1`, `directed_f1`, `shd` | Only with `--truth` (simulated data): accuracy of the connections, of the arrows, and the number of wrong edges |

**How to use it:**
- A graph that misses real dependences predicts worse, so a low score means missing structure.
- Extra edges barely hurt prediction with 80,000 training rows. When two methods are within each other's interval,
  prefer the sparser graph.
- Held-out likelihood cannot judge arrow directions: graphs that differ only in direction can score exactly the same.
- Agreement between methods (`skeleton_jaccard`) is a useful stability check. Edges found by all three are the most
  trustworthy.

## 5. Settings that matter

| Option | Default | When to change it |
|---|---|---|
| `--penalty` | 1 (standard BIC) | Applies to `dagma_hc` and `fges`. **With 100,000 rows, real data often give very dense graphs**, because BIC then accepts even tiny dependences. Raise it (2 to 10) for sparser graphs |
| `--min-partial-corr` | off | The same idea in interpretable units: only keep edges whose partial correlation is at least this, e.g. `0.03` or `0.05`. Overrides `--penalty` |
| `--k` | 20 | `dagma_hc`: candidate neighbours per variable from each source. More finds more true edges but runs slower |
| `--tabu` | 0 | `dagma_hc`: set to 10 for tabu search, which may take a few downhill moves to escape a local optimum. In simulations it found higher-BIC, more accurate graphs at no extra cost (skeleton F1 +0.03 to +0.09) |
| `--candidates` | `dagma+pcorr` | `dagma_hc`: `dagma` uses only DAGMA's proposals; the default adds each variable's largest partial correlations |
| `--restarts` | 5 | `dagma_hc`: number of hill-climbing runs with different tie-breaking; the best-scoring graph is kept |
| `--lambda1`, `--w-threshold` | 0.03, 0.3 | DAGMA's own sparsity weight and edge cut-off (the library defaults) |
| `--threads`, `--max-degree`, `--java-heap` | all cores, none, 16g | `fges` runtime and memory. `--max-degree 20` bounds runtime on dense data |
| `--device` | `cuda` if available | `dagma`: `cpu` works but is slow above ~500 columns |
| `--data-type` | `auto` | `continuous` or `categorical`. `auto` calls a table categorical if every column has at most 20 distinct integer values |

**Categorical data.**
- `dagma_hc` uses a categorical BIC (one probability table per parent configuration).
- `dagma` treats the category codes as numbers, which is crude.
- `fges` here is wired for continuous data only. Tetrad has discrete scores (BDeu) if you need them.

**Mixed data** is treated as continuous, with codes as numbers. A common alternative is to one-hot encode
categorical columns, dropping one level each.

**Time series.**
- `--split last` makes the benchmark honest: train on the past, test on the future.
- The methods still treat rows as independent, so they look only for effects within a single time step.
- For lagged effects, look at time-series methods such as PCMCI (tigramite) or DYNOTEARS.

## 6. How the methods work

**DAGMA (`dagma`).**
- **The model:** each variable is a weighted sum of the others plus noise. The weight matrix `W` encodes the graph:
  `W[i, j] != 0` means `i -> j`.
- **The fit:** it chooses `W` to minimize prediction error plus a sparsity penalty. A smooth function of `W` that is
  zero only when there are no cycles forces the result to be a DAG; this is what makes plain gradient descent
  possible.
- **The output:** edges with `|W| >= 0.3`.
- **Cost:** each step inverts a 2,500 x 2,500 matrix, hence the GPU.
- **Weaknesses:** the 0.3 cut-off is arbitrary, and directions can depend on how variables are scaled. We
  standardize every column first, so units don't matter.
- **Implementation:** `causal_discovery/dagma_torch.py` is a line-by-line PyTorch port of the official
  implementation, checked to give the same result (see NOTICE).

**DAGMA + BIC hill climbing (`dagma_hc`).**
- **The BIC (Bayesian Information Criterion)** scores a whole graph: how well it predicts the data,
  `-(number of parameters / 2) * log(rows)`. Higher is better. It rewards fit but charges for every extra edge.
- **Hill climbing** starts from an empty graph and repeatedly makes the single change that most improves the BIC:
  add, delete or reverse one arrow. It stops when no change helps.
- **Why DAGMA is involved:** trying every pair of 2,500 variables at each step is too slow. So DAGMA's weights
  propose the `k` most likely neighbours of each variable, and hill climbing only considers those pairs, in both
  directions.
- **Why partial correlations are added:** in our simulations, DAGMA's lists alone missed 3–13% of the true
  neighbours, and hill climbing can never add an edge that isn't on the list. A variable's *partial correlation*
  with another (their correlation after removing the linear effect of all other variables) is non-zero for every
  true neighbour in linear data. So each variable's `k` largest partial correlations are added to the list.
  Computing them takes one inversion of the covariance matrix. With them, the lists contained every true neighbour,
  and the hill-climbing result improved substantially (skeleton F1 0.52–0.80 → 0.69–0.91 on 50–400 variables). Use
  `--candidates dagma` for DAGMA's lists alone.
- **Why restarts:** hill climbing is greedy, and early ties (`A -> B` and `B -> A` often score identically) can
  steer it into a worse graph. It is run `--restarts` times with different tie-breaking, and the highest-BIC graph
  is kept.
- **Code:** `causal_discovery/hill_climb.py` and `scores.py`.

**fGES (`fges`).**
- **What it searches:** a greedy search like hill climbing, but over *equivalence classes*. Each class is a set of
  graphs that explain the data equally well. This removes the arbitrary direction choices.
- **How it searches:** a forward phase adds edges while the BIC improves; a backward phase removes them while it
  improves.
- **Guarantee and scale:** it is consistent, finding the right class with enough data under the assumptions in
  section 1. It has been run on 30,000 variables in minutes and a million in hours (Ramsey et al. 2017).
- **Output:** the class (a "CPDAG"). `info.json` reports how many edges it left undirected. The code picks one DAG
  from the class for scoring; all DAGs in a class score the same.
- **Code:** `causal_discovery/fges.py`, which calls Tetrad through py-tetrad.

## 7. Using it from Python

```python
from causal_discovery.data import load_table, Preprocessor
from causal_discovery import methods

df = load_table("mydata.parquet")
prep = Preprocessor().fit(df).transform(df)     # standardizes continuous columns
A, info = methods.fges(prep, penalty=2.0)       # or methods.dagma(prep), methods.dagma_hc(prep, k=20)
edges = [(prep.names[i], prep.names[j]) for i, j in zip(*A.nonzero())]
```

## 8. Repository layout

```
causal_discovery/
  data.py          loading, data-type detection, preprocessing, train/test splits
  dagma_torch.py   DAGMA (linear) in PyTorch
  scores.py        BIC for continuous (Gaussian) and categorical data
  hill_climb.py    hill climbing over candidate pairs (with optional tabu search)
  fges.py          fGES via Tetrad
  methods.py       the three methods behind one interface
  evaluate.py      held-out log-likelihood, accuracy against a known graph
  benchmark.py     the train/test benchmark
  simulate.py      synthetic data with a known graph
  __main__.py      command line (simulate / run / benchmark)
tests/             quick checks (pytest)
```

## 9. References

- Bello, Aragam, Ravikumar. *DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity
  characterization.* NeurIPS 2022. https://arxiv.org/abs/2209.08037
- Ramsey, Glymour, Sanchez-Romero, Glymour. *A million variables and more: the Fast Greedy Equivalence Search
  algorithm.* Int. J. Data Science and Analytics, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5380925
- Chickering. *Optimal structure identification with greedy search.* JMLR 2002 (GES).
- Tsamardinos, Brown, Aliferis. *The max-min hill-climbing Bayesian network structure learning algorithm.* Machine
  Learning 2006 (restrict-then-search, as in `dagma_hc`).
- Reisach, Seiler, Weichwald. *Beware of the simulated DAG!* NeurIPS 2021 (why standardizing matters).
- Tetrad: https://github.com/cmu-phil/tetrad. py-tetrad: https://github.com/cmu-phil/py-tetrad

## License

Apache-2.0 (see LICENSE). `dagma_torch.py` is adapted from DAGMA (Apache-2.0). Tetrad, used by `fges`, is GPL-3.0
and is downloaded by pip, not included here (see NOTICE).
