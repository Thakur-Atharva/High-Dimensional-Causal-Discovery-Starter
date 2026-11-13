"""Fast checks (CPU, under a minute): python -m pytest -q"""
import numpy as np
import pytest
import torch

from causal_discovery import methods
from causal_discovery.benchmark import is_dag, run_benchmark
from causal_discovery.dagma_torch import DagmaLinear
from causal_discovery.data import Preprocessor, split_rows
from causal_discovery.evaluate import heldout_loglik, structure_metrics
from causal_discovery.hill_climb import hill_climb
from causal_discovery.simulate import make_dataset


def prep_of(df):
    return Preprocessor().fit(df).transform(df)


def test_splits():
    tr, te = split_rows(100, "last", 0.2)
    assert list(te) == list(range(80, 100)) and len(tr) == 80
    tr, te = split_rows(100, "random", 0.2, seed=1)
    assert len(te) == 20 and not set(tr) & set(te)


def test_dagma_returns_dag_with_signal():
    df, A = make_dataset(10, 3000, seed=1)
    W = DagmaLinear(device="cpu", dtype=torch.float64).fit(X=prep_of(df).X, max_iter=20_000, warm_iter=10_000)
    assert is_dag(W != 0)
    assert structure_metrics(W != 0, A)["skeleton_recall"] > 0.3


def test_hill_climbing_recovers_chain():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(5000)
    x1 = x0 + rng.standard_normal(5000)
    x2 = x1 + rng.standard_normal(5000)
    import pandas as pd
    p = prep_of(pd.DataFrame({"a": x0, "b": x1, "c": x2}))
    H = hill_climb(methods.bic_score(p), 3)
    assert {tuple(sorted(e)) for e in zip(*np.nonzero(H))} == {(0, 1), (1, 2)}


def test_categorical_pipeline():
    df, A = make_dataset(8, 5000, "categorical", seed=2)
    p = prep_of(df)
    assert p.data_type == "categorical"
    H, info = methods.dagma_hc(p, k=7, restarts=2, verbose=False, max_iter=5000, warm_iter=2000, device="cpu")
    assert is_dag(H) and structure_metrics(H, A)["skeleton_f1"] > 0.5


def test_heldout_prefers_true_graph():
    df, A = make_dataset(10, 4000, seed=3)
    tr, te = split_rows(len(df), "random", 0.25)
    pre = Preprocessor().fit(df.iloc[tr])
    train, test = pre.transform(df.iloc[tr]), pre.transform(df.iloc[te])
    assert heldout_loglik(A, train, test).mean() > heldout_loglik(np.zeros_like(A), train, test).mean()


def test_benchmark_runs(tmp_path):
    df, A = make_dataset(8, 2000, seed=4)
    res = run_benchmark(df, ["dagma_hc"], "random", truth=A, out_dir=tmp_path,
                        options={"dagma_fit": {"device": "cpu", "max_iter": 5000, "warm_iter": 2000},
                                 "dagma_hc": {"k": 7, "restarts": 2}})
    assert (tmp_path / "results.md").exists() and res["methods"][1]["acyclic"]


def test_fges_if_available():
    pytest.importorskip("pytetrad")
    import shutil
    if shutil.which("java") is None:
        pytest.skip("java not installed")
    from causal_discovery.fges import fges
    df, A = make_dataset(15, 5000, seed=5)
    H, info = fges(prep_of(df), threads=2, max_heap="2g")
    assert is_dag(H) and structure_metrics(H, A)["skeleton_f1"] > 0.6
