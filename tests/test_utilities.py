"""Checks for utilities/ (the step-presence pipeline): CPU, about a minute."""
import json
import logging

import numpy as np
import pandas as pd
import pytest
import torch

from causal_discovery.dagma_torch import is_acyclic_mask
from causal_discovery.masked_lasso import MaskedLasso
from utilities.encoding import (build_mask, check_time_order, collapse_edges, estimate_gpu_gb, fit_encoding,
                                indicator_matrix)
from utilities.presence import ColumnInfo, filter_and_pool, find_tied_steps, load_presence_matrix, parse_columns
from utilities.simulate_presence import simulate_presence


def step_of(name: str) -> int:
    return int(name.split("_", 1)[0])


def test_simulate_presence_format():
    df, truth = simulate_presence(n_lots=500, n_steps=40, seed=1)
    assert df.columns[0] == "lot_id" and df["lot_id"].is_unique
    mach = [c for c in df.columns[1:] if c.endswith("::mach")]
    pres = [c for c in df.columns[1:] if c not in mach]
    assert len(pres) == len(mach) == 40
    assert set(np.unique(df[pres].to_numpy())) <= {0, 1, 2}
    assert all(((df[p] == 0) == (df[p + "::mach"] == "")).all() for p in pres)
    assert {"NA", "None"} <= set(df[mach].to_numpy().ravel())
    assert len(truth) > 0 and all(step_of(s) < step_of(t) for s, t in truth.itertuples(index=False))


def test_simulate_machine_distribution():
    df, _ = simulate_presence(n_lots=3000, n_steps=500, rare_steps=0, seed=2)
    k = np.array([df[c].replace("", np.nan).nunique() for c in df.columns if c.endswith("::mach")])
    assert k.min() >= 1 and k.max() <= 11 and 3 <= k.mean() <= 5


# ----- Task 4: parse, load, filter --------------------------------------------------------------------------------

def test_parse_columns():
    parsed, unparsed = parse_columns(["lot_id", "1_ST_A:step_x", "1_ST_A:step_x::mach", "12_B:c:d", "bad", "3_Q"])
    by = {p.name: p for p in parsed}
    assert by["1_ST_A:step_x::mach"] == ColumnInfo("1_ST_A:step_x::mach", "1_ST_A:step_x", 1, "ST_A", "step_x",
                                                   "machine")
    assert (by["12_B:c:d"].stage, by["12_B:c:d"].step, by["3_Q"].step) == ("B", "c:d", "")
    assert by["1_ST_A:step_x"].kind == "presence"
    assert unparsed == ["bad"]


def test_load_machine_names_verbatim(tmp_path):        # Review Focus 1
    p = tmp_path / "m.csv"
    p.write_text("\ufefflot_id,1_S:a,1_S:a::mach\nL1,1,NA\nL2,1,None\nL3,1,007\nL4,0, \nL5,0,\n", encoding="utf-8")
    df, _ = load_presence_matrix(p)
    assert df["1_S:a::mach"].tolist() == ["NA", "None", "007", "absent", "absent"] and "lot_id" in df


def test_load_float_presence(tmp_path):                # Review Focus 2
    p = tmp_path / "f.csv"
    p.write_text("lot_id,1_S:a,1_S:a::mach\nL1,1.0,M\nL2,0.0,\nL3,2.0,M\n")
    df, _ = load_presence_matrix(p)
    assert df["1_S:a"].tolist() == [1, 0, 2] and df["1_S:a"].dtype == np.int8


@pytest.mark.parametrize("cell", ["3", ""])
def test_load_rejects_bad_presence(tmp_path, cell):
    p = tmp_path / "b.csv"
    p.write_text(f"lot_id,1_S:a,1_S:a::mach\nL1,{cell},M\nL2,1,M\n")
    with pytest.raises(ValueError, match="1_S:a"):
        load_presence_matrix(p)


def test_load_rejects_unparsed(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("lot_id,weird,1_S:a,1_S:a::mach\nL1,x,1,M\n")
    with pytest.raises(ValueError, match="weird"):
        load_presence_matrix(p)
    df, _ = load_presence_matrix(p, ignore_unparsed=True)
    assert "weird" not in df


def _toy_steps(n=1000):
    """step 1: 150 lots on A (120) / B (30); step 2: 50 lots; step 3: every lot on C; step 4: presence only, 500 lots"""
    lots = np.arange(n)
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in lots]})
    v1 = lots < 150
    df["1_S:a"] = v1.astype(np.int8)
    df["1_S:a::mach"] = np.where(lots < 120, "A", np.where(v1, "B", "absent"))
    v2 = lots >= n - 50
    df["2_S:b"] = v2.astype(np.int8)
    df["2_S:b::mach"] = np.where(v2, "D", "absent")
    df["3_S:c"] = np.ones(n, np.int8)
    df["3_S:c::mach"] = "C"
    df["4_S:d"] = (lots % 2).astype(np.int8)
    infos, _ = parse_columns(list(df.columns))
    return df, infos


def test_filter_rare_steps_pool_and_constant():
    df, infos = _toy_steps()
    kept, table, summary = filter_and_pool(df, infos, min_lots=100)
    t = table.set_index("name")
    assert t.loc[["2_S:b", "2_S:b::mach"], "drop_reason"].eq("rare_step").all()
    assert t.loc[["3_S:c", "3_S:c::mach"], "drop_reason"].eq("constant").all()
    assert set(kept["1_S:a::mach"]) == {"A", "other", "absent"}
    assert list(kept.columns) == ["lot_id", "1_S:a::mach", "1_S:a", "4_S:d"]
    assert t.loc["1_S:a", "lots_visited"] == 150 and t.loc["1_S:a::mach", "n_categories"] == 3
    assert filter_and_pool(df, infos, min_lots=40)[1].set_index("name").loc["2_S:b", "kept"]


def test_step_with_one_column_kept(caplog):            # Review Focus 3
    df, infos = _toy_steps()
    with caplog.at_level(logging.WARNING):
        kept, table, _ = filter_and_pool(df, infos, min_lots=100)
    assert "4_S:d" in kept and "4_S:d" in caplog.text


def test_find_tied_steps():
    parsed, _ = parse_columns(["5_A:x", "5_A:x::mach", "5_B:y", "6_C:z"])
    assert find_tied_steps(parsed) == {5: ["5_A:x", "5_B:y"]}


# ----- Task 5: encoding, merge, mask, collapse -------------------------------------------------------------------

def _table(df):
    """Kept-columns table for a hand-built frame (no step filtering)."""
    infos, _ = parse_columns(list(df.columns))
    _, table, _ = filter_and_pool(df, infos, min_lots=1, min_machine_lots=1)
    return table[table.kept].reset_index(drop=True)


def _step(df, num, key, visited, machines=None, rng=None, revisit=None):
    """Add a step: presence (0/1, or 2 where ``revisit``) and, if ``machines`` is given, a machine column."""
    name = f"{num}_{key}"
    pres = visited.astype(np.int8)
    if revisit is not None:
        pres = pres + (visited & revisit).astype(np.int8)
    if machines is not None:
        pick = rng.choice(machines, len(visited)) if len(machines) > 1 else np.full(len(visited), machines[0])
        df[name + "::mach"] = np.where(visited, pick, "absent")
    df[name] = pres


def _mask_frame(with_tie=True, n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in range(n)]})
    _step(df, 1, "S:a", rng.random(n) < 0.6, ["A", "B"], rng, revisit=rng.random(n) < 0.3)
    _step(df, 2, "S:b", rng.random(n) < 0.5, ["C", "D"], rng)
    if with_tie:
        _step(df, 2, "T:x", rng.random(n) < 0.5)
    return df


def test_reference_levels():
    lots = np.arange(100)
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in lots]})
    df["1_S:a::mach"] = np.where(lots < 20, "absent", np.where(lots < 60, "A", "B"))
    df["1_S:a"] = np.where(lots < 20, 0, np.where(lots < 90, 1, 2)).astype(np.int8)
    df["2_S:b::mach"] = np.where((lots >= 50) & (lots < 90), "D", "C")
    enc = fit_encoding(df, _table(df))
    assert enc.levels["1_S:a::mach"] == ["absent", "A", "B"] and set(enc.levels["1_S:a"]) == {"1", "2"}
    assert enc.levels["2_S:b::mach"] == ["C", "D"]


def test_merge_identical_and_complement():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in range(n)]})
    _step(df, 1, "S:a", rng.random(n) < 0.5, ["M"], rng)              # machine M == visited
    v2 = rng.random(n) < 0.5
    _step(df, 2, "S:b", v2, ["X", "Y"], rng)
    _step(df, 3, "S:c", ~v2, ["Z1", "Z2"], rng)                        # visited exactly when step 2 is not
    enc = fit_encoding(df, _table(df))
    assert "1_S:a=1" not in enc.names and "1_S:a::mach=M" not in enc.names and "1_S:a::mach=absent" in enc.names
    ident = enc.identical.set_index("column")
    assert ident.loc["1_S:a"].rep_column == "1_S:a::mach" and ident.loc["1_S:a"].complement
    assert ident.loc["1_S:a"].rep_category == "absent"
    # step 3 is visited exactly when step 2 is not: its "absent" is step 2's "not absent"
    assert ident.loc["3_S:c::mach"].rep_column == "2_S:b::mach" and ident.loc["3_S:c::mach"].complement


def test_merge_keeps_full_weight():
    rng = np.random.default_rng(2)
    n = 5000
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in range(n)]})
    v1 = rng.random(n) < 0.5
    _step(df, 1, "S:a", v1, ["M"], rng)                                # "ran on M" == "visited step 1"
    _step(df, 2, "S:b", rng.random(n) < np.where(v1, 0.75, 0.25))      # correlation ~0.5 with step 1
    enc = fit_encoding(df, _table(df))
    X = indicator_matrix(df, enc)
    X = (X - X.mean(0)) / X.std(0)
    las = MaskedLasso(device="cpu", dtype=torch.float64)
    las.fit(X.T @ X / len(X), build_mask(enc))
    edges, _ = collapse_edges(las.W_raw, enc, 0.3)
    hit = edges[(edges.source == "1_S:a::mach") & (edges.target == "2_S:b") & (edges.kind == "learned")]
    assert len(hit) == 1 and abs(hit.weight.iloc[0]) > 0.4             # not split into two halves below 0.3


def test_build_mask_rules():
    enc = fit_encoding(df := _mask_frame(), _table(df))
    m = build_mask(enc)
    i = {name: k for k, name in enumerate(enc.names)}
    assert m[i["1_S:a::mach=A"], i["1_S:a=1"]] and not m[i["1_S:a=1"], i["1_S:a::mach=A"]]
    assert m[i["1_S:a=1"], i["2_S:b::mach=C"]] and not m[i["2_S:b::mach=C"], i["1_S:a=1"]]
    assert m[i["2_S:b::mach=C"], i["2_T:x=1"]] and m[i["2_T:x=1"], i["2_S:b::mach=C"]]
    assert not m[i["1_S:a::mach=A"], i["1_S:a::mach=B"]] and not m.diagonal().any()
    assert not is_acyclic_mask(m)
    enc2 = fit_encoding(df2 := _mask_frame(with_tie=False), _table(df2))
    assert is_acyclic_mask(build_mask(enc2))
    assert build_mask(enc, time_order=False)[i["2_S:b::mach=C"], i["1_S:a=1"]]


def test_collapse_edges_and_identical():
    df = _mask_frame(with_tie=False)
    _step(df, 3, "S:c", np.random.default_rng(5).random(len(df)) < 0.5, ["Q"], np.random.default_rng(5))
    enc = fit_encoding(df, _table(df))
    i = {name: k for k, name in enumerate(enc.names)}
    W = np.zeros((len(enc.names),) * 2)
    W[i["1_S:a::mach=A"], i["2_S:b::mach=C"]] = 0.5
    W[i["1_S:a::mach=B"], i["2_S:b::mach=C"]] = -0.7
    W[i["1_S:a=1"], i["2_S:b::mach=C"]] = 0.1
    edges, ind = collapse_edges(W, enc, 0.3)
    r = edges[edges.kind == "learned"].iloc[0]
    assert (r.source, r.target, r.weight, r.n_indicator_edges) == ("1_S:a::mach", "2_S:b::mach", -0.7, 2)
    assert "A→C" in r.detail and "B→C" in r.detail
    ident = edges[edges.kind == "identical"]
    assert len(ind) == 2 and set(ident.target) == {"2_S:b", "3_S:c"}      # presence == not absent
    assert list(edges.columns) == ["source", "target", "weight", "n_indicator_edges", "kind", "detail"]


def test_column_fully_merged_and_partnerless():       # Review Focus 3
    rng = np.random.default_rng(3)
    df = _mask_frame(with_tie=False)
    v4 = rng.random(len(df)) < 0.5
    _step(df, 4, "S:d", v4)                                            # presence only
    _step(df, 5, "S:e", v4)                                            # presence only, always with step 4
    enc = fit_encoding(df, _table(df))
    edges, _ = collapse_edges(np.zeros((len(enc.names),) * 2), enc, 0.3)
    assert not any(n.startswith("5_S:e=") for n in enc.names)
    assert ((edges.source == "4_S:d") & (edges.target == "5_S:e") & (edges.kind == "identical")).any()


def test_check_time_order():
    cols = _table(_mask_frame(with_tie=False))
    check_time_order(pd.DataFrame({"source": ["1_S:a::mach", "1_S:a"], "target": ["1_S:a", "2_S:b::mach"]}), cols)
    with pytest.raises(ValueError, match="2_S:b"):
        check_time_order(pd.DataFrame({"source": ["2_S:b"], "target": ["1_S:a"]}), cols)
    with pytest.raises(ValueError):
        check_time_order(pd.DataFrame({"source": ["1_S:a"], "target": ["1_S:a::mach"]}), cols)


def test_estimate_gpu_gb():
    assert estimate_gpu_gb(10_000, "dagma") == pytest.approx(4.8)
    assert estimate_gpu_gb(10_000, "lasso") == pytest.approx(3.2)


# ----- Task 6: preprocessing CLI -----------------------------------------------------------------------------------

def test_preprocess_cli(tmp_path):
    from utilities.preprocess import main as preprocess_main
    df, _ = simulate_presence(n_lots=800, n_steps=30, rare_steps=3, seed=4)
    df.to_csv(tmp_path / "p.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / "p.csv"), "--out", str(tmp_path / "prep")]) == 0
    s = json.loads((tmp_path / "prep" / "summary.json").read_text())
    cols = pd.read_csv(tmp_path / "prep" / "columns.csv")
    data = pd.read_parquet(tmp_path / "prep" / "data.parquet")
    assert s["steps_dropped_rare"] == 3 and (cols.drop_reason == "rare_step").sum() == 6
    kept = cols[cols.kept].assign(k=cols.kind.eq("presence"))
    assert list(data.columns[1:]) == kept.sort_values(["step_number", "step_key", "k"]).name.tolist()
    assert s["encoded_columns"] > 0 and s["gpu_gb_dagma"] > s["gpu_gb_lasso"] and s["runtime_note"]
    assert data["lot_id"].tolist() == df["lot_id"].tolist()
    assert preprocess_main(["--input", str(tmp_path / "p.csv"), "--out", str(tmp_path / "prep2"),
                            "--min-lots", "200"]) == 0
    assert json.loads((tmp_path / "prep2" / "summary.json").read_text())["kept_columns"] <= s["kept_columns"]


# ----- Task 7: held-out scoring ------------------------------------------------------------------------------------

def test_frequency_model_exact():
    from utilities.scoring import heldout_logprob
    tr = pd.DataFrame({"a": ["x", "x", "y"]})
    te = pd.DataFrame({"a": ["x", "z"]})
    lp = heldout_logprob(tr, te, ["a"], {}, jobs=1)
    np.testing.assert_allclose(lp[:, 0], np.log([3 / 5, 1 / 6]))


def _xy(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.choice(list("abcd"), n)
    y = np.where(rng.random(n) < noise, rng.choice(list("abcd"), n), x)
    return pd.DataFrame({"x": x, "y": y})


def test_parent_improves_heldout():
    from utilities.scoring import compare, heldout_logprob
    df = _xy(2000, 0.05, 0)
    tr, te = df.iloc[:1600], df.iloc[1600:]
    lp0 = heldout_logprob(tr, te, ["x", "y"], {}, jobs=1)
    lp1 = heldout_logprob(tr, te, ["x", "y"], {"y": ["x"]}, jobs=1)
    c = compare(lp1, lp0)
    assert c["gain"] > 0.3 and c["gain"] - c["ci95"] > 0


def test_many_parents_still_gain():          # a probability-table model scores below the empty graph here
    from utilities.scoring import compare, heldout_logprob
    rng = np.random.default_rng(1)
    n = 3750
    df = pd.DataFrame({f"p{i}": rng.choice(list("vwxyz"), n) for i in range(12)})
    df["y"] = np.where(rng.random(n) < 0.8, df["p0"], rng.choice(list("vwxyz"), n))
    tr, te = df.iloc[:3000], df.iloc[3000:]
    c = compare(heldout_logprob(tr, te, ["y"], {"y": [f"p{i}" for i in range(12)]}, jobs=1),
                heldout_logprob(tr, te, ["y"], {}, jobs=1))
    assert c["gain"] > 0


def test_unseen_parent_category_is_finite():
    from utilities.scoring import heldout_logprob
    df = _xy(500, 0.1, 2)
    tr, te = df.iloc[:400], df.iloc[400:].copy()
    te.loc[te.index[:5], "x"] = "zzz"
    assert np.isfinite(heldout_logprob(tr, te, ["y"], {"y": ["x"]}, jobs=1)).all()


# ----- Task 8: fitting and run_dagma ------------------------------------------------------------------------------

def _prepare(tmp_path, df, name="prep"):
    from utilities.preprocess import main as preprocess_main
    df.to_csv(tmp_path / f"{name}.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / f"{name}.csv"), "--out", str(tmp_path / name)]) == 0
    return tmp_path / name


@pytest.fixture
def prepared(tmp_path):
    df, truth = simulate_presence(n_lots=1500, n_steps=25, n_effects=6, seed=5)
    prep = _prepare(tmp_path, df)
    return prep, pd.read_csv(prep / "columns.csv"), truth


def test_benchmark_and_fit_e2e(prepared, tmp_path):
    from utilities.run_dagma import main as run_main
    prep, cols, truth = prepared
    assert run_main(["benchmark", "--prepared", str(prep), "--out", str(tmp_path / "b"),
                     "--solver", "lasso", "--device", "cpu", "--jobs", "1"]) == 0
    res = json.loads((tmp_path / "b" / "results.json").read_text())
    g = {r["graph"]: r for r in res["graphs"]}
    assert set(g) == {"empty graph", "within-step only"} | {f"dagma (threshold {t})" for t in (0.1, 0.2, 0.3, 0.4, 0.5)}
    assert g["within-step only"]["gain_vs_empty"] > 0
    assert g["dagma (threshold 0.3)"]["gain_vs_within_step"] > 0
    assert (tmp_path / "b" / "results.md").exists() and (tmp_path / "b" / "per_column.csv").exists()
    assert run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "f"),
                     "--solver", "lasso", "--device", "cpu"]) == 0
    edges = pd.read_csv(tmp_path / "f" / "edges.csv")
    check_time_order(edges, cols)
    found = {(step_of(s), step_of(t)) for s, t in zip(edges.source, edges.target)}
    recall = np.mean([(step_of(s), step_of(t)) in found for s, t in truth.itertuples(index=False)])
    assert recall >= 0.8
    info = json.loads((tmp_path / "f" / "info.json").read_text())
    A = np.load(tmp_path / "f" / "adjacency.npy")
    assert A.shape == (len(info["columns"]),) * 2 and A.sum() == len(edges[["source", "target"]].drop_duplicates())


def test_fit_dagma_solver_runs(prepared, tmp_path):
    from utilities.run_dagma import main as run_main
    prep, _, _ = prepared
    assert run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "d"), "--device", "cpu",
                     "--warm-iter", "2000", "--max-iter", "4000"]) == 0
    info = json.loads((tmp_path / "d" / "info.json").read_text())
    assert info["skipped_acyclicity"] is True and info["peak_gpu_gb"] is None        # None on CPU


def test_prepare_problem_matches_fit(prepared):
    from utilities.fitting import prepare_problem
    prep, cols, _ = prepared
    data = pd.read_parquet(prep / "data.parquet")
    enc, cov, mask = prepare_problem(data, cols[cols.kept])
    d = len(enc.names)
    assert cov.shape == mask.shape == (d, d) and np.allclose(np.diag(cov), 1.0) and is_acyclic_mask(mask)


def test_lasso_with_tied_steps_exits(tmp_path):
    from utilities.run_dagma import main as run_main
    df, _ = simulate_presence(n_lots=600, n_steps=10, rare_steps=0, seed=6)
    df = df.rename(columns={c: "2" + c[1:] for c in df.columns if c.startswith("3_")})   # steps 2 and 3 share N=2
    prep = _prepare(tmp_path, df)
    with pytest.raises(SystemExit, match="share a step number"):
        run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "x"), "--solver", "lasso", "--device", "cpu"])


def test_gpu_memory_check(monkeypatch):                 # Review Focus 5
    from utilities.fitting import check_gpu_memory
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 * 1024**3, 24 * 1024**3))
    with pytest.raises(SystemExit, match="--min-lots"):
        check_gpu_memory(20_000, "dagma", "cuda")
    check_gpu_memory(100, "dagma", "cuda")             # fits: no error
    with pytest.raises(SystemExit) as stop:
        check_gpu_memory(20_000, "lasso", "cuda")
    assert "use --solver lasso" not in str(stop.value)  # already the low-memory solver


# ----- Task 9: interactive viewer -----------------------------------------------------------------------------------

@pytest.fixture
def edges_and_columns(prepared, tmp_path):
    from utilities.run_dagma import main as run_main
    prep, _, _ = prepared
    assert run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "f"), "--solver", "lasso",
                     "--device", "cpu"]) == 0
    return str(tmp_path / "f" / "edges.csv"), str(prep / "columns.csv")


def _node_count(fig):
    return sum(len(t.x) for t in fig.data if t.meta == "nodes")


def test_viewer_pipeline_edges(tmp_path, edges_and_columns):
    from utilities.plot_edges import main as plot_main
    e_path, c_path = edges_and_columns
    out = tmp_path / "g.html"
    assert plot_main(["--edges", e_path, "--columns", c_path, "--out", str(out)]) == 0
    html = out.read_text()
    assert "stxHighlight" in html and 'id="stx-search"' in html
    assert pd.read_csv(e_path).source.iloc[0] in html


def test_viewer_package_edges(tmp_path):                       # Review Focus 4
    from utilities.plot_edges import build_figure, load_nodes
    pd.DataFrame({"source": ["X1", "X2"], "target": ["X2", "X3"]}).to_csv(tmp_path / "e.csv", index=False)
    edges = pd.read_csv(tmp_path / "e.csv")
    nodes = load_nodes(edges, None)
    assert set(nodes.name) == {"X1", "X2", "X3"} and nodes.step_number.isna().all()
    assert _node_count(build_figure(edges, nodes)) == 3


def test_viewer_max_edges_and_isolated(tmp_path, capsys, edges_and_columns):
    from utilities.plot_edges import load_nodes, main as plot_main
    _, c_path = edges_and_columns
    cols = pd.read_csv(c_path)
    names = cols[cols.kept].name.tolist()
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a in names for b in names if step_of(a) < step_of(b)]
    pick = rng.choice(len(pairs), 100, replace=False)
    e100 = pd.DataFrame({"source": [pairs[k][0] for k in pick], "target": [pairs[k][1] for k in pick],
                         "weight": rng.uniform(-1, 1, 100)})
    e100.to_csv(tmp_path / "e100.csv", index=False)
    assert plot_main(["--edges", str(tmp_path / "e100.csv"), "--columns", c_path, "--max-edges", "10",
                      "--out", str(tmp_path / "m.html")]) == 0
    assert "kept 10 of 100" in capsys.readouterr().out
    top10 = e100.reindex(e100.weight.abs().sort_values(ascending=False).index).head(10)
    nodes = load_nodes(top10, c_path)
    assert (nodes.in_degree + nodes.out_degree).gt(0).all() and len(nodes) < len(names)


def test_viewer_js_parses(tmp_path, edges_and_columns):
    import re
    import shutil
    import subprocess
    from utilities.plot_edges import main as plot_main
    e_path, c_path = edges_and_columns
    plot_main(["--edges", e_path, "--columns", c_path, "--out", str(tmp_path / "j.html"), "--cdn"])
    html = (tmp_path / "j.html").read_text()
    js = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)[-1]   # the plot call with the injected script
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    (tmp_path / "s.js").write_text(js)
    subprocess.run(["node", "--check", str(tmp_path / "s.js")], check=True)


def test_viewer_node_ids_are_plain_json():
    # the page's click/highlight script reads node ids from the figure's JSON; plotly encodes numpy arrays as base64
    from utilities.plot_edges import build_figure, load_nodes
    edges = pd.DataFrame({"source": ["1_S:a::mach", "1_S:a"], "target": ["1_S:a", "2_S:b"]})
    fig = build_figure(edges, load_nodes(edges, None))
    data = json.loads(fig.to_json())["data"]
    ids = [i for t in data if t.get("meta") == "nodes" for i in t["customdata"]]
    assert sorted(ids) == [0, 1, 2]


def test_viewer_range_fits_arcs_and_labels_stages():
    from utilities.plot_edges import build_figure, load_nodes
    edges = pd.DataFrame({"source": ["1_A:x::mach", "1_A:x::mach", "2_A:y", "3_B:z::mach"],
                          "target": ["1_A:x", "30_C:w", "40_C:v", "3_B:z"]})
    fig = build_figure(edges, load_nodes(edges, None))
    lo, hi = fig.layout.yaxis.range
    arc_y = [v for t in fig.data if t.meta == "edges" for v in t.y if v is not None]
    assert lo <= min(arc_y) and hi >= max(arc_y) and hi < 2.75      # fitted, not the old fixed range
    stages = [t for t in fig.data if t.meta == "stages"][0]
    assert list(stages.text) == ["A", "B", "C"]


# ----- final-review fixes ------------------------------------------------------------------------------------------

def test_covariance_blocked_matches_direct():
    # X.T @ X (BLAS syrk) segfaulted at d >= ~32,700 on the Slurm node; the blocked product must give the same matrix
    from utilities.fitting import covariance
    X = np.random.default_rng(0).normal(size=(50, 23))
    np.testing.assert_allclose(covariance(X, block=7), X.T @ X / 50, rtol=1e-12, atol=1e-12)


def test_viewer_no_edges(tmp_path):
    from utilities.plot_edges import main as plot_main
    pd.DataFrame(columns=["source", "target", "weight"]).to_csv(tmp_path / "e.csv", index=False)
    with pytest.raises(SystemExit, match="no edges"):
        plot_main(["--edges", str(tmp_path / "e.csv"), "--out", str(tmp_path / "e.html")])


def test_viewer_min_weight_without_weight_column(tmp_path):
    from utilities.plot_edges import main as plot_main
    pd.DataFrame({"source": ["X1", "X2"], "target": ["X2", "X3"]}).to_csv(tmp_path / "e.csv", index=False)
    assert plot_main(["--edges", str(tmp_path / "e.csv"), "--min-weight", "0.4", "--out", str(tmp_path / "e.html")]) == 0
    assert "X3" in (tmp_path / "e.html").read_text()


def test_lasso_nonconvergence_reported(prepared, tmp_path):
    from utilities.run_dagma import main as run_main
    prep, _, _ = prepared
    assert run_main(["benchmark", "--prepared", str(prep), "--out", str(tmp_path / "b"), "--solver", "lasso",
                     "--device", "cpu", "--jobs", "1", "--lasso-max-iter", "1"]) == 0
    info = json.loads((tmp_path / "b" / "info.json").read_text())
    assert info["converged"] is False and info["kkt_violation"] > 0
    assert "did not converge" in (tmp_path / "b" / "results.md").read_text()


def test_load_rejects_duplicate_headers(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("lot_id,1_S:a,1_S:a,1_S:a::mach\nL1,1,1,M\n")
    with pytest.raises(ValueError, match="duplicate.*1_S:a"):
        load_presence_matrix(p)


def test_no_gpu_stops_large_fit(prepared, tmp_path, monkeypatch):
    import utilities.fitting as F
    from utilities.run_dagma import main as run_main
    prep, _, _ = prepared
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(F, "CPU_MAX_D", 10)
    with pytest.raises(SystemExit, match="No GPU"):
        run_main(["fit", "--prepared", str(prep), "--out", str(tmp_path / "f"), "--solver", "lasso"])


def test_ignore_unparsed_recorded(tmp_path):
    from utilities.preprocess import main as preprocess_main
    df, _ = simulate_presence(n_lots=300, n_steps=10, rare_steps=0, seed=7)
    df.insert(1, "weird", "x")
    df.to_csv(tmp_path / "w.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / "w.csv"), "--out", str(tmp_path / "p"), "--ignore-unparsed"]) == 0
    cols = pd.read_csv(tmp_path / "p" / "columns.csv").set_index("name")
    assert cols.loc["weird", "drop_reason"] == "unparsed" and not cols.loc["weird", "kept"]
    assert json.loads((tmp_path / "p" / "summary.json").read_text())["columns_unparsed"] == 1


def test_preprocess_reports_near_duplicates(tmp_path):
    from utilities.preprocess import main as preprocess_main
    df, _ = simulate_presence(n_lots=2000, n_steps=12, rare_steps=0, seed=8)
    a, b = df.columns[7], df.columns[9]                     # presence columns of steps 4 and 5
    rng = np.random.default_rng(0)
    va = rng.random(len(df)) < 0.5                          # step 4: half the lots
    df[a], df[a + "::mach"] = va.astype(np.int8), np.where(va, rng.choice(["M4a", "M4b"], len(df)), "")
    vb = np.where(rng.random(len(df)) < 0.01, ~va, va)      # step 5: the same lots, 1% differ (corr ~0.98)
    df[b], df[b + "::mach"] = vb.astype(np.int8), np.where(vb, "M5", "")
    df.to_csv(tmp_path / "n.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / "n.csv"), "--out", str(tmp_path / "p")]) == 0
    assert json.loads((tmp_path / "p" / "summary.json").read_text())["near_duplicate_indicator_pairs"] >= 1


def test_near_duplicates_ignore_same_step_pairs(tmp_path):
    # with few revisits, "visited once" ~ "not absent" within a step: expected, not a cross-step near-duplicate
    from utilities.preprocess import main as preprocess_main
    df, _ = simulate_presence(n_lots=2000, n_steps=12, rare_steps=0, seed=8)
    df.to_csv(tmp_path / "s.csv", index=False)
    assert preprocess_main(["--input", str(tmp_path / "s.csv"), "--out", str(tmp_path / "p")]) == 0
    assert json.loads((tmp_path / "p" / "summary.json").read_text())["near_duplicate_indicator_pairs"] == 0
