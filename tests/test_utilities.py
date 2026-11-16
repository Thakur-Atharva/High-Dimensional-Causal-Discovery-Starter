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
    _step(df, 1, "S:a", rng.random(n) < 0.6, ["A", "B"], rng)
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
    assert set(enc.levels["1_S:a::mach"]) == {"A", "B"} and set(enc.levels["1_S:a"]) == {"1", "2"}
    assert enc.levels["2_S:b::mach"] == ["D"]


def test_merge_identical_and_complement():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"lot_id": [f"L{i}" for i in range(n)]})
    _step(df, 1, "S:a", rng.random(n) < 0.5, ["M"], rng)              # machine M == visited
    v2 = rng.random(n) < 0.5
    _step(df, 2, "S:b", v2, ["X", "Y"], rng)
    _step(df, 3, "S:c", ~v2, ["Z1", "Z2"], rng)                        # visited exactly when step 2 is not
    enc = fit_encoding(df, _table(df))
    assert "1_S:a=1" not in enc.names and "1_S:a::mach=M" in enc.names
    ident = enc.identical.set_index("column")
    assert ident.loc["1_S:a"].rep_column == "1_S:a::mach" and not ident.loc["1_S:a"].complement
    assert ident.loc["3_S:c"].rep_column == "2_S:b" and ident.loc["3_S:c"].complement


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
    assert len(hit) == 1 and hit.weight.iloc[0] > 0.4                  # not split into two halves below 0.3


def test_build_mask_rules():
    enc = fit_encoding(df := _mask_frame(), _table(df))
    m = build_mask(enc)
    i = {name: k for k, name in enumerate(enc.names)}
    assert m[i["1_S:a::mach=A"], i["1_S:a=1"]] and not m[i["1_S:a=1"], i["1_S:a::mach=A"]]
    assert m[i["1_S:a=1"], i["2_S:b=1"]] and not m[i["2_S:b=1"], i["1_S:a=1"]]
    assert m[i["2_S:b=1"], i["2_T:x=1"]] and m[i["2_T:x=1"], i["2_S:b=1"]]
    assert not m[i["1_S:a::mach=A"], i["1_S:a::mach=B"]] and not m.diagonal().any()
    assert not is_acyclic_mask(m)
    enc2 = fit_encoding(df2 := _mask_frame(with_tie=False), _table(df2))
    assert is_acyclic_mask(build_mask(enc2))
    assert build_mask(enc, time_order=False)[i["2_S:b=1"], i["1_S:a=1"]]


def test_collapse_edges_and_identical():
    df = _mask_frame(with_tie=False)
    _step(df, 3, "S:c", np.random.default_rng(5).random(len(df)) < 0.5, ["Q"], np.random.default_rng(5))
    enc = fit_encoding(df, _table(df))
    i = {name: k for k, name in enumerate(enc.names)}
    W = np.zeros((len(enc.names),) * 2)
    W[i["1_S:a::mach=A"], i["2_S:b=1"]] = 0.5
    W[i["1_S:a::mach=B"], i["2_S:b=1"]] = -0.7
    W[i["1_S:a=1"], i["2_S:b=1"]] = 0.1
    edges, ind = collapse_edges(W, enc, 0.3)
    r = edges[edges.kind == "learned"].iloc[0]
    assert (r.source, r.target, r.weight, r.n_indicator_edges) == ("1_S:a::mach", "2_S:b", -0.7, 2)
    assert "A→1" in r.detail and "B→1" in r.detail
    assert len(ind) == 2 and (edges.kind == "identical").sum() == 1
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
