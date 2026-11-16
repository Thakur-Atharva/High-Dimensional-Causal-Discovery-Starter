"""Checks for utilities/ (the step-presence pipeline): CPU, about a minute."""
import logging

import numpy as np
import pandas as pd
import pytest

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
