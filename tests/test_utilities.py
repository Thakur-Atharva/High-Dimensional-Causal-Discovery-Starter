"""Checks for utilities/ (the step-presence pipeline): CPU, about a minute."""
import numpy as np
import pandas as pd

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
