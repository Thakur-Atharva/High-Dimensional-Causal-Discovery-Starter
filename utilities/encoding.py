"""From categorical step columns to DAGMA's input and back.

* One-hot encoding: one 0/1 indicator per category. Machine columns keep every level, ``absent`` first; presence
  columns drop ``0`` (or, if 0 never occurs, their most common level).
* Identical indicators (equal, or exact complements, on the rows being fitted) are merged into the earliest one in
  time. Otherwise the L1 penalty splits an effect's weight between the copies, and both halves can fall below the
  edge threshold.
* The time-order mask over indicators.
* Collapsing indicator-level weights back to one edge per pair of original columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .presence import ABSENT, KIND_ORDER

IDENTICAL_COLUMNS = ["rep_column", "rep_category", "column", "category", "complement"]
EDGE_COLUMNS = ["source", "target", "weight", "n_indicator_edges", "kind", "detail"]


@dataclass
class Encoding:
    """The indicators DAGMA sees (representatives only), in time order."""
    names: list[str]                 # "column=category"
    column: list[str]
    category: list[str]
    step_number: np.ndarray
    step_key: list[str]
    kind: list[str]
    levels: dict[str, list[str]]     # per original column: its non-reference categories
    aliases: dict[int, list[str]] = field(default_factory=dict)   # representative index -> merged indicator names
    identical: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=IDENTICAL_COLUMNS))


def _kept(columns: pd.DataFrame) -> pd.DataFrame:
    cols = columns[columns["kept"]] if "kept" in columns else columns
    order = cols.assign(_k=cols["kind"].map(KIND_ORDER)).sort_values(["step_number", "_k", "step_key", "name"])
    return order.drop(columns="_k")


def _reference(values: pd.Series, kind: str) -> str:
    preferred = ABSENT if kind == "machine" else "0"
    counts = values.value_counts()
    if preferred in counts.index:
        return preferred
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def fit_encoding(df: pd.DataFrame, columns: pd.DataFrame) -> Encoding:
    """Decide the indicators from the rows of ``df`` (the training rows): levels, references, merges."""
    cand = []                                                   # (column row, category, 0/1 vector)
    levels: dict[str, list[str]] = {}
    for row in _kept(columns).itertuples(index=False):
        s = df[row.name].astype(str)
        cats = sorted(s.unique())
        if row.kind == "machine":
            # Every level, "absent" first: "absent" then carries "visited this step?" in one indicator (the presence
            # level 1 usually merges into it), and every machine keeps its own. Dropping "absent" instead spreads an
            # effect on visiting over all of the step's machine indicators, which in simulations hid half the edges.
            levels[row.name] = sorted(cats, key=lambda c: (c != ABSENT, c))
        else:
            ref = _reference(s, row.kind)
            levels[row.name] = [c for c in cats if c != ref]
        for c in levels[row.name]:
            v = (s == c).to_numpy()
            if v.any() and not v.all():
                cand.append((row, c, v))
    keep, rep_of_key, aliases, ident = [], {}, {}, []
    for row, c, v in cand:
        canon = np.packbits(~v if v[0] else v).tobytes()        # equal or complementary vectors share this key
        rep = rep_of_key.get(canon)
        if rep is None:
            rep_of_key[canon] = len(keep)
            keep.append((row, c, v))
            continue
        rrow, rc, rv = keep[rep]
        aliases.setdefault(rep, []).append(f"{row.name}={c}")
        if rrow.name != row.name:
            ident.append((rrow.name, rc, row.name, c, bool(rv[0] != v[0])))
    return Encoding(
        names=[f"{r.name}={c}" for r, c, _ in keep], column=[r.name for r, _, _ in keep],
        category=[c for _, c, _ in keep], step_number=np.array([r.step_number for r, _, _ in keep], dtype=np.int64),
        step_key=[r.step_key for r, _, _ in keep], kind=[r.kind for r, _, _ in keep], levels=levels,
        aliases=aliases, identical=pd.DataFrame(ident, columns=IDENTICAL_COLUMNS))


def indicator_matrix(df: pd.DataFrame, enc: Encoding) -> np.ndarray:
    """n x d 0/1 matrix of the representative indicators for the rows of ``df``."""
    X = np.zeros((len(df), len(enc.names)), dtype=np.float64)
    cache: dict[str, np.ndarray] = {}
    for j, (col, cat) in enumerate(zip(enc.column, enc.category)):
        if col not in cache:
            cache[col] = df[col].astype(str).to_numpy()
        X[:, j] = cache[col] == cat
    return X


def build_mask(enc: Encoding, time_order: bool = True, block: int = 2048) -> np.ndarray:
    """Allowed edges between indicators (``mask[i, j]``: i -> j may exist).

    With ``time_order``: from an earlier step to a later one; within a step, machine -> presence only; between
    different steps sharing a step number, both ways (the time order cannot separate them). Never between indicators
    of the same column, never i -> i.
    """
    d = len(enc.names)
    sn = enc.step_number
    key = pd.factorize(pd.Series(enc.step_key))[0]
    col = pd.factorize(pd.Series(enc.column))[0]
    mach = np.array([k == "machine" for k in enc.kind])
    mask = np.zeros((d, d), dtype=bool)
    for a in range(0, d, block):
        b = min(a + block, d)
        if time_order:
            si, sj = sn[a:b, None], sn[None, :]
            same_step = key[a:b, None] == key[None, :]
            allowed = (si < sj) | ((si == sj) & ~same_step) | ((si == sj) & same_step & mach[a:b, None] & ~mach[None, :])
        else:
            allowed = np.ones((b - a, d), dtype=bool)
        mask[a:b] = allowed & (col[a:b, None] != col[None, :])
    return mask


def collapse_edges(W: np.ndarray, enc: Encoding, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Column-level edges (learned and identical) and the indicator-level learned edges behind them."""
    W = np.asarray(W)
    src, dst = np.nonzero((np.abs(W) >= threshold) & (W != 0))
    ind = pd.DataFrame({"source": [enc.column[i] for i in src], "source_category": [enc.category[i] for i in src],
                        "target": [enc.column[j] for j in dst], "target_category": [enc.category[j] for j in dst],
                        "weight": W[src, dst]})
    rows = []
    if len(ind):
        ind["src_idx"], ind["dst_idx"] = src, dst
        for (s, t), g in ind.groupby(["source", "target"], sort=False):
            g = g.reindex(g["weight"].abs().sort_values(ascending=False).index)
            parts = []
            for r in g.head(5).itertuples(index=False):
                text = f"{r.source_category}→{r.target_category} ({r.weight:+.2f})"
                al = enc.aliases.get(r.src_idx, []) + enc.aliases.get(r.dst_idx, [])
                parts.append(text + (f" [≡ {', '.join(al)}]" if al else ""))
            rows.append((s, t, float(g["weight"].iloc[0]), len(g), "learned", "; ".join(parts)))
        ind = ind.drop(columns=["src_idx", "dst_idx"])
    for (s, t), g in enc.identical.groupby(["rep_column", "column"], sort=False):
        detail = "; ".join(f"{r.rep_category}≡{'not ' if r.complement else ''}{r.category}"
                           for r in g.itertuples(index=False))
        rows.append((s, t, -1.0 if bool(g["complement"].iloc[0]) else 1.0, len(g), "identical", detail))
    return pd.DataFrame(rows, columns=EDGE_COLUMNS), ind


def check_time_order(edges: pd.DataFrame, columns: pd.DataFrame) -> None:
    """Raise ValueError if any edge goes back in time or from a presence column to its own step's machine column."""
    info = columns.set_index("name")[["step_number", "step_key", "kind"]]
    bad = []
    for s, t in zip(edges["source"], edges["target"]):
        if s not in info.index or t not in info.index:
            bad.append(f"{s} -> {t} (unknown column)")
            continue
        a, b = info.loc[s], info.loc[t]
        ok = (a.step_number < b.step_number
              or (a.step_number == b.step_number and a.step_key != b.step_key)
              or (a.step_key == b.step_key and a.kind == "machine" and b.kind == "presence"))
        if not ok:
            bad.append(f"{s} -> {t}")
    if bad:
        raise ValueError(f"{len(bad)} edge(s) violate the time order: {bad[:10]}")


def estimate_gpu_gb(d: int, solver: str, acyclic: bool = True) -> float:
    """GPU memory for a fit on d indicators in float32, in d x d matrices with ~10% headroom over measured peaks.

    Measured at d = 10,093 and 33,745: lasso 8.25, DAGMA with the acyclicity term skipped 8.0, DAGMA's full path
    (needed when the mask has cycles, e.g. steps sharing a number) 17.0.
    """
    matrices = 9 if (solver == "lasso" or acyclic) else 18
    if solver not in ("lasso", "dagma"):
        raise ValueError("solver must be 'dagma' or 'lasso'")
    return matrices * d * d * 4 / 1e9
