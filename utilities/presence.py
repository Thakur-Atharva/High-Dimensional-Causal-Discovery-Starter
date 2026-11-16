"""Reading and filtering the step-presence matrix.

Expected columns: an id column (``lot_id``) and, per process step, ``N_stage:step`` (presence: 0, 1 or 2) and
``N_stage:step::mach`` (the machine; an empty cell means the lot was not processed there). ``N`` is the step's
position in time.
"""
from __future__ import annotations

import csv
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMN_RE = re.compile(r"^(\d+)_(.+?)(::mach)?$")
ABSENT = "absent"          # empty machine cell: the lot did not go through this step
OTHER = "other"            # machines used on too few lots, pooled
KIND_ORDER = {"machine": 0, "presence": 1}     # within a step, the machine column comes first in time


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    step_key: str          # the column name without "::mach"; identifies the step
    step_number: int       # N, the position in time
    stage: str
    step: str
    kind: str              # "presence" or "machine"


def parse_columns(names: Sequence[str], id_column: str = "lot_id") -> tuple[list[ColumnInfo], list[str]]:
    """Split column names into parsed step columns and names that don't match the pattern (the id column is skipped)."""
    parsed, unparsed = [], []
    for name in names:
        name = str(name)
        if name == id_column:
            continue
        m = COLUMN_RE.match(name)
        if m is None:
            unparsed.append(name)
            continue
        stage, _, step = m.group(2).partition(":")
        kind = "machine" if m.group(3) else "presence"
        key = name[: -len("::mach")] if kind == "machine" else name
        parsed.append(ColumnInfo(name, key, int(m.group(1)), stage, step, kind))
    return parsed, unparsed


def load_presence_matrix(path: str | Path, id_column: str = "lot_id",
                         ignore_unparsed: bool = False) -> tuple[pd.DataFrame, list[ColumnInfo]]:
    """Read the CSV: presence columns as int8 (0/1/2), machine columns as text with empty cells set to ``absent``.

    Machine names are kept verbatim: ``NA``, ``None`` or ``007`` stay machine names. Stops with a list of offending
    columns if a presence value is missing or not 0/1/2, or if a column name doesn't match the pattern (unless
    ``ignore_unparsed``).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    dups = [name for name, k in Counter(header).items() if k > 1]
    if dups:                                   # pandas would silently rename them to "name.1", "name.2", ...
        raise ValueError(f"{len(dups)} column name(s) appear more than once in the header (duplicate): {dups[:10]}")
    if id_column not in header:
        raise ValueError(f"id column {id_column!r} not found; the first columns are {header[:5]} (see --id-column)")
    infos, unparsed = parse_columns(header, id_column)
    if unparsed and not ignore_unparsed:
        raise ValueError(f"{len(unparsed)} column name(s) don't match 'N_stage:step' or 'N_stage:step::mach': "
                         f"{unparsed[:10]}. Fix them, or pass --ignore-unparsed to skip them.")
    if unparsed:
        log.warning("skipping %d column(s) whose names don't match the pattern: %s", len(unparsed), unparsed[:10])
    if not infos:
        raise ValueError("no step columns found")
    presence = [i.name for i in infos if i.kind == "presence"]
    machine = [i.name for i in infos if i.kind == "machine"]
    dtype = {c: str for c in machine} | {id_column: str}
    # na_filter=False: no value is treated as missing, so a machine called "NA" stays "NA"; empty cells are "".
    df = pd.read_csv(path, usecols=[id_column] + presence + machine, dtype=dtype, na_filter=False,
                     encoding="utf-8-sig", low_memory=False)
    bad = {}
    out = {id_column: df[id_column]}
    for c in presence:
        v = pd.to_numeric(df[c], errors="coerce")
        invalid = v.isna() | ~v.isin([0, 1, 2])
        if invalid.any():
            bad[c] = df[c][invalid].iloc[0]
        else:
            out[c] = v.astype(np.int8)
    if bad:
        examples = ", ".join(f"{c} (e.g. {v!r})" for c, v in list(bad.items())[:10])
        raise ValueError(f"{len(bad)} presence column(s) have values other than 0, 1 or 2, or empty cells: {examples}")
    for c in machine:
        m = df[c].astype(str).str.strip()
        out[c] = m.where(m != "", ABSENT)
    if not df[id_column].is_unique:
        log.warning("%s values are not unique (%d duplicates)", id_column, int(df[id_column].duplicated().sum()))
    order = [id_column] + [i.name for i in infos]
    table = pd.DataFrame({c: out[c] for c in order})
    table.attrs["unparsed"] = unparsed         # recorded in columns.csv by utilities.preprocess
    return table, infos


def _steps(infos: list[ColumnInfo]) -> list[tuple[str, dict[str, ColumnInfo]]]:
    """Steps in time order, each with its presence and/or machine column."""
    steps: dict[str, dict[str, ColumnInfo]] = defaultdict(dict)
    for info in infos:
        steps[info.step_key][info.kind] = info
    return sorted(steps.items(), key=lambda kv: (next(iter(kv[1].values())).step_number, kv[0]))


def filter_and_pool(df: pd.DataFrame, infos: list[ColumnInfo], id_column: str = "lot_id", min_lots: int = 100,
                    min_machine_lots: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop rarely visited steps, pool rarely used machines into ``other``, drop constant columns.

    A step is visited by a lot when its presence is >= 1 (or, for a step without a presence column, when its machine
    is not ``absent``). Returns the kept table (id column, then kept columns in time order with the machine column
    first within a step), a table describing every step column, and summary counts.
    """
    min_machine_lots = min_lots if min_machine_lots is None else min_machine_lots
    rows, kept = [], {id_column: df[id_column]}
    pooled = 0
    for key, parts in _steps(infos):
        pres, mach = parts.get("presence"), parts.get("machine")
        if pres is None or mach is None:
            log.warning("step %s has only its %s column; keeping it", key, "machine" if pres is None else "presence")
        visited = (df[pres.name] > 0) if pres is not None else (df[mach.name] != ABSENT)
        n_visited = int(visited.sum())
        rare = n_visited < min_lots
        for info in (mach, pres):
            if info is None:
                continue
            col = df[info.name]
            if info.kind == "machine" and not rare:
                counts = col.value_counts()
                small = [m for m, c in counts.items() if c < min_machine_lots and m != ABSENT]
                if small:
                    col = col.where(~col.isin(small), OTHER)
                    pooled += len(small)
            n_cat = int(col.nunique())
            reason = "rare_step" if rare else ("constant" if n_cat <= 1 else "")
            rows.append({"name": info.name, "step_key": key, "step_number": info.step_number, "stage": info.stage,
                         "step": info.step, "kind": info.kind, "lots_visited": n_visited, "n_categories": n_cat,
                         "kept": not reason, "drop_reason": reason})
            if not reason:
                kept[info.name] = col
    table = pd.DataFrame(rows)
    summary = {"rows": int(len(df)), "original_columns": len(infos), "kept_columns": int(table["kept"].sum()),
               "steps_total": len({i.step_key for i in infos}),
               "steps_dropped_rare": int(table.loc[table.drop_reason == "rare_step", "step_key"].nunique()),
               "columns_dropped_constant": int((table.drop_reason == "constant").sum()),
               "machines_pooled": pooled}
    return pd.DataFrame(kept), table, summary


def find_tied_steps(infos: list[ColumnInfo]) -> dict[int, list[str]]:
    """Step numbers shared by more than one step (these make the time order only partial)."""
    keys: dict[int, set[str]] = defaultdict(set)
    for info in infos:
        keys[info.step_number].add(info.step_key)
    return {n: sorted(k) for n, k in sorted(keys.items()) if len(k) > 1}
