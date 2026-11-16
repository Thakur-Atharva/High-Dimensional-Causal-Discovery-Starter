"""A synthetic presence matrix in the real format, with planted edges, for tests, timing runs and dry runs.

    python -m utilities.simulate_presence --lots 9000 --steps 5000 --out data/sim_presence.csv

Writes the table and ``<name>_truth.csv`` (the planted cross-step effects, ``source,target`` at column level).

What it contains:
* ``lot_id``, then two columns per step k, in time order: ``k_<stage>:<step>`` (presence: 0 not visited, 1 visited,
  2 visited twice) and ``k_<stage>:<step>::mach`` (the machine, empty when not visited). Stage and step names contain
  ``_`` and ``:``, as real names may.
* 1 to 11 machines per step, with a long right tail and a mean of about 4. Some machines are literally named ``NA`` and
  ``None``, to check that they are not read as missing.
* A few rarely visited steps, at least two single-machine steps without revisits (whose machine and presence columns
  carry the same information), and one pair of steps that are always visited together.
* Planted effects: lots that ran on a chosen machine at step a visit a later step b with probability 0.9, other lots
  with probability 0.2.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _machines_per_step(rng: np.random.Generator, n_steps: int, mean: float, max_machines: int) -> np.ndarray:
    """1 + negative binomial (long right tail), capped at max_machines; mean about ``mean``."""
    extra = mean - 1.0
    p = 2.0 / (2.0 + extra)                              # negative binomial with n = 2 has mean 2 (1 - p) / p
    return np.minimum(1 + rng.negative_binomial(2, p, size=n_steps), max_machines)


def simulate_presence(n_lots: int = 2000, n_steps: int = 60, mean_machines: float = 4.0, max_machines: int = 11,
                      rare_steps: int = 3, n_effects: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table, truth). ``truth`` lists the planted effects as (machine column of a) -> (presence column of b)."""
    if n_steps < 8:
        raise ValueError("n_steps must be at least 8")
    rng = np.random.default_rng(seed)
    steps = np.arange(1, n_steps + 1)
    base = [f"{k}_STG_{(k - 1) // 10 + 1}:step:{k}" for k in steps]
    k_mach = _machines_per_step(rng, n_steps, mean_machines, max_machines)
    visit_p = rng.uniform(0.3, 1.0, n_steps)
    revisit_p = rng.uniform(0.0, 0.1, n_steps)

    # special steps (indices into steps), chosen from different parts of the line
    idx = rng.permutation(n_steps)
    single = idx[:2]                                     # single machine, no revisits
    k_mach[single], revisit_p[single] = 1, 0.0
    together = int(idx[2]) if idx[2] < n_steps - 1 else int(idx[2]) - 1   # step together+1 copies step together
    rare = [i for i in idx[3:] if i not in (together, together + 1)][:rare_steps]
    visit_p[rare] = 0.01
    special = set(single) | {together, together + 1} | set(rare)

    # planted effects: source step a (well visited), target step b > a with base visit probability 0.2
    effects: list[tuple[int, int]] = []
    used_targets: set[int] = set()
    candidates = [i for i in range(n_steps) if i not in special]
    for _ in range(50 * n_effects):
        if len(effects) == n_effects or len(candidates) < 2:
            break
        a, b = sorted(rng.choice(candidates, 2, replace=False))
        if b in used_targets or a in used_targets or any(a == e[1] or b == e[0] for e in effects):
            continue
        effects.append((int(a), int(b)))
        used_targets.add(int(b))
        visit_p[a] = max(visit_p[a], 0.9)
        visit_p[b] = 0.2
        revisit_p[b] = 0.0

    # two well-visited multi-machine steps whose most-used machine is literally named "NA" / "None"
    named = [int(i) for i in idx if k_mach[i] >= 2 and i not in special and visit_p[i] >= 0.3][:2]
    names_na = dict(zip(named, ["NA", "None"]))
    effect_machine: dict[int, str] = {}
    cols: dict[str, np.ndarray] = {"lot_id": np.array([f"LOT{i:06d}" for i in range(n_lots)])}
    visited_by_step: dict[int, np.ndarray] = {}
    machine_by_step: dict[int, np.ndarray] = {}
    for i in range(n_steps):
        k = int(k_mach[i])
        machines = [f"T{i + 1:05d}{chr(65 + m)}" for m in range(k)]
        weights = 0.5 / k + 0.5 * rng.dirichlet(np.full(k, 2.0))
        if i in names_na:
            machines[int(np.argmax(weights))] = names_na[i]
        if i == together + 1:
            visited = visited_by_step[together].copy()
        else:
            p = np.full(n_lots, visit_p[i])
            for a, b in effects:
                if b == i:
                    p = np.where(machine_by_step[a] == effect_machine[a], 0.9, 0.2)
            visited = rng.random(n_lots) < p
        presence = visited.astype(np.int8) + (visited & (rng.random(n_lots) < revisit_p[i])).astype(np.int8)
        machine = np.where(visited, rng.choice(np.array(machines, dtype=object), n_lots, p=weights / weights.sum()), "")
        if any(a == i for a, _ in effects):
            effect_machine[i] = machines[int(np.argmax(weights))]
        visited_by_step[i], machine_by_step[i] = visited, machine
        cols[base[i]] = presence
        cols[base[i] + "::mach"] = machine.astype(object)
    table = pd.DataFrame(cols)
    truth = pd.DataFrame({"source": [base[a] + "::mach" for a, _ in effects],
                          "target": [base[b] for _, b in effects]})
    return table, truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m utilities.simulate_presence", description=__doc__.split("\n\n")[0])
    ap.add_argument("--lots", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--mean-machines", type=float, default=4.0)
    ap.add_argument("--rare-steps", type=int, default=3)
    ap.add_argument("--effects", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output CSV; the planted edges go to <name>_truth.csv beside it")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table, truth = simulate_presence(args.lots, args.steps, args.mean_machines, rare_steps=args.rare_steps,
                                     n_effects=args.effects, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    truth.to_csv(out.with_name(out.stem + "_truth.csv"), index=False)
    log.info("wrote %s (%d lots x %d columns) and %d planted edges", out, len(table), table.shape[1] - 1, len(truth))
    return 0


if __name__ == "__main__":
    sys.exit(main())
