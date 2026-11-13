"""Greedy hill climbing on a decomposable score (BIC), restricted to candidate pairs of variables.

Searching all d (d - 1) possible edges is too slow at d = 2,500. Instead, a cheap first stage (here DAGMA) proposes,
for every variable, a short list of likely neighbours. Hill climbing then decides which of those pairs get an edge,
and in which direction, by repeatedly applying the single change that most improves the score:

    add  i -> j      delete  i -> j      reverse  i -> j  (to j -> i)

until no change helps. This "restrict, then maximize" design is the same as Sparse Candidate (Friedman et al. 1999)
and MMHC (Tsamardinos et al. 2006). Both directions of every candidate pair are always considered, so the first
stage's edge directions do not matter.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)
NEG = -np.inf


def candidate_pairs(W: np.ndarray, k: int = 10, min_abs: float = 0.0) -> list[tuple[int, int]]:
    """Undirected candidate pairs: for each variable, its k strongest neighbours by max(|W[i, j]|, |W[j, i]|)."""
    S = np.maximum(np.abs(W), np.abs(W).T)
    np.fill_diagonal(S, 0.0)
    d = S.shape[0]
    k = min(k, d - 1)
    pairs = set()
    top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    for j in range(d):
        for i in top[j]:
            if S[j, i] > min_abs:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def partial_correlations(X: np.ndarray) -> np.ndarray:
    """Partial correlation of every pair given all other variables, from the inverse covariance matrix.

    For linear data, a variable's parents and children always have non-zero partial correlation with it, so the
    largest ones are good candidate neighbours. (Its parents' other children do too, so these are a superset.)
    """
    C = np.cov(np.asarray(X, dtype=np.float64), rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(C)
    s = np.sqrt(np.abs(np.diag(P)))
    R = -P / np.outer(s, s)
    np.fill_diagonal(R, 0.0)
    return R


def all_pairs(d: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


class HillClimber:
    def __init__(self, score, d: int, pairs, max_parents: int | None = None, prefer: np.ndarray | None = None,
                 verbose: bool = False) -> None:
        self.score, self.d, self.verbose = score, d, verbose
        self.max_parents = max_parents if max_parents is not None else d
        src, dst = [], []
        for a, b in pairs:
            src += [a, b]
            dst += [b, a]
        self.src, self.dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
        E = len(self.src)
        self.rev = np.arange(E) ^ 1                      # edge 2m is a -> b, edge 2m + 1 is b -> a
        self.in_e = [[] for _ in range(d)]
        self.out_e = [[] for _ in range(d)]
        for e in range(E):
            self.in_e[self.dst[e]].append(e)
            self.out_e[self.src[e]].append(e)
        # Ties: adding i -> j and j -> i often raise the BIC by exactly the same amount (BIC cannot see the direction
        # of an isolated edge), and the choice changes everything downstream. Ties go to the direction with the
        # larger prefer[i, j] (e.g. DAGMA's |W|); without ``prefer``, to the lower-numbered move.
        pr = np.zeros(E) if prefer is None else np.asarray(prefer, dtype=np.float64)[self.src, self.dst]
        self.priority = np.concatenate([pr, pr, -pr])     # add, delete, reverse
        self.present = np.zeros(E, dtype=bool)
        self.parents = [set() for _ in range(d)]
        self.children = [set() for _ in range(d)]
        self.local = np.array([score.local(j, ()) for j in range(d)])
        self.d_add = np.full(E, NEG)
        self.d_del = np.full(E, NEG)
        self.d_rev = np.full(E, NEG)
        for e in range(E):
            self._refresh(e)

    # ----- score changes of the three moves on ordered candidate edge e = (i -> j) --------------------------------
    def _refresh(self, e: int) -> None:
        i, j = self.src[e], self.dst[e]
        P = self.parents[j]
        if self.present[e]:
            self.d_add[e] = NEG
            without = self.score.local(j, P - {i}) - self.local[j]
            self.d_del[e] = without
            if len(self.parents[i]) < self.max_parents:
                self.d_rev[e] = without + self.score.local(i, self.parents[i] | {j}) - self.local[i]
            else:
                self.d_rev[e] = NEG
        else:
            self.d_del[e] = self.d_rev[e] = NEG
            if self.present[self.rev[e]] or len(P) >= self.max_parents:
                self.d_add[e] = NEG
            else:
                self.d_add[e] = self.score.local(j, P | {i}) - self.local[j]

    def _refresh_node(self, v: int) -> None:
        for e in self.in_e[v]:
            self._refresh(e)
            self._refresh(self.rev[e])
        for e in self.out_e[v]:
            if self.present[e]:
                self._refresh(e)                          # reversal of v -> x depends on v's parents

    def _reaches(self, start: int, target: int, skip: tuple[int, int] | None = None) -> bool:
        """Is there a directed path start ~> target (optionally ignoring the edge ``skip``)?"""
        stack, seen = [start], {start}
        while stack:
            u = stack.pop()
            for w in self.children[u]:
                if skip is not None and (u, w) == skip:
                    continue
                if w == target:
                    return True
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return False

    def _legal(self, kind: int, e: int) -> bool:
        i, j = int(self.src[e]), int(self.dst[e])
        if kind == 0:                                     # add i -> j: illegal if j already reaches i
            return not self._reaches(j, i)
        if kind == 1:                                     # delete never creates a cycle
            return True
        return not self._reaches(i, j, skip=(i, j))       # reverse: illegal if another path i ~> j remains

    @staticmethod
    def _closure(start: int, nbrs) -> set:
        """``start`` plus everything reachable from it through ``nbrs`` (parents: ancestors; children: descendants)."""
        seen, stack = {start}, [start]
        while stack:
            for w in nbrs[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    def _first_legal(self, order: np.ndarray, blocked: np.ndarray) -> int | None:
        E = len(self.src)
        for m in order:
            m = int(m)
            if self._legal(m // E, m % E):
                return m
            blocked[m] = True
        return None

    def _apply(self, kind: int, e: int) -> None:
        i, j = int(self.src[e]), int(self.dst[e])
        if kind == 0:
            self.present[e] = True
            self.parents[j].add(i); self.children[i].add(j)
        else:
            self.present[e] = False
            self.parents[j].discard(i); self.children[i].discard(j)
            if kind == 2:
                r = self.rev[e]
                self.present[r] = True
                self.parents[i].add(j); self.children[j].add(i)
        for v in ({j} if kind < 2 else {i, j}):
            self.local[v] = self.score.local(v, self.parents[v])
        for v in ({j} if kind < 2 else {i, j}):
            self._refresh_node(v)
        self._refresh(e)
        self._refresh(self.rev[e])

    def _restore(self, present: np.ndarray) -> None:
        self.present = present.copy()
        self.parents = [set() for _ in range(self.d)]
        self.children = [set() for _ in range(self.d)]
        for e in np.nonzero(self.present)[0]:
            self.parents[self.dst[e]].add(int(self.src[e])); self.children[self.src[e]].add(int(self.dst[e]))
        self.local = np.array([self.score.local(j, self.parents[j]) for j in range(self.d)])
        for e in range(len(self.src)):
            self._refresh(e)

    def run(self, max_steps: int | None = None, tol: float = 1e-9, tabu: int = 0) -> np.ndarray:
        """Climb until no single change improves the score.

        tabu > 0 turns this into tabu search (as in bnlearn's ``tabu``): at a peak, keep taking the best change even if
        it lowers the score, but never undo one of the last ``tabu`` changes; stop after ``tabu`` changes in a row
        without a new best score, and return the best graph seen.
        """
        t0 = time.perf_counter()
        steps, counts = 0, [0, 0, 0]
        E = len(self.src)
        best_score, best_present, since_best = self.local.sum(), self.present.copy(), 0
        recent: list[tuple[int, int]] = []                # (kind, edge) moves that would undo a recent change
        # Moves found to close a cycle. Adding edges only creates paths, so such a move stays illegal until an
        # edge on that cycle is deleted or reversed; remembering them avoids re-checking them at every step.
        blocked = np.zeros(3 * E, dtype=bool)
        while max_steps is None or steps < max_steps:
            gains = np.concatenate([self.d_add, self.d_del, self.d_rev])
            floor = -np.inf if tabu else tol             # tabu search may take downhill moves
            ok = (gains > floor) & ~blocked
            if tabu and recent:
                for kind, e in recent:
                    ok[kind * E + e] = False
            n_ok = int(ok.sum())
            if n_ok == 0:
                break
            cand = np.nonzero(ok)[0]
            if n_ok > 64:
                kth = np.partition(-gains[cand], 63)[63]
                cand = cand[-gains[cand] <= kth + 1e-6]
            order = cand[np.lexsort((cand, -self.priority[cand], -np.round(gains[cand], 6)))]
            chosen = self._first_legal(order, blocked)
            if chosen is None:                            # all of the best would close a cycle: scan the rest
                rest = np.nonzero(ok & ~blocked)[0]
                rest = rest[np.lexsort((rest, -self.priority[rest], -np.round(gains[rest], 6)))]
                chosen = self._first_legal(rest, blocked)
                if chosen is None:
                    break
            kind, e = divmod(chosen, E)
            if gains[chosen] <= tol and not tabu:
                break
            if kind:
                # Removing i -> j can only unblock moves whose cycle ran through it: add a -> b with a downstream of
                # j and b upstream of i, or reverse i' -> j' with i' upstream of i and j' downstream of j.
                i, j = int(self.src[e]), int(self.dst[e])
                up = np.zeros(self.d, dtype=bool); up[list(self._closure(i, self.parents))] = True
                down = np.zeros(self.d, dtype=bool); down[list(self._closure(j, self.children))] = True
            self._apply(kind, e)
            if kind:
                blocked[:E][down[self.src] & up[self.dst]] = False
                blocked[2 * E:][up[self.src] & down[self.dst]] = False
            steps += 1
            counts[kind] += 1
            if tabu:
                undo = (1, e) if kind == 0 else (0, e) if kind == 1 else (2, int(self.rev[e]))
                recent = (recent + [undo])[-tabu:]
                cur = self.local.sum()
                if cur > best_score + tol:
                    best_score, best_present, since_best = cur, self.present.copy(), 0
                else:
                    since_best += 1
                    if since_best > tabu:
                        break
            if self.verbose and steps % 500 == 0:
                log.info("  hill climbing step %d: %d edges, BIC %.1f (%.0f s)", steps, int(self.present.sum()),
                         self.local.sum(), time.perf_counter() - t0)
        if tabu and not np.array_equal(best_present, self.present):
            self._restore(best_present)
        self.steps, self.seconds = steps, time.perf_counter() - t0
        self.moves = dict(zip(("add", "delete", "reverse"), counts))
        A = np.zeros((self.d, self.d), dtype=np.int8)
        A[self.src[self.present], self.dst[self.present]] = 1
        return A


def hill_climb(score, d: int, pairs=None, max_parents: int | None = None, max_steps: int | None = None,
               tabu: int = 0, prefer: np.ndarray | None = None, verbose: bool = False) -> np.ndarray:
    """Run BIC hill climbing (or tabu search if tabu > 0) from the empty graph over ``pairs`` (all pairs if None).
    Returns a 0/1 adjacency matrix."""
    hc = HillClimber(score, d, all_pairs(d) if pairs is None else pairs, max_parents=max_parents, prefer=prefer,
                     verbose=verbose)
    A = hc.run(max_steps=max_steps, tabu=tabu)
    hill_climb.last = hc
    return A
