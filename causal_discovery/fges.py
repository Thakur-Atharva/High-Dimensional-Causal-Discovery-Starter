"""fGES (Fast Greedy Equivalence Search) from CMU's Tetrad, called through py-tetrad.

fGES searches over *Markov equivalence classes* (groups of DAGs that imply the same independences) instead of single
DAGs. That avoids the arbitrary direction choices that make plain hill climbing fragile.

It needs Java 21+ and the py-tetrad package (see README).
- **Input to Tetrad:** the covariance matrix, not the raw table. The linear-Gaussian BIC only needs the covariance,
  and copying 100,000 x 2,500 values into Java one by one is very slow.
- **Separate process:** Tetrad runs in a child process, so the Java virtual machine never shares a process with
  NumPy's linear-algebra libraries (we saw intermittent crashes when they did), and its memory is released when the
  search ends.

Output: fGES returns a CPDAG, an equivalence class in which some edges are left undirected because the data cannot
orient them. ``fges`` returns that CPDAG and also one DAG from the class, needed to fit and score the model. All DAGs
in a class have the same likelihood, so which one is chosen does not affect the benchmark.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import numpy as np

from .scores import resolve_penalty

log = logging.getLogger(__name__)


def _search(S: np.ndarray, n: int, penalty: float, pairs, max_degree: int, threads: int, max_heap: str,
            faithfulness: bool, verbose: bool):
    """Runs inside the child process: start Java, run fGES, return the CPDAG and one DAG as 0/1 matrices."""
    import importlib.resources as ir
    import jpype
    import jpype.imports  # noqa: F401  (enables "from edu.cmu... import ...")
    jar = str(ir.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
    jpype.startJVM(f"-Xmx{max_heap}", "--enable-native-access=ALL-UNNAMED", classpath=[jar])
    import java.util as ju
    from edu.cmu.tetrad.data import ContinuousVariable, CovarianceMatrix
    from edu.cmu.tetrad.graph import EdgeListGraph, Edges, GraphTransforms
    from edu.cmu.tetrad.search import Fges
    from edu.cmu.tetrad.search.score import SemBicScore

    d = S.shape[0]
    names = [f"V{j}" for j in range(d)]               # Tetrad-safe names; mapped back by position
    nodes = ju.ArrayList()
    for name in names:
        nodes.add(ContinuousVariable(name))
    score = SemBicScore(CovarianceMatrix(nodes, jpype.JArray(jpype.JDouble, 2)(S.tolist()), n))
    score.setPenaltyDiscount(float(penalty))
    alg = Fges(score)
    alg.setFaithfulnessAssumed(faithfulness)
    if max_degree > 0:
        alg.setMaxDegree(int(max_degree))
    alg.setNumThreads(int(threads))
    alg.setVerbose(bool(verbose))
    if pairs is not None:
        bound = EdgeListGraph(nodes)
        for a, b in pairs:
            bound.addUndirectedEdge(nodes.get(int(a)), nodes.get(int(b)))
        alg.setBoundGraph(bound)
    t0 = time.perf_counter()
    cpdag = alg.search()
    seconds = time.perf_counter() - t0
    dag = GraphTransforms.dagFromCpdag(cpdag)
    index = {name: j for j, name in enumerate(names)}

    def to_matrices(g):
        directed = np.zeros((d, d), dtype=np.int8)
        undirected = np.zeros((d, d), dtype=np.int8)
        for e in g.getEdges():
            a, b = index[str(e.getNode1().getName())], index[str(e.getNode2().getName())]
            if Edges.isDirectedEdge(e):
                tail = index[str(Edges.getDirectedEdgeTail(e).getName())]
                directed[tail, b if tail == a else a] = 1
            else:
                undirected[a, b] = undirected[b, a] = 1
        return directed, undirected

    cp_dir, cp_und = to_matrices(cpdag)
    A, und = to_matrices(dag)
    return A, und, cp_dir, cp_und, seconds


def fges(prep, penalty: float = 1.0, min_partial_corr: float | None = None, max_degree: int = -1,
         threads: int | None = None, max_heap: str = "16g", bound_k: int | None = None,
         faithfulness: bool = True, verbose: bool = False):
    """Run fGES with the linear-Gaussian BIC (Tetrad's SEM-BIC) on standardized continuous data.

    penalty    : BIC penalty multiplier ("penalty discount"); 1 is the textbook BIC. Larger gives sparser graphs.
    min_partial_corr : alternative to ``penalty``: keep only edges with partial correlation above about this value.
    max_degree : cap on the number of neighbours per variable (-1: none). Lowering it bounds runtime.
    threads    : CPU threads (default: the cores this process may use).
    max_heap   : Java memory limit, e.g. "16g"; use "64g" at thousands of variables.
    bound_k    : if > 0, only pairs among each variable's bound_k largest partial correlations may get an edge
                 (Tetrad's "bound graph"). With many rows, fGES otherwise starts from nearly every pair of variables,
                 which is slow above a few hundred variables (see README). None (default): 20 from 500 variables
                 up, otherwise no bound.
    """
    if prep.data_type != "continuous":
        raise ValueError("fGES here uses the linear-Gaussian score; for categorical data see README ('fGES on "
                         "categorical data').")
    X = prep.X
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    S = np.ascontiguousarray(Xc.T @ Xc / n)
    del Xc
    penalty = resolve_penalty(penalty, min_partial_corr, n)
    pairs = None
    if bound_k is None:
        bound_k = 20 if X.shape[1] >= 500 else 0
    if bound_k > 0:
        from .hill_climb import candidate_pairs, partial_correlations
        pairs = candidate_pairs(partial_correlations(X), k=bound_k)
        log.info("fGES: search bounded to %d candidate pairs (top %d partial correlations per variable)",
                 len(pairs), bound_k)
    if not threads:                                   # cores this process may use (respects Slurm/cgroups)
        threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    with mp.get_context("spawn").Pool(1) as pool:
        A, und, cp_dir, cp_und, seconds = pool.apply(
            _search, (S, n, penalty, pairs, max_degree, threads, max_heap, faithfulness, verbose))
    if und.any():
        raise RuntimeError("Tetrad's DAG extension left undirected edges")
    log.info("fGES: %d edges (%d directed, %d undirected in the CPDAG) in %.1f s", int(A.sum()), int(cp_dir.sum()),
             int(cp_und.sum() // 2), seconds)
    return A, {"seconds": seconds, "cpdag_directed": int(cp_dir.sum()), "cpdag_undirected": int(cp_und.sum() // 2),
               "cpdag": (cp_dir, cp_und), "penalty": penalty, "bound_k": bound_k}
