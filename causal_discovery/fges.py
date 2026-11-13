"""fGES (Fast Greedy Equivalence Search) from CMU's Tetrad, called through py-tetrad.

fGES searches over *Markov equivalence classes* (groups of DAGs that imply the same independences) instead of single
DAGs. That avoids the arbitrary direction choices that make plain hill climbing fragile, and it is the
best-documented method at this scale (Ramsey et al. 2017: 30,000 variables in minutes, 1,000,000 in hours).

It needs Java 21+ and the py-tetrad package (see README). We pass Tetrad the correlation matrix rather than the raw
table: the linear-Gaussian BIC only needs the covariance, and copying 100,000 x 2,500 values into Java one by one is
very slow.

Output: fGES returns a CPDAG, an equivalence class in which some edges are left undirected because the data cannot
orient them. ``fges`` returns that CPDAG and also one DAG from the class, needed to fit and score the model. All DAGs
in a class have the same likelihood, so which one is chosen does not affect the benchmark.
"""
from __future__ import annotations

import logging
import math
import os
import time

import numpy as np

log = logging.getLogger(__name__)
_JVM_STARTED = False


def start_jvm(max_heap: str = "16g") -> None:
    """Start the Java virtual machine with the Tetrad jar (once per Python process)."""
    global _JVM_STARTED
    import importlib.resources as ir
    import jpype
    import jpype.imports  # noqa: F401  (enables "from edu.cmu... import ...")
    if jpype.isJVMStarted():
        _JVM_STARTED = True
        return
    jar = str(ir.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
    jpype.startJVM(f"-Xmx{max_heap}", "--enable-native-access=ALL-UNNAMED", classpath=[jar])
    _JVM_STARTED = True


def penalty_for_min_partial_corr(r: float, n: int) -> float:
    """Penalty discount c such that BIC adds an edge only if its partial correlation exceeds about r.

    An edge is accepted iff  -n log(1 - r^2) > c log(n).
    """
    return -n * math.log(1 - r * r) / math.log(n)


def fges(prep, penalty: float = 1.0, max_degree: int = -1, threads: int | None = None, max_heap: str = "16g",
         faithfulness: bool = True, verbose: bool = False):
    """Run fGES with the linear-Gaussian BIC (Tetrad's SEM-BIC) on standardized continuous data.

    penalty    : BIC penalty multiplier ("penalty discount"); 1 is the textbook BIC. Larger gives sparser graphs.
    max_degree : cap on the number of neighbours per variable (-1: none). Lowering it bounds runtime.
    threads    : CPU threads (default: all cores).
    """
    if prep.data_type != "continuous":
        raise ValueError("fGES here uses the linear-Gaussian score; for categorical data see README ('fGES on "
                         "categorical data').")
    start_jvm(max_heap)
    import jpype
    import java.util as ju
    from edu.cmu.tetrad.data import ContinuousVariable, CovarianceMatrix
    from edu.cmu.tetrad.graph import Edges, GraphTransforms
    from edu.cmu.tetrad.search import Fges
    from edu.cmu.tetrad.search.score import SemBicScore

    X = prep.X
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    S = np.ascontiguousarray(Xc.T @ Xc / n)
    names = [f"V{j}" for j in range(d)]               # Tetrad-safe names; mapped back by position
    nodes = ju.ArrayList()
    for name in names:
        nodes.add(ContinuousVariable(name))
    cov = CovarianceMatrix(nodes, jpype.JArray(jpype.JDouble, 2)(S.tolist()), n)
    score = SemBicScore(cov)
    score.setPenaltyDiscount(float(penalty))
    alg = Fges(score)
    alg.setFaithfulnessAssumed(faithfulness)
    if max_degree > 0:
        alg.setMaxDegree(int(max_degree))
    if not threads:                                   # cores this process may use (respects Slurm/cgroups)
        threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    alg.setNumThreads(int(threads))
    alg.setVerbose(bool(verbose))
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
    if und.any():
        raise RuntimeError("Tetrad's DAG extension left undirected edges")
    log.info("fGES: %d edges (%d directed, %d undirected in the CPDAG) in %.1f s", int(A.sum()), int(cp_dir.sum()),
             int(cp_und.sum() // 2), seconds)
    return A, {"seconds": seconds, "cpdag_directed": int(cp_dir.sum()), "cpdag_undirected": int(cp_und.sum() // 2),
               "cpdag": (cp_dir, cp_und), "penalty": penalty}
