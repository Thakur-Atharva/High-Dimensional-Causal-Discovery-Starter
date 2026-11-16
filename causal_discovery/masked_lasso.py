"""DAGMA's objective under an acyclic edge mask, solved directly (``--solver lasso``).

When a mask of allowed edges is itself acyclic (for example, edges may only go forward in time), every allowed W is a
DAG, DAGMA's acyclicity term h(W) is identically zero, and DAGMA's central path only rescales one problem:

    minimize over masked W:   0.5 * tr((I - W)^T C (I - W)) + lambda1 * |W|_1        (C = covariance)

This is d separate lasso regressions (each variable on the variables it may have as parents). DAGMA solves it with
Adam steps of fixed size; here it is solved with FISTA (accelerated proximal gradient) on all columns at once, which
reaches the same minimiser in far fewer iterations. The result is thresholded at ``w_threshold`` exactly as DAGMA's.

Each iteration is one d x d matrix product, so a GPU helps at thousands of variables.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

from .dagma_torch import default_device, is_acyclic_mask

log = logging.getLogger(__name__)


class MaskedLasso:
    """Same inputs and outputs as ``DagmaLinear`` with a mask: ``W_raw``, ``W_est``, ``seconds``, ``h_final`` (0).

    device : "cuda", "cpu" or a torch.device; defaults to the GPU when one is visible.
    dtype  : torch.float32 on GPU, torch.float64 on CPU by default.
    """

    def __init__(self, device=None, dtype: torch.dtype | None = None, verbose: bool = False) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype or (torch.float32 if self.device.type == "cuda" else torch.float64)
        self.verbose = verbose

    def fit(self, cov: np.ndarray, mask: np.ndarray, lambda1: float = 0.03, w_threshold: float = 0.3,
            max_iter: int = 20_000, check_every: int = 100, kkt_tol: float | None = None) -> np.ndarray:
        """Solve the masked problem; return W thresholded at ``w_threshold`` (W[i, j] != 0 means i -> j).

        Stops when the largest violation of the lasso optimality conditions is at most ``kkt_tol`` (default
        ``0.01 * lambda1``), checked every ``check_every`` iterations, or after ``max_iter`` iterations.
        """
        mask = np.asarray(mask) != 0
        d = mask.shape[0]
        if mask.shape != (d, d) or np.shape(cov) != (d, d):
            raise ValueError(f"cov and mask must both be d x d; got {np.shape(cov)} and {mask.shape}")
        mask = mask.copy()
        np.fill_diagonal(mask, False)
        if not is_acyclic_mask(mask):
            raise ValueError("MaskedLasso needs an acyclic mask (the allowed edges contain a cycle); use DAGMA instead")
        kkt_tol = 1e-2 * lambda1 if kkt_tol is None else kkt_tol
        t0 = time.perf_counter()
        C = torch.as_tensor(np.asarray(cov), dtype=self.dtype, device=self.device)
        M = torch.as_tensor(mask, device=self.device).to(self.dtype)
        del mask
        L = self._largest_eigenvalue(C) * 1.01
        step, thr = 1.0 / L, lambda1 / L
        W = torch.zeros_like(C)
        Y = torch.zeros_like(C)
        t = 1.0
        self.converged, k = False, 0
        for k in range(1, max_iter + 1):
            Z = C @ Y                                     # gradient at Y is C @ Y - C
            Z.sub_(C).mul_(-step).add_(Y)                 # Y - step * gradient
            W_new = torch.sign(Z).mul_(Z.abs_().sub_(thr).clamp_(min=0.0))   # soft threshold
            W_new.mul_(M)
            del Z
            # gradient-based adaptive restart (O'Donoghue & Candes 2015): drop momentum when it points uphill
            restart = bool(((Y - W_new) * (W_new - W)).sum() > 0)
            t_new = 1.0 if restart else (1 + math.sqrt(1 + 4 * t * t)) / 2
            Y = W_new if restart else W_new + ((t - 1) / t_new) * (W_new - W)
            W, t = W_new, t_new
            if k % check_every == 0 or k == max_iter:
                viol = self._kkt_violation(C, M, W, lambda1)
                if self.verbose and k % (10 * check_every) == 0:
                    log.info("  lasso iter %6d  KKT violation %.2e  (%.0f s)", k, viol, time.perf_counter() - t0)
                if viol <= kkt_tol:
                    self.converged = True
                    break
        if not self.converged:
            log.warning("MaskedLasso: not converged after %d iterations (KKT violation %.2e > %.2e)", k, viol, kkt_tol)
        self.iterations = k
        self.kkt_violation = float(viol)
        self.seconds = time.perf_counter() - t0
        self.h_final = 0.0
        self.W_raw = W.detach().cpu().numpy().astype(np.float64)
        W_est = self.W_raw.copy()
        W_est[np.abs(W_est) < w_threshold] = 0.0
        self.W_est = W_est
        log.info("MaskedLasso: %d variables, %d iterations, %.1f s, converged=%s", d, k, self.seconds, self.converged)
        return W_est

    @staticmethod
    def _largest_eigenvalue(C: torch.Tensor, iters: int = 100) -> float:
        v = torch.ones(C.shape[0], dtype=C.dtype, device=C.device) / math.sqrt(C.shape[0])
        for _ in range(iters):
            v = C @ v
            v /= torch.linalg.vector_norm(v)
        return float(v @ (C @ v))

    @staticmethod
    def _kkt_violation(C: torch.Tensor, M: torch.Tensor, W: torch.Tensor, lambda1: float) -> float:
        """Largest violation of the lasso optimality conditions over allowed entries."""
        G = C @ W - C
        active = W != 0
        viol = torch.where(active, (G + lambda1 * torch.sign(W)).abs(), (G.abs() - lambda1).clamp(min=0.0))
        return float((viol * M).max())
