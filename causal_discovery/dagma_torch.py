"""DAGMA (linear model, least-squares loss) in PyTorch, so it runs on a GPU.

This is a line-by-line port of ``dagma.linear.DagmaLinear`` (dagma 1.1.1, Kevin Bello, Apache-2.0,
https://github.com/kevinsbello/dagma) from NumPy/SciPy to PyTorch. The algorithm, defaults and stopping rules are the
same; see NOTICE. Two practical changes:

* ``fit`` accepts either the data matrix or its covariance. The least-squares loss only needs X^T X / n, so the
  100,000 x 2,500 data matrix never has to sit on the GPU.
* The raw (unthresholded) weight matrix is kept in ``W_raw``. The hybrid method uses it to propose candidate edges.

Each iteration inverts a d x d matrix, so the cost per iteration is O(d^3). At d = 2,500 that is a few milliseconds on
a modern GPU and roughly a second on a CPU, hence the GPU recommendation.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_acyclic_mask(mask: np.ndarray) -> bool:
    """True if the directed graph of allowed edges (``mask[i, j]`` = i -> j may exist) has no cycle.

    Kahn's algorithm, vectorized: repeatedly remove every node with no remaining incoming edge.
    """
    A = np.asarray(mask, dtype=bool)
    indeg = A.sum(axis=0).astype(np.int64)
    alive = np.ones(A.shape[0], dtype=bool)
    while alive.any():
        free = np.flatnonzero(alive & (indeg == 0))
        if free.size == 0:
            return False
        alive[free] = False
        indeg -= A[free].sum(axis=0)
    return True


class DagmaLinear:
    """DAGMA with a linear structural equation model and least-squares loss.

    Parameters
    ----------
    device : "cuda", "cpu" or a torch.device. Defaults to the GPU when one is visible.
    dtype : torch.float32 (default on GPU; much faster) or torch.float64 (default on CPU; the original's precision).
    verbose : log the objective every ``checkpoint`` iterations.
    """

    def __init__(self, device=None, dtype: torch.dtype | None = None, verbose: bool = False) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype or (torch.float32 if self.device.type == "cuda" else torch.float64)
        self.verbose = verbose
        # Entries of the inverse of an M-matrix are >= 0. In float32, exact zeros can come out as tiny negatives, so
        # the out-of-domain test allows a small tolerance (the original float64 code allows none).
        self._neg_tol = 1e-6 if self.dtype == torch.float32 else 0.0

    # ----- pieces of the objective -------------------------------------------------------------------------------
    def _score(self, W: torch.Tensor) -> float:
        if self.Id is None:                     # skip path: no dense identity; 0.5 tr((I-W)^T C (I-W)) expanded
            CW = self.cov @ W
            return 0.5 * (torch.trace(self.cov) - 2 * torch.trace(CW) + (W * CW).sum()).item()
        dif = self.Id - W
        return 0.5 * torch.trace(dif.T @ (self.cov @ dif)).item()

    def _h(self, W: torch.Tensor, s: float = 1.0) -> float:
        if self.Id is None:
            M = -(W * W)
            M.diagonal().add_(s)
        else:
            M = s * self.Id - W * W
        return (-torch.linalg.slogdet(M)[1] + self.d * math.log(s)).item()

    def _func(self, W: torch.Tensor, mu: float, s: float = 1.0):
        score = self._score(W)
        h = 0.0 if self.skipped_acyclicity else self._h(W, s)
        return mu * (score + self.lambda1 * W.abs().sum().item()) + h, score, h

    # ----- inner solver: argmin_W  mu * (score + lambda1 |W|_1) + h(W), by Adam ---------------------------------
    def _minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999):
        obj_prev = 1e16
        opt_m = torch.zeros_like(W)
        opt_v = torch.zeros_like(W)
        grad = torch.zeros_like(W)
        log.debug("minimize: mu=%g lr=%g s=%g lambda1=%g max_iter=%d", mu, lr, s, self.lambda1, max_iter)
        for it in range(1, max_iter + 1):
            if self.skipped_acyclicity:
                # The mask alone makes every feasible W a DAG: h(W) = 0 and its gradient is 0 on every allowed entry,
                # and sI - W o W is always an M-matrix. So only the score and the L1 term move W. Same Adam update as
                # below, written in place so that large d fits in GPU memory.
                G = self.cov @ W
                G.sub_(self.cov).mul_(mu).add_(torch.sign(W), alpha=mu * self.lambda1)
                opt_m.mul_(beta_1).add_(G, alpha=1 - beta_1)
                opt_v.mul_(beta_2).addcmul_(G, G, value=1 - beta_2)
                torch.sqrt(opt_v, out=G)
                G.div_(math.sqrt(1 - beta_2 ** it)).add_(1e-8)      # sqrt(v_hat) + eps
                torch.div(opt_m, G, out=G)
                W.sub_(G, alpha=lr / (1 - beta_1 ** it))           # m_hat / (sqrt(v_hat) + eps)
                W.mul_(self.mask_exc)
                del G
                if it % self.checkpoint == 0 or it == max_iter:
                    obj_new, score, h = self._func(W, mu, s)
                    if self.verbose and it % (10 * self.checkpoint) == 0:
                        log.info("  iter %6d  score %.5e  obj %.5e  (%.0f s)", it, score, obj_new,
                                 time.perf_counter() - self._t0)
                    if abs((obj_prev - obj_new) / obj_prev) <= tol:
                        break
                    obj_prev = obj_new
                continue
            M = torch.linalg.inv(s * self.Id - W * W) + 1e-16
            while bool((M < -self._neg_tol).any()):           # sI - W o W is no longer an M-matrix
                if it == 1 or s <= 0.9:
                    log.debug("W left the domain for s=%g at iteration %d", s, it)
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                M = torch.linalg.inv(s * self.Id - W * W) + 1e-16
                log.debug("learning rate decreased to %g", lr)
            G_score = -mu * (self.cov @ (self.Id - W))
            Gobj = G_score + mu * self.lambda1 * torch.sign(W) + 2 * W * M.T
            opt_m = opt_m * beta_1 + (1 - beta_1) * Gobj        # Adam
            opt_v = opt_v * beta_2 + (1 - beta_2) * Gobj * Gobj
            m_hat = opt_m / (1 - beta_1 ** it)
            v_hat = opt_v / (1 - beta_2 ** it)
            grad = m_hat / (torch.sqrt(v_hat) + 1e-8)
            W -= lr * grad
            W *= self.mask_exc
            if it % self.checkpoint == 0 or it == max_iter:
                obj_new, score, h = self._func(W, mu, s)
                if self.verbose and it % (10 * self.checkpoint) == 0:
                    log.info("  iter %6d  score %.5e  h %.3e  obj %.5e  (%.0f s)", it, score, h, obj_new,
                             time.perf_counter() - self._t0)
                if abs((obj_prev - obj_new) / obj_prev) <= tol:
                    break
                obj_prev = obj_new
        return W, True

    # ----- public API ----------------------------------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray | None = None,
        cov: np.ndarray | None = None,
        lambda1: float = 0.03,
        w_threshold: float = 0.3,
        T: int = 5,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        s: Sequence[float] | float = (1.0, 0.9, 0.8, 0.7, 0.6),
        warm_iter: int = 30_000,
        max_iter: int = 60_000,
        lr: float = 3e-4,
        checkpoint: int = 1000,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        exclude_edges: Sequence[tuple[int, int]] | None = None,
        mask: np.ndarray | None = None,
        skip_acyclicity: bool | str = "auto",
    ) -> np.ndarray:
        """Fit DAGMA and return the thresholded weighted adjacency matrix (W[i, j] != 0 means i -> j).

        Pass either ``X`` (n x d data; it is centred here) or ``cov`` (d x d matrix X^T X / n of centred data).
        Defaults are those of dagma 1.1.1. The unthresholded result is stored in ``self.W_raw``.

        mask : optional d x d 0/1 array of *allowed* edges (``mask[i, j]`` = 1 means i -> j may appear). Every other
            entry of W is held at zero throughout, e.g. to forbid edges that go backwards in time.
        skip_acyclicity : if the mask alone rules out cycles, DAGMA's acyclicity term h(W) and its gradient are exactly
            zero for every allowed W, so computing them changes nothing but costs a d x d inversion per iteration.
            "auto" (default) skips them exactly when the mask is acyclic; True requires an acyclic mask; False never
            skips. Without a mask nothing is skipped. ``self.skipped_acyclicity`` records which path ran.
        """
        if (X is None) == (cov is None):
            raise ValueError("pass exactly one of X or cov")
        if cov is None:
            Xt = torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)
            Xt = Xt - Xt.mean(dim=0, keepdim=True)
            cov_t = Xt.T @ Xt / Xt.shape[0]
            del Xt
        else:
            cov_t = torch.as_tensor(np.asarray(cov), dtype=self.dtype, device=self.device)
        self.cov, self.lambda1, self.checkpoint = cov_t, lambda1, checkpoint
        self.d = cov_t.shape[0]
        skip = False
        if mask is not None:
            mask = np.asarray(mask) != 0
            if mask.shape != (self.d, self.d):
                raise ValueError(f"mask must be {self.d} x {self.d}, got {mask.shape}")
            mask = mask.copy()
            np.fill_diagonal(mask, False)
            if exclude_edges:
                r, c = zip(*exclude_edges)
                mask[list(r), list(c)] = False
            if skip_acyclicity == "auto":
                skip = is_acyclic_mask(mask)
            elif skip_acyclicity is True:
                if not is_acyclic_mask(mask):
                    raise ValueError("skip_acyclicity=True needs an acyclic mask, but the allowed edges contain a cycle")
                skip = True
            elif skip_acyclicity is not False:
                raise ValueError("skip_acyclicity must be 'auto', True or False")
            self.mask_exc = torch.as_tensor(mask, device=self.device).to(self.dtype)
            del mask
        else:
            self.mask_exc = torch.ones(self.d, self.d, dtype=self.dtype, device=self.device)
            if exclude_edges:
                r, c = zip(*exclude_edges)
                self.mask_exc[list(r), list(c)] = 0.0
        self.skipped_acyclicity = skip
        self.Id = None if skip else torch.eye(self.d, dtype=self.dtype, device=self.device)
        log.info("DAGMA: %s", "mask is acyclic, so the acyclicity term is skipped (it is identically zero)" if skip
                 else ("masked, full acyclicity term" if self.mask_exc.min() == 0 else "no mask"))
        s = list(s) if isinstance(s, (list, tuple)) else [float(s)] * T
        if len(s) < T:
            s = s + (T - len(s)) * [s[-1]]
        W = torch.zeros(self.d, self.d, dtype=self.dtype, device=self.device)
        mu = mu_init
        self._t0 = time.perf_counter()
        for i in range(int(T)):
            lr_adam, success = lr, False
            inner = int(max_iter) if i == T - 1 else int(warm_iter)
            if self.verbose:
                log.info("DAGMA outer iteration %d/%d (mu=%g, s=%g)", i + 1, T, mu, s[i])
            while not success:                    # the skip path never leaves the domain, so it can update W in place
                W_temp, success = self._minimize(W if skip else W.clone(), mu, inner, s[i], lr=lr_adam,
                                                 beta_1=beta_1, beta_2=beta_2)
                if not success:
                    log.debug("retrying with larger s")
                    lr_adam *= 0.5
                    s[i] += 0.1
            W = W_temp
            mu *= mu_factor
        self.h_final = self._h(W)
        self.score_final = self._score(W)
        self.seconds = time.perf_counter() - self._t0
        self.W_raw = W.detach().cpu().numpy().astype(np.float64)
        W_est = self.W_raw.copy()
        W_est[np.abs(W_est) < w_threshold] = 0.0
        self.W_est = W_est
        return W_est
