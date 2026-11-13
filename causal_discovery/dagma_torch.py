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
        dif = self.Id - W
        return 0.5 * torch.trace(dif.T @ (self.cov @ dif)).item()

    def _h(self, W: torch.Tensor, s: float = 1.0) -> float:
        M = s * self.Id - W * W
        return (-torch.linalg.slogdet(M)[1] + self.d * math.log(s)).item()

    def _func(self, W: torch.Tensor, mu: float, s: float = 1.0):
        score, h = self._score(W), self._h(W, s)
        return mu * (score + self.lambda1 * W.abs().sum().item()) + h, score, h

    # ----- inner solver: argmin_W  mu * (score + lambda1 |W|_1) + h(W), by Adam ---------------------------------
    def _minimize(self, W, mu, max_iter, s, lr, tol=1e-6, beta_1=0.99, beta_2=0.999):
        obj_prev = 1e16
        opt_m = torch.zeros_like(W)
        opt_v = torch.zeros_like(W)
        grad = torch.zeros_like(W)
        log.debug("minimize: mu=%g lr=%g s=%g lambda1=%g max_iter=%d", mu, lr, s, self.lambda1, max_iter)
        for it in range(1, max_iter + 1):
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
    ) -> np.ndarray:
        """Fit DAGMA and return the thresholded weighted adjacency matrix (W[i, j] != 0 means i -> j).

        Pass either ``X`` (n x d data; it is centred here) or ``cov`` (d x d matrix X^T X / n of centred data).
        Defaults are those of dagma 1.1.1. The unthresholded result is stored in ``self.W_raw``.
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
        self.Id = torch.eye(self.d, dtype=self.dtype, device=self.device)
        self.mask_exc = torch.ones(self.d, self.d, dtype=self.dtype, device=self.device)
        if exclude_edges:
            r, c = zip(*exclude_edges)
            self.mask_exc[list(r), list(c)] = 0.0
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
            while not success:
                W_temp, success = self._minimize(W.clone(), mu, inner, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2)
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
