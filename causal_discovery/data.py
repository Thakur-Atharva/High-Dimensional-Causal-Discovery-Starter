"""Loading a table, deciding whether it is continuous or categorical, and train/test splits."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_table(path: str | Path, columns: list[str] | None = None, index_col: str | None = None) -> pd.DataFrame:
    """Read .csv, .parquet or .feather. Rows are samples, columns are variables."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path, columns=columns)
    elif suffix == ".feather":
        df = pd.read_feather(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=columns)
    if index_col is not None:
        df = df.drop(columns=[index_col])
    return df


def infer_data_type(df: pd.DataFrame, max_categories: int = 20) -> str:
    """'categorical' if every column is non-numeric or integer-valued with few distinct values, else 'continuous'."""
    def is_cat(s: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(s):
            return True
        v = s.dropna().to_numpy()
        return bool(np.all(np.mod(v, 1) == 0)) and s.nunique() <= max_categories

    flags = [is_cat(df[c]) for c in df.columns]
    if all(flags):
        return "categorical"
    if any(flags):
        log.warning("%d of %d columns look categorical but the rest are continuous; treating everything as "
                    "continuous (categorical codes are used as numbers). See README, 'Mixed data'.",
                    sum(flags), len(flags))
    return "continuous"


@dataclass
class Prepared:
    """Data ready for the methods.

    continuous: ``X`` is float64 and standardized with ``mean`` / ``std`` (computed on the rows passed to ``fit``).
    categorical: ``X`` is int codes 0..k-1 and ``categories[j]`` lists the original values of column j.
    """
    data_type: str
    names: list[str]
    X: np.ndarray
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    categories: list[np.ndarray] = field(default_factory=list)


class Preprocessor:
    """Fit on training rows, then transform any rows the same way (no test-set information leaks into training)."""

    def __init__(self, data_type: str = "auto", standardize: bool = True, max_categories: int = 20) -> None:
        self.data_type, self.standardize, self.max_categories = data_type, standardize, max_categories

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"{len(bad)} columns contain missing values (e.g. {bad[:5]}); impute or drop them first.")
        if self.data_type == "auto":
            self.data_type = infer_data_type(df, self.max_categories)
        self.names = list(map(str, df.columns))
        if self.data_type == "continuous":
            X = df.to_numpy(dtype=np.float64)
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) if self.standardize else np.ones(X.shape[1])
            const = self.std == 0
            if const.any():
                raise ValueError(f"constant columns cannot be used: {[n for n, c in zip(self.names, const) if c][:10]}")
        elif self.data_type == "categorical":
            self.categories = [np.unique(df[c].to_numpy()) for c in df.columns]
        else:
            raise ValueError("data_type must be 'auto', 'continuous' or 'categorical'")
        return self

    def transform(self, df: pd.DataFrame) -> Prepared:
        if self.data_type == "continuous":
            X = (df.to_numpy(dtype=np.float64) - self.mean) / self.std
            return Prepared("continuous", self.names, X, self.mean, self.std)
        codes = np.empty(df.shape, dtype=np.int64)
        for j, c in enumerate(df.columns):
            v = df[c].to_numpy()
            idx = np.searchsorted(self.categories[j], v)
            idx = np.clip(idx, 0, len(self.categories[j]) - 1)
            unseen = self.categories[j][idx] != v
            if unseen.any():                    # a value never seen in training: map it to the most common code
                idx[unseen] = np.bincount(idx[~unseen]).argmax() if (~unseen).any() else 0
            codes[:, j] = idx
        return Prepared("categorical", self.names, codes, categories=self.categories)


def split_rows(n: int, how: str = "random", test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train/test row indices.

    how="random": shuffle rows, hold out ``test_frac`` of them (independent samples).
    how="last":   hold out the last ``test_frac`` of rows in file order (time-ordered data: train on the past,
                  test on the future).
    """
    n_test = int(round(n * test_frac))
    if how == "random":
        perm = np.random.default_rng(seed).permutation(n)
        return np.sort(perm[n_test:]), np.sort(perm[:n_test])
    if how == "last":
        return np.arange(n - n_test), np.arange(n - n_test, n)
    raise ValueError("how must be 'random' or 'last'")
