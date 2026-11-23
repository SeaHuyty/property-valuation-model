"""Feature preprocessing for the price MLP.

Fit on the training split only, then reuse the saved state for validation, test and
inference, so no statistics leak from held-out rows.

  categorical  -> integer ids (0 = "other": rare in training or unseen, 1.. = known values)
  numeric      -> optional log1p, then standardized with training mean / std
  binary       -> float 0/1, unchanged
  target       -> log(price_usd)

`commune` is keyed as "district|commune" because the same commune name appears in
several districts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OTHER = 0


def _cat_values(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype("string").fillna("Unknown").str.strip()
    if col == "commune" and "district" in df:
        s = df["district"].astype("string").fillna("Unknown").str.strip() + "|" + s
    return s


class Preprocessor:
    def __init__(self, categorical, numeric, binary, log=(), min_count=5, max_embedding_dim=16):
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.binary = list(binary)
        self.log = set(log)
        self.min_count = min_count
        self.max_embedding_dim = max_embedding_dim
        self.vocab: dict[str, dict[str, int]] = {}
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    # ---- fitting
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for col in self.categorical:
            counts = _cat_values(df, col).value_counts()
            keep = counts[counts >= self.min_count].index
            self.vocab[col] = {v: i + 1 for i, v in enumerate(sorted(keep))}
        for col in self.numeric:
            x = self._num(df, col)
            self.mean[col] = float(x.mean())
            self.std[col] = float(x.std()) or 1.0
        return self

    def _num(self, df, col):
        x = df[col].astype(float).clip(lower=0)
        return np.log1p(x) if col in self.log else x

    # ---- transforming
    def transform(self, df: pd.DataFrame):
        """Return (cat int64 [n, n_cat], num float32 [n, n_num + n_bin])."""
        cats = []
        for col in self.categorical:
            v = _cat_values(df, col)
            cats.append(v.map(self.vocab[col]).fillna(OTHER).astype(int).to_numpy())
        cat = np.stack(cats, axis=1).astype(np.int64) if cats else np.zeros((len(df), 0), np.int64)

        nums = [((self._num(df, c) - self.mean[c]) / self.std[c]).to_numpy() for c in self.numeric]
        nums += [df[c].astype(float).to_numpy() for c in self.binary]
        num = np.stack(nums, axis=1).astype(np.float32)
        return cat, num

    @staticmethod
    def target(price):
        return np.log(np.asarray(price, dtype=float)).astype(np.float32)

    @staticmethod
    def inverse_target(y):
        return np.exp(np.asarray(y, dtype=float))

    # ---- shape info for the network
    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.categorical]

    @property
    def embedding_dims(self) -> list[int]:
        # fastai rule of thumb, capped
        return [min(self.max_embedding_dim, round(1.6 * n ** 0.56)) for n in self.cardinalities]

    @property
    def n_numeric(self) -> int:
        return len(self.numeric) + len(self.binary)

    # ---- persistence
    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("categorical", "numeric", "binary", "min_count", "max_embedding_dim",
                 "vocab", "mean", "std")} | {"log": sorted(self.log)}

    @classmethod
    def from_state(cls, s: dict) -> "Preprocessor":
        p = cls(s["categorical"], s["numeric"], s["binary"], s["log"], s["min_count"], s["max_embedding_dim"])
        p.vocab, p.mean, p.std = s["vocab"], s["mean"], s["std"]
        return p
