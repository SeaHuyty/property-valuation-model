"""Data split and metrics shared by the neural network and the baselines,
so every model is scored on the same held-out rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LAB = Path(__file__).resolve().parents[1]


def load_splits(path, test_size=0.2, val_size=0.1, seed=42):
    """Return train, val, test DataFrames, stratified by property_type.
    val is carved out of the training part (used only for early stopping / tuning)."""
    path = Path(path)
    d = pd.read_csv(path if path.is_absolute() else LAB / path, encoding="utf-8-sig", low_memory=False)
    train, test = train_test_split(d, test_size=test_size, random_state=seed, stratify=d["property_type"])
    train, val = train_test_split(train, test_size=val_size, random_state=seed, stratify=train["property_type"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def regression_metrics(y_true, y_pred) -> dict:
    """Metrics in USD (RMSE, MAE, R2, MAPE) plus RMSE on log price."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "mape": float(np.mean(np.abs(err) / y) * 100),
        "median_ape": float(np.median(np.abs(err) / y) * 100),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(y)) ** 2))),
    }
