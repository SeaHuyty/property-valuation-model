"""Data split and metrics shared by the neural network and the baselines,
so every model is scored on the same held-out rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


METRICS = ["rmse", "mae", "r2", "mape", "median_ape", "rmse_log"]


def evaluate(df, price, target="price_usd") -> dict:
    """regression_metrics overall and per property type (types with >= 10 rows)."""
    price = np.asarray(price, float)
    out = {"overall": regression_metrics(df[target], price)}
    for t, idx in df.groupby("property_type").groups.items():
        if len(idx) >= 10:
            out[t] = regression_metrics(df.loc[idx, target], price[idx]) | {"n": len(idx)}
    return out


def summarize(seeds, runs) -> dict:
    """mean / std / per-seed values for every split x group x metric."""
    summary = {"seeds": list(seeds)}
    for split in ("train", "val", "test"):
        groups = sorted(set().union(*(r[split].keys() for r in runs)))
        summary[split] = {}
        for g in groups:
            rows = [r[split][g] for r in runs if g in r[split]]
            summary[split][g] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                     "values": v}
                                 for k in METRICS for v in [[row[k] for row in rows]]}
    return summary


# ---- shared driver for the notebook baselines (engines/median_baseline, linear_regression, ridge_regression)
CONFIG = LAB / "configs" / "neural_network_config.yml"  # baselines reuse its data split + feature lists
LOG_CLIP = (np.log(1_000), np.log(100_000_000))       # keeps a wild extrapolation from overflowing exp()


def load_config(path=CONFIG) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def run_seeds(fit, cfg, out, seeds=None, verbose=True):
    """Fit and score a baseline on every seed's split, save it like the NN does.

    fit(train, val, cfg) -> predict, where predict(df) returns log price.
    Writes <out>/seed_<s>/{metrics.json, test_predictions.csv}, the first seed's files again in <out>/,
    and <out>/summary.json. Returns (summary, [predict per seed]).
    """
    dc = cfg["data"]
    seeds = list(seeds or dc.get("seeds") or [dc["seed"]])
    out = Path(out)
    runs, fitted = [], []
    for i, seed in enumerate(seeds):
        train, val, test = load_splits(dc["path"], dc["test_size"], dc["val_size"], seed)
        predict = fit(train, val, cfg)
        price = lambda df: np.exp(np.clip(predict(df), *LOG_CLIP))
        metrics = {s: evaluate(df, price(df), dc["target"]) for s, df in (("train", train), ("val", val), ("test", test))}
        preds = test[["property_type", "district", dc["target"]]].rename(columns={dc["target"]: "actual"})
        preds["predicted"] = price(test)
        for folder in [out / f"seed_{seed}"] + ([out] if i == 0 else []):
            folder.mkdir(parents=True, exist_ok=True)
            json.dump(metrics, open(folder / "metrics.json", "w"), indent=2)
            preds.to_csv(folder / "test_predictions.csv", index=False)
        runs.append(metrics)
        fitted.append(predict)
        if verbose:
            o = metrics["test"]["overall"]
            print(f"[seed {seed}] test MdAPE {o['median_ape']:.1f}%  RMSE log {o['rmse_log']:.3f}  R2 {o['r2']:.3f}")
    summary = summarize(seeds, runs)
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    return summary, fitted


def compare_table(seeds, models_dir=LAB / "models"):
    """Test mean +- std of every run under models/ trained on these seeds, best MdAPE first."""
    rows = {}
    for f in sorted(Path(models_dir).rglob("summary.json")):
        s = json.load(open(f))
        if s["seeds"] != list(seeds):
            continue  # other seeds = other test rows, not comparable
        o = s["test"]["overall"]
        pm = lambda k, fmt: f"{o[k]['mean']:{fmt}} ± {o[k]['std']:{fmt}}"
        rows[f.parent.relative_to(models_dir).as_posix()] = {
            "MdAPE %": pm("median_ape", ".1f"), "RMSE log": pm("rmse_log", ".3f"), "R²": pm("r2", ".3f"),
            "MAE $": pm("mae", ",.0f"), "_k": o["median_ape"]["mean"]}
    return pd.DataFrame(rows).T.sort_values("_k").drop(columns="_k")
