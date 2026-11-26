"""Train the price MLP.

Usage (from laboratory/):
    python -m engines.neural_network.train                      # seeds from config (data.seeds)
    python -m engines.neural_network.train --seeds 42           # single quick run
    python -m engines.neural_network.train --seeds 42 7 13      # repeated hold-out
    python -m engines.neural_network.train --name more_dropout  # save to <output_dir>/more_dropout/

Each seed gives a different train/val/test split *and* a different weight init, so
one run's score is noisy (test R2 swings ~0.15 between splits). With several seeds the
script reports mean +- std, which is what model comparisons should use.

Writes to <output_dir>/ (default models/neural_network/), or <output_dir>/<name>/ with --name
so experiments don't overwrite each other:
    model.pt              model from the first seed (what inference loads)
    metrics.json          metrics of the first seed
    history.csv           loss per epoch of the first seed
    test_predictions.csv  test rows of the first seed: actual vs predicted price
    seed_<s>/             the four files above per seed (only with >1 seed)
    summary.json          mean / std / per-seed values of each metric (only with >1 seed)

notebooks/compare_models.ipynb reads these folders and plots runs side by side.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from engines.common import LAB, evaluate as score, load_splits, summarize
from engines.neural_network.networks import PriceMLP
from engines.neural_network.preprocessing import Preprocessor


def tensors(pre, df, target):
    cat, num = pre.transform(df)
    return torch.from_numpy(cat), torch.from_numpy(num), torch.from_numpy(pre.target(df[target]))


@torch.no_grad()
def predict_log(model, cat, num, batch=4096):
    model.eval()
    return torch.cat([model(cat[i:i + batch], num[i:i + batch]) for i in range(0, len(cat), batch)]).numpy()


def evaluate(model, pre, df, cat, num, target):
    return score(df, pre.inverse_target(predict_log(model, cat, num)), target)


def train_one(cfg, seed):
    """Train on the split given by `seed`. Returns (checkpoint dict, metrics, history, test predictions)."""
    dc, fc, mc, tc = cfg["data"], cfg["features"], cfg["model"], cfg["train"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df, val_df, test_df = load_splits(dc["path"], dc["test_size"], dc["val_size"], seed)
    pre = Preprocessor(fc["categorical"], fc["numeric"], fc["binary"], fc.get("log", []),
                       fc["min_count"], fc["max_embedding_dim"]).fit(train_df)
    tr, va, te = (tensors(pre, df, dc["target"]) for df in (train_df, val_df, test_df))

    model_args = {"cardinalities": pre.cardinalities, "embedding_dims": pre.embedding_dims,
                  "n_numeric": pre.n_numeric, "numeric_hidden": mc["numeric_hidden"],
                  "hidden": mc["hidden"], "dropout": mc["dropout"],
                  "embedding_dropout": mc["embedding_dropout"], "activation": mc["activation"]}
    model = PriceMLP(**model_args, target_mean=float(tr[2].mean()))
    tqdm.write(f"[seed {seed}] rows train {len(train_df)}  val {len(val_df)}  test {len(test_df)}  "
               f"params {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.HuberLoss(delta=tc["huber_delta"]) if tc["loss"] == "huber" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tc["lr_factor"], patience=tc["lr_patience"])
    # drop_last: a final batch of 1 row would break BatchNorm
    loader = DataLoader(TensorDataset(*tr), batch_size=tc["batch_size"], shuffle=True, drop_last=True)

    best, best_val, best_epoch, wait, history = None, float("inf"), 0, 0, []
    t0 = time.time()
    bar = tqdm(range(1, tc["epochs"] + 1), desc=f"seed {seed}", unit="ep", dynamic_ncols=True)
    for epoch in bar:
        model.train()
        total = 0.0
        for cat, num, y in tqdm(loader, desc=f"epoch {epoch}", leave=False, unit="batch", dynamic_ncols=True):
            opt.zero_grad()
            loss = loss_fn(model(cat, num), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / (len(loader) * tc["batch_size"])
        val_pred = torch.from_numpy(predict_log(model, va[0], va[1]))
        val_loss = loss_fn(val_pred, va[2]).item()
        val_rmse_log = float(torch.sqrt(torch.mean((val_pred - va[2]) ** 2)))
        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_rmse_log": val_rmse_log, "lr": lr})

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, best, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", rmse_log=f"{val_rmse_log:.3f}",
                        best=f"{best_val:.4f}@{best_epoch}", lr=f"{lr:.0e}")
        if wait >= tc["patience"]:
            break
    bar.close()
    tqdm.write(f"[seed {seed}] stopped at epoch {epoch}, best val {best_val:.4f} at epoch {best_epoch} "
               f"({time.time() - t0:.0f}s)")

    model.load_state_dict(best)
    metrics = {name: evaluate(model, pre, df, t[0], t[1], dc["target"])
               for name, df, t in (("train", train_df, tr), ("val", val_df, va), ("test", test_df, te))}
    ckpt = {"state_dict": best, "preprocessor": pre.state_dict(), "config": cfg | {"seed": seed},
            "model_args": model_args}
    preds = test_df[["property_type", "district", dc["target"]]].rename(columns={dc["target"]: "actual"})
    preds["predicted"] = pre.inverse_target(predict_log(model, te[0], te[1]))
    return ckpt, metrics, history, preds


def save_run(out, ckpt, metrics, history, preds):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out / "model.pt")
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    preds.to_csv(out / "test_predictions.csv", index=False)


def print_run(metrics):
    print(f"\n{'split':<6}{'RMSE $':>12}{'MAE $':>12}{'R2':>8}{'MAPE %':>9}{'MdAPE %':>9}{'RMSE log':>10}")
    for name, m in metrics.items():
        o = m["overall"]
        print(f"{name:<6}{o['rmse']:>12,.0f}{o['mae']:>12,.0f}{o['r2']:>8.3f}{o['mape']:>9.1f}"
              f"{o['median_ape']:>9.1f}{o['rmse_log']:>10.3f}")
    print("\ntest by property type")
    for t, m in metrics["test"].items():
        if t != "overall":
            print(f"  {t:<11} n={m['n']:>5}  MAE ${m['mae']:>10,.0f}  MdAPE {m['median_ape']:>5.1f}%  R2 {m['r2']:.3f}")


def print_summary(summary):
    n = len(summary["seeds"])
    print(f"\n=== mean +- std over {n} seeds {summary['seeds']} ===")
    print(f"{'split':<6}{'RMSE $':>20}{'MAE $':>20}{'R2':>15}{'MdAPE %':>14}{'RMSE log':>16}")
    for split in ("train", "val", "test"):
        o = summary[split]["overall"]
        f = lambda k, fmt: f"{o[k]['mean']:{fmt}} +- {o[k]['std']:{fmt}}"
        print(f"{split:<6}{f('rmse', ',.0f'):>20}{f('mae', ',.0f'):>20}{f('r2', '.3f'):>15}"
              f"{f('median_ape', '.1f'):>14}{f('rmse_log', '.3f'):>16}")
    print("\ntest by property type")
    for t, m in summary["test"].items():
        if t != "overall":
            print(f"  {t:<11} MdAPE {m['median_ape']['mean']:>5.1f} +- {m['median_ape']['std']:<4.1f}%  "
                  f"RMSE log {m['rmse_log']['mean']:.3f} +- {m['rmse_log']['std']:.3f}  "
                  f"R2 {m['r2']['mean']:.3f} +- {m['r2']['std']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(LAB / "configs" / "neural_network_config.yml"))
    ap.add_argument("--seeds", type=int, nargs="+", help="override data.seeds from the config")
    ap.add_argument("--name", help="experiment name: save to <output_dir>/<name>/ instead of <output_dir>/")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seeds = args.seeds or cfg["data"].get("seeds") or [cfg["data"]["seed"]]

    out = LAB / cfg["output_dir"]
    if args.name:
        out = out / args.name
    runs = []
    for i, seed in enumerate(seeds):
        ckpt, metrics, history, preds = train_one(cfg, seed)
        runs.append(metrics)
        if len(seeds) > 1:
            save_run(out / f"seed_{seed}", ckpt, metrics, history, preds)
        if i == 0:  # first seed's model is the one inference loads
            save_run(out, ckpt, metrics, history, preds)
        if len(seeds) == 1:
            print_run(metrics)

    if len(seeds) > 1:
        summary = summarize(seeds, runs)
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
        print_summary(summary)
    else:
        (out / "summary.json").unlink(missing_ok=True)  # don't leave a stale multi-seed summary
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
