# Model comparison notebook, NN base results

- **Date:** 2026-11-23 23:12
- **Commit:** `e1b6af7`
- **Author:** SeaHuyty

## What
- `train.py` runs all seeds (42, 7, 13, 21, 99) and saves per-seed and summary results.
- First NN results saved under `models/neural_network/base/` (on `data.csv`).
- `compare_models.ipynb` to put runs side by side.

## Why / result
NN base (test, mean of 5 seeds): MdAPE 23.0%, RMSE(log) 0.455, R² 0.683, MAE $121k.

## Files
- `laboratory/engines/neural_network/train.py`
- `laboratory/models/neural_network/base/`
- `laboratory/notebooks/compare_models.ipynb`
