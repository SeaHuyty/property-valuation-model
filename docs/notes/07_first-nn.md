# First neural network training

- **Date:** 2026-11-23 20:28
- **Commit:** `64b4864`
- **Author:** SeaHuyty

## What
- PyTorch MLP: embeddings for categoricals + numeric branch (`networks.py`), preprocessing (`preprocessing.py`), training loop with early stopping (`train.py`), `inference.py`.
- Config `configs/neural_network_config.yml`: data path, features, split, seeds.
- `engines/common.py`: first shared helpers (splits).
- Removed the one-off data scripts from `scripts/` (their output is already in `data/`).

## Why / result
First trained model on log price.

## Files
- `laboratory/engines/neural_network/*`
- `laboratory/configs/neural_network_config.yml`
- `laboratory/engines/common.py`
