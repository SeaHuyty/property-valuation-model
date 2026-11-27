# Shared scoring and seed-run helpers

- **Date:** 2026-11-26 19:26
- **Commit:** `0633e5a`
- **Author:** SeaHuyty

## What
- Moved scoring (`regression_metrics`) and the per-seed run loop (`run_seeds`) from `train.py` into `engines/common.py`.

## Why / result
Every model (NN and baselines) now uses the exact same splits and metrics.

## Files
- `laboratory/engines/common.py`
- `laboratory/engines/neural_network/train.py`
