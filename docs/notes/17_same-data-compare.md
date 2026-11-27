# Compare only same-data runs, champion sections

- **Date:** 2026-11-26 22:16
- **Commit:** `069faaf`
- **Author:** SeaHuyty

## What
- `compare_models` only compares runs trained on the same data and seeds (paired per seed on identical test rows).
- Added champion sections: error by type, price decile, district, bias, and "good enough" targets.

## Why / result
Avoid comparing models trained on different data, which gave misleading gaps.

## Files
- `laboratory/evaluate/compare_models.ipynb`
