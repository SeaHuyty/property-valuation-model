# Organize folders, add training docs

- **Date:** 2026-11-23 22:28
- **Commit:** `a338e40`
- **Author:** SeaHuyty

## What
- Docs moved into `docs/data/` and `docs/train/`; scrapers moved to `laboratory/scripts/scrape/`.
- `docs/train/seed.md`: why run several seeds (one split's score is mostly luck; test R² swung ~0.15 between seeds).
- `docs/train/save_best_in_epoch.md`: early stopping keeps the best val-loss epoch.
- `docs/train/evaluation.md`: how to compare a baseline fairly (same splits, same metrics, same summary format).

## Why / result
Set the rule used later for every change: same seeds on both sides, count a gain only if it beats the noise.

## Files
- `docs/data/`, `docs/train/`
- `laboratory/scripts/scrape/`
