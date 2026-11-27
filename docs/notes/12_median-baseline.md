# Median price-per-m² baseline

- **Date:** 2026-11-26 19:26
- **Commit:** `0ef28a8`
- **Author:** SeaHuyty

## What
- No-learning baseline: median price/m² by location and type × area.
- Three variants by minimum group size: `base`, `min_group_1`, `min_group_20`.

## Why / result
Test MdAPE (on `data.csv`): base 28.1%, min_group_1 28.0%, min_group_20 28.3%. The floor any model must beat.

## Files
- `laboratory/engines/median_baseline/median_baseline.ipynb`
- `laboratory/models/median_baseline/`
