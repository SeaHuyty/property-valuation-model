# Data cleaning, imputation, normalization

- **Date:** 2026-11-22 23:00
- **Commit:** `12ecfb5`
- **Author:** SeaHuyty

## What
- `clean_dataset.py`: drop project pages, bad prices (< $5k, > $20M, impossible $/m²) and duplicates; merge size columns; group property types; unify district/commune spellings; amenity lists → 0/1 flags.
- `handle_missing.py`: not-applicable values → 0; `area_m2` (20%), bedrooms (22%), bathrooms (37%) imputed with the median of the most specific group, plus `*_missing` flags. Pre-imputation table saved as `data_unimputed.csv`.
- `docs/dataset.md` documents the pipeline and columns.

## Why / result
Result: `data/clean/data.csv`, 12,547 rows × 25 columns, no missing values, 95% Phnom Penh. Known caveats: imputation medians fit on the full data (leak), khmer24 amenity flags incomplete, only 16 Land rows.

## Files
- `laboratory/scripts/clean_dataset.py`
- `laboratory/scripts/handle_missing.py`
- `docs/dataset.md`
- `data/clean/data.csv`, `data_unimputed.csv`
