# Data, feature analysis and ablation notebooks

- **Date:** 2026-11-26 22:16
- **Commit:** `9282b30`
- **Author:** SeaHuyty

## What
- `data_analysis.ipynb`: explore the data (skew, outliers, duplicates).
- `feature_analysis.ipynb`: each feature vs price.
- `feature_ablation.ipynb`: retrain without a feature to test if it matters.

## Why / result
Found: area drives ~60% of log-price variance, then location. Correlation misled: dropping "redundant" features made the model worse, so the full feature set stays.

## Files
- `laboratory/data/analysis/*.ipynb`
