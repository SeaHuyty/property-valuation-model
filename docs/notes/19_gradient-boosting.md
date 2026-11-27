# Uncommitted: filtered data, gradient boosting champion

- **Date:** up to 2026-11-27 (not committed yet)
- **Commit:** none, working tree
- **Author:** SeaHuyty (with Claude)

## What
- `filter_data.ipynb`: removed duplicates with different prices (22% of rows, a test leak), 157 price/m² outliers (robust z-score within type, cut 3.5), 38 prices under $15k, and the 16 Land rows. Sanity rules on area, bedrooms and bathrooms. 12,280 → 10,299 rows (`data/clean/data_filtered.csv`).
- Area recovery: 453 non-condo rows got area = land width × length.
- Rejected: features parsed from raw listing text (no gain), tighter outlier cut 2.5 (no gain on fixed test rows).
- NN retrained on filtered data (`models/neural_network/filtered/`).
- Baselines rerun on filtered data with the same seeds.
- New `engines/gradient_boosting/gradient_boosting.ipynb`: HistGradientBoosting ensemble of 10 models per seed (5 absolute-error + 5 squared-error), averaged in log space.
- `docs/process.md` and `docs/findings/` written.

## Result (test, mean of 5 seeds, filtered data)
| Model | MdAPE | RMSE (log) | R² | MAE |
|---|---|---|---|---|
| Gradient boosting ensemble | 21.1% | 0.415 | 0.783 | $106k |
| Neural network | 23.0% | 0.426 | 0.765 | $112k |
| Ridge + interactions | 26.1% | 0.465 | 0.725 | $124k |
| Linear + interactions | 26.1% | 0.466 | 0.727 | $123k |
| Median price/m² | 28.0% | 0.520 | 0.675 | $135k |

Gradient boosting is the champion and beats the NN on 5/5 seeds.

## Open
- Inference still loads the NN, and its default path `models/neural_network/model.pt` doesn't exist. Need a final gradient boosting model trained on all the data and wired in.
- Option: switch from 5 random seeds to stratified 5-fold.
