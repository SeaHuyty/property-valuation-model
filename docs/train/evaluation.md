For a fair comparison with a baseline, say a group median, linear regression or LightGBM:
1. Use the same splits: call load_splits(path, test_size, val_size, seed) from engines/common.py with the same seeds as the config.
2. Use the same metrics: regression_metrics() from engines/common.py.
3. Save the results in the same summary.json format, so you can put them side by side.
4. Compare the test mean ± std.

If the baseline uses different splits, the numbers can't be compared, because the gap between splits (about 0.15 R²) is bigger than most model differences.