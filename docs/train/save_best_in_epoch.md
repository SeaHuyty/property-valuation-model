Early stopping is in train.py.

Epoch check, per run (train.py:107-114):
- After each epoch, it computes the loss on the validation set.
- When the val loss improves by more than 1e-5, it saves a copy of the weights (best) and resets the wait counter.
- When the val loss doesn't improve for patience epochs in a row, it stops.
- At the end it loads best, the weights from the best epoch rather than the last one, then computes the test metrics.
- ReduceLROnPlateau also lowers the learning rate when the val loss stops improving, before training stops.

So each run already stops before overfitting. history.csv shows the train and val loss per epoch if you want to look.

Seeds, across runs: early stopping picks the best epoch for one split. It can't tell you whether that split was easy or hard. Seeds answer that by repeating the whole run, with early stopping inside each one, on different splits.

┌────────────┬────────────────┬──────────────────────┐
│            │ early stopping │    several seeds     │
├────────────┼────────────────┼──────────────────────┤
│ answers    │ when to stop   │ is this score real   │
│            │ this run       │ or luck of the split │
├────────────┼────────────────┼──────────────────────┤
│ uses       │ val set        │ val + test across    │
│            │                │ splits               │
├────────────┼────────────────┼──────────────────────┤
│ protects   │ overfitting    │ fooling yourself in  │
│ against    │                │ comparisons          │
└────────────┴────────────────┴──────────────────────┘

Both are needed: early stopping inside each run, seeds around the runs.