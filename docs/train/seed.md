Why need to run several seed?

One run's score is mostly luck of the split, so one seed can't tell you whether a change helped.

Why: each seed picks a different random train/val/test split and a different starting weight set. The test set is small, about 1.9k rows, and some types like Land, Villa and Shophouse have only a few hundred rows in it. Whether a few expensive outliers land in test moves the score a lot: test R² swung about 0.15 between seeds.

Problem with one seed: you change something, like adding bathrooms or tweaking dropout, and R² goes 0.80 → 0.83. You can't tell if the change helped or the split just got easier.

With several seeds: you get a mean ± std, for example R² 0.81 ± 0.05 vs 0.82 ± 0.05. That gap is inside the noise, so the change doesn't matter. Count a change as real only when the gap is bigger than about one std, with the same seeds used on both sides.

When to use which:
- --seeds 42: quick check that the code runs and the loss goes down.
- --seeds 42 7 13, or 5 or more seeds: comparing features, hyperparameters or data changes, and reporting final numbers.

Final model: the seeds are for measuring, not for picking a winner. Inference loads the first seed's model.