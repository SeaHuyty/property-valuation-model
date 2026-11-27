┌───────────────────────────┬───────────────────────┐
│           Data            │     Usually wins      │
├───────────────────────────┼───────────────────────┤
│ Images, text, audio,      │ NN, by a lot          │
│ video                     │                       │
├───────────────────────────┼───────────────────────┤
│ Tables (rows × columns)   │ Gradient boosting     │
│ with ~1k–1M rows          │ (XGBoost, LightGBM,   │
│                           │ HGB)                  │
├───────────────────────────┼───────────────────────┤
│ Huge tables (millions of  │ NN sometimes catches  │
│ rows) with lots of tuning │ up                    │
└───────────────────────────┴───────────────────────┘

Our data is a table with 10,299 rows, which is gradient boosting's home ground. Two well-known studies found the same thing:
- Grinsztajn et al., 2022, "Why do tree-based models still outperform deep learning on tabular data?"
- Shwartz-Ziv & Armon, 2021, "Tabular data: deep learning is not all you need"

## Why trees fit this data:
- Small data. 10k rows is little for an NN, which needs many rows to learn well. Trees learn fine from 10k.
- Step-shaped prices. Price jumps by commune, by type, or past an area threshold. A tree split is a step. An NN draws smooth curves, so it struggles with sharp jumps.
- Location columns with many values. Commune and borey have hundreds of values, many seen only a few times. Trees group these by price directly. An NN must learn an embedding for each one, and rare ones get barely trained.
- Weak or noisy columns. Amenity flags carry little signal. Trees mostly ignore weak columns, while an NN spends effort fitting them and overfits.
- Missing values. Trees handle NaN natively. The NN needs imputation and *_missing flags.
- Tuning cost. An NN needs tuning of layers, dropout, learning rate and embedding sizes. Trees work well with only a little tuning.

---

The performance of gradient boosting (GBDT) and neural networks (NNs) on tabular data with 10,000 rows can vary significantly based on the specific dataset and its characteristics.

1. GBDTs: They are generally better at handling skewed or heavy-tailed feature distributions and other forms of dataset irregularities. They are also more efficient for structured datasets, such as those with up to 10 million rows, due to their ability to partition the feature space into axis-aligned rectangles, which mirrors how humans typically structure tabular data. 
2. NNs: They excel in specific scenarios, such as extremely high-cardinality categorical features or multimodal integration (e.g., tabular + image). However, they may struggle with datasets that do not exhibit these characteristics. 

In the context of a 10,000-row dataset, practitioners should consider the dataset's properties and the specific requirements of their machine learning tasks. Light hyperparameter tuning on a GBDT may be more effective than selecting between GBDTs and NNs, depending on the dataset's characteristics.