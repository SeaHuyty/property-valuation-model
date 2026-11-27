## Model Training Process

1. Scrape data
 - Python scrapers (`laboratory/scripts/scrape/`) for 3 Cambodian listing sites: khmer24.com, realestate.com.kh, cambodiaproperty.asia.
 - Collected: location (province, district, commune, borey), property type, area / land dimensions, bedrooms, bathrooms, floor, amenities, title deed, price (USD).
 - Raw: ~13,900 listings (khmer24 7,645, realestate.com.kh 6,192, cambodiaproperty 39).

2. Consolidate the scraped data from 3 different sites
 - Mapped each site's fields into one schema → `data/consolidated/listings_consolidated.csv` (13,876 rows).
 - Unified district / commune spellings across sites, grouped property types into common classes, normalized borey names.
 - Merged size columns into one area + land width/length; split amenity lists into 0/1 flags.

3. Data Analysis, Preprocessing and Imputation
 - Notebooks: `data_analysis` (explore), `filter_data` (remove outliers/duplicates), `feature_analysis` (feature vs price), `feature_ablation` (test a feature by removing it).
 - Data issues: 22% of rows were duplicates with different prices (test leak), 157 price/m² outliers, 38 prices under $15k (deposits/rent), only 16 Land rows.
 - Cleaning: filter + dedupe took 12,280 → 10,299 rows (`data_filtered.csv`). 7 types (Condo 3,864, Villa 2,049, Flat 1,379, House 1,237, Shophouse 1,055, Twin Villa 402, Link House 313); 95% Phnom Penh; median price $168.5k.
   - Area recovery: for 453 non-condo rows with a missing area but known land width × length, area = width × length (for non-condos `area_m2` is land area). Improved R² and MAE on 5/5 seeds.
   - Sanity rules (dropped): area below a per-type minimum (Condo/Flat 25 m², House/Shophouse/Link House 30 m², Villa/Twin Villa 50 m²), under 10 m² per bedroom, over 12 bedrooms, bathrooms > bedrooms + 2. NN MdAPE −0.7 pts, better on 5/5 seeds.
   - Price/m² outliers: robust z-score (MAD) within each type, cut at 3.5. A tighter cut (2.5) looked better but gave no gain once the test rows were held fixed, so it was rejected.
   - Rejected: extra features parsed from the raw listing text (no gain).
   - Rule for every change: compare on the same test rows, paired per seed; keep only if better on most seeds.
 - Imputation: area 19%, bedrooms 22%, bathrooms 37% missing. Imputed by type, with `*_missing` flags kept (they carry signal).
 - Features: area dominates (~60% of log-price variance), then location (commune, district, type). Amenity flags weak.
 - Ablation: correlation misled. Dropping "redundant-looking" features made the model worse. Keep the full feature set; row cleaning matters more than feature selection.

4. Train on neural network
 - MLP (PyTorch): embeddings for categoricals (type, district, commune, borey, title deed) + dense numeric branch → 256-128-64 layers, BatchNorm, dropout 0.25.
 - Target = log price (prices span $15k–$15M). Huber loss, AdamW, early stopping on validation.
 - Split 70 / 10 / 20 (train/val/test), stratified by type. Repeated over 5 seeds, reported as mean ± std.
 - Result on data_filtered: test MdAPE 23.0%, RMSE(log) 0.426, R² 0.765, MAE $111.7k (`models/neural_network/filtered`).

5. Train on other models (baselines)
 - Same data, same 5 splits, same metrics as the NN.
 - Median price-per-m² by location/type (no learning): MdAPE 28.0%.
 - Linear regression: 26.6%; with type × numeric interactions: 26.1%.
 - Ridge regression: 26.7%; with interactions: 26.1%.

6. Train gradient boosting (`engines/gradient_boosting/gradient_boosting.ipynb`)
 - sklearn `HistGradientBoostingRegressor` on log price, same features; categoricals split natively (codes for values seen ≥ 3 times in train, max 250 per column).
 - Noise reduction: one model depends on its random seed, so the run averages 10 models (in log space): 5 × absolute-error loss (fits the median, robust to odd listings) + 5 × squared-error loss with bigger trees (127 leaves). `max_features=0.5` makes the copies differ. All early-stop on validation; settings picked on validation MdAPE.
 - Each part tested on the 5 seeds (test MdAPE): single model 21.9% → tuned 21.4% → averaged abs + sq 21.0%.
 - Rejected: location price/m² (target-encoded) features (worse, 22.3%); blending in the NN (no MdAPE gain).
 - Result: test MdAPE 21.1%, RMSE(log) 0.415, R² 0.783, MAE $106.2k. Beats the NN on 5/5 seeds.

7. Evaluate the model to pick the champion model
 - `evaluate/compare_models.ipynb`: compares runs on identical test rows only (paired per seed), plus error by type, by price decile, by district, and bias.
 - Metrics: MdAPE (typical % error), RMSE on log price, R², MAE.
 - Champion: gradient boosting ensemble. Beats the NN by 1.9 pts MdAPE and every baseline by 5+ pts.

   | Model | MdAPE | RMSE (log) | R² | MAE |
   |---|---|---|---|---|
   | Gradient boosting ensemble | 21.1% | 0.415 | 0.783 | $106k |
   | Neural network | 23.0% | 0.426 | 0.765 | $112k |
   | Ridge + interactions | 26.1% | 0.465 | 0.725 | $124k |
   | Linear + interactions | 26.1% | 0.466 | 0.727 | $123k |
   | Median price/m² | 28.0% | 0.520 | 0.675 | $135k |

 - Targets (none met yet): MdAPE ≤ 20% (21.1%), within ±20% ≥ 50% (48.0%), off by ×1.5+ ≤ 10% (25.0%).
 - 80% of actual prices fall within ×0.63–×1.61 of the estimate.
 - Noise floor: identical listings differ by 13.5% MdAPE, so no model can get much lower. Champion is 1.6× that floor. The ×1.5+ target looks out of reach with these features.
 - By type: Twin Villa 12%, Link House 17% (best); House 34%, Flat 26%, Shophouse 26% (worst). The House listings don't carry enough detail to price them well.
 - Weak districts: Ta Khmau 36%, 7 Makara 33%, Siem Reap 31%, Daun Penh 29%; few rows outside Phnom Penh.
 - Next: the app's inference still loads the NN, so the gradient boosting model has to be wired in.
