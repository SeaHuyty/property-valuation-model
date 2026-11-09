# Methodology

## Step 1: Data Collection (Web Scraping)

Build automated web scrapers using Python (Scrapy, BeautifulSoup, or Selenium) to scrape listings nationwide across Cambodia, extracting:

- **Location:** Province, Khan/District, Sangkat/Commune
- **Property Type:** Borey, Villa, Shophouse, Condo, Apartment, Land
- **Numerical Features:** Land area (or land width and length), building size, number of bedrooms, bathrooms, and floors (when available; most listings omit it)
- **Target Variable:** Listed price (standardized into USD)

## Step 2: Data Cleaning and Preprocessing

- **Outlier Removal:** Filter out non-serious listings (e.g., $1 "call for price" bait listings) and extreme, unrealistic data-entry errors.
- **Dimension Parsing:** Many listings give width and length instead of a total size. Parse unstructured dimension strings (e.g., "4.2m x 16m") or separate width/length fields and multiply them into total area in square meters. If a listing already gives a total area, use it directly. Apply this to both land area and building size.
- **Missing Value Imputation:** For properties where certain features do not apply (e.g., land plots have 0 bedrooms and 0 bathrooms), apply domain-specific default values rather than dropping rows.
- **Floors Handling:** Because most listings do not state the number of floors, do not drop rows with a missing value. Instead, add a binary `floors_missing` indicator and impute the value using the property-type default (e.g., 0 for land, 1 for single-storey houses) or the median for that property type. Alternatively, estimate floors from building size divided by land area where both are available. If the field is still too sparse after scraping, drop it as a feature.
- **Feature Scaling:** Apply `StandardScaler` or `MinMaxScaler` to numerical inputs, and log-transform the target price variable to handle price skewness.

## Step 3: Model Architecture (Deep Neural Network)

Implement a Multi-Layer Perceptron (MLP) using PyTorch or TensorFlow/Keras:

- **Categorical Input Branches:** Categorical features (Location: Province/District; Property Type) pass through Embedding Layers to reduce dimensionality and learn latent representations.
- **Numerical Input Branch:** Continuous features pass directly through dense layers.
- **Concatenation Layer:** Combines the flattened embedding vectors with the numerical features.
- **Dense Hidden Layers:** Fully connected layers with ReLU or LeakyReLU activation, paired with Batch Normalization and Dropout (e.g., 20–30%) to prevent overfitting.
- **Output Layer:** A single linear neuron predicting the continuous price value.

## Step 4: Baseline Comparison & Benchmarking

To scientifically evaluate the neural network, the same preprocessed data will be used to train standard baseline models:

1. **Baseline 1:** Multiple Linear Regression / Ridge Regression
2. **Baseline 2:** Random Forest Regressor
3. **Baseline 3:** XGBoost / LightGBM

## Step 5: Model Evaluation Metrics

Evaluate performance on an unseen test set (e.g., an 80/20 train-test split) using standard regression metrics:

| Metric   | Description                                                     |
| -------- | --------------------------------------------------------------- |
| **RMSE** | Root Mean Squared Error: measures general prediction error magnitude |
| **MAE**  | Mean Absolute Error: represents the average error in USD        |
| **R²**   | Coefficient of Determination: how much price variance the model explains |
| **MAPE** | Mean Absolute Percentage Error: error as a percentage of actual property value |

## Step 6: System Deployment & Web Interface

### Backend API (FastAPI)

- Save the trained neural network weights and export an inference pipeline.
- Expose a `/predict` endpoint that takes property attributes as a JSON payload and returns the predicted price and a confidence interval.

### Frontend Web App (Vue)

- **Responsive form:** Users select their province, district, and property type, and enter dimensions/rooms.
- **Interactive dashboard:** Visualizes the estimated price, estimated price per square meter, and comparable properties from the training dataset.
