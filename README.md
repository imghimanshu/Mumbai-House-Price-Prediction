# 🏠 Mumbai House Price Estimator

An end-to-end machine learning project that predicts house prices in Mumbai. It covers the complete pipeline — raw data cleaning, exploratory analysis, feature engineering, model training/comparison, and a deployed **Flask web app** where a user picks a locality from a dropdown, types in the property's numbers, and gets an instant price estimate.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Real estate prices in Mumbai vary drastically by locality, property type, and size. This project builds a regression model on ~72,000 real listings and serves it through a lightweight web form:

- **Categorical fields** (Locality, City Region, Property Type, Furnishing) are dropdowns, because the model only knows the exact categories it was trained on — anything else is meaningless to it.
- **Numeric fields** (Area, BHK, Bathrooms, Balconies, Property Age) are free-typed number boxes, validated on the server before they ever reach the model.

The example above shows a real run: a 1000 sqft, 3BHK, unfurnished Independent House in Borivali, 1 year old, estimated at **₹2,04,90,236**.

---

## 🗺️ The Full Pipeline, Step by Step

This is the complete journey from raw CSV to a working web app, in the order it actually happens.

### 1. Data Collection
The starting point is `mumbai-house-price-data.csv` — **71,938 raw property listings** scraped/aggregated with fields like price, area, locality, bedroom/bathroom/balcony counts, property age, city region, property type, furnishing status, and geo-coordinates.

### 2. Data Cleaning
Raw scraped real-estate data is messy — placeholder prices, data-entry typos, and physically impossible listings all creep in. Cleaning was done in stages, each one checked against the row count before/after so nothing was silently dropped in bulk:

| Step | What it removes | Why |
|---|---|---|
| Drop `price >= 999,999,999` | Corrupted/placeholder prices | Clearly not real sale prices |
| Drop `price <= 100,000` | Unrealistically cheap listings | Data-entry errors, not real Mumbai prices |
| Drop `bedroom_num == 0` | Listings with no bedrooms | Incomplete/invalid records |
| Drop `total_floors` column | A near-constant column (99.9% identical value) | Zero predictive signal, pure noise |
| **Locality-wise price-per-sqft outlier removal** | For each locality, keep only listings within ±1 standard deviation of that locality's mean price/sqft | The single biggest cleaning step — removes mispriced outliers *relative to their own neighborhood*, which is more meaningful than a single global cutoff |
| Drop `area_per_bedroom < 120 sqft` | Physically implausible layouts (e.g. a "4BHK" in 200 sqft) | Sanity check on room sizing |

**Result:** 71,938 → **53,128 clean rows** — about a 26% reduction, mostly from the locality-wise outlier filter, which is expected and healthy for this kind of noisy listings data.

### 3. Exploratory Data Analysis (EDA)
Before modeling, the price distribution was visualized to understand its shape.

**Output 1 — Raw price distribution**

<img width="701" height="490" alt="5d56d174-0193-486e-b572-81d6bd25ee9e" src="https://github.com/user-attachments/assets/2cd8cc99-2587-4eb4-92ef-c34d1ac62e70" />

Raw `price` is heavily right-skewed — the vast majority of homes cluster under ₹5 crore, but a long tail of ultra-luxury properties stretches the x-axis out to over ₹80 crore. A handful of extreme values like this can quietly dominate a model's error metric and drag predictions for ordinary homes off-target, so this plot is what first flagged the need for a transform.

**Output 2 — Log-transformed price distribution**

<img width="690" height="490" alt="c06943fa-b0ee-4b42-8851-8c5bbd2895a1" src="https://github.com/user-attachments/assets/083ad39b-0eb2-48c5-b72b-0efa3053c5f8" />

Applying `log1p(price)` reshapes the same data into something close to a bell curve. This one chart drove the most important modeling decision in the whole project: **train the model on `log_price`, not raw price** (see step 4). A roughly normal, symmetric target is exactly what regression models are built to predict well.

A correlation matrix across numeric columns (price, area, bedroom_num, bathroom_num, balcony_num, age) was also checked to confirm the expected relationships (area and bedroom count correlate positively with price) before engineering new features.

### 4. Feature Engineering
This is where raw columns become model-ready inputs:

- **Log-transform the target:** `log_price = log1p(price)`. Regression models generally assume roughly normal, homoscedastic targets — feeding them a heavily skewed target like raw price leads to the model over-focusing on a handful of huge-value outliers. Training on `log_price` and inverting with `expm1()` at prediction time fixes this and was key to getting good R² scores.
- **Rare locality bucketing:** any locality with fewer than 10 listings was relabeled `"Other"`. Without this, one-hot encoding would create dozens of near-empty, high-noise columns the model can't learn anything reliable from. This left 125 meaningful localities plus one `"Other"` catch-all.
- **One-hot encoding:** `locality`, `city`, `property_type`, and `furnished` were converted to 0/1 dummy columns (`pd.get_dummies(..., drop_first=True)`), since these are categorical and have no inherent numeric ordering.
- **Derived feature — `area_per_bedroom = area / bedroom_num`:** captures how "spacious" a home is independent of its raw size. A 600 sqft 1BHK and a 2400 sqft 4BHK have the same area_per_bedroom, and this feature ended up being meaningfully predictive on its own.
- **Dropped columns:** `title`, `price`, `price_per_sqft` were removed from the feature set — `title` is unstructured text, and `price`/`price_per_sqft` would leak the answer directly into the input.

**Final feature matrix:** 53,128 rows × **140 columns**.

### 5. Train/Test Split
An 80/20 split (`train_test_split(..., test_size=0.2, random_state=42)`) gave:
- **Training set:** 42,502 rows
- **Test set:** 10,626 rows

The fixed `random_state` makes the split reproducible — anyone re-running the notebook gets the exact same split and comparable metrics.

### 6. Model Training & Comparison
Three regressors were trained on the same data and evaluated on the same untouched test set:

| Model | R² Score | MAE (actual ₹) |
|---|---|---|
| Linear Regression | 0.938 | ₹37,96,717 |
| **Random Forest** ⭐ | **0.979** | **₹15,94,761** |
| XGBoost | 0.964 | ₹23,71,339 |

**Random Forest was chosen as the final, deployed model.** It has both the highest R² (explains 97.9% of price variance on unseen data) and by far the lowest average error. Linear Regression sets a reasonable baseline, but real estate pricing has non-linear interactions (e.g. the value of an extra bedroom differs wildly by locality) that tree ensembles capture far better. XGBoost was trained with only lightly-tuned hyperparameters, so there's likely room to close the gap with proper tuning — see [Limitations](#-known-limitations--next-steps).

**Output 3 — Random Forest: predicted vs. actual price**

<img width="694" height="706" alt="c812150c-548f-4e16-b4e3-c0b61ba80de2" src="https://github.com/user-attachments/assets/008b320d-fb14-4af7-9607-a2371745ff3e" />

Each point is one test-set property; the x-axis is its real price, the y-axis is what the model predicted, both on a log scale, with the red dashed line marking a perfect prediction. The tight clustering along that diagonal — from under ₹10 lakh all the way past ₹30 crore — is the visual confirmation behind the 0.979 R² score: the model isn't just accurate on average, it stays accurate across the entire price range, from budget homes to ultra-luxury listings, with only a small scatter of outliers.

### 7. Feature Importance
Random Forest's built-in importances reveal what actually drives its predictions:

| Feature | Importance |
|---|---|
| `bedroom_num` | 50.7% |
| `longitude` | 23.4% |
| `latitude` | 15.7% |
| `area` | 5.0% |
| `area_per_bedroom` | 2.4% |
| `age` | 0.4% |

Location (via coordinates) and bedroom count together account for the overwhelming majority of the model's decision-making — which matches real-world intuition about how Mumbai real estate is priced.

> ⚠️ **Important caveat:** the deployed app currently sends a single fixed lat/long (Mumbai city center) for every prediction, regardless of the chosen locality. Since these two features alone make up ~39% of the model's importance, this is the single biggest accuracy gap in the current app — see [Limitations](#-known-limitations--next-steps).

### 8. Model Persistence
Once Random Forest was selected, it was serialized for reuse outside the notebook:

```python
joblib.dump(rf, 'mumbai_house_price_model.pkl')
model_columns = list(X_train.columns)
json.dump(model_columns, open('model_columns.json', 'w'))
```

- **`mumbai_house_price_model.pkl`** — the trained Random Forest, loaded once at Flask app startup.
- **`model_columns.json`** — the exact ordered list of 140 feature columns the model expects. This is critical: the web app has to build a one-row DataFrame with *exactly* these columns, in this order, or `model.predict()` will fail or (worse) silently misinterpret the input.

### 9. Deployment — the Flask Web App
`app.py` is a single self-contained file — no `templates/`, no `static/`, no JavaScript, no client-side framework. Everything (HTML, CSS, form logic) lives in one Python file, which keeps the app trivially easy to run anywhere Python and a few pip packages are available.

**How a request flows through the app:**

1. **App startup:** loads `mumbai_house_price_model.pkl` and `model_columns.json` once. It then derives the dropdown options directly from the column names (e.g. every column starting with `locality_` becomes one dropdown option) — so the dropdowns can never drift out of sync with what the model actually supports.
2. **GET /** — renders the empty form.
3. **User fills the form:**
   - **Dropdowns:** Locality, City Region, Property Type, Furnishing — constrained to the model's known categories, so there's no way to submit a category the model has never seen.
   - **Free numeric inputs:** Area, BHK, Bathrooms, Balconies, Property Age — plain `<input type="number">` boxes. The user can type any value, not just a preset list.
4. **POST /** — the form submits back to the same route:
   - **`validate_numeric_fields()`** parses every numeric field with `float()` inside a try/except, and checks it against a sane range (e.g. Area between 100–20,000 sqft, BHK between 1–20). Anything that fails — empty, non-numeric text, or out-of-range — is collected into a list of human-readable error messages, and *no prediction is attempted* until every field is valid.
   - **`build_feature_vector()`** builds a single-row `pandas.DataFrame` with all 140 columns initialized to 0, then:
     - Fills in the five validated numeric fields directly.
     - Computes `area_per_bedroom` the same way it was computed during training.
     - Sets the default latitude/longitude.
     - Sets the one matching `locality_*`, `city_*`, `property_type_*`, and `furnished_*` column to `1` (mirroring one-hot encoding at inference time).
   - **`model.predict(X)`** returns a `log_price` prediction; `np.expm1()` converts it back to actual rupees.
   - The page re-renders with the result shown, previous selections preserved (so the user doesn't have to re-fill the form after seeing an error or a result).
5. **Error handling:** any validation failure or unexpected exception during prediction is caught and displayed in a clearly styled error box, rather than crashing the app or showing a raw stack trace.

### 10. Running It
```bash
pip install flask scikit-learn pandas numpy joblib
python app.py
```
Then open `http://localhost:5000` — pick your dropdowns, type your numbers, and click **Calculate Estimate Price**.

**Final Output — the live app**

<img width="521" height="741" alt="Screenshot 2026-07-10 203249" src="https://github.com/user-attachments/assets/4c158ca6-0203-4bb3-a03e-f5534684e617" />

This is the end result of everything above: a 1000 sqft, 3BHK, unfurnished Independent House in Borivali, 1 year old, 3 bathrooms, 3 balconies — submitted through the real form and estimated by the real trained model at **₹2,04,90,236**. Every number on this screen traces back through the pipeline: the dropdown options come straight from `model_columns.json`, the typed numbers pass through `validate_numeric_fields()`, and the price comes from `mumbai_house_price_model.pkl`'s prediction, un-logged back into rupees.

---

## 🖼️ All Outputs at a Glance

| # | Output | What it shows |
|---|---|---|
| 1 | `assets/price_distribution.png` | Raw price is heavily skewed — motivates the log-transform |
| 2 | `assets/log_price_distribution.png` | Log-transformed price — confirms the transform worked, used as the actual model target |
| 3 | `assets/rf_predicted_vs_actual.png` | Random Forest's predictions vs. real prices on the test set — visual proof of the 0.979 R² |
| **Final** | `assets/app_screenshot.png` | The deployed Flask app producing a real prediction end-to-end |

---

## 📁 Project Structure

```
mumbai-house-price-app/
├── app.py                              # Flask app — dropdowns + free numeric inputs
├── mumbai_house_price_prediction.ipynb # Full notebook: cleaning → EDA → modeling
├── model_columns.json                  # Ordered list of feature columns the model expects
├── requirements.txt                    # Python dependencies
├── assets/
│   ├── price_distribution.png          # Output 1 — raw price distribution
│   ├── log_price_distribution.png      # Output 2 — log-transformed price distribution
│   ├── rf_predicted_vs_actual.png      # Output 3 — Random Forest predicted vs actual
│   └── app_screenshot.png              # Final Output — the live app
├── .gitignore
└── README.md
```

> **Not included in the repo** (regenerate locally by running the notebook, or version with [Git LFS](https://git-lfs.com/)):
> - `mumbai-house-price-data.csv` — raw dataset
> - `mumbai_house_data_cleaned.csv` — cleaned dataset
> - `mumbai_house_price_model.pkl` — trained model, **required** for `app.py` to run

---

## 🚀 Getting Started

### 1. Clone
```bash
git clone https://github.com/imghimanshu/Mumbai-House-Price-Prediction.git
cd Mumbai-House-Price-Prediction
```

### 2. Environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model
Place `mumbai-house-price-data.csv` in the project root, then run every cell of `mumbai_house_price_prediction.ipynb` top to bottom. This regenerates `mumbai_house_price_model.pkl` (and `model_columns.json`, already included).

### 4. Run the app
```bash
python app.py
```
Visit **http://localhost:5000**.

---

## ✅ Input Validation Rules

| Field | Type | Rule |
|---|---|---|
| Locality / City / Property Type / Furnishing | Dropdown | Must be one of the model's trained categories |
| Area | Number | 100 – 20,000 sqft |
| BHK (Bedrooms) | Number | 1 – 20 |
| Bathrooms | Number | 1 – 20 |
| Balconies | Number | 0 – 10 |
| Property Age | Number | 0 – 100 years |

Any missing field, non-numeric text, or out-of-range value is rejected with a specific error message — the model is never called with invalid input.

---

## ⚠️ Known Limitations & Next Steps

- **Fixed lat/long:** every prediction currently uses a single default coordinate (Mumbai city center) instead of the true coordinates of the selected locality — despite latitude/longitude together driving ~39% of the model's decisions. Mapping each locality to its real coordinates is the highest-impact improvement available.
- **XGBoost wasn't tuned:** trained with reasonable-but-arbitrary hyperparameters and still underperformed Random Forest; a proper search (Optuna/GridSearchCV) might change the ranking.
- **No cross-validation:** metrics come from a single 80/20 split; k-fold CV would give a more robust estimate of real-world performance.
- **Outlier handling is univariate:** locality-wise price/sqft filtering is simple and effective but could be extended with multivariate outlier detection (e.g. Isolation Forest).
- **Local-only deployment:** currently runs via `python app.py` on localhost; containerizing with Docker and deploying to Render/Railway/Heroku would make it publicly accessible without the user running Python themselves.

---

## 🧰 Tech Stack

- **Data & modeling:** pandas, numpy, scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Web app:** Flask (pure Python, no JS)
- **Model persistence:** joblib

---

## 📄 License

MIT License — free to use, modify, and build on.
