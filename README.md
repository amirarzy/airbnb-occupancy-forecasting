# Airbnb Occupancy Rate Forecasting

This project builds a machine learning pipeline to forecast the **occupancy rate of Airbnb listings** based on listing metadata and facilities information.

It was originally developed for an **ML challenge**, and implemented in Python using `pandas`, `scikit-learn`, `XGBoost` and `optuna`.  
Best model: XGBoost regressor with Optuna tuning, reaching a validation **MAE of 0.138** on Airbnb occupancy prediction. [web:144][web:147]

---

## 📊 Problem & Data

- Goal: predict the continuous **occupancy** value for each Airbnb listing.  
- Input data: JSON files (`train.json`, `test.json`) containing:
  - Listing attributes (room type, listing type, cancellation policy, etc.)
  - Text field of **facilities/amenities**
  - Host information and other metadata

The raw JSON files are loaded and converted into tabular `pandas` DataFrames (`df_train`, `df_test`) for further processing. [web:147]

---

## 🧩 Feature Engineering

Key feature engineering steps:

- Parsed the free‑text **facilities** field and created binary features for amenity categories  
  (e.g. cooking, food storage, laundry, safety, internet, climate, bathroom, parking, leisure, child‑friendly, work space, breakfast, etc.).
- Engineered a **host_count** feature that measures how many listings each host manages across the whole dataset (train + test), capturing host experience / portfolio size.
- Handled missing values for text and categorical fields (e.g. replacing with `"Unknown"`).
- Encoded categorical variables (`room_type`, `listing_type`, `cancellation`) using `LabelEncoder`.

These steps transform mixed JSON and text data into a clean numerical feature matrix suitable for tree‑based models like XGBoost. [web:128][web:131][web:147]

---

## 🤖 Modeling

The main model is an **XGBoost Regressor**:

- Trained an initial baseline XGBoost model and evaluated with **Mean Absolute Error (MAE)** on a validation split.
- Performed **hyperparameter tuning** with `optuna` to optimize:
  - `n_estimators`, `learning_rate`, `max_depth`, `subsample`
  - `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`
- Used **5‑Fold cross‑validation** to obtain stable MAE estimates and out‑of‑fold predictions.  
- The tuned XGBoost model achieved a **validation MAE of 0.138**. [web:144][web:147]
- Trained a final model on the full training set using the best hyperparameters and generated predictions for the test set.

---

## 📦 Output

- Final predictions are stored as a list of dictionaries:
  - `{"occupancy": <predicted_value>}` for each listing.
- The script saves:
  - `predicted.json` – raw predictions  
  - `predicted.zip` – zipped version for submission to the challenge platform.

---

## 🚀 How to Run (outline)

1. Place `train.json` and `test.json` in a folder (e.g. `ML Challenge/`).
2. Install dependencies:
   pip install pandas numpy xgboost optuna scikit-learn
3. Run the notebook or script to:
- load data,
- engineer features,
- train the XGBoost model with tuned hyperparameters,
- generate `predicted.json` and `predicted.zip`.

   
