# 🍽️ IntelliCanteen — Meal Attendance Forecasting

**A Machine Learning project** predicting daily student canteen attendance (Breakfast, Lunch, and Dinner) using time-series feature engineering and gradient-boosted trees.

## Team

- **OUADAH Lina Selma** 
- **BELAMRI Chakib**
- **MEZIGHECHE Malak Yasmine** 
- **HENNI Mohammed Yassine** 

## 🔗 Live Demo

Try the interactive prediction app here: **[LINK HERE]**

## Overview

IntelliCanteen forecasts how many students will show up for each meal at a given restaurant on a given day. The goal is to help canteen management anticipate demand, reduce food waste, and improve staffing/preparation decisions.

The project is split into three independent notebooks — one per meal — since each meal has its own attendance patterns, seasonality, and volume:

| Notebook | Description |
|---|---|
| `breakfast_count_prediction.ipynb` | Breakfast attendance forecasting |
| `lunch_count_prediction.ipynb` | Lunch attendance forecasting |
| `dinner_count_prediction.ipynb` | Dinner attendance forecasting |

## Pipeline

Each notebook follows the same structured pipeline:

1. **Data loading & EDA** — exploring restaurant, date, and meal-composition data
2. **Log transformation** on the target (`new_count`) to correct right-skewed distribution
3. **Feature engineering** — calendar features (weekends, Ramadan, academic holidays), meal-content features (keyword flags / Word2Vec embeddings)
4. **Lag & momentum features** (time-aware) — lag_1, lag_7, lag_14, lag_21, rolling means, EWMA divergence, trend/velocity indicators
5. **Distribution check** across years to validate a unified modeling approach
6. **Time-based Train / Validation / Test split** (chronological, no shuffling, to avoid leakage)
7. **Baseline models** — XGBoost, LightGBM, ExtraTrees, RandomForest
8. **Hyperparameter tuning** with Optuna
9. **Feature importance & residual diagnostics** on the winning model

## Results

| Meal | Model | Test R² | Test MAE |
|---|---|---|---|
| **Breakfast** | XGBoost (Word2Vec features) | **0.8561** | ~58 visitors |
| **Lunch** | XGBoost (tuned) | **0.8083** | ~74.8 visitors |
| **Dinner** | XGBoost (tuned) | **0.9075** | ~55.7 visitors |

All three models use chronological (time-based) train/validation/test splits to ensure realistic, leakage-free evaluation.

## Tech Stack

- **Modeling:** XGBoost, LightGBM, ExtraTrees, RandomForest, Optuna (hyperparameter tuning)
- **Data processing:** pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

An end-to-end ML system predicting daily breakfast, lunch, and dinner attendance across university restaurants using time-series features, calendar signals (holidays, Ramadan, weekends), and tuned XGBoost models — reaching up to **0.91 R²** on unseen data. Deployed as an interactive Streamlit app.
