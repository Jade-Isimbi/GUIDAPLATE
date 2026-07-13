# GuidaPlate — Production Models

Production artifacts required by the live app. Do not remove.

| File | Size | Role | Loaded by |
|---|---|---|---|
| `xgboost_v3.pkl` | 757 KB | Tier 1 day-scale risk classifier | backend/models/xgboost_model.py |
| `xgboost_v3_meal.pkl` | 605 KB | Tier 1 meal-scale risk classifier (default ON via `GUIDAPLATE_MEAL_XGB`) | backend/models/xgboost_model.py |
| `lstm_v3_final.keras` | 412 KB | Tier 2 sequence **risk** classifier (**trend** = post-hoc heuristic on hidden states) | backend/models/lstm_model.py |
| `lstm_v3_scaler.pkl` | 1 KB | LSTM input scaler | backend/models/lstm_model.py |
| `lstm_v3_label_encoder.pkl` | 1 KB | LSTM label encoder | backend/models/lstm_model.py |

## Research-only artifacts (not loaded at runtime)

| File | Size | Role | Used by |
|---|---|---|---|
| `transition_matrix.json` | 1 KB | Meal-to-meal transition probs (offline experiments) | notebooks/scripts only — not the production API |

## Abandoned / not deployed (archived)

Trained and evaluated, but **not** loaded by any live API. Weekly Trend uses nutrient aggregates + LSTM only.

| File | Size | Role | Notes |
|---|---|---|---|
| `archive/weekly_rf.pkl` | 663 KB | Tier 3 weekly Random Forest | Trained in `notebooks/archive/11_weekly_tier3.ipynb`. Offline check: `scripts/archive/verify_tier3.py`. |
| `archive/weekly_rf_config.json` | 1 KB | Metadata for the archived RF | Companion to `archive/weekly_rf.pkl`. |

## Other archived models

Non-production models (v1, v2, ablations, GRU experiments) are also under `models/archive/`.
