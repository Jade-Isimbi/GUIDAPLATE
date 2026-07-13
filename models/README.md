# GuidaPlate — Production Models

Production artifacts required by the live app. Do not remove.

| File | Size | Role | Loaded by |
|---|---|---|---|
| `xgboost_v3.pkl` | 757 KB | Tier 1 day-scale risk classifier | backend/models/xgboost_model.py |
| `xgboost_v3_meal.pkl` | 605 KB | Tier 1 meal-scale risk classifier (default ON via `GUIDAPLATE_MEAL_XGB`) | backend/models/xgboost_model.py |
| `lstm_v3_final.keras` | 412 KB | Tier 2 sequence **risk** classifier (**trend** = post-hoc heuristic on hidden states) | backend/models/lstm_model.py |
| `lstm_v3_scaler.pkl` | 1 KB | LSTM input scaler | backend/models/lstm_model.py |
| `lstm_v3_label_encoder.pkl` | 1 KB | LSTM label encoder | backend/models/lstm_model.py |

## Abandoned / not deployed (`archive/`)

Trained and evaluated, but **not** loaded by any live API. Weekly Trend uses nutrient aggregates + LSTM only.

| File | Role | Notes |
|---|---|---|
| `archive/weekly_rf.pkl` + `weekly_rf_config.json` | Tier 3 weekly Random Forest | Offline check: `scripts/archive/verify_tier3.py` |
| `archive/transition_matrix.json` | Meal-to-meal transition probs | Offline trend/HMM experiments only |
| Other `archive/*` | v1/v2 XGB/LSTM, ablations, GRU | Superseded by v3 production set above |
