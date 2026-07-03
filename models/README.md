# GuidaPlate — Production Models

Seven files required by the
production app. Do not remove.

| File | Size | Role | Loaded by |
|---|---|---|---|
| `xgboost_v3.pkl` | 757 KB | Tier 1 meal risk classifier | backend/models/xgboost_model.py |
| `lstm_v3_final.keras` | 412 KB | Tier 2 sequence pattern + trend | backend/models/lstm_model.py |
| `lstm_v3_scaler.pkl` | 1 KB | LSTM input scaler | backend/models/lstm_model.py |
| `lstm_v3_label_encoder.pkl` | 1 KB | LSTM label encoder | backend/models/lstm_model.py |
| `weekly_rf.pkl` | 663 KB | Tier 3 weekly RF classifier | backend/api/weekly_trend.py |
| `weekly_rf_config.json` | 1 KB | Weekly RF metadata | backend/api/weekly_trend.py |
| `transition_matrix.json` | 1 KB | Meal-to-meal transition probs | backend/api/next_meal.py |

## Archived models
Non-production models (v1, v2,
ablations, GRU experiments) are
in models/archive/.
