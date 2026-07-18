# GuidaPlate — Production Models

Only production and retained research artifacts live here.

| File | Role | Loaded by live API? |
|---|---|---|
| `xgboost_v3_meal_noscore.pkl` | **Live** meal-scale classifier (13 features; no `clinical_score`) | Yes |
| `xgboost_v3_meal_noscore_meta.pkl` | Metadata for the live meal model | Yes (support) |
| `xgboost_v3_meal.pkl` | Legacy meal-scale research artifact (includes `clinical_score`) | No |
| `xgboost_v3.pkl` | Day-scale research/evaluation artifact | No |
| `lstm_v3_final.keras` | Sequence risk classifier | Yes |
| `lstm_v3_scaler.pkl` | LSTM input scaler | Yes |
| `lstm_v3_label_encoder.pkl` | LSTM label encoder | Yes |

Live holdout (Model C): **96.89%** accuracy.  
Legacy meal (with `clinical_score`): **99.86%**.

If meal-model inference fails, the API uses the meal-scale rule fallback
(0 exceedances → LOW, 1 → MODERATE, ≥2 → HIGH).
