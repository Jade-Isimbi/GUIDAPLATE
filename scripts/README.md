# GuidaPlate — Scripts

Standalone CLI utilities for ML reproduction and maintenance.
**Not imported by the production app at runtime.**

Run from the project root with `venv311` active:

```bash
source venv311/bin/activate
python scripts/run_model_comparison.py
```

## Evidence / maintenance (keep)

| Script | Purpose |
|---|---|
| `run_model_comparison.py` | Regenerates master model comparison metrics |
| `run_overfitting_analysis.py` | Regenerates train/test gap analysis |
| `run_lstm_ablations_controlled.py` | Controlled LSTM ablations A2/B2/C2 |
| `generate_testing_evidence.py` | Builds `docs/testing/` PNG evidence from `outputs/` |
| `generate_food_database_ts.py` | Syncs FOODS + RWANDAN_FOOD_IDS in `foodDatabase.ts` |
| `generate_clinical_constants_ts.py` | Syncs STAGE_THRESHOLDS + KDOQI limits TS mirrors |
| `generate_lstm_v3_notebook.py` | Regenerates `notebooks/05c_lstm_v3_improved.ipynb` |
| `generate_v3_notebooks.py` | Regenerates notebooks `03c` / `04c` |
| `train_xgboost_v3_meal.py` | Train meal-scale XGBoost |
| `test_xgboost_v3_meal.py` | Meal model offline tests |
| `test_meal_xgboost_clinical_guidelines.py` | Meal clinical-guideline checks |
| `eval_xgboost_v3_day_ablation.py` | Day-model clinical_score ablation |

## Archive (legacy / one-off)

Under `scripts/archive/` — reproduction of older experiments only.
`ROOT` paths use `parents[2]` (repo root).

| Script | Purpose |
|---|---|
| `archive/verify_tier3.py` | Abandoned Tier 3 RF smoke load |
| `archive/backfill_legacy_patients.py` | One-time legacy DB migration |
| `archive/run_labels_v2.py` | Regenerate v2 risk labels |
| `archive/run_lstm_v2.py` | Train LSTM v2 |
| `archive/run_lstm_ablations.py` | LSTM ablation study (non-controlled) |
| `archive/patch_lstm_v2_notebooks.py` | Patch 03b/05b notebook cells |
| `archive/append_ablation_section.py` | Append ablation cells to 05b |
| `archive/append_ablation_controlled.py` | Append controlled ablation cells |
| `archive/lstm_augmented_retrain.py` | Augmented LSTM v1 retrain |
| `archive/run_trend_prediction_experiment.py` | Trend GRU v1 experiment |
| `archive/run_trend_prediction_v2.py` | Trend GRU v2 experiment |
