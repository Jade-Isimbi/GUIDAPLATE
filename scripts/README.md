# GuidaPlate — Scripts

Standalone CLI utilities for ML reproduction and maintenance.
**Not imported by the production app at runtime.**

```bash
source venv311/bin/activate
python scripts/run_model_comparison.py
```

## Active scripts

| Script | Purpose |
|---|---|
| `train_xgboost_v3_meal.py` | Train meal-scale XGBoost |
| `test_xgboost_v3_meal.py` | Meal model offline tests |
| `test_meal_xgboost_clinical_guidelines.py` | Meal clinical-guideline checks |
| `eval_xgboost_v3_day_ablation.py` | Day-model clinical_score ablation |
| `run_model_comparison.py` | Master model comparison metrics |
| `run_overfitting_analysis.py` | Train/test gap analysis |
| `run_lstm_ablations_controlled.py` | Controlled LSTM ablations |
| `generate_testing_evidence.py` | Build `docs/testing/` evidence images |
| `generate_xgboost_training_evidence.py` | Training evidence figures |
| `generate_xgboost_v3_meal_eval_figures.py` | Meal evaluation figures |
| `generate_food_database_ts.py` | Sync frontend food database constants |
| `generate_clinical_constants_ts.py` | Sync frontend clinical constants |
| `generate_lstm_v3_notebook.py` | Regenerate LSTM notebook |
| `generate_v3_notebooks.py` | Regenerate notebooks `03c` / `04c` |
| `seed_demo_weekly_pattern.py` | Seed demo weekly dietary pattern data |

## Archive (`scripts/archive/`)

Legacy and one-off experiment scripts only (older LSTM/XGB runs, trend GRU,
clinical_score ablation experiments, Tier 3 RF checks). Not required to run the app.
