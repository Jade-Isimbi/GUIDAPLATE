# notebooks/

Jupyter notebooks for the GuidaPlate
data pipeline and model development.

## Production pipeline (v3)

Run in this order:

1. `01_data_exploration.ipynb` — NHANES cohort + food database EDA
2. `03c_labels_v3_clinical_score.ipynb` — v3 clinical-score risk labels
3. `04c_xgboost_v3_raw_features.ipynb` — XGBoost v3 training
4. `05c_lstm_v3_improved.ipynb` — LSTM v3 training
5. `06_model_comparison.ipynb` — full model comparison + McNemar

## Supporting research

| Notebook | Purpose |
|---|---|
| `03_statistical_analysis.ipynb` | Kruskal-Wallis, Spearman, exceedance tests |
| `03b_labels_v2_sequence_aware.ipynb` | v2 label pipeline (ablation base) |
| `04_xgboost_training.ipynb` | XGBoost v1 baseline |
| `04b_xgboost_v2_improved.ipynb` | XGBoost v2 comparison |
| `05_lstm_training.ipynb` | LSTM v1 baseline |
| `05b_lstm_v2_improved.ipynb` | LSTM v2 + ablation study |

## Archived notebooks

Superseded notebooks are in `notebooks/archive/`.

Includes `11_weekly_tier3.ipynb` — Tier 3 weekly RF training/eval
(**abandoned / not deployed**; model artifacts in `models/archive/weekly_rf.pkl`).

## Prerequisites

Before running any notebook:

1. Place `food_database.csv` in `backend/data/`
2. Download all NHANES XPT files to `data/raw/nhanes/`
3. Install requirements: `pip install -r requirements.txt`

## Notes

- All notebooks import config using: `import config`
- `config.py` at repo root redirects to `backend/config.py`
- Do not run notebooks out of order within each pipeline section
