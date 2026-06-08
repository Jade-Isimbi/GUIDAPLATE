# notebooks/

Jupyter notebooks for the GuidaPlate
data pipeline and model development.

## Notebook Order

Run in this order:

| Notebook | Purpose |
|---|---|
| 01_data_exploration.ipynb | Explore NHANES cohort and food database |
| 02_nhanes_preprocessing.ipynb | Merge NHANES files, calculate eGFR, assign CKD stages |
| 03_statistical_analysis.ipynb | Run 5 statistical tests (Spearman, Kruskal-Wallis, etc.) |
| 04_xgboost_training.ipynb | Train XGBoost 9-feature 3-class classifier |
| 05_lstm_training.ipynb | Build LSTM sequences from NHANES IFF files |
| 06_evaluation.ipynb | Full evaluation and McNemar test |

## Prerequisites

Before running any notebook:
1. Place food_database.csv in backend/data/
2. Download all 6 NHANES XPT files to data/raw/nhanes/
3. Install requirements: pip install -r requirements.txt

## Notes

- All notebooks import config using: import config
- config.py at repo root redirects to backend/config.py
- Do not run notebooks out of order
- Synthetic data notebooks have been archived
