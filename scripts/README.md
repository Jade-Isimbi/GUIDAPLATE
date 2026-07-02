# GuidaPlate — Scripts

Standalone CLI utilities for
ML reproduction and maintenance.
Not imported by the production
app or notebooks at runtime.

## Reproduction scripts
Run these to regenerate outputs/
stats/ and outputs/figures/ from
trained models:

| Script | Purpose |
|---|---|
| `run_model_comparison.py` | Regenerates master model comparison metrics (feeds docs/testing/07_model_comparison/) |
| `run_overfitting_analysis.py` | Regenerates train/test gap analysis (feeds docs/testing/03_overfitting_analysis/) |
| `run_lstm_ablations_controlled.py` | Reruns controlled LSTM ablations A2/B2/C2 |
| `generate_testing_evidence.py` | Builds docs/testing/ PNG evidence from outputs/stats/ and outputs/figures/ |

## Maintenance scripts
| Script | Purpose |
|---|---|
| `generate_food_database_ts.py` | Syncs frontend/src/data/foodDatabase.ts from backend/data/food_database.csv — run after CSV updates |
| `backfill_legacy_patients.py` | One-time DB migration for legacy user rows — only needed on old databases |

## Notebook generators
| Script | Purpose |
|---|---|
| `generate_lstm_v3_notebook.py` | Regenerates notebooks/05c_lstm_v3_improved.ipynb |
| `generate_v3_notebooks.py` | Regenerates notebooks/03c and 04c |

## Legacy scripts (v2 / ablation / trend)
Still in `scripts/` for notebook reproduction:

| Script | Purpose |
|---|---|
| `run_labels_v2.py` | Regenerate v2 risk labels |
| `run_lstm_v2.py` | Train LSTM v2 |
| `run_lstm_ablations.py` | LSTM ablation study |
| `patch_lstm_v2_notebooks.py` | Patch 03b/05b notebook cells |
| `append_ablation_section.py` | Append ablation cells to 05b |
| `append_ablation_controlled.py` | Append controlled ablation cells |
| `lstm_augmented_retrain.py` | Augmented LSTM v1 retrain |
| `run_trend_prediction_experiment.py` | Trend GRU v1 experiment |
| `run_trend_prediction_v2.py` | Trend GRU v2 experiment |

## Usage
Run from the project root with
venv311 active:
```bash
source venv311/bin/activate
python scripts/run_model_comparison.py
```
