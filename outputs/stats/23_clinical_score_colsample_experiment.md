# Clinical-score column-subsampling experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

## Verdict

**PROMISING_MIDDLE_GROUND: accuracy stayed within 2 percentage points of production, clinical_score dominance fell by at least 10 points, and the four raw nutrients gained at least 10 points of total gain.**

Only `colsample_bytree` changed: production **1.0** → experiment **0.5**.
The feature set, split, class weights, and all other model hyperparameters match
the production day model.

## Holdout performance

Same stratified split: `test_size=0.2`, `random_state=42`, n=296.

| Model | Accuracy | Macro F1 | Weighted AUC |
|---|---:|---:|---:|
| Production day | 98.99% | 0.9853 | 0.9975 |
| Colsample 0.5 experiment | 98.99% | 0.9852 | 0.9979 |
| Existing no-score ablation | 93.24% | 0.9241 | 0.9915 |

- Experiment vs production: **+0.00 percentage points**
- Experiment vs no-score ablation: **+5.74 percentage points**

## Normalized gain importance

This is gain-based feature importance, matching the method behind Figure 5.16.

| Feature | Production | Colsample 0.5 | Change |
|---|---:|---:|---:|
| `clinical_score` | 98.64% | 71.38% | -27.25 pp |
| `k_p_product` | 0.00% | 8.43% | +8.43 pp |
| `phosphorus` | 1.01% | 8.14% | +7.13 pp |
| `protein_per_kg` | 0.00% | 3.59% | +3.59 pp |
| `potassium` | 0.00% | 2.17% | +2.17 pp |
| `stage_numeric` | 0.00% | 2.01% | +2.01 pp |
| `ckd_stage_encoded` | 0.00% | 1.78% | +1.78 pp |
| `sodium` | 0.36% | 1.47% | +1.11 pp |
| `protein_sodium_ratio` | 0.00% | 1.03% | +1.03 pp |

- `clinical_score`: **98.64% → 71.38%** (-27.25 percentage points)
- Four raw nutrients combined: **1.36% → 15.37%** (+14.00 percentage points)

## Artifacts

- `models/xgboost_v3_colsample_05_experiment.pkl` — joblib dictionary with `not_for_production=True`
- `outputs/stats/23_clinical_score_colsample_experiment.json`
- `outputs/figures/xgb_v3_colsample_05_gain_importance.png`

Protected production and prior experimental artifacts were hash-verified before
and after training and remained unchanged.
