# Meal-scale clinical-score column-subsampling experiment

**Status: EXPERIMENT ONLY — not for production.**

## Verdict

**NEGATIVE_DOMINANCE: clinical_score still dominates; column subsampling did not materially reduce circularity.**

## Day-scale comparison

SMALLER_EFFECT: meal-scale reduced clinical_score dominance less than day-scale. Day: accuracy delta +0.00 pp, clinical_score 98.64%→71.38% (-27.25 pp), raw nutrients 1.36%→15.37% (+14.00 pp). Meal: accuracy 99.86% (delta +0.00 pp), clinical_score 80.05%→72.73% (-7.32 pp), raw nutrients 10.35%→15.32% (+4.97 pp).

Only `colsample_bytree` changed: production meal **0.9** → experiment **0.5**.
(Note: production meal already used 0.9, unlike day production which used 1.0.)
The feature set, split, class weights, and all other model hyperparameters match
the production meal model.

## Holdout performance

Same stratified split: `test_size=0.2`, `random_state=42`, n=2153.

| Model | Accuracy | Macro F1 | Weighted AUC |
|---|---:|---:|---:|
| Production meal | 99.86% | 0.9982 | 1.0000 |
| Colsample 0.5 experiment | 99.86% | 0.9982 | 1.0000 |
| Existing no-score ablation | 77.52% | 0.7602 | 0.9491 |

- Experiment vs production: **+0.00 percentage points**
- Experiment vs no-score ablation: **+22.34 percentage points**

## Normalized gain importance

This is gain-based feature importance, matching the method behind Figure 5.16.

| Feature | Production | Colsample 0.5 | Change |
|---|---:|---:|---:|
| `clinical_score` | 80.05% | 72.73% | -7.32 pp |
| `k_p_product` | 7.76% | 8.65% | +0.88 pp |
| `phosphorus` | 8.00% | 7.74% | -0.26 pp |
| `potassium` | 0.38% | 3.65% | +3.27 pp |
| `protein_per_kg` | 1.25% | 2.72% | +1.48 pp |
| `stage_numeric` | 0.00% | 1.34% | +1.34 pp |
| `ckd_stage_encoded` | 1.44% | 1.26% | -0.17 pp |
| `sodium` | 0.73% | 1.21% | +0.48 pp |
| `protein_sodium_ratio` | 0.40% | 0.70% | +0.31 pp |

- `clinical_score`: **80.05% → 72.73%** (-7.32 percentage points)
- Four raw nutrients combined: **10.35% → 15.32%** (+4.97 percentage points)
- `k_p_product`: **7.76% → 8.65%** (+0.88 percentage points)

## Artifacts

- `models/xgboost_v3_meal_colsample_05_experiment.pkl` — joblib dictionary with `not_for_production=True`
- `outputs/stats/24_meal_clinical_score_colsample_experiment.json`
- `outputs/figures/xgb_v3_meal_colsample_05_gain_importance.png`

Protected production and all four prior experimental artifacts were hash-verified
before and after training and remained unchanged.
