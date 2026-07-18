# Meal-scale model without clinical_score feature

**Status: EXPERIMENT ONLY — not for production.**

## Honest framing

Labels remain thresholded from `clinical_score` (inherent label circularity).
This experiment only removes `clinical_score` from the **features**, so the
model cannot directly read the labeling formula. Production meal features also
omitted occasion; `clinical_score` was the only place Snack vs Dinner scale
entered the model — that is the main hypothesis for the 99.86% → 77.52% drop.

## Quality band

**STRONG_NONCIRCULAR_FEATURE_SET**

Winner: **C_noscore_occasion_plus_caps** at **96.89%**
(gap to production +2.97 pp; lift vs ablation +19.37 pp).

## Holdout comparison (n=2153)

| Arm | #feats | Accuracy | Macro F1 | Weighted AUC | vs ablation |
|---|---:|---:|---:|---:|---:|
| Production meal (with clinical_score) | 9 | 99.86% | 0.9982 | 1.0000 | — |
| Published no-score ablation | 8 | 77.52% | 0.7602 | 0.9491 | 0.00 pp |
| A_noscore_8feat_retuned | 8 | 75.10% | 0.7399 | 0.9467 | -2.42 pp |
| B_noscore_plus_occasion | 9 | 96.24% | 0.9536 | 0.9986 | +18.72 pp |
| C_noscore_occasion_plus_caps | 13 | 96.89% | 0.9608 | 0.9988 | +19.37 pp |

## Winner gain importance

- `meal_cap_protein_per_kg`: 21.89%
- `k_p_product`: 16.68%
- `phosphorus`: 13.99%
- `meal_cap_phosphorus`: 10.85%
- `meal_cap_potassium`: 9.90%
- `occasion_encoded`: 6.64%
- `protein_per_kg`: 5.67%
- `sodium`: 4.01%
- `meal_cap_sodium`: 3.41%
- `potassium`: 2.49%
- `stage_numeric`: 1.88%
- `protein_sodium_ratio`: 1.33%
- `ckd_stage_encoded`: 1.26%

## Artifact

`models/xgboost_v3_meal_noscore_occasion_experiment.pkl` — `not_for_production=True`.

Protected production + five prior experimental pickles were SHA-256 verified
unchanged before and after.
