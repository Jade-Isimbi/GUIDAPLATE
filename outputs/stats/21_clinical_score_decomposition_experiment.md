# Clinical-score decomposition experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

Production pickles untouched:
- `models/xgboost_v3.pkl` SHA-256 `0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5`
- `models/xgboost_v3_meal.pkl` SHA-256 `564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db`

## Identity check

`k_severity + p_severity + protein_severity + na_severity == clinical_score`

- max |sum − recomputed clinical_score|: **0.000e+00** (pass ≤ 1e-9)
- max |sum − CSV clinical_score|: **4.441e-16**

## Holdout accuracy (same split: test_size=0.2, random_state=42, n_test=296)

| Model | Accuracy | F1 macro | AUC weighted |
|-------|----------|----------|--------------|
| Production day (`xgboost_v3.pkl`) | 98.9865% | 0.9853 | 0.9975 |
| Decomposed experiment (12 features) | 93.9189% | 0.9269 | 0.9896 |
| Delta (decomp − prod) | -5.0676% | — | — |

Chapter 5 reported production day accuracy: **98.99%**.

## SHAP — severity components (new model)

### HIGH
| Rank | Feature | mean \|SHAP\| |
|------|---------|---------------|
| 1 | p_severity | 1.019477 |
| 2 | protein_severity | 0.433500 |
| 3 | k_severity | 0.062721 |
| 4 | na_severity | 0.000000 |

### MODERATE
| Rank | Feature | mean \|SHAP\| |
|------|---------|---------------|
| 1 | p_severity | 1.006534 |
| 2 | protein_severity | 0.379148 |
| 3 | k_severity | 0.049528 |
| 4 | na_severity | 0.000000 |

## Artifact

`models/xgboost_v3_decomposed_experiment.pkl` — joblib dict with `not_for_production=True`.
