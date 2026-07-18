# Potassium residualization experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

## Correlation after train-only residualization

Training split correlations of `potassium_unique`:

- with `p_severity`: 0.0000000000
- with `protein_severity`: 0.0000000000
- with `na_severity`: 0.0000000000

## Holdout accuracy (n=296)

| Model | Accuracy |
|---|---:|
| Production day | 98.9865% |
| Decomposition experiment | 93.9189% |
| Residualized experiment | 94.5946% |

## SHAP target terms

### HIGH

1. `p_severity` — 1.028517
2. `protein_severity` — 0.443694
3. `potassium_unique` — 0.008528
4. `na_severity` — 0.000000

### MODERATE

1. `p_severity` — 1.028468
2. `protein_severity` — 0.384862
3. `potassium_unique` — 0.008216
4. `na_severity` — 0.000000

Decomposition HIGH `k_severity` reference: **0.062721**.

## Artifact

`models/xgboost_v3_potassium_residualized_experiment.pkl` — contains both model and fitted residualizer;
`not_for_production=True`.
