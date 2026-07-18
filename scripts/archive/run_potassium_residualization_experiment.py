#!/usr/bin/env python3
"""
EXPERIMENT ONLY — residualized potassium feature for SHAP diagnostics.

Fits potassium ~ p_severity + protein_severity + na_severity on the day-scale
training split only, then replaces k_severity with potassium_unique.

Does not modify production or decomposition artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from scripts.run_clinical_score_decomposition_experiment import (
    BASE_FEATURES,
    DAY_PKL,
    FEATURES_PROD,
    MEAL_PKL,
    RANDOM_STATE,
    RISK_ENCODE,
    TEST_SIZE,
    build_frame,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATS = ROOT / "outputs" / "stats"
DECOMP_PKL = MODELS / "xgboost_v3_decomposed_experiment.pkl"
DECOMP_JSON = STATS / "21_clinical_score_decomposition_experiment.json"
OUT_PKL = MODELS / "xgboost_v3_potassium_residualized_experiment.pkl"
OUT_JSON = STATS / "22_potassium_residualization_experiment.json"
OUT_MD = STATS / "22_potassium_residualization_experiment.md"

PROTECTED = {
    DAY_PKL: "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5",
    MEAL_PKL: "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db",
    DECOMP_PKL: "19df7c874bc0817e8d2ca80fb9ea50223e2ca3a87aa33577da77d55f70b422a8",
}

RESIDUAL_PREDICTORS = ["p_severity", "protein_severity", "na_severity"]
RESIDUAL_FEATURE = "potassium_unique"
FEATURES_RESIDUALIZED = BASE_FEATURES + [
    RESIDUAL_FEATURE,
    "p_severity",
    "protein_severity",
    "na_severity",
]


def assert_protected(when: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, expected in PROTECTED.items():
        digest = sha256_file(path)
        hashes[path.name] = digest
        if digest != expected:
            raise RuntimeError(
                f"FATAL [{when}]: protected artifact changed: {path.name}\n"
                f"expected {expected}\ngot      {digest}"
            )
        print(f"[{when}] {path.name} SHA-256 OK: {digest}")
    return hashes


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "auc_weighted": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def shap_ranked(model, X: pd.DataFrame, class_idx: int) -> list[tuple[str, float]]:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contributions = np.asarray(
        model.get_booster().predict(dmat, pred_contribs=True)
    )
    mean_abs = np.abs(
        contributions[:, class_idx, : X.shape[1]]
    ).mean(axis=0)
    return sorted(
        zip(X.columns.tolist(), mean_abs.tolist()),
        key=lambda pair: -pair[1],
    )


def main() -> int:
    before = assert_protected("before")

    df = build_frame()
    y = df["y"]

    # Split indices exactly as in production/decomposition experiments.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()
    print(f"split: train={len(train)} test={len(test)}")

    # Fit on training data only; use the fitted regression unchanged on holdout.
    residualizer = LinearRegression()
    residualizer.fit(train[RESIDUAL_PREDICTORS], train["potassium"])
    train[RESIDUAL_FEATURE] = (
        train["potassium"]
        - residualizer.predict(train[RESIDUAL_PREDICTORS])
    )
    test[RESIDUAL_FEATURE] = (
        test["potassium"]
        - residualizer.predict(test[RESIDUAL_PREDICTORS])
    )

    corr_train = train[
        [RESIDUAL_FEATURE, *RESIDUAL_PREDICTORS]
    ].corr(method="pearson")
    corr_test = test[
        [RESIDUAL_FEATURE, *RESIDUAL_PREDICTORS]
    ].corr(method="pearson")
    print("\nTrain correlations:")
    print(corr_train.to_string(float_format=lambda value: f"{value:.10f}"))
    print("\nHoldout correlations:")
    print(corr_test.to_string(float_format=lambda value: f"{value:.10f}"))

    # Reuse the exact hyperparameter dictionary saved by decomposition experiment.
    decomp_payload = joblib.load(DECOMP_PKL)
    if not decomp_payload.get("not_for_production"):
        raise RuntimeError("Decomposition artifact is not marked not_for_production")
    hyperparams = dict(decomp_payload["hyperparams"])

    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sample_weight = compute_sample_weight(
        class_weight=class_weight,
        y=train["y"],
    )
    model = xgb.XGBClassifier(**hyperparams)
    model.fit(
        train[FEATURES_RESIDUALIZED],
        train["y"],
        sample_weight=sample_weight,
    )
    y_pred = model.predict(test[FEATURES_RESIDUALIZED])
    y_prob = model.predict_proba(test[FEATURES_RESIDUALIZED])
    residual_metrics = metrics(test["y"].to_numpy(), y_pred, y_prob)

    prod = joblib.load(DAY_PKL)
    prod_pred = prod.predict(test[FEATURES_PROD])
    prod_prob = prod.predict_proba(test[FEATURES_PROD])
    prod_metrics = metrics(test["y"].to_numpy(), prod_pred, prod_prob)

    decomp_results = json.loads(DECOMP_JSON.read_text())
    decomp_metrics = decomp_results["experiment_day_holdout"]

    print("\nAccuracy:")
    print(f"  production:    {prod_metrics['accuracy']:.10f}")
    print(f"  decomposition: {decomp_metrics['accuracy']:.10f}")
    print(f"  residualized:  {residual_metrics['accuracy']:.10f}")

    target_terms = [
        RESIDUAL_FEATURE,
        "p_severity",
        "protein_severity",
        "na_severity",
    ]
    shap_results: dict[str, dict] = {}
    for class_name, class_idx in (("HIGH", 2), ("MODERATE", 1)):
        ranked = shap_ranked(
            model,
            test[FEATURES_RESIDUALIZED],
            class_idx,
        )
        target_ranked = [
            (name, value)
            for name, value in ranked
            if name in target_terms
        ]
        shap_results[class_name] = {
            "target_terms_ranked": target_ranked,
            "overall_top5": ranked[:5],
        }
        print(f"\nSHAP {class_name} target terms:")
        for rank, (name, value) in enumerate(target_ranked, start=1):
            print(f"  {rank}. {name:18s} {value:.10f}")

    payload = {
        "model": model,
        "residualizer": residualizer,
        "residual_predictors": RESIDUAL_PREDICTORS,
        "residual_target": "potassium",
        "feature_names": FEATURES_RESIDUALIZED,
        "experiment": "potassium_residualization_day_v3",
        "not_for_production": True,
        "note": (
            "EXPERIMENT ONLY. Do not deploy. Existing production and "
            "decomposition artifacts remain unchanged."
        ),
        "hyperparams": hyperparams,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    }
    joblib.dump(payload, OUT_PKL)

    after = assert_protected("after")
    results = {
        "status": "EXPERIMENT_ONLY_NOT_FOR_PRODUCTION",
        "protected_artifacts_unchanged": before == after,
        "protected_sha256": after,
        "features": FEATURES_RESIDUALIZED,
        "n_features": len(FEATURES_RESIDUALIZED),
        "residualizer": {
            "fit_scope": "training_split_only",
            "target": "potassium",
            "predictors": RESIDUAL_PREDICTORS,
            "intercept": float(residualizer.intercept_),
            "coefficients": {
                name: float(value)
                for name, value in zip(
                    RESIDUAL_PREDICTORS,
                    residualizer.coef_,
                )
            },
        },
        "correlation_train": corr_train.to_dict(),
        "correlation_holdout": corr_test.to_dict(),
        "production_day_holdout": prod_metrics,
        "decomposition_day_holdout": decomp_metrics,
        "residualized_day_holdout": residual_metrics,
        "accuracy_delta_vs_production": float(
            residual_metrics["accuracy"] - prod_metrics["accuracy"]
        ),
        "accuracy_delta_vs_decomposition": float(
            residual_metrics["accuracy"] - decomp_metrics["accuracy"]
        ),
        "shap": shap_results,
        "decomposition_high_k_severity_reference": 0.06272097676992416,
        "artifact": str(OUT_PKL.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    high = shap_results["HIGH"]["target_terms_ranked"]
    moderate = shap_results["MODERATE"]["target_terms_ranked"]
    OUT_MD.write_text(
        f"""# Potassium residualization experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

## Correlation after train-only residualization

Training split correlations of `potassium_unique`:

- with `p_severity`: {corr_train.loc[RESIDUAL_FEATURE, 'p_severity']:.10f}
- with `protein_severity`: {corr_train.loc[RESIDUAL_FEATURE, 'protein_severity']:.10f}
- with `na_severity`: {corr_train.loc[RESIDUAL_FEATURE, 'na_severity']:.10f}

## Holdout accuracy (n=296)

| Model | Accuracy |
|---|---:|
| Production day | {prod_metrics['accuracy']:.4%} |
| Decomposition experiment | {decomp_metrics['accuracy']:.4%} |
| Residualized experiment | {residual_metrics['accuracy']:.4%} |

## SHAP target terms

### HIGH

{chr(10).join(f'{rank}. `{name}` — {value:.6f}' for rank, (name, value) in enumerate(high, 1))}

### MODERATE

{chr(10).join(f'{rank}. `{name}` — {value:.6f}' for rank, (name, value) in enumerate(moderate, 1))}

Decomposition HIGH `k_severity` reference: **0.062721**.

## Artifact

`{OUT_PKL.relative_to(ROOT)}` — contains both model and fitted residualizer;
`not_for_production=True`.
"""
    )
    print(f"\nWrote {OUT_PKL}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
