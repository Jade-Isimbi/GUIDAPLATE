#!/usr/bin/env python3
"""
EXPERIMENT ONLY — constrain clinical_score visibility with column subsampling.

Trains the production day-scale nine-feature XGBoost on the exact production
split and weighting scheme, changing only colsample_bytree from 1.0 to 0.5.

Does not modify production, decomposition, or residualization artifacts.

Outputs:
  - models/xgboost_v3_colsample_05_experiment.pkl
  - outputs/stats/23_clinical_score_colsample_experiment.json
  - outputs/stats/23_clinical_score_colsample_experiment.md
  - outputs/figures/xgb_v3_colsample_05_gain_importance.png
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from scripts.run_clinical_score_decomposition_experiment import (
    FEATURES_PROD,
    RANDOM_STATE,
    RISK_CLASSES,
    RISK_ENCODE,
    TEST_SIZE,
    build_frame,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATS = ROOT / "outputs" / "stats"
FIGURES = ROOT / "outputs" / "figures"

DAY_PKL = MODELS / "xgboost_v3.pkl"
MEAL_PKL = MODELS / "xgboost_v3_meal.pkl"
DECOMP_PKL = MODELS / "xgboost_v3_decomposed_experiment.pkl"
RESIDUAL_PKL = MODELS / "xgboost_v3_potassium_residualized_experiment.pkl"

OUT_PKL = MODELS / "xgboost_v3_colsample_05_experiment.pkl"
OUT_JSON = STATS / "23_clinical_score_colsample_experiment.json"
OUT_MD = STATS / "23_clinical_score_colsample_experiment.md"
OUT_FIGURE = FIGURES / "xgb_v3_colsample_05_gain_importance.png"

COLSAMPLE_EXPERIMENT = 0.5
ABLATION_ACCURACY = 0.9324324324324325
RAW_NUTRIENTS = ["potassium", "phosphorus", "protein_per_kg", "sodium"]

PROTECTED = {
    DAY_PKL: "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5",
    MEAL_PKL: "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db",
    DECOMP_PKL: "19df7c874bc0817e8d2ca80fb9ea50223e2ca3a87aa33577da77d55f70b422a8",
    RESIDUAL_PKL: "165cd163ef9768a0f4328b24ed18ad3c473b3b7a808dfca8974b24a8cb73a685",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protected(when: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path, expected in PROTECTED.items():
        actual = sha256_file(path)
        hashes[path.name] = actual
        if actual != expected:
            raise RuntimeError(
                f"FATAL [{when}]: protected artifact changed: {path.name}\n"
                f"expected {expected}\ngot      {actual}"
            )
        print(f"[{when}] {path.name} SHA-256 OK: {actual}")
    return hashes


def production_hyperparams(model: xgb.XGBClassifier) -> dict:
    params = model.get_params()
    keys = (
        "objective",
        "num_class",
        "eval_metric",
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "min_child_weight",
        "random_state",
        "verbosity",
    )
    return {
        key: params[key]
        for key in keys
        if key in params and params[key] is not None
    }


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "auc_weighted": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(RISK_CLASSES)
        },
    }


def normalized_gain(model: xgb.XGBClassifier) -> dict[str, float]:
    """
    Return normalized gain importance, matching XGBClassifier.feature_importances_
    used for Figure 5.16 while retaining explicit feature names and zero-gain rows.
    """
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    gains = {feature: float(raw.get(feature, 0.0)) for feature in FEATURES_PROD}
    total = sum(gains.values())
    if total <= 0:
        return {feature: 0.0 for feature in FEATURES_PROD}
    return {feature: value / total for feature, value in gains.items()}


def save_gain_figure(
    production_gain: dict[str, float],
    experiment_gain: dict[str, float],
) -> None:
    frame = pd.DataFrame(
        {
            "Production (colsample=1.0)": production_gain,
            "Experiment (colsample=0.5)": experiment_gain,
        }
    ).loc[FEATURES_PROD]
    frame = frame.sort_values("Experiment (colsample=0.5)", ascending=True)
    axis = frame.plot(kind="barh", figsize=(10, 6), width=0.8)
    axis.set_title("Day XGBoost normalized gain importance")
    axis.set_xlabel("Share of total gain")
    axis.set_ylabel("Feature")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.tight_layout()
    plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = assert_protected("before")

    frame = build_frame()
    X = frame[FEATURES_PROD]
    y = frame["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"split: train={len(X_train)} test={len(X_test)}")

    production = joblib.load(DAY_PKL)
    production_params = production_hyperparams(production)
    if float(production_params["colsample_bytree"]) != 1.0:
        raise RuntimeError(
            "Expected production colsample_bytree=1.0, got "
            f"{production_params['colsample_bytree']!r}"
        )

    experiment_params = dict(production_params)
    experiment_params["colsample_bytree"] = COLSAMPLE_EXPERIMENT
    changed_params = {
        key: {
            "production": production_params.get(key),
            "experiment": experiment_params.get(key),
        }
        for key in sorted(set(production_params) | set(experiment_params))
        if production_params.get(key) != experiment_params.get(key)
    }
    if changed_params != {
        "colsample_bytree": {
            "production": 1.0,
            "experiment": COLSAMPLE_EXPERIMENT,
        }
    }:
        raise RuntimeError(f"Unexpected hyperparameter changes: {changed_params}")

    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sample_weight = compute_sample_weight(class_weight=class_weight, y=y_train)
    experiment = xgb.XGBClassifier(**experiment_params)
    experiment.fit(X_train, y_train, sample_weight=sample_weight)

    production_metrics = metrics(
        y_test.to_numpy(),
        production.predict(X_test),
        production.predict_proba(X_test),
    )
    experiment_metrics = metrics(
        y_test.to_numpy(),
        experiment.predict(X_test),
        experiment.predict_proba(X_test),
    )

    production_gain = normalized_gain(production)
    experiment_gain = normalized_gain(experiment)
    raw_production = sum(production_gain[name] for name in RAW_NUTRIENTS)
    raw_experiment = sum(experiment_gain[name] for name in RAW_NUTRIENTS)

    accuracy_delta_production = (
        experiment_metrics["accuracy"] - production_metrics["accuracy"]
    )
    accuracy_delta_ablation = experiment_metrics["accuracy"] - ABLATION_ACCURACY
    clinical_drop = (
        production_gain["clinical_score"] - experiment_gain["clinical_score"]
    )
    raw_gain_increase = raw_experiment - raw_production

    # Transparent operational definition for the requested "promising middle ground."
    promising = (
        accuracy_delta_production >= -0.02
        and clinical_drop >= 0.10
        and raw_gain_increase >= 0.10
    )
    if promising:
        verdict = (
            "PROMISING_MIDDLE_GROUND: accuracy stayed within 2 percentage points "
            "of production, clinical_score dominance fell by at least 10 points, "
            "and the four raw nutrients gained at least 10 points of total gain."
        )
    elif accuracy_delta_production < -0.04:
        verdict = (
            "NEGATIVE_ACCURACY: accuracy moved substantially toward the no-score "
            "ablation result."
        )
    elif clinical_drop < 0.10:
        verdict = (
            "NEGATIVE_DOMINANCE: clinical_score still dominates; column "
            "subsampling did not materially reduce circularity."
        )
    else:
        verdict = (
            "MIXED_RESULT: dominance changed, but the predefined accuracy/raw-gain "
            "criteria for a promising middle ground were not all met."
        )

    save_gain_figure(production_gain, experiment_gain)

    artifact_payload = {
        "model": experiment,
        "feature_names": FEATURES_PROD,
        "experiment": "clinical_score_colsample_05_day_v3",
        "not_for_production": True,
        "note": (
            "EXPERIMENT ONLY. Do not deploy. Production and all prior experiment "
            "artifacts remain protected and unchanged."
        ),
        "hyperparams": experiment_params,
        "changed_from_production": changed_params,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "class_weight": class_weight,
    }
    joblib.dump(artifact_payload, OUT_PKL)

    after = assert_protected("after")
    results = {
        "status": "EXPERIMENT_ONLY_NOT_FOR_PRODUCTION",
        "verdict": verdict,
        "promising_middle_ground": promising,
        "promising_criteria": {
            "accuracy_within_production_percentage_points": 2.0,
            "minimum_clinical_score_share_drop_percentage_points": 10.0,
            "minimum_raw_nutrient_share_increase_percentage_points": 10.0,
        },
        "protected_artifacts_unchanged": before == after,
        "protected_sha256": after,
        "features": FEATURES_PROD,
        "n_features": len(FEATURES_PROD),
        "split": artifact_payload["split"],
        "class_weight": class_weight,
        "production_hyperparams": production_params,
        "experiment_hyperparams": experiment_params,
        "changed_from_production": changed_params,
        "production_day_holdout": production_metrics,
        "experiment_day_holdout": experiment_metrics,
        "no_clinical_score_ablation_reference": {
            "accuracy": ABLATION_ACCURACY,
            "source": "outputs/stats/10_xgboost_v3_day_ablation.json",
        },
        "accuracy_delta_experiment_minus_production": float(
            accuracy_delta_production
        ),
        "accuracy_delta_experiment_minus_ablation": float(accuracy_delta_ablation),
        "gain_importance_method": (
            "normalized booster gain; equivalent to the default gain-based "
            "XGBClassifier.feature_importances_ method used for Figure 5.16"
        ),
        "production_normalized_gain": production_gain,
        "experiment_normalized_gain": experiment_gain,
        "clinical_score_gain_share": {
            "production": production_gain["clinical_score"],
            "experiment": experiment_gain["clinical_score"],
            "drop": clinical_drop,
        },
        "raw_nutrient_total_gain_share": {
            "production": raw_production,
            "experiment": raw_experiment,
            "increase": raw_gain_increase,
        },
        "raw_nutrient_gain_comparison": {
            name: {
                "production": production_gain[name],
                "experiment": experiment_gain[name],
                "change": experiment_gain[name] - production_gain[name],
            }
            for name in RAW_NUTRIENTS
        },
        "artifact": str(OUT_PKL.relative_to(ROOT)),
        "figure": str(OUT_FIGURE.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")

    gain_rows = "\n".join(
        "| `{}` | {:.2%} | {:.2%} | {:+.2f} pp |".format(
            feature,
            production_gain[feature],
            experiment_gain[feature],
            100 * (experiment_gain[feature] - production_gain[feature]),
        )
        for feature in sorted(
            FEATURES_PROD,
            key=lambda name: experiment_gain[name],
            reverse=True,
        )
    )
    OUT_MD.write_text(
        f"""# Clinical-score column-subsampling experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

## Verdict

**{verdict}**

Only `colsample_bytree` changed: production **1.0** → experiment **0.5**.
The feature set, split, class weights, and all other model hyperparameters match
the production day model.

## Holdout performance

Same stratified split: `test_size=0.2`, `random_state=42`, n={len(y_test)}.

| Model | Accuracy | Macro F1 | Weighted AUC |
|---|---:|---:|---:|
| Production day | {production_metrics['accuracy']:.2%} | {production_metrics['f1_macro']:.4f} | {production_metrics['auc_weighted']:.4f} |
| Colsample 0.5 experiment | {experiment_metrics['accuracy']:.2%} | {experiment_metrics['f1_macro']:.4f} | {experiment_metrics['auc_weighted']:.4f} |
| Existing no-score ablation | {ABLATION_ACCURACY:.2%} | 0.9241 | 0.9915 |

- Experiment vs production: **{100 * accuracy_delta_production:+.2f} percentage points**
- Experiment vs no-score ablation: **{100 * accuracy_delta_ablation:+.2f} percentage points**

## Normalized gain importance

This is gain-based feature importance, matching the method behind Figure 5.16.

| Feature | Production | Colsample 0.5 | Change |
|---|---:|---:|---:|
{gain_rows}

- `clinical_score`: **{production_gain['clinical_score']:.2%} → {experiment_gain['clinical_score']:.2%}** ({-100 * clinical_drop:+.2f} percentage points)
- Four raw nutrients combined: **{raw_production:.2%} → {raw_experiment:.2%}** ({100 * raw_gain_increase:+.2f} percentage points)

## Artifacts

- `{OUT_PKL.relative_to(ROOT)}` — joblib dictionary with `not_for_production=True`
- `{OUT_JSON.relative_to(ROOT)}`
- `{OUT_FIGURE.relative_to(ROOT)}`

Protected production and prior experimental artifacts were hash-verified before
and after training and remained unchanged.
"""
    )

    print("\nHoldout accuracy")
    print(f"  production:       {production_metrics['accuracy']:.6f}")
    print(f"  colsample 0.5:    {experiment_metrics['accuracy']:.6f}")
    print(f"  no-score ablation:{ABLATION_ACCURACY:.6f}")
    print("\nNormalized gain")
    print(
        "  clinical_score: "
        f"{production_gain['clinical_score']:.2%} -> "
        f"{experiment_gain['clinical_score']:.2%}"
    )
    print(
        "  raw nutrients:  "
        f"{raw_production:.2%} -> {raw_experiment:.2%}"
    )
    print(f"\n{verdict}")
    print(f"\nWrote {OUT_PKL}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
