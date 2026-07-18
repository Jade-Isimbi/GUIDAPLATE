#!/usr/bin/env python3
"""
EXPERIMENT ONLY — meal-scale clinical_score column-subsampling.

Trains the production meal-scale nine-feature XGBoost on the exact production
split and weighting scheme, changing only colsample_bytree from the production
value (0.9) to 0.5.

Does not modify production or any prior experimental artifacts.

Outputs:
  - models/xgboost_v3_meal_colsample_05_experiment.pkl
  - outputs/stats/24_meal_clinical_score_colsample_experiment.json
  - outputs/stats/24_meal_clinical_score_colsample_experiment.md
  - outputs/figures/xgb_v3_meal_colsample_05_gain_importance.png
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

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATS = ROOT / "outputs" / "stats"
FIGURES = ROOT / "outputs" / "figures"

DAY_PKL = MODELS / "xgboost_v3.pkl"
MEAL_PKL = MODELS / "xgboost_v3_meal.pkl"
DECOMP_PKL = MODELS / "xgboost_v3_decomposed_experiment.pkl"
RESIDUAL_PKL = MODELS / "xgboost_v3_potassium_residualized_experiment.pkl"
DAY_COLSAMPLE_PKL = MODELS / "xgboost_v3_colsample_05_experiment.pkl"
LABELS = STATS / "05_risk_labels_v3_meal.csv"

OUT_PKL = MODELS / "xgboost_v3_meal_colsample_05_experiment.pkl"
OUT_JSON = STATS / "24_meal_clinical_score_colsample_experiment.json"
OUT_MD = STATS / "24_meal_clinical_score_colsample_experiment.md"
OUT_FIGURE = FIGURES / "xgb_v3_meal_colsample_05_gain_importance.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
COLSAMPLE_EXPERIMENT = 0.5
ABLATION_ACCURACY = 0.77519739897817
DAY_COLSAMPLE_JSON = STATS / "23_clinical_score_colsample_experiment.json"

RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

FEATURES = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "ckd_stage_encoded",
    "stage_numeric",
    "k_p_product",
    "protein_sodium_ratio",
    "clinical_score",
]
RAW_NUTRIENTS = ["potassium", "phosphorus", "protein_per_kg", "sodium"]

PROTECTED = {
    DAY_PKL: "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5",
    MEAL_PKL: "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db",
    DECOMP_PKL: "19df7c874bc0817e8d2ca80fb9ea50223e2ca3a87aa33577da77d55f70b422a8",
    RESIDUAL_PKL: "165cd163ef9768a0f4328b24ed18ad3c473b3b7a808dfca8974b24a8cb73a685",
    DAY_COLSAMPLE_PKL: "40ed2493150ee7ca42598619cc090e31f39ec63356151ba387597ddf55ca82e4",
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


def build_frame() -> pd.DataFrame:
    """Match meal-training FE from scripts/train_xgboost_v3_meal.py."""
    df = pd.read_csv(LABELS)
    for col in ("potassium", "phosphorus", "protein_per_kg", "sodium", "clinical_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=[
            "risk_label",
            "clinical_score",
            "potassium",
            "phosphorus",
            "protein_per_kg",
            "sodium",
            "ckd_stage",
        ]
    ).copy()
    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
    df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
    df["protein_sodium_ratio"] = df["protein_per_kg"] / (
        df["sodium"] / 1000.0 + 1e-6
    )
    df["y"] = df["risk_label"].map(RISK_ENCODE)
    df = df.dropna(subset=["ckd_stage_encoded", "stage_numeric", "y"])
    df["y"] = df["y"].astype(int)
    return df


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
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    gains = {feature: float(raw.get(feature, 0.0)) for feature in FEATURES}
    total = sum(gains.values())
    if total <= 0:
        return {feature: 0.0 for feature in FEATURES}
    return {feature: value / total for feature, value in gains.items()}


def save_gain_figure(
    production_gain: dict[str, float],
    experiment_gain: dict[str, float],
) -> None:
    frame = pd.DataFrame(
        {
            "Production meal (colsample=0.9)": production_gain,
            "Experiment (colsample=0.5)": experiment_gain,
        }
    ).loc[FEATURES]
    frame = frame.sort_values("Experiment (colsample=0.5)", ascending=True)
    axis = frame.plot(kind="barh", figsize=(10, 6), width=0.8)
    axis.set_title("Meal XGBoost normalized gain importance")
    axis.set_xlabel("Share of total gain")
    axis.set_ylabel("Feature")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    plt.tight_layout()
    plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close()


def compare_to_day(
    meal_acc: float,
    meal_acc_delta: float,
    meal_clinical_prod: float,
    meal_clinical_exp: float,
    meal_raw_prod: float,
    meal_raw_exp: float,
) -> str:
    day = json.loads(DAY_COLSAMPLE_JSON.read_text())
    day_acc_delta = day["accuracy_delta_experiment_minus_production"]
    day_clinical_drop = day["clinical_score_gain_share"]["drop"]
    day_raw_increase = day["raw_nutrient_total_gain_share"]["increase"]
    meal_clinical_drop = meal_clinical_prod - meal_clinical_exp
    meal_raw_increase = meal_raw_exp - meal_raw_prod

    if meal_acc_delta < -0.01 and abs(day_acc_delta) < 1e-9:
        pattern = (
            "ACCURACY_COST: meal-scale paid an accuracy cost that day-scale "
            "did not (day accuracy unchanged)."
        )
    elif meal_clinical_drop > day_clinical_drop + 0.05:
        pattern = (
            "LARGER_EFFECT: meal-scale reduced clinical_score dominance more "
            "than day-scale."
        )
    elif meal_clinical_drop < day_clinical_drop - 0.05:
        pattern = (
            "SMALLER_EFFECT: meal-scale reduced clinical_score dominance less "
            "than day-scale."
        )
    else:
        pattern = (
            "SIMILAR_PATTERN: meal-scale shows a comparable accuracy/dominance "
            "trade-off to day-scale."
        )

    return (
        f"{pattern} "
        f"Day: accuracy delta {100 * day_acc_delta:+.2f} pp, "
        f"clinical_score {day['clinical_score_gain_share']['production']:.2%}→"
        f"{day['clinical_score_gain_share']['experiment']:.2%} "
        f"({-100 * day_clinical_drop:+.2f} pp), "
        f"raw nutrients {day['raw_nutrient_total_gain_share']['production']:.2%}→"
        f"{day['raw_nutrient_total_gain_share']['experiment']:.2%} "
        f"({100 * day_raw_increase:+.2f} pp). "
        f"Meal: accuracy {meal_acc:.2%} "
        f"(delta {100 * meal_acc_delta:+.2f} pp), "
        f"clinical_score {meal_clinical_prod:.2%}→{meal_clinical_exp:.2%} "
        f"({-100 * meal_clinical_drop:+.2f} pp), "
        f"raw nutrients {meal_raw_prod:.2%}→{meal_raw_exp:.2%} "
        f"({100 * meal_raw_increase:+.2f} pp)."
    )


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = assert_protected("before")

    frame = build_frame()
    X = frame[FEATURES]
    y = frame["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"split: train={len(X_train)} test={len(X_test)}")
    if len(X_test) != 2153:
        raise RuntimeError(
            f"Expected meal holdout n=2153, got {len(X_test)}. "
            "Split or labels may have drifted from production meal training."
        )

    production = joblib.load(MEAL_PKL)
    production_params = production_hyperparams(production)
    prod_colsample = float(production_params["colsample_bytree"])
    if abs(prod_colsample - 0.9) > 1e-9:
        raise RuntimeError(
            "Expected production meal colsample_bytree=0.9, got "
            f"{prod_colsample!r}"
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
            "production": 0.9,
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
    kp_production = production_gain["k_p_product"]
    kp_experiment = experiment_gain["k_p_product"]

    accuracy_delta_production = (
        experiment_metrics["accuracy"] - production_metrics["accuracy"]
    )
    accuracy_delta_ablation = experiment_metrics["accuracy"] - ABLATION_ACCURACY
    clinical_drop = (
        production_gain["clinical_score"] - experiment_gain["clinical_score"]
    )
    raw_gain_increase = raw_experiment - raw_production
    kp_gain_increase = kp_experiment - kp_production

    promising = (
        accuracy_delta_production >= -0.02
        and clinical_drop >= 0.10
        and raw_gain_increase >= 0.10
    )
    if promising:
        verdict = (
            "PROMISING_MIDDLE_GROUND: accuracy stayed within 2 percentage points "
            "of production meal, clinical_score dominance fell by at least 10 "
            "points, and the four raw nutrients gained at least 10 points of "
            "total gain."
        )
    elif accuracy_delta_production < -0.04:
        verdict = (
            "NEGATIVE_ACCURACY: accuracy moved substantially toward the "
            "no-score meal ablation result."
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

    day_comparison = compare_to_day(
        experiment_metrics["accuracy"],
        accuracy_delta_production,
        production_gain["clinical_score"],
        experiment_gain["clinical_score"],
        raw_production,
        raw_experiment,
    )

    save_gain_figure(production_gain, experiment_gain)

    artifact_payload = {
        "model": experiment,
        "feature_names": FEATURES,
        "experiment": "clinical_score_colsample_05_meal_v3",
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
            "label_source": str(LABELS.relative_to(ROOT)),
        },
        "class_weight": class_weight,
    }
    joblib.dump(artifact_payload, OUT_PKL)

    after = assert_protected("after")
    results = {
        "status": "EXPERIMENT_ONLY_NOT_FOR_PRODUCTION",
        "verdict": verdict,
        "promising_middle_ground": promising,
        "day_comparison": day_comparison,
        "promising_criteria": {
            "accuracy_within_production_percentage_points": 2.0,
            "minimum_clinical_score_share_drop_percentage_points": 10.0,
            "minimum_raw_nutrient_share_increase_percentage_points": 10.0,
        },
        "protected_artifacts_unchanged": before == after,
        "protected_sha256": after,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "split": artifact_payload["split"],
        "class_weight": class_weight,
        "production_hyperparams": production_params,
        "experiment_hyperparams": experiment_params,
        "changed_from_production": changed_params,
        "note_production_colsample": (
            "Production meal already used colsample_bytree=0.9 (unlike day "
            "production which used 1.0). Experiment further reduces it to 0.5."
        ),
        "production_meal_holdout": production_metrics,
        "experiment_meal_holdout": experiment_metrics,
        "no_clinical_score_ablation_reference": {
            "accuracy": ABLATION_ACCURACY,
            "source": "outputs/stats/10_xgboost_v3_meal_deep_eval.json",
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
        "k_p_product_gain_share": {
            "production": kp_production,
            "experiment": kp_experiment,
            "increase": kp_gain_increase,
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
            FEATURES,
            key=lambda name: experiment_gain[name],
            reverse=True,
        )
    )
    OUT_MD.write_text(
        f"""# Meal-scale clinical-score column-subsampling experiment

**Status: EXPERIMENT ONLY — not for production.**

## Verdict

**{verdict}**

## Day-scale comparison

{day_comparison}

Only `colsample_bytree` changed: production meal **0.9** → experiment **0.5**.
(Note: production meal already used 0.9, unlike day production which used 1.0.)
The feature set, split, class weights, and all other model hyperparameters match
the production meal model.

## Holdout performance

Same stratified split: `test_size=0.2`, `random_state=42`, n={len(y_test)}.

| Model | Accuracy | Macro F1 | Weighted AUC |
|---|---:|---:|---:|
| Production meal | {production_metrics['accuracy']:.2%} | {production_metrics['f1_macro']:.4f} | {production_metrics['auc_weighted']:.4f} |
| Colsample 0.5 experiment | {experiment_metrics['accuracy']:.2%} | {experiment_metrics['f1_macro']:.4f} | {experiment_metrics['auc_weighted']:.4f} |
| Existing no-score ablation | {ABLATION_ACCURACY:.2%} | 0.7602 | 0.9491 |

- Experiment vs production: **{100 * accuracy_delta_production:+.2f} percentage points**
- Experiment vs no-score ablation: **{100 * accuracy_delta_ablation:+.2f} percentage points**

## Normalized gain importance

This is gain-based feature importance, matching the method behind Figure 5.16.

| Feature | Production | Colsample 0.5 | Change |
|---|---:|---:|---:|
{gain_rows}

- `clinical_score`: **{production_gain['clinical_score']:.2%} → {experiment_gain['clinical_score']:.2%}** ({-100 * clinical_drop:+.2f} percentage points)
- Four raw nutrients combined: **{raw_production:.2%} → {raw_experiment:.2%}** ({100 * raw_gain_increase:+.2f} percentage points)
- `k_p_product`: **{kp_production:.2%} → {kp_experiment:.2%}** ({100 * kp_gain_increase:+.2f} percentage points)

## Artifacts

- `{OUT_PKL.relative_to(ROOT)}` — joblib dictionary with `not_for_production=True`
- `{OUT_JSON.relative_to(ROOT)}`
- `{OUT_FIGURE.relative_to(ROOT)}`

Protected production and all four prior experimental artifacts were hash-verified
before and after training and remained unchanged.
"""
    )

    print("\nHoldout accuracy")
    print(f"  production meal:  {production_metrics['accuracy']:.6f}")
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
    print(
        "  k_p_product:    "
        f"{kp_production:.2%} -> {kp_experiment:.2%}"
    )
    print(f"\n{verdict}")
    print(f"\nDay comparison: {day_comparison}")
    print(f"\nWrote {OUT_PKL}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
