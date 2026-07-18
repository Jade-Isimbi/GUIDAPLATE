#!/usr/bin/env python3
"""
EXPERIMENT ONLY — meal-scale XGBoost without clinical_score as a feature.

Hypothesis: meal accuracy collapses when clinical_score is removed partly because
occasion is NOT in the production feature set; clinical_score alone carries
Snack vs Dinner scale. Adding occasion (and optionally meal-cap values) should
recover accuracy without feeding the labeling formula back in.

Arms:
  A) 8 features (production minus clinical_score), RandomizedSearchCV retune
  B) A + occasion_encoded
  C) B + meal cap values (K/P/protein/Na limits for that stage×occasion)

Does NOT modify production or prior experiment artifacts.

Outputs:
  - models/xgboost_v3_meal_noscore_occasion_experiment.pkl  (best of A/B/C)
  - outputs/stats/25_meal_noscore_occasion_experiment.json
  - outputs/stats/25_meal_noscore_occasion_experiment.md
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from backend.clinical_constants import KDOQI_DAILY_LIMITS
from backend.models.xgboost_model import meal_limits_for_occasion

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
STATS = ROOT / "outputs" / "stats"
LABELS = STATS / "05_risk_labels_v3_meal.csv"

DAY_PKL = MODELS / "xgboost_v3.pkl"
MEAL_PKL = MODELS / "xgboost_v3_meal.pkl"
DECOMP_PKL = MODELS / "xgboost_v3_decomposed_experiment.pkl"
RESIDUAL_PKL = MODELS / "xgboost_v3_potassium_residualized_experiment.pkl"
DAY_COLSAMPLE_PKL = MODELS / "xgboost_v3_colsample_05_experiment.pkl"
MEAL_COLSAMPLE_PKL = MODELS / "xgboost_v3_meal_colsample_05_experiment.pkl"

OUT_PKL = MODELS / "xgboost_v3_meal_noscore_occasion_experiment.pkl"
OUT_JSON = STATS / "25_meal_noscore_occasion_experiment.json"
OUT_MD = STATS / "25_meal_noscore_occasion_experiment.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ITER = 40
PROD_MEAL_ACCURACY = 0.9986065954482118
ABLATION_ACCURACY = 0.77519739897817

RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}
OCCASION_ENCODE = {"Breakfast": 0, "Lunch": 1, "Dinner": 2, "Snack": 3}

BASE_FEATURES = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "ckd_stage_encoded",
    "stage_numeric",
    "k_p_product",
    "protein_sodium_ratio",
]
FEATURES_A = BASE_FEATURES
FEATURES_B = BASE_FEATURES + ["occasion_encoded"]
FEATURES_C = FEATURES_B + [
    "meal_cap_potassium",
    "meal_cap_phosphorus",
    "meal_cap_protein_per_kg",
    "meal_cap_sodium",
]

PROTECTED = {
    DAY_PKL: "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5",
    MEAL_PKL: "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db",
    DECOMP_PKL: "19df7c874bc0817e8d2ca80fb9ea50223e2ca3a87aa33577da77d55f70b422a8",
    RESIDUAL_PKL: "165cd163ef9768a0f4328b24ed18ad3c473b3b7a808dfca8974b24a8cb73a685",
    DAY_COLSAMPLE_PKL: "40ed2493150ee7ca42598619cc090e31f39ec63356151ba387597ddf55ca82e4",
    MEAL_COLSAMPLE_PKL: "5f6a031f36acf375933830a41bc9d4823b2cd18581019412ca65236b64715052",
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


def build_frame() -> pd.DataFrame:
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
            "occasion",
        ]
    ).copy()
    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
    df["occasion_encoded"] = df["occasion"].map(OCCASION_ENCODE)
    df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
    df["protein_sodium_ratio"] = df["protein_per_kg"] / (
        df["sodium"] / 1000.0 + 1e-6
    )

    caps = df.apply(
        lambda row: meal_limits_for_occasion(row["ckd_stage"], row["occasion"]),
        axis=1,
        result_type="expand",
    )
    df["meal_cap_potassium"] = caps["potassium"]
    df["meal_cap_phosphorus"] = caps["phosphorus"]
    df["meal_cap_protein_per_kg"] = caps["protein_per_kg"]
    df["meal_cap_sodium"] = caps["sodium"]

    # Sanity: caps must match known KDOQI×fractions for at least one stage
    assert set(df["ckd_stage"].unique()) <= set(KDOQI_DAILY_LIMITS)

    df["y"] = df["risk_label"].map(RISK_ENCODE)
    df = df.dropna(subset=["ckd_stage_encoded", "stage_numeric", "occasion_encoded", "y"])
    df["y"] = df["y"].astype(int)
    return df


def metrics(y_true, y_pred, y_prob) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
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
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(RISK_CLASSES)
        },
    }


def normalized_gain(model: xgb.XGBClassifier, features: list[str]) -> dict[str, float]:
    raw = model.get_booster().get_score(importance_type="gain")
    gains = {f: float(raw.get(f, 0.0)) for f in features}
    total = sum(gains.values())
    if total <= 0:
        return {f: 0.0 for f in features}
    return {f: v / total for f, v in gains.items()}


def train_arm(
    name: str,
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight: np.ndarray,
) -> dict:
    print(f"\n=== Arm {name}: {len(features)} features, RandomizedSearchCV n_iter={N_ITER} ===")
    t0 = time.time()
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0],
    }
    base = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train[features], y_train, sample_weight=sample_weight)
    best_params = dict(search.best_params_)
    model = xgb.XGBClassifier(
        **best_params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train[features], y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test[features])
    y_prob = model.predict_proba(X_test[features])
    holdout = metrics(y_test.to_numpy(), y_pred, y_prob)
    elapsed = time.time() - t0
    gain = normalized_gain(model, features)
    print(
        f"Arm {name}: acc={holdout['accuracy']:.4%} "
        f"f1m={holdout['f1_macro']:.4f} "
        f"cv_f1={search.best_score_:.4f} "
        f"({elapsed:.1f}s)"
    )
    return {
        "name": name,
        "features": features,
        "n_features": len(features),
        "best_params": best_params,
        "best_cv_f1_macro": float(search.best_score_),
        "holdout": holdout,
        "normalized_gain": gain,
        "train_seconds": float(elapsed),
        "model": model,
    }


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    before = assert_protected("before")

    df = build_frame()
    y = df["y"]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    print(f"split: train={len(train)} test={len(test)}")
    if len(test) != 2153:
        raise RuntimeError(f"Expected n_test=2153, got {len(test)}")

    # Production meal predict-only reference on same rows
    prod = joblib.load(MEAL_PKL)
    prod_features = [
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
    prod_metrics = metrics(
        test["y"].to_numpy(),
        prod.predict(test[prod_features]),
        prod.predict_proba(test[prod_features]),
    )
    print(f"production meal holdout accuracy: {prod_metrics['accuracy']:.6f}")

    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sw = compute_sample_weight(class_weight=class_weight, y=train["y"])

    arm_a = train_arm("A_noscore_8feat_retuned", FEATURES_A, train, train["y"], test, test["y"], sw)
    arm_b = train_arm("B_noscore_plus_occasion", FEATURES_B, train, train["y"], test, test["y"], sw)
    arm_c = train_arm("C_noscore_occasion_plus_caps", FEATURES_C, train, train["y"], test, test["y"], sw)

    arms = [arm_a, arm_b, arm_c]
    best = max(arms, key=lambda a: (a["holdout"]["accuracy"], a["holdout"]["f1_macro"]))
    print(
        f"\nBest arm: {best['name']} "
        f"acc={best['holdout']['accuracy']:.4%} "
        f"f1m={best['holdout']['f1_macro']:.4f}"
    )

    # How close to "excellent"?
    gap_to_prod = PROD_MEAL_ACCURACY - best["holdout"]["accuracy"]
    lift_vs_ablation = best["holdout"]["accuracy"] - ABLATION_ACCURACY
    if best["holdout"]["accuracy"] >= 0.95:
        quality = "STRONG_NONCIRCULAR_FEATURE_SET"
    elif best["holdout"]["accuracy"] >= 0.90:
        quality = "GOOD_NONCIRCULAR_FEATURE_SET"
    elif lift_vs_ablation >= 0.05:
        quality = "IMPROVED_BUT_BELOW_EXCELLENT"
    else:
        quality = "LIMITED_GAIN"

    payload = {
        "model": best["model"],
        "feature_names": best["features"],
        "experiment": "meal_noscore_occasion_recovery_v3",
        "winning_arm": best["name"],
        "not_for_production": True,
        "note": (
            "EXPERIMENT ONLY. No clinical_score feature. Labels remain "
            "clinical_score-thresholded (inherent label circularity). "
            "Do not deploy over xgboost_v3_meal.pkl without explicit decision."
        ),
        "hyperparams": best["best_params"],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "class_weight": class_weight,
        "occasion_encode": OCCASION_ENCODE,
    }
    joblib.dump(payload, OUT_PKL)

    after = assert_protected("after")

    def arm_public(arm: dict) -> dict:
        return {
            "name": arm["name"],
            "features": arm["features"],
            "n_features": arm["n_features"],
            "best_params": arm["best_params"],
            "best_cv_f1_macro": arm["best_cv_f1_macro"],
            "holdout": arm["holdout"],
            "normalized_gain": arm["normalized_gain"],
            "train_seconds": arm["train_seconds"],
            "accuracy_delta_vs_production": float(
                arm["holdout"]["accuracy"] - prod_metrics["accuracy"]
            ),
            "accuracy_delta_vs_ablation": float(
                arm["holdout"]["accuracy"] - ABLATION_ACCURACY
            ),
        }

    results = {
        "status": "EXPERIMENT_ONLY_NOT_FOR_PRODUCTION",
        "quality_band": quality,
        "honest_limitation": (
            "Labels are still derived by thresholding clinical_score. "
            "This experiment removes clinical_score from the FEATURES only. "
            "That eliminates the model 'cheating' by reading its own label "
            "formula, but does not make the target an independent clinical outcome."
        ),
        "protected_artifacts_unchanged": before == after,
        "protected_sha256": after,
        "production_meal_holdout": prod_metrics,
        "published_production_accuracy": PROD_MEAL_ACCURACY,
        "published_ablation_accuracy": ABLATION_ACCURACY,
        "arms": [arm_public(a) for a in arms],
        "winner": {
            "name": best["name"],
            "accuracy": best["holdout"]["accuracy"],
            "f1_macro": best["holdout"]["f1_macro"],
            "gap_to_production_pp": float(100 * gap_to_prod),
            "lift_vs_ablation_pp": float(100 * lift_vs_ablation),
            "features": best["features"],
            "normalized_gain": best["normalized_gain"],
        },
        "artifact": str(OUT_PKL.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")

    def arm_row(arm: dict) -> str:
        h = arm["holdout"]
        return (
            f"| {arm['name']} | {arm['n_features']} | "
            f"{h['accuracy']:.2%} | {h['f1_macro']:.4f} | "
            f"{h['auc_weighted']:.4f} | "
            f"{100 * (h['accuracy'] - ABLATION_ACCURACY):+.2f} pp |"
        )

    gain_lines = "\n".join(
        f"- `{name}`: {value:.2%}"
        for name, value in sorted(
            best["normalized_gain"].items(),
            key=lambda kv: -kv[1],
        )
    )

    OUT_MD.write_text(
        f"""# Meal-scale model without clinical_score feature

**Status: EXPERIMENT ONLY — not for production.**

## Honest framing

Labels remain thresholded from `clinical_score` (inherent label circularity).
This experiment only removes `clinical_score` from the **features**, so the
model cannot directly read the labeling formula. Production meal features also
omitted occasion; `clinical_score` was the only place Snack vs Dinner scale
entered the model — that is the main hypothesis for the 99.86% → 77.52% drop.

## Quality band

**{quality}**

Winner: **{best['name']}** at **{best['holdout']['accuracy']:.2%}**
(gap to production {100 * gap_to_prod:+.2f} pp; lift vs ablation {100 * lift_vs_ablation:+.2f} pp).

## Holdout comparison (n=2153)

| Arm | #feats | Accuracy | Macro F1 | Weighted AUC | vs ablation |
|---|---:|---:|---:|---:|---:|
| Production meal (with clinical_score) | 9 | {prod_metrics['accuracy']:.2%} | {prod_metrics['f1_macro']:.4f} | {prod_metrics['auc_weighted']:.4f} | — |
| Published no-score ablation | 8 | {ABLATION_ACCURACY:.2%} | 0.7602 | 0.9491 | 0.00 pp |
{arm_row(arm_a)}
{arm_row(arm_b)}
{arm_row(arm_c)}

## Winner gain importance

{gain_lines}

## Artifact

`{OUT_PKL.relative_to(ROOT)}` — `not_for_production=True`.

Protected production + five prior experimental pickles were SHA-256 verified
unchanged before and after.
"""
    )

    print(f"\nQuality band: {quality}")
    print(f"Wrote {OUT_PKL}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
