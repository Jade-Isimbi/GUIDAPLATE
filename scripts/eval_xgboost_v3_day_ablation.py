#!/usr/bin/env python3
"""
Day XGBoost v3 clinical_score ablation (read-only on production pickle).

Mirrors meal deep-eval protocol in outputs/stats/10_xgboost_v3_meal_deep_eval.json:
  - full model metrics from the EXISTING models/xgboost_v3.pkl (predict only)
  - temporary retrain WITHOUT clinical_score on the same 04c train/test split
  - temporary retrain on raw nutrients + stage only (6 features)

NEVER writes to models/xgboost_v3.pkl. Verifies protected SHA256 before/after.

Usage (repo root):
  ./venv311/bin/python3 scripts/eval_xgboost_v3_day_ablation.py
"""

from __future__ import annotations

import hashlib
import json
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "stats"
DAY_PKL = ROOT / "models" / "xgboost_v3.pkl"
LABELS = STATS / "05_risk_labels_v3.csv"
OUT = STATS / "10_xgboost_v3_day_ablation.json"

PROTECTED_SHA256 = (
    "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

FEATURES_FULL = [
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
FEATURES_NO_SCORE = [f for f in FEATURES_FULL if f != "clinical_score"]
FEATURES_RAW6 = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "ckd_stage_encoded",
    "stage_numeric",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_protected(when: str) -> str:
    digest = sha256_file(DAY_PKL)
    if digest != PROTECTED_SHA256:
        raise RuntimeError(
            f"Day pickle hash mismatch {when}: got {digest}, expected {PROTECTED_SHA256}"
        )
    print(f"[{when}] xgboost_v3.pkl SHA256 OK: {digest}")
    return digest


def build_frame() -> pd.DataFrame:
    df = pd.read_csv(LABELS)
    df = df.dropna(subset=["risk_label", "clinical_score"]).copy()
    # Match 04c: require nutrient columns present for feature matrix
    for col in ("potassium", "phosphorus", "protein_per_kg", "sodium"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["potassium", "phosphorus", "protein_per_kg", "sodium"])
    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
    df["k_p_product"] = df["potassium"] * df["phosphorus"]
    df["protein_sodium_ratio"] = df["protein_per_kg"] / df["sodium"].replace(0, np.nan)
    df["protein_sodium_ratio"] = df["protein_sodium_ratio"].fillna(0.0)
    df["y"] = df["risk_label"].map(RISK_ENCODE)
    df = df.dropna(subset=["ckd_stage_encoded", "stage_numeric", "y"])
    df["y"] = df["y"].astype(int)
    return df


def metrics_block(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    per_class = {}
    for i, label in enumerate(RISK_CLASSES):
        per_class[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "auc_weighted": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        ),
        "auc_macro": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        ),
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def train_temp(X_train, y_train, X_test, y_test, feature_names: list[str], name: str):
    """In-memory clone using production hyperparams — never dumped to disk."""
    prod = joblib.load(DAY_PKL)
    params = prod.get_params()
    # Strip fitted-only / unsettable noise
    keep = {
        k: params[k]
        for k in (
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
        if k in params and params[k] is not None
    }
    clf = xgb.XGBClassifier(**keep)
    sw = compute_sample_weight("balanced", y_train)
    clf.fit(X_train[feature_names], y_train, sample_weight=sw)
    y_pred = clf.predict(X_test[feature_names])
    y_prob = clf.predict_proba(X_test[feature_names])
    block = metrics_block(name, y_test.to_numpy(), y_pred, y_prob)
    block["n_features"] = len(feature_names)
    block["features"] = feature_names
    block["protocol"] = (
        "temporary_retrain_same_split_as_04c; never written to models/xgboost_v3.pkl"
    )
    return block


def eval_production(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model = joblib.load(DAY_PKL)
    X = X_test[FEATURES_FULL]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    block = metrics_block(
        "day_xgb_full_production_pkl",
        y_test.to_numpy(),
        y_pred,
        y_prob,
    )
    block["n_features"] = len(FEATURES_FULL)
    block["features"] = FEATURES_FULL
    block["protocol"] = "predict_only_on_models/xgboost_v3.pkl"
    block["artifact"] = "xgboost_v3.pkl"
    return block


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    before = assert_protected("before")

    df = build_frame()
    X = df[FEATURES_FULL]
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    full = eval_production(X_test, y_test)
    no_score = train_temp(
        X_train, y_train, X_test, y_test, FEATURES_NO_SCORE, "day_xgb_no_clinical_score"
    )
    raw6 = train_temp(
        X_train, y_train, X_test, y_test, FEATURES_RAW6, "day_xgb_raw6"
    )

    after = assert_protected("after")
    if before != after:
        raise RuntimeError("Hash changed during ablation — aborting write")

    payload = {
        "model": "XGBoost v3 day",
        "artifact": "xgboost_v3.pkl",
        "protected_sha256": PROTECTED_SHA256,
        "sha256_verified_unchanged": True,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "label_source": "outputs/stats/05_risk_labels_v3.csv",
            "matches_notebook": "notebooks/04c_xgboost_v3_raw_features.ipynb",
        },
        "protocol_note": (
            "Full metrics: production pickle predict-only. "
            "Ablations: temporary in-memory retrain on the same train split "
            "without writing models/xgboost_v3.pkl — same protocol family as "
            "meal ablation_no_clinical_score / ablation_raw6 in "
            "10_xgboost_v3_meal_deep_eval.json."
        ),
        "day_model": full,
        "ablation_no_clinical_score": no_score,
        "ablation_raw6": raw6,
        "published_holdout_csv": {
            "path": "outputs/stats/10_xgboost_v3_metrics.csv",
            "accuracy": 0.9899,
            "f1_macro": 0.9853,
            "auc_roc": 0.9975,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {OUT}")
    print(
        f"FULL  acc={full['accuracy']:.4f} f1m={full['f1_macro']:.4f} "
        f"auc_w={full['auc_weighted']:.4f}"
    )
    print(
        f"NO_CS acc={no_score['accuracy']:.4f} f1m={no_score['f1_macro']:.4f} "
        f"auc_w={no_score['auc_weighted']:.4f}"
    )
    print(
        f"RAW6  acc={raw6['accuracy']:.4f} f1m={raw6['f1_macro']:.4f} "
        f"auc_w={raw6['auc_weighted']:.4f}"
    )


if __name__ == "__main__":
    main()
