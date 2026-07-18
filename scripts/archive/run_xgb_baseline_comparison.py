#!/usr/bin/env python3
"""
XGBoost-family baseline comparison (day-scale, risk labels v3).

Trains RF / Decision Tree / SVC / Logistic Regression on the same holdout
protocol as production day XGBoost, and scores models/xgboost_v3.pkl
predict-only on that test split.

Never writes to models/xgboost_v3.pkl (or any production model).

Usage (repo root):
  ./venv311/bin/python3 scripts/run_xgb_baseline_comparison.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "stats"
LABELS = STATS / "05_risk_labels_v3.csv"
DAY_PKL = ROOT / "models" / "xgboost_v3.pkl"
OUT_CSV = STATS / "13_xgb_family_comparison.csv"

PROTECTED_SHA256 = (
    "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_X_SHAPE = (1476, 9)
EXPECTED_TRAIN = 1180
EXPECTED_TEST = 296

RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
# Literal maps from notebooks/04c_xgboost_v3_raw_features.ipynb
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

# Literal FEATURES_V3 order from 04c
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
            f"FATAL [{when}]: models/xgboost_v3.pkl SHA-256 mismatch!\n"
            f"  expected: {PROTECTED_SHA256}\n"
            f"  got:      {digest}\n"
            "Stopping — do not trust this run as a baseline."
        )
    print(f"[{when}] xgboost_v3.pkl SHA-256 OK: {digest}")
    return digest


def build_frame() -> pd.DataFrame:
    """
    Load labels CSV (step 1) and apply 04c feature engineering literally.

    Source: notebooks/04c_xgboost_v3_raw_features.ipynb cells for STAGE_ENCODE,
    stage_numeric map, k_p_product, protein_sodium_ratio, FEATURES_V3.
    Label filter / nutrient dropna mirrors 04c Section 1 (and ablation), but
    reads only 05_risk_labels_v3.csv as requested (no cohort merge).
    """
    df = pd.read_csv(LABELS)
    df = df.dropna(subset=["risk_label", "clinical_score"]).copy()
    nutrient_cols = ["potassium", "phosphorus", "protein_per_kg", "sodium"]
    for col in nutrient_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=nutrient_cols)

    # --- literal from 04c feature construction cell ---
    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
    df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
    df["protein_sodium_ratio"] = df["protein_per_kg"] / (
        df["sodium"] / 1000 + 1e-6
    )
    # clinical_score already loaded from v3 labels file

    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    df = df.dropna(subset=["ckd_stage_encoded", "stage_numeric", "risk_encoded"])
    df["risk_encoded"] = df["risk_encoded"].astype(int)
    return df


def per_class_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = cm.sum()
    out: dict[str, float] = {}
    for i, label in enumerate(RISK_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        out[label] = float(tn / denom) if denom > 0 else float("nan")
    return out


def metrics_row(
    model_name: str,
    hyperparameters: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    spec = per_class_specificity(y_true, y_pred)
    row: dict = {
        "model": model_name,
        "hyperparameters": hyperparameters,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # one-vs-rest, weighted average (project convention in 10_xgboost_v3_metrics.csv)
        "auc_roc_ovr_weighted": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        ),
    }
    for i, label in enumerate(RISK_CLASSES):
        row[f"{label}_precision"] = float(prec[i])
        row[f"{label}_recall"] = float(rec[i])
        row[f"{label}_specificity"] = spec[label]
    return row


def main() -> int:
    STATS.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists():
        print(f"STOP: refusing to overwrite existing {OUT_CSV}")
        return 1

    assert_protected("before")

    # 1–2. Load + FE
    df = build_frame()
    if list(FEATURES) != [
        "potassium",
        "phosphorus",
        "protein_per_kg",
        "sodium",
        "ckd_stage_encoded",
        "stage_numeric",
        "k_p_product",
        "protein_sodium_ratio",
        "clinical_score",
    ]:
        print("STOP: FEATURES order drift")
        return 1

    X = df[FEATURES]
    y = df["risk_encoded"]

    # 3. Shape check
    if X.shape != EXPECTED_X_SHAPE:
        print(
            f"STOP: engineered feature matrix shape {X.shape} "
            f"!= expected {EXPECTED_X_SHAPE}. Not proceeding."
        )
        print(f"  n_rows after dropna: {len(df)}")
        print(f"  columns: {list(X.columns)}")
        return 1
    print(f"[OK] engineered X shape: {X.shape}")
    print(f"[OK] feature order: {list(X.columns)}")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if len(X_train) != EXPECTED_TRAIN or len(X_test) != EXPECTED_TEST:
        print(
            f"STOP: split sizes train={len(X_train)} test={len(X_test)} "
            f"!= expected {EXPECTED_TRAIN}/{EXPECTED_TEST}. Not proceeding."
        )
        return 1
    print(f"[OK] split: train={len(X_train)} test={len(X_test)}")

    y_test_np = y_test.to_numpy()
    rows: list[dict] = []

    # 5. Train candidates (hyperparameters recorded in CSV)
    candidates: list[tuple[str, str, object]] = [
        (
            "RandomForestClassifier",
            (
                "n_estimators=200, max_depth=12, min_samples_leaf=2, "
                "class_weight='balanced', random_state=42, n_jobs=-1"
            ),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "DecisionTreeClassifier",
            (
                "max_depth=10, min_samples_leaf=5, "
                "class_weight='balanced', random_state=42"
            ),
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "SVC",
            (
                "Pipeline(StandardScaler, "
                "SVC(kernel='rbf', C=1.0, gamma='scale', "
                "probability=True, class_weight='balanced', random_state=42))"
            ),
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "svc",
                        SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "LogisticRegression",
            (
                "Pipeline(StandardScaler, "
                "LogisticRegression(max_iter=2000, solver='lbfgs', "
                "class_weight='balanced', random_state=42)); "
                "sklearn 1.9 multinomial default for multi-class"
            ),
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    for name, hp, clf in candidates:
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        rows.append(metrics_row(name, hp, y_test_np, y_pred, y_prob))
        print(
            f"  {name}: acc={rows[-1]['accuracy']:.4f} "
            f"f1_macro={rows[-1]['f1_macro']:.4f} "
            f"auc={rows[-1]['auc_roc_ovr_weighted']:.4f}"
        )

    # 6. Production XGBoost predict-only
    print("Scoring production xgboost_v3.pkl (predict-only)...")
    xgb_model = joblib.load(DAY_PKL)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)
    rows.append(
        metrics_row(
            "XGBoost_v3_production",
            "predict_only models/xgboost_v3.pkl (not retrained)",
            y_test_np,
            y_pred_xgb,
            y_prob_xgb,
        )
    )
    print(
        f"  XGBoost_v3_production: acc={rows[-1]['accuracy']:.4f} "
        f"f1_macro={rows[-1]['f1_macro']:.4f} "
        f"auc={rows[-1]['auc_roc_ovr_weighted']:.4f}"
    )

    # 7–8. Save NEW CSV only
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # 9. Hash confirmation
    after = assert_protected("after")
    print(f"HASH_CONFIRMATION={after}")
    print(f"HASH_MATCHES_EXPECTED={after == PROTECTED_SHA256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
