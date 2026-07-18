#!/usr/bin/env python3
"""
EXPERIMENT ONLY — clinical_score decomposition for SHAP potassium visibility.

Trains a day-scale XGBoost with clinical_score split into four weighted
severity terms. Does NOT write or modify:
  - models/xgboost_v3.pkl
  - models/xgboost_v3_meal.pkl

Outputs:
  - models/xgboost_v3_decomposed_experiment.pkl  (NOT for production)
  - outputs/stats/21_clinical_score_decomposition_experiment.json
  - outputs/stats/21_clinical_score_decomposition_experiment.md

Usage (repo root):
  ./venv311/bin/python3 scripts/run_clinical_score_decomposition_experiment.py
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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from backend.clinical_constants import (
    CLINICAL_SEVERITY_WEIGHTS,
    KDOQI_DAILY_LIMITS,
)
from backend.models.xgboost_model import compute_clinical_score

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "stats"
MODELS = ROOT / "models"
LABELS = STATS / "05_risk_labels_v3.csv"
DAY_PKL = MODELS / "xgboost_v3.pkl"
MEAL_PKL = MODELS / "xgboost_v3_meal.pkl"
OUT_PKL = MODELS / "xgboost_v3_decomposed_experiment.pkl"
OUT_JSON = STATS / "21_clinical_score_decomposition_experiment.json"
OUT_MD = STATS / "21_clinical_score_decomposition_experiment.md"

PROTECTED_DAY = (
    "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5"
)
PROTECTED_MEAL = (
    "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

# Production 9-feature template minus clinical_score, plus 4 severity terms.
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
SEVERITY_FEATURES = [
    "k_severity",
    "p_severity",
    "protein_severity",
    "na_severity",
]
FEATURES_DECOMP = BASE_FEATURES + SEVERITY_FEATURES  # 12
FEATURES_PROD = BASE_FEATURES + ["clinical_score"]  # 9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_production_untouched(when: str) -> dict[str, str]:
    digests = {}
    for path, expected, label in (
        (DAY_PKL, PROTECTED_DAY, "xgboost_v3.pkl"),
        (MEAL_PKL, PROTECTED_MEAL, "xgboost_v3_meal.pkl"),
    ):
        digest = sha256_file(path)
        digests[label] = digest
        if digest != expected:
            raise RuntimeError(
                f"FATAL [{when}]: {label} SHA-256 changed!\n"
                f"  expected: {expected}\n"
                f"  got:      {digest}\n"
                "Experiment must not modify production pickles."
            )
        print(f"[{when}] {label} SHA-256 OK: {digest}")
    return digests


def piecewise_severity(ratio: float) -> float:
    """Same f() as compute_clinical_score (unweighted branch)."""
    if ratio > 1.0:
        return 1.0 + (ratio - 1.0) * 2.0
    return ratio


def severity_components(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> dict[str, float]:
    limits = KDOQI_DAILY_LIMITS[ckd_stage]
    values = {
        "potassium": potassium,
        "phosphorus": phosphorus,
        "protein_per_kg": protein_per_kg,
        "sodium": sodium,
    }
    weight_keys = {
        "k_severity": ("potassium", "potassium"),
        "p_severity": ("phosphorus", "phosphorus"),
        "protein_severity": ("protein_per_kg", "protein"),
        "na_severity": ("sodium", "sodium"),
    }
    out: dict[str, float] = {}
    for feat, (nutrient, wkey) in weight_keys.items():
        weight = CLINICAL_SEVERITY_WEIGHTS[wkey]
        ratio = values[nutrient] / limits[nutrient]
        out[feat] = float(weight * piecewise_severity(ratio))
    return out


def build_frame() -> pd.DataFrame:
    """Match production day FE from 04c / backend (not the ablation-script variant)."""
    df = pd.read_csv(LABELS)
    for col in ("potassium", "phosphorus", "protein_per_kg", "sodium"):
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
    # Recompute clinical_score from production function for identity check
    df["clinical_score_recomputed"] = df.apply(
        lambda r: compute_clinical_score(
            float(r["potassium"]),
            float(r["phosphorus"]),
            float(r["protein_per_kg"]),
            float(r["sodium"]),
            r["ckd_stage"],
        ),
        axis=1,
    )
    comps = df.apply(
        lambda r: severity_components(
            float(r["potassium"]),
            float(r["phosphorus"]),
            float(r["protein_per_kg"]),
            float(r["sodium"]),
            r["ckd_stage"],
        ),
        axis=1,
        result_type="expand",
    )
    for col in SEVERITY_FEATURES:
        df[col] = comps[col]
    df["severity_sum"] = df[SEVERITY_FEATURES].sum(axis=1)
    df["y"] = df["risk_label"].map(RISK_ENCODE)
    df = df.dropna(subset=["ckd_stage_encoded", "stage_numeric", "y"])
    df["y"] = df["y"].astype(int)
    return df


def clone_prod_hyperparams() -> dict:
    prod = joblib.load(DAY_PKL)
    params = prod.get_params()
    keep_keys = (
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
        k: params[k]
        for k in keep_keys
        if k in params and params[k] is not None
    }


def metrics(y_true, y_pred, y_prob) -> dict:
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


def shap_mean_abs(model, X: pd.DataFrame, class_idx: int) -> list[tuple[str, float]]:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = np.asarray(model.get_booster().predict(dmat, pred_contribs=True))
    # (n, n_class, n_feat+1)
    vals = np.abs(contribs[:, class_idx, : X.shape[1]]).mean(axis=0)
    ranked = sorted(zip(X.columns.tolist(), vals.tolist()), key=lambda t: -t[1])
    return [(n, float(v)) for n, v in ranked]


def main() -> int:
    STATS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    before = assert_production_untouched("before")

    df = build_frame()
    # Identity: severity sum == clinical_score
    max_abs_vs_recomputed = float(
        (df["severity_sum"] - df["clinical_score_recomputed"]).abs().max()
    )
    max_abs_vs_csv = float((df["severity_sum"] - df["clinical_score"]).abs().max())
    print(f"max |severity_sum - clinical_score_recomputed| = {max_abs_vs_recomputed:.3e}")
    print(f"max |severity_sum - clinical_score (CSV)|         = {max_abs_vs_csv:.3e}")
    if max_abs_vs_recomputed > 1e-9:
        raise RuntimeError("Decomposition does not reconstruct clinical_score")

    sample_rows = (
        df[
            [
                "SEQN",
                "ckd_stage",
                "clinical_score_recomputed",
                "severity_sum",
                *SEVERITY_FEATURES,
            ]
        ]
        .head(5)
        .to_dict(orient="records")
    )

    X_prod = df[FEATURES_PROD]
    X_decomp = df[FEATURES_DECOMP]
    y = df["y"]

    Xtr_p, Xte_p, ytr, yte = train_test_split(
        X_prod, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    Xtr_d, Xte_d, ytr2, yte2 = train_test_split(
        X_decomp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    assert list(ytr.index) == list(ytr2.index)
    assert list(yte.index) == list(yte2.index)
    print(f"split: train={len(ytr)} test={len(yte)}")

    # Production pickle predict-only (reference)
    prod = joblib.load(DAY_PKL)
    y_pred_prod = prod.predict(Xte_p)
    y_prob_prod = prod.predict_proba(Xte_p)
    prod_metrics = metrics(yte.to_numpy(), y_pred_prod, y_prob_prod)
    print(f"production day holdout accuracy: {prod_metrics['accuracy']:.6f}")

    # Train experimental model with production hyperparams
    hyper = clone_prod_hyperparams()
    print("cloned hyperparams:", {k: hyper[k] for k in ("max_depth", "n_estimators", "learning_rate")})
    # Match 04c: upweight MODERATE
    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sw = compute_sample_weight(class_weight=class_weight, y=ytr2)
    clf = xgb.XGBClassifier(**hyper)
    clf.fit(Xtr_d, ytr2, sample_weight=sw)
    y_pred = clf.predict(Xte_d)
    y_prob = clf.predict_proba(Xte_d)
    decomp_metrics = metrics(yte2.to_numpy(), y_pred, y_prob)
    print(f"decomposed experiment holdout accuracy: {decomp_metrics['accuracy']:.6f}")
    delta = decomp_metrics["accuracy"] - prod_metrics["accuracy"]
    print(f"accuracy delta (decomp - prod): {delta:+.6f}")

    # SHAP on decomposed model
    shap_by_class: dict[str, dict] = {}
    for cls_name, idx in (("HIGH", 2), ("MODERATE", 1)):
        ranked = shap_mean_abs(clf, Xte_d, idx)
        severity_only = [(n, v) for n, v in ranked if n in SEVERITY_FEATURES]
        raw_nutrients = [
            (n, v)
            for n, v in ranked
            if n in ("potassium", "phosphorus", "protein_per_kg", "sodium")
        ]
        shap_by_class[cls_name] = {
            "overall_top5": ranked[:5],
            "severity_ranked": severity_only,
            "severity_top2": severity_only[:2],
            "raw_nutrient_ranked": raw_nutrients,
        }
        print(f"\nSHAP {cls_name} severity ranked:")
        for n, v in severity_only:
            print(f"  {n:18s} {v:.6f}")
        print(f"  top-2 severity: {severity_only[0][0]}, {severity_only[1][0]}")

    # Production SHAP raw-nutrient ranks (reference: K was never top-2)
    shap_prod_ref: dict[str, list] = {}
    for cls_name, idx in (("HIGH", 2), ("MODERATE", 1)):
        ranked = shap_mean_abs(prod, Xte_p, idx)
        raw = [
            (n, v)
            for n, v in ranked
            if n in ("potassium", "phosphorus", "protein_per_kg", "sodium")
        ]
        shap_prod_ref[cls_name] = raw
        print(f"\nPROD SHAP {cls_name} raw nutrients:")
        for n, v in raw:
            print(f"  {n:18s} {v:.6f}")

    # Persist experiment artifact only
    payload = {
        "model": clf,
        "feature_names": FEATURES_DECOMP,
        "experiment": "clinical_score_decomposition_day_v3",
        "not_for_production": True,
        "note": (
            "EXPERIMENT ONLY. Do not deploy. Production remains "
            "models/xgboost_v3.pkl and models/xgboost_v3_meal.pkl."
        ),
        "hyperparams": hyper,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
        },
    }
    joblib.dump(payload, OUT_PKL)
    print(f"\nWrote experimental artifact: {OUT_PKL}")

    after = assert_production_untouched("after")

    results = {
        "status": "EXPERIMENT_ONLY_NOT_FOR_PRODUCTION",
        "production_artifacts_unchanged": before == after,
        "production_sha256": after,
        "identity_check": {
            "max_abs_severity_sum_minus_clinical_score_recomputed": max_abs_vs_recomputed,
            "max_abs_severity_sum_minus_clinical_score_csv": max_abs_vs_csv,
            "sample_rows": sample_rows,
            "pass": max_abs_vs_recomputed <= 1e-9,
        },
        "features_decomposed": FEATURES_DECOMP,
        "n_features": len(FEATURES_DECOMP),
        "production_day_holdout": {
            **prod_metrics,
            "reported_chapter5_accuracy": 0.9898648648648649,
            "artifact": "xgboost_v3.pkl",
            "protocol": "predict_only",
        },
        "experiment_day_holdout": {
            **decomp_metrics,
            "artifact": str(OUT_PKL.relative_to(ROOT)),
            "protocol": "retrain_same_split_cloned_prod_hyperparams_mod_weight_4x",
        },
        "accuracy_delta_decomp_minus_prod": float(delta),
        "shap_decomposed_model": shap_by_class,
        "shap_production_raw_nutrients_reference": shap_prod_ref,
        "interpretation_notes": [
            "k_severity / p_severity / protein_severity / na_severity sum to clinical_score.",
            "Compare severity_ranked within the new model to see if potassium component rises.",
            "Production reference: potassium raw SHAP was never top-2 among the four nutrients.",
        ],
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown summary
    sev_high = shap_by_class["HIGH"]["severity_ranked"]
    sev_mod = shap_by_class["MODERATE"]["severity_ranked"]
    md = f"""# Clinical-score decomposition experiment (day-scale)

**Status: EXPERIMENT ONLY — not for production.**

Production pickles untouched:
- `models/xgboost_v3.pkl` SHA-256 `{after['xgboost_v3.pkl']}`
- `models/xgboost_v3_meal.pkl` SHA-256 `{after['xgboost_v3_meal.pkl']}`

## Identity check

`k_severity + p_severity + protein_severity + na_severity == clinical_score`

- max |sum − recomputed clinical_score|: **{max_abs_vs_recomputed:.3e}** (pass ≤ 1e-9)
- max |sum − CSV clinical_score|: **{max_abs_vs_csv:.3e}**

## Holdout accuracy (same split: test_size=0.2, random_state=42, n_test={len(yte)})

| Model | Accuracy | F1 macro | AUC weighted |
|-------|----------|----------|--------------|
| Production day (`xgboost_v3.pkl`) | {prod_metrics['accuracy']:.4%} | {prod_metrics['f1_macro']:.4f} | {prod_metrics['auc_weighted']:.4f} |
| Decomposed experiment (12 features) | {decomp_metrics['accuracy']:.4%} | {decomp_metrics['f1_macro']:.4f} | {decomp_metrics['auc_weighted']:.4f} |
| Delta (decomp − prod) | {delta:+.4%} | — | — |

Chapter 5 reported production day accuracy: **98.99%**.

## SHAP — severity components (new model)

### HIGH
| Rank | Feature | mean \\|SHAP\\| |
|------|---------|---------------|
| 1 | {sev_high[0][0]} | {sev_high[0][1]:.6f} |
| 2 | {sev_high[1][0]} | {sev_high[1][1]:.6f} |
| 3 | {sev_high[2][0]} | {sev_high[2][1]:.6f} |
| 4 | {sev_high[3][0]} | {sev_high[3][1]:.6f} |

### MODERATE
| Rank | Feature | mean \\|SHAP\\| |
|------|---------|---------------|
| 1 | {sev_mod[0][0]} | {sev_mod[0][1]:.6f} |
| 2 | {sev_mod[1][0]} | {sev_mod[1][1]:.6f} |
| 3 | {sev_mod[2][0]} | {sev_mod[2][1]:.6f} |
| 4 | {sev_mod[3][0]} | {sev_mod[3][1]:.6f} |

## Artifact

`{OUT_PKL.relative_to(ROOT)}` — joblib dict with `not_for_production=True`.
"""
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
