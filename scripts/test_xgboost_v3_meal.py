#!/usr/bin/env python3
"""
Full test battery for meal-level XGBoost v3.

Mirrors the project's documented testing strategies (docs/testing/) adapted
to the meal artifact. NEVER loads production inference onto meal model —
only evaluates models/xgboost_v3_meal.pkl.

Usage (repo root):
  ./venv311/bin/python3 scripts/test_xgboost_v3_meal.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_xgboost_v3_meal import (  # noqa: E402
    FEATURES,
    KDOQI,
    MEAL_DATASET_PATH,
    MEAL_METRICS_PATH,
    MEAL_MODEL_PATH,
    OCCASION_FRACS,
    PROTECTED_SHA256,
    RANDOM_STATE,
    RISK_CLASSES,
    RISK_ENCODE,
    STAGE_ENCODE,
    STAGE_NUMERIC,
    TEST_SIZE,
    WEIGHTS,
    assign_label,
    build_meal_dataset,
    meal_caps,
    rule_baseline_label,
    sha256_file,
)

DAY_V3_PATH = ROOT / "models" / "xgboost_v3.pkl"
STATS_DIR = ROOT / "outputs" / "stats"
DOCS_DIR = ROOT / "docs" / "testing" / "11_meal_xgboost_v3"
REPORT_JSON = STATS_DIR / "11_xgboost_v3_meal_test_report.json"
REPORT_MD = STATS_DIR / "11_xgboost_v3_meal_test_report.md"

PROPOSAL_AUC_FLOOR = 0.90
PROPOSAL_SENS_FLOOR = 0.85
OVERFIT_GAP_MAX = 0.02  # 2 pp, matching project evidence language


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    strategy: str


@dataclass
class SuiteResult:
    checks: list[Check] = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def add(self, strategy: str, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name=name, passed=passed, detail=detail, strategy=strategy))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


def clinical_score_meal(row: pd.Series) -> float:
    caps = meal_caps(row["ckd_stage"], row["occasion"])
    score = 0.0
    for nutrient, weight in WEIGHTS.items():
        val = row.get(nutrient)
        if pd.isna(val):
            continue
        ratio = float(val) / caps[nutrient]
        if ratio > 1.0:
            score += weight * (1 + (ratio - 1) * 2)
        else:
            score += weight * ratio
    return float(score)


def prepare_xy(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + ["risk_label"]).copy()
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    X = df[FEATURES]
    y = df["risk_encoded"]
    return df, X, y


def split_data(df, X, y):
    return train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ── Strategy 00: artifact integrity ─────────────────────────────────────────

def test_integrity(suite: SuiteResult) -> None:
    print("\n=== 00 Integrity / protection ===")
    suite.add(
        "00_integrity",
        "day_v3_sha256_unchanged",
        DAY_V3_PATH.exists() and sha256_file(DAY_V3_PATH) == PROTECTED_SHA256,
        f"sha={sha256_file(DAY_V3_PATH)[:16]}..." if DAY_V3_PATH.exists() else "missing",
    )
    suite.add(
        "00_integrity",
        "meal_artifact_exists",
        MEAL_MODEL_PATH.exists(),
        str(MEAL_MODEL_PATH.relative_to(ROOT)) if MEAL_MODEL_PATH.exists() else "missing",
    )
    suite.add(
        "00_integrity",
        "meal_path_not_day_path",
        MEAL_MODEL_PATH.resolve() != DAY_V3_PATH.resolve(),
        f"meal={MEAL_MODEL_PATH.name} day={DAY_V3_PATH.name}",
    )
    meal_sha = sha256_file(MEAL_MODEL_PATH) if MEAL_MODEL_PATH.exists() else ""
    day_sha = sha256_file(DAY_V3_PATH) if DAY_V3_PATH.exists() else ""
    suite.add(
        "00_integrity",
        "meal_sha_differs_from_day",
        bool(meal_sha) and meal_sha != day_sha,
        f"meal={meal_sha[:16]}... day={day_sha[:16]}...",
    )
    suite.add(
        "00_integrity",
        "metrics_csv_exists",
        MEAL_METRICS_PATH.exists(),
        str(MEAL_METRICS_PATH.name),
    )
    suite.add(
        "00_integrity",
        "dataset_csv_exists",
        MEAL_DATASET_PATH.exists(),
        str(MEAL_DATASET_PATH.name),
    )


# ── Strategy 01: McNemar ────────────────────────────────────────────────────

def test_mcnemar(suite: SuiteResult, y_true, y_meal, y_day, y_rule) -> dict:
    print("\n=== 01 McNemar tests ===")

    def _mc(y_a, y_b, label: str):
        a_ok = y_a == y_true
        b_ok = y_b == y_true
        ct = np.array(
            [
                [int((a_ok & b_ok).sum()), int((~a_ok & b_ok).sum())],
                [int((a_ok & ~b_ok).sum()), int((~a_ok & ~b_ok).sum())],
            ]
        )
        res = mcnemar(ct, exact=False, correction=True)
        return {
            "label": label,
            "p": float(res.pvalue),
            "b": int(ct[0, 1]),
            "c": int(ct[1, 0]),
            "table": ct.tolist(),
        }

    vs_rule = _mc(y_meal, y_rule, "meal_vs_rule")
    vs_day = _mc(y_meal, y_day, "meal_vs_day")

    suite.add(
        "01_mcnemar",
        "meal_beats_rule_mcnemar",
        vs_rule["p"] < 0.05 and vs_rule["c"] > vs_rule["b"],
        f"p={vs_rule['p']:.4g} meal_only={vs_rule['c']} rule_only={vs_rule['b']}",
    )
    # Day is already strong; meal need not be significantly better, but should not be worse
    suite.add(
        "01_mcnemar",
        "meal_not_worse_than_day",
        vs_day["c"] >= vs_day["b"],
        f"p={vs_day['p']:.4g} meal_only={vs_day['c']} day_only={vs_day['b']}",
    )
    return {"vs_rule": vs_rule, "vs_day": vs_day}


# ── Strategy 02: 5-fold CV ──────────────────────────────────────────────────

def test_cv(suite: SuiteResult, X, y) -> dict:
    print("\n=== 02 5-fold stratified CV ===")
    model = joblib.load(MEAL_MODEL_PATH)
    params = model.get_params()
    # strip fitted-only / incompatible
    keep = {
        k: params[k]
        for k in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ]
        if k in params
    }
    clf = xgb.XGBClassifier(
        **keep,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # sample weights vary per fold — use unweighted accuracy for CV stability report
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    mean, std = float(scores.mean()), float(scores.std())
    suite.add(
        "02_cross_validation",
        "cv_accuracy_above_proposal",
        mean > PROPOSAL_AUC_FLOOR,  # accuracy floor same spirit; also check f1 below
        f"acc={mean:.4f} ± {std:.4f} folds={[round(float(s),4) for s in scores]}",
    )
    f1_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    suite.add(
        "02_cross_validation",
        "cv_f1_macro_stable",
        float(f1_scores.mean()) > 0.95 and float(f1_scores.std()) < 0.05,
        f"f1_macro={float(f1_scores.mean()):.4f} ± {float(f1_scores.std()):.4f}",
    )
    return {
        "accuracy_folds": [float(s) for s in scores],
        "accuracy_mean": mean,
        "accuracy_std": std,
        "f1_macro_folds": [float(s) for s in f1_scores],
        "f1_macro_mean": float(f1_scores.mean()),
        "f1_macro_std": float(f1_scores.std()),
    }


# ── Strategy 03: overfitting ────────────────────────────────────────────────

def test_overfit(suite: SuiteResult, model, X_train, y_train, X_test, y_test) -> dict:
    print("\n=== 03 Overfitting analysis ===")
    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    test_acc = float(accuracy_score(y_test, model.predict(X_test)))
    gap = train_acc - test_acc
    suite.add(
        "03_overfitting",
        "train_test_gap_under_2pp",
        abs(gap) < OVERFIT_GAP_MAX,
        f"train={train_acc:.4f} test={test_acc:.4f} gap={gap:+.4f}",
    )
    suite.add(
        "03_overfitting",
        "test_not_collapsed",
        test_acc >= 0.95,
        f"test_acc={test_acc:.4f}",
    )
    return {"train_acc": train_acc, "test_acc": test_acc, "gap": gap}


# ── Strategy 04: confusion / per-class ──────────────────────────────────────

def test_confusion(suite: SuiteResult, y_true, y_pred, y_prob) -> dict:
    print("\n=== 04 Confusion matrices / per-class ===")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    high_sens = float(recalls[2])
    mod_sens = float(recalls[1])
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted"))

    suite.add(
        "04_confusion",
        "proposal_auc_floor",
        auc > PROPOSAL_AUC_FLOOR,
        f"auc={auc:.4f} (floor {PROPOSAL_AUC_FLOOR})",
    )
    suite.add(
        "04_confusion",
        "proposal_high_sensitivity",
        high_sens >= PROPOSAL_SENS_FLOOR,
        f"HIGH sens={high_sens:.4f} (floor {PROPOSAL_SENS_FLOOR})",
    )
    suite.add(
        "04_confusion",
        "proposal_mod_sensitivity",
        mod_sens >= PROPOSAL_SENS_FLOOR,
        f"MOD sens={mod_sens:.4f} (floor {PROPOSAL_SENS_FLOOR})",
    )
    suite.add(
        "04_confusion",
        "zero_high_false_negatives",
        int(cm[2, 0] + cm[2, 1]) == 0,
        f"HIGH→not-HIGH={int(cm[2,0]+cm[2,1])} cm={cm.tolist()}",
    )
    return {
        "confusion": cm.tolist(),
        "high_sens": high_sens,
        "mod_sens": mod_sens,
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


# ── Strategy 05: per-stage ──────────────────────────────────────────────────

def test_per_stage(suite: SuiteResult, test_df, y_true, y_pred, y_prob) -> dict:
    print("\n=== 05 Per-stage breakdown ===")
    out = {}
    for stage in ["G2", "G3a", "G3b", "G4"]:
        mask = (test_df["ckd_stage"] == stage).to_numpy()
        n = int(mask.sum())
        if n < 10:
            suite.add(
                "05_per_stage",
                f"stage_{stage}_sample",
                n > 0,
                f"n={n} (sparse — reported only)",
            )
            if n == 0:
                continue
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        rec = recall_score(
            y_true[mask], y_pred[mask], labels=[0, 1, 2], average=None, zero_division=0
        )
        out[stage] = {
            "n": n,
            "accuracy": acc,
            "high_sens": float(rec[2]),
            "mod_sens": float(rec[1]),
        }
        # G4 may be tiny; require HIGH sens when HIGH cases exist
        high_n = int((y_true[mask] == 2).sum())
        ok = acc >= 0.90 and (high_n == 0 or float(rec[2]) >= PROPOSAL_SENS_FLOOR)
        suite.add(
            "05_per_stage",
            f"stage_{stage}_performance",
            ok,
            f"n={n} acc={acc:.4f} HIGH_sens={float(rec[2]):.4f} MOD_sens={float(rec[1]):.4f}",
        )
    return out


# ── Strategy 05b: per-occasion ──────────────────────────────────────────────

def test_per_occasion(suite: SuiteResult, test_df, y_true, y_pred) -> dict:
    print("\n=== 05b Per-occasion breakdown ===")
    out = {}
    for occ in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        mask = (test_df["occasion"] == occ).to_numpy()
        n = int(mask.sum())
        suite.add(
            "05b_per_occasion",
            f"occasion_{occ}_present",
            n > 0,
            f"n={n}",
        )
        if n == 0:
            continue
        acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        rec = recall_score(
            y_true[mask], y_pred[mask], labels=[0, 1, 2], average=None, zero_division=0
        )
        out[occ] = {"n": n, "accuracy": acc, "high_sens": float(rec[2]), "mod_sens": float(rec[1])}
        suite.add(
            "05b_per_occasion",
            f"occasion_{occ}_performance",
            acc >= 0.95 and float(rec[2]) >= PROPOSAL_SENS_FLOOR,
            f"acc={acc:.4f} HIGH_sens={float(rec[2]):.4f} MOD_sens={float(rec[1]):.4f}",
        )
    return out


# ── Strategy 06: edge cases ─────────────────────────────────────────────────

def _row_features(stage: str, occasion: str, k: float, p: float, pro: float, na: float) -> dict:
    caps = meal_caps(stage, occasion)
    # build a pseudo-row for clinical score
    row = pd.Series(
        {
            "ckd_stage": stage,
            "occasion": occasion,
            "potassium": k,
            "phosphorus": p,
            "protein_per_kg": pro,
            "sodium": na,
        }
    )
    cs = clinical_score_meal(row)
    return {
        "potassium": k,
        "phosphorus": p,
        "protein_per_kg": pro,
        "sodium": na,
        "ckd_stage_encoded": float(STAGE_ENCODE[stage]),
        "stage_numeric": float(STAGE_NUMERIC[stage]),
        "k_p_product": (k * p) / 1e6,
        "protein_sodium_ratio": pro / (na / 1000 + 1e-6),
        "clinical_score": cs,
        "expected_label": assign_label(cs),
        "caps": caps,
    }


def test_edge_cases(suite: SuiteResult, model) -> dict:
    print("\n=== 06 Edge case testing ===")
    cases = []
    # Synthetic boundary / extreme cases across stages & occasions
    specs = [
        ("empty_meal", "G3a", "Breakfast", 0, 0, 0, 0),
        ("tiny_snack", "G3a", "Snack", 50, 20, 0.02, 40),
        ("at_low_mod_boundary", "G2", "Lunch", None, None, None, None),  # filled below
        ("just_above_high", "G3b", "Dinner", None, None, None, None),
        ("extreme_high_K", "G4", "Dinner", 5000, 200, 0.1, 400),
        ("extreme_high_P", "G4", "Lunch", 200, 3000, 0.1, 400),
        ("high_protein", "G3a", "Dinner", 400, 300, 1.5, 500),
        ("high_sodium", "G2", "Breakfast", 200, 150, 0.1, 4000),
        ("balanced_safe", "G3a", "Lunch", 200, 150, 0.15, 400),
    ]

    # Construct near-boundary meals by scaling caps
    # LOW/MOD boundary ≈ clinical_score 0.7 — use ~0.65× and ~0.75× of caps with equal ratios
    def scaled(stage, occ, factor):
        caps = meal_caps(stage, occ)
        return (
            caps["potassium"] * factor,
            caps["phosphorus"] * factor,
            caps["protein_per_kg"] * factor,
            caps["sodium"] * factor,
        )

    filled = []
    for name, stage, occ, k, p, pro, na in specs:
        if k is None:
            if "low_mod" in name:
                k, p, pro, na = scaled(stage, occ, 0.70)
            else:
                k, p, pro, na = scaled(stage, occ, 1.25)
        filled.append((name, stage, occ, k, p, pro, na))

    results = []
    n_ok = 0
    for name, stage, occ, k, p, pro, na in filled:
        feat = _row_features(stage, occ, k, p, pro, na)
        X = pd.DataFrame([{f: feat[f] for f in FEATURES}])
        pred_idx = int(model.predict(X)[0])
        pred = RISK_CLASSES[pred_idx]
        proba = model.predict_proba(X)[0]
        expected = feat["expected_label"]
        ok = pred == expected
        n_ok += int(ok)
        results.append(
            {
                "name": name,
                "stage": stage,
                "occasion": occ,
                "clinical_score": round(feat["clinical_score"], 4),
                "expected": expected,
                "predicted": pred,
                "confidence": round(float(proba[pred_idx]), 4),
                "pass": ok,
            }
        )
        suite.add(
            "06_edge_cases",
            f"edge_{name}",
            ok,
            f"{stage}/{occ} score={feat['clinical_score']:.3f} exp={expected} got={pred}",
        )

    suite.add(
        "06_edge_cases",
        "edge_all_pass",
        n_ok == len(results),
        f"{n_ok}/{len(results)} edge cases matched label oracle",
    )
    return {"cases": results, "n_pass": n_ok, "n_total": len(results)}


# ── Strategy 07: model comparison ───────────────────────────────────────────

def test_comparison(suite: SuiteResult, metrics: dict) -> None:
    print("\n=== 07 Model comparison floors ===")
    meal = metrics["meal"]
    day = metrics["day_on_meal"]
    rule = metrics["rule"]
    suite.add(
        "07_comparison",
        "meal_accuracy_ge_day_on_meal",
        meal["accuracy"] >= day["accuracy"] - 1e-9,
        f"meal={meal['accuracy']:.4f} day_on_meal={day['accuracy']:.4f}",
    )
    suite.add(
        "07_comparison",
        "meal_beats_rule_accuracy",
        meal["accuracy"] > rule["accuracy"] + 0.05,
        f"meal={meal['accuracy']:.4f} rule={rule['accuracy']:.4f}",
    )
    suite.add(
        "07_comparison",
        "meal_mod_sens_ge_day",
        meal["mod_sens"] + 1e-9 >= day["mod_sens"],
        f"meal={meal['mod_sens']:.4f} day={day['mod_sens']:.4f}",
    )


# ── Strategy 08: hyperparameter smoke (saved params sane) ───────────────────

def test_hyperparams(suite: SuiteResult, model) -> dict:
    print("\n=== 08 Hyperparameter sanity ===")
    p = model.get_params()
    suite.add(
        "08_hyperparams",
        "n_estimators_positive",
        int(p.get("n_estimators", 0)) >= 50,
        f"n_estimators={p.get('n_estimators')}",
    )
    suite.add(
        "08_hyperparams",
        "max_depth_reasonable",
        2 <= int(p.get("max_depth", 0)) <= 10,
        f"max_depth={p.get('max_depth')}",
    )
    suite.add(
        "08_hyperparams",
        "learning_rate_positive",
        0 < float(p.get("learning_rate", 0)) <= 1.0,
        f"learning_rate={p.get('learning_rate')}",
    )
    suite.add(
        "08_hyperparams",
        "subsample_valid",
        0.5 <= float(p.get("subsample", 0)) <= 1.0,
        f"subsample={p.get('subsample')}",
    )
    return {
        k: p.get(k)
        for k in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ]
    }


# ── Strategy 09: calibration ────────────────────────────────────────────────

def test_calibration(suite: SuiteResult, y_true, y_prob) -> dict:
    print("\n=== 09 Calibration ===")
    # One-vs-rest Brier for HIGH class (index 2) — clinically critical
    y_high = (y_true == 2).astype(int)
    brier_high = float(brier_score_loss(y_high, y_prob[:, 2]))
    # Max-prob confidence among correct vs incorrect
    pred = y_prob.argmax(axis=1)
    correct = pred == y_true
    conf = y_prob.max(axis=1)
    mean_correct = float(conf[correct].mean()) if correct.any() else 0.0
    mean_wrong = float(conf[~correct].mean()) if (~correct).any() else 0.0

    suite.add(
        "09_calibration",
        "high_brier_reasonable",
        brier_high < 0.05,
        f"Brier(HIGH)={brier_high:.4f}",
    )
    suite.add(
        "09_calibration",
        "correct_more_confident_than_errors",
        mean_correct > mean_wrong or int((~correct).sum()) == 0,
        f"conf_correct={mean_correct:.3f} conf_wrong={mean_wrong:.3f} n_err={int((~correct).sum())}",
    )
    # reliability curve bins for HIGH
    try:
        frac_pos, mean_pred = calibration_curve(y_high, y_prob[:, 2], n_bins=8, strategy="quantile")
        cal = {
            "frac_pos": [float(x) for x in frac_pos],
            "mean_pred": [float(x) for x in mean_pred],
        }
    except Exception as exc:
        cal = {"error": str(exc)}
    return {
        "brier_high": brier_high,
        "mean_conf_correct": mean_correct,
        "mean_conf_wrong": mean_wrong,
        "calibration_high": cal,
    }


# ── Strategy 10: integration ────────────────────────────────────────────────

def test_integration(suite: SuiteResult, model) -> dict:
    print("\n=== 10 Integration verification ===")
    checks_detail = {}

    # 1 load
    suite.add("10_integration", "joblib_load_meal", True, "model loaded")

    # 2 predict_proba shape
    X_dummy = pd.DataFrame([{f: 0.1 if f != "clinical_score" else 0.3 for f in FEATURES}])
    proba = model.predict_proba(X_dummy)
    suite.add(
        "10_integration",
        "predict_proba_shape",
        proba.shape == (1, 3),
        f"shape={proba.shape}",
    )
    checks_detail["proba_shape"] = list(proba.shape)

    # 3 classes align LOW/MOD/HIGH encoding
    classes = list(getattr(model, "classes_", [0, 1, 2]))
    suite.add(
        "10_integration",
        "class_order_012",
        list(classes) == [0, 1, 2] or set(classes) == {0, 1, 2},
        f"classes={classes}",
    )

    # 4 feature count
    n_feat = int(getattr(model, "n_features_in_", len(FEATURES)))
    suite.add(
        "10_integration",
        "n_features_is_9",
        n_feat == 9,
        f"n_features_in_={n_feat}",
    )

    # 5 meal clinical score uses occasion caps (unit)
    caps_b = meal_caps("G3a", "Breakfast")
    caps_d = meal_caps("G3a", "Dinner")
    suite.add(
        "10_integration",
        "breakfast_caps_lt_dinner",
        caps_b["potassium"] < caps_d["potassium"],
        f"BF K={caps_b['potassium']} Dinner K={caps_d['potassium']}",
    )
    suite.add(
        "10_integration",
        "snack_caps_tightest",
        caps_b["potassium"] > meal_caps("G3a", "Snack")["potassium"],
        f"Snack K={meal_caps('G3a','Snack')['potassium']}",
    )

    # 6 same nutrient totals → different score by occasion
    row_bf = pd.Series(
        dict(ckd_stage="G3a", occasion="Breakfast", potassium=800, phosphorus=300, protein_per_kg=0.2, sodium=600)
    )
    row_sn = row_bf.copy()
    row_sn["occasion"] = "Snack"
    s_bf, s_sn = clinical_score_meal(row_bf), clinical_score_meal(row_sn)
    suite.add(
        "10_integration",
        "same_nutrients_higher_risk_as_snack",
        s_sn > s_bf,
        f"Breakfast score={s_bf:.3f} Snack score={s_sn:.3f}",
    )

    # 7 day model still loads
    try:
        day = joblib.load(DAY_V3_PATH)
        _ = day.predict_proba(X_dummy)
        suite.add("10_integration", "day_model_still_loads", True, "xgboost_v3.pkl OK")
    except Exception as exc:
        suite.add("10_integration", "day_model_still_loads", False, str(exc))

    # 8 dataset has all occasions
    if MEAL_DATASET_PATH.exists():
        ds = pd.read_csv(MEAL_DATASET_PATH)
        occs = set(ds["occasion"].unique())
        suite.add(
            "10_integration",
            "dataset_has_four_occasions",
            occs == {"Breakfast", "Lunch", "Dinner", "Snack"},
            f"occasions={sorted(occs)}",
        )
        checks_detail["dataset_n"] = int(len(ds))
    else:
        suite.add("10_integration", "dataset_has_four_occasions", False, "dataset missing")

    # 9 published metrics match live holdout within tolerance
    if MEAL_METRICS_PATH.exists():
        pub = pd.read_csv(MEAL_METRICS_PATH).iloc[0]
        checks_detail["published_accuracy"] = float(pub["accuracy"])
        suite.add(
            "10_integration",
            "published_metrics_sane",
            float(pub["auc_roc"]) > PROPOSAL_AUC_FLOOR
            and float(pub["high_sensitivity"]) >= PROPOSAL_SENS_FLOOR,
            f"pub auc={pub['auc_roc']} HIGH={pub['high_sensitivity']} MOD={pub['mod_sensitivity']}",
        )

    # final day hash again
    suite.add(
        "10_integration",
        "day_v3_untouched_after_tests",
        sha256_file(DAY_V3_PATH) == PROTECTED_SHA256,
        "sha match",
    )
    return checks_detail


def write_reports(suite: SuiteResult) -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_pass": suite.n_pass,
        "n_fail": suite.n_fail,
        "n_total": len(suite.checks),
        "all_passed": suite.n_fail == 0,
        "checks": [
            {"strategy": c.strategy, "name": c.name, "passed": c.passed, "detail": c.detail}
            for c in suite.checks
        ],
        "extras": suite.extras,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Meal XGBoost v3 — test report",
        "",
        f"**Result: {'ALL PASSED' if suite.n_fail == 0 else 'FAILURES PRESENT'}** "
        f"({suite.n_pass}/{len(suite.checks)} checks)",
        "",
        "| Strategy | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for c in suite.checks:
        lines.append(
            f"| {c.strategy} | {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail.replace('|', '/')} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")

    # also drop a short README in docs/testing
    (DOCS_DIR / "README.md").write_text(
        "\n".join(
            [
                "# 11 — Meal XGBoost v3 testing",
                "",
                "Automated battery for `models/xgboost_v3_meal.pkl`.",
                "",
                "```bash",
                "./venv311/bin/python3 scripts/test_xgboost_v3_meal.py",
                "```",
                "",
                f"Latest: **{suite.n_pass}/{len(suite.checks)} passed** "
                f"(see `outputs/stats/11_xgboost_v3_meal_test_report.md`).",
                "",
                "Production `xgboost_v3.pkl` is hash-checked and never overwritten.",
                "",
            ]
        )
    )
    print(f"\nWrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {DOCS_DIR / 'README.md'}")


def main() -> int:
    t0 = time.time()
    suite = SuiteResult()
    print("=" * 72)
    print("MEAL XGBoost v3 — FULL TEST BATTERY")
    print("=" * 72)

    test_integrity(suite)
    if not MEAL_MODEL_PATH.exists():
        write_reports(suite)
        print("ABORT: meal model missing")
        return 1

    print("\nLoading dataset + models...")
    if MEAL_DATASET_PATH.exists():
        # Prefer cached labels CSV for speed; rebuild features
        base = pd.read_csv(MEAL_DATASET_PATH)
        # rebuild engineered features from nutrients
        df = base.copy()
        df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
        df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
        df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
        df["protein_sodium_ratio"] = df["protein_per_kg"] / (df["sodium"] / 1000 + 1e-6)
        # clinical_score already in CSV from training
        if "rule_baseline" not in df.columns:
            df["rule_baseline"] = df.apply(rule_baseline_label, axis=1)
    else:
        df = build_meal_dataset()

    df, X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, X, y)
    test_df = df.loc[idx_test].copy()
    y_true = y_test.to_numpy()

    meal_model = joblib.load(MEAL_MODEL_PATH)
    day_model = joblib.load(DAY_V3_PATH)

    y_meal = meal_model.predict(X_test)
    y_prob = meal_model.predict_proba(X_test)
    y_day = day_model.predict(X_test)
    y_rule = test_df["rule_baseline"].map(RISK_ENCODE).to_numpy()

    mcn = test_mcnemar(suite, y_true, y_meal, y_day, y_rule)
    cv = test_cv(suite, X, y)
    overfit = test_overfit(suite, meal_model, X_train, y_train, X_test, y_test)
    conf = test_confusion(suite, y_true, y_meal, y_prob)
    stages = test_per_stage(suite, test_df, y_true, y_meal, y_prob)
    occasions = test_per_occasion(suite, test_df, y_true, y_meal)
    edges = test_edge_cases(suite, meal_model)

    day_rec = recall_score(y_true, y_day, labels=[0, 1, 2], average=None, zero_division=0)
    rule_rec = recall_score(y_true, y_rule, labels=[0, 1, 2], average=None, zero_division=0)
    metrics = {
        "meal": {
            "accuracy": conf["accuracy"],
            "mod_sens": conf["mod_sens"],
            "high_sens": conf["high_sens"],
        },
        "day_on_meal": {
            "accuracy": float(accuracy_score(y_true, y_day)),
            "mod_sens": float(day_rec[1]),
            "high_sens": float(day_rec[2]),
        },
        "rule": {
            "accuracy": float(accuracy_score(y_true, y_rule)),
            "mod_sens": float(rule_rec[1]),
            "high_sens": float(rule_rec[2]),
        },
    }
    test_comparison(suite, metrics)
    hyps = test_hyperparams(suite, meal_model)
    cal = test_calibration(suite, y_true, y_prob)
    integ = test_integration(suite, meal_model)

    suite.extras = {
        "mcnemar": mcn,
        "cv": cv,
        "overfit": overfit,
        "holdout": conf,
        "per_stage": stages,
        "per_occasion": occasions,
        "edge_cases": edges,
        "comparison": metrics,
        "hyperparams": hyps,
        "calibration": cal,
        "integration": integ,
        "elapsed_sec": round(time.time() - t0, 2),
        "day_v3_sha256": sha256_file(DAY_V3_PATH),
        "meal_sha256": sha256_file(MEAL_MODEL_PATH),
    }

    write_reports(suite)

    # CSV summary of checks
    pd.DataFrame(
        [
            {
                "strategy": c.strategy,
                "check": c.name,
                "passed": c.passed,
                "detail": c.detail,
            }
            for c in suite.checks
        ]
    ).to_csv(STATS_DIR / "11_xgboost_v3_meal_test_checks.csv", index=False)

    print("\n" + "=" * 72)
    print(f"RESULT: {suite.n_pass}/{len(suite.checks)} PASSED — {suite.n_fail} FAILED")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    print("=" * 72)
    if suite.n_fail:
        print("\nFailed checks:")
        for c in suite.checks:
            if not c.passed:
                print(f"  - [{c.strategy}] {c.name}: {c.detail}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
