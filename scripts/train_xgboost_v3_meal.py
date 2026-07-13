#!/usr/bin/env python3
"""
Train meal-level XGBoost v3 (parallel artifact).

- NEVER overwrites models/xgboost_v3.pkl
- Writes models/xgboost_v3_meal.pkl + meal metrics/dataset
- Labels/features use OCCASION_RULES fractions × daily KDOQI (honest meal scale)

Usage (from repo root):
  ./venv311/bin/python3 scripts/train_xgboost_v3_meal.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
NHANES_DIR = ROOT / "data" / "raw" / "nhanes"
COHORT_PATH = ROOT / "data" / "processed" / "ckd_cohort_final.csv"

DAY_V3_PATH = MODEL_DIR / "xgboost_v3.pkl"
MEAL_MODEL_PATH = MODEL_DIR / "xgboost_v3_meal.pkl"
MEAL_DATASET_PATH = STATS_DIR / "05_risk_labels_v3_meal.csv"
MEAL_METRICS_PATH = STATS_DIR / "10_xgboost_v3_meal_metrics.csv"
DAY_METRICS_PATH = STATS_DIR / "10_xgboost_v3_metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}

# Match notebooks/04c + backend/models/xgboost_model.py
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

KDOQI = {
    "G2": {"potassium": 3500.0, "phosphorus": 1000.0, "protein_per_kg": 0.8, "sodium": 2300.0},
    "G3a": {"potassium": 3000.0, "phosphorus": 800.0, "protein_per_kg": 0.6, "sodium": 2300.0},
    "G3b": {"potassium": 3000.0, "phosphorus": 800.0, "protein_per_kg": 0.6, "sodium": 2300.0},
    "G4": {"potassium": 2500.0, "phosphorus": 700.0, "protein_per_kg": 0.55, "sodium": 2300.0},
}

WEIGHTS = {"potassium": 0.35, "phosphorus": 0.30, "protein_per_kg": 0.25, "sodium": 0.10}

# OCCASION_RULES nutrient_caps (K, P, protein, Na)
OCCASION_FRACS = {
    "Breakfast": (0.25, 0.25, 0.30, 0.25),
    "Lunch": (0.40, 0.40, 0.40, 0.40),
    "Dinner": (0.40, 0.40, 0.40, 0.40),
    "Snack": (0.15, 0.15, 0.10, 0.15),
}

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

PROTECTED_SHA256 = "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_day_v3_untouched(when: str) -> None:
    if not DAY_V3_PATH.exists():
        raise FileNotFoundError(f"Missing protected model: {DAY_V3_PATH}")
    digest = sha256_file(DAY_V3_PATH)
    if digest != PROTECTED_SHA256:
        raise RuntimeError(
            f"PROTECTED xgboost_v3.pkl hash changed ({when})!\n"
            f" expected {PROTECTED_SHA256}\n"
            f" got      {digest}"
        )
    print(f"[OK] xgboost_v3.pkl untouched ({when}): {digest[:16]}...")


def load_iff(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat

        df, _ = pyreadstat.read_xport(str(path))
        return df
    except Exception:
        return pd.read_sas(path, format="xport")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
        alt = c.replace(".", "_")
        if alt in df.columns:
            return alt
    raise KeyError(f"None of {candidates} in {list(df.columns)[:25]}")


def standardize_iff(df: pd.DataFrame, day: int) -> pd.DataFrame:
    prefix = f"DR{day}"
    # DR*_030Z = name of eating occasion (1=Breakfast…6=Snack…).
    # DR*_020 in cycle J is time-of-day (seconds), not occasion — use as fallback only.
    rename = {
        pick_col(df, [f"{prefix}.030Z", f"{prefix}_030Z"]): "occasion_code",
        pick_col(df, [f"{prefix}.020", f"{prefix}_020"]): "meal_time",
        pick_col(df, [f"{prefix}IPOTA"]): "potassium",
        pick_col(df, [f"{prefix}IPHOS"]): "phosphorus",
        pick_col(df, [f"{prefix}IPROT"]): "protein",
        pick_col(df, [f"{prefix}ISODI"]): "sodium",
    }
    out = df.rename(columns=rename)
    return out[
        ["SEQN", "occasion_code", "meal_time", "potassium", "phosphorus", "protein", "sodium"]
    ].copy()


def map_occasion_from_time(meal_time) -> str | None:
    """Fallback when DR*_030Z missing: time-of-day → Breakfast/Lunch/Dinner."""
    if pd.isna(meal_time):
        return None
    if isinstance(meal_time, dt.time):
        code = meal_time.hour * 3600 + meal_time.minute * 60 + meal_time.second
    else:
        try:
            code = float(meal_time)
        except (TypeError, ValueError):
            return None
    if code < 39600:
        return "Breakfast"
    if code < 61200:
        return "Lunch"
    return "Dinner"


def map_occasion(occasion_code, meal_time=None) -> str | None:
    """Map NHANES DR*_030Z eating-occasion name to GuidaPlate occasion."""
    if pd.isna(occasion_code):
        return map_occasion_from_time(meal_time)
    try:
        code = int(round(float(occasion_code)))
    except (TypeError, ValueError):
        return map_occasion_from_time(meal_time)

    # CDC DRXIFT_J codebook (selected):
    # 1 Breakfast, 2 Lunch, 3 Dinner, 4 Supper, 5 Brunch,
    # 6 Snack, 7 Drink, 8 Infant feeding, 9 Extended consumption,
    # 10 Desayuno, 11 Almuerzo, 12 Comida, 13 Cena,
    # 14 Entre comida, 15 Botana, 16 Bocadillo, 17 Tentempie, 18 Bebida,
    # 19 Other / 99 Don't know
    if code in (1, 5, 10):
        return "Breakfast"
    if code in (2, 11):
        return "Lunch"
    if code in (3, 4, 12, 13):
        return "Dinner"
    if code in (6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 91, 99):
        return "Snack"
    return map_occasion_from_time(meal_time)


def meal_caps(stage: str, occasion: str) -> dict[str, float]:
    daily = KDOQI[stage]
    fk, fp, fpro, fna = OCCASION_FRACS[occasion]
    return {
        "potassium": daily["potassium"] * fk,
        "phosphorus": daily["phosphorus"] * fp,
        "protein_per_kg": daily["protein_per_kg"] * fpro,
        "sodium": daily["sodium"] * fna,
    }


def compute_clinical_score_meal(row: pd.Series) -> float:
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


def assign_label(score: float) -> str:
    if score >= 1.2:
        return "HIGH"
    if score >= 0.7:
        return "MODERATE"
    return "LOW"


def rule_baseline_label(row: pd.Series) -> str:
    caps = meal_caps(row["ckd_stage"], row["occasion"])
    exceeded = 0
    for nutrient in ["potassium", "phosphorus", "protein_per_kg", "sodium"]:
        if pd.notna(row.get(nutrient)) and float(row[nutrient]) > caps[nutrient]:
            exceeded += 1
    if exceeded >= 2:
        return "HIGH"
    if exceeded == 1:
        return "MODERATE"
    return "LOW"


def build_meal_dataset() -> pd.DataFrame:
    print("Loading cohort...")
    cohort = pd.read_csv(COHORT_PATH)
    cohort = cohort[cohort["ckd_stage"].isin(STAGE_ENCODE)].copy()
    ckd_seqns = set(cohort["SEQN"])

    print("Loading DR1IFF / DR2IFF (may take a minute)...")
    iff1 = standardize_iff(load_iff(NHANES_DIR / "DR1IFF_J.xpt"), day=1)
    iff2 = standardize_iff(load_iff(NHANES_DIR / "DR2IFF_J.xpt"), day=2)
    iff1 = iff1[iff1["SEQN"].isin(ckd_seqns)].copy()
    iff2 = iff2[iff2["SEQN"].isin(ckd_seqns)].copy()
    iff1["day"] = 1
    iff2["day"] = 2
    foods = pd.concat([iff1, iff2], ignore_index=True)
    for col in ["potassium", "phosphorus", "protein", "sodium"]:
        foods[col] = pd.to_numeric(foods[col], errors="coerce").fillna(0.0)
    foods["occasion"] = foods.apply(
        lambda r: map_occasion(r["occasion_code"], r["meal_time"]), axis=1
    )
    foods = foods.dropna(subset=["occasion"])

    meal = (
        foods.groupby(["SEQN", "day", "occasion"], as_index=False)[
            ["potassium", "phosphorus", "protein", "sodium"]
        ]
        .sum()
    )
    # Drop empty meals
    meal = meal[
        (meal["potassium"] + meal["phosphorus"] + meal["protein"] + meal["sodium"]) > 0
    ].copy()

    meta = cohort[["SEQN", "weight_kg", "ckd_stage"]].drop_duplicates("SEQN")
    df = meal.merge(meta, on="SEQN", how="inner")
    df = df[df["weight_kg"].notna() & (df["weight_kg"] > 0)].copy()
    df["protein_per_kg"] = df["protein"] / df["weight_kg"]

    df["clinical_score"] = df.apply(compute_clinical_score_meal, axis=1)
    df["risk_label"] = df["clinical_score"].apply(assign_label)
    df["rule_baseline"] = df.apply(rule_baseline_label, axis=1)

    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
    df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
    df["protein_sodium_ratio"] = df["protein_per_kg"] / (df["sodium"] / 1000 + 1e-6)

    print(f"Meal rows: {len(df)}")
    print("Occasion counts:")
    print(df["occasion"].value_counts())
    print("Label distribution:")
    print(df["risk_label"].value_counts().reindex(RISK_CLASSES))
    return df


def mcnemar_p(y_true: np.ndarray, y_model: np.ndarray, y_base: np.ndarray) -> tuple[float, int, int]:
    model_correct = y_model == y_true
    base_correct = y_base == y_true
    b = int((~model_correct & base_correct).sum())  # baseline only
    c = int((model_correct & ~base_correct).sum())  # model only
    table = [[0, b], [c, 0]]
    # statsmodels wants 2x2 with both correct/incorrect; use discordant form
    ct = np.array([[int((model_correct & base_correct).sum()), b], [c, int((~model_correct & ~base_correct).sum())]])
    try:
        res = mcnemar(ct, exact=False, correction=True)
        p = float(res.pvalue)
    except Exception:
        p = float("nan")
    return p, b, c


def train_and_eval(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=FEATURES + ["risk_label"]).copy()
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    X = df[FEATURES]
    y = df["risk_encoded"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sample_weight_train = compute_sample_weight(class_weight=class_weight, y=y_train)

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
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_distributions,
        n_iter=50,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    print("RandomizedSearchCV (50 iter, f1_macro)...")
    t0 = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weight_train)
    print(f"Done in {time.time() - t0:.1f}s | Best CV F1 macro: {search.best_score_:.4f}")
    print("Best params:", search.best_params_)

    best = xgb.XGBClassifier(
        **search.best_params_,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    best.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)
    y_true = y_test.to_numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")

    # Per-class recall = sensitivity
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    high_sens = float(recalls[RISK_ENCODE["HIGH"]])
    mod_sens = float(recalls[RISK_ENCODE["MODERATE"]])

    # McNemar vs meal rule baseline on same test rows
    base_labels = df.loc[idx_test, "rule_baseline"].map(RISK_ENCODE).to_numpy()
    p_mc, b, c = mcnemar_p(y_true, y_pred, base_labels)

    print("\nClassification report (meal-labeled test):")
    print(classification_report(y_true, y_pred, target_names=RISK_CLASSES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    # Save model ONLY to meal path
    if MEAL_MODEL_PATH.resolve() == DAY_V3_PATH.resolve():
        raise RuntimeError("Refusing to write meal model onto day v3 path")
    joblib.dump(best, MEAL_MODEL_PATH)
    print(f"Saved meal model: {MEAL_MODEL_PATH}")

    metrics = {
        "model": "XGBoost v3 meal",
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro": round(float(f1_m), 4),
        "auc_roc": round(float(auc), 4),
        "high_sensitivity": round(high_sens, 4),
        "mod_sensitivity": round(mod_sens, 4),
        "mcnemar_p": round(float(p_mc), 4) if p_mc == p_mc else None,
        "mcnemar_b": b,
        "mcnemar_c": c,
        "n_features": len(FEATURES),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_meal_rows_total": int(len(df)),
        "best_cv_f1_macro": round(float(search.best_score_), 4),
        "artifact": str(MEAL_MODEL_PATH.name),
        "day_v3_untouched": True,
        "label_scale": "occasion_fraction_of_daily_kdoqi",
    }
    return metrics, best


def print_comparison(meal_metrics: dict) -> None:
    day = {
        "accuracy": 0.9899,
        "f1_weighted": 0.9899,
        "f1_macro": 0.9853,
        "auc_roc": 0.9975,
        "high_sensitivity": 1.0,
        "mod_sensitivity": 0.9694,
    }
    if DAY_METRICS_PATH.exists():
        day_df = pd.read_csv(DAY_METRICS_PATH)
        if len(day_df):
            row = day_df.iloc[0]
            for k in day:
                if k in row and pd.notna(row[k]):
                    day[k] = float(row[k])

    print("\n" + "=" * 78)
    print("COMPARISON (different tasks — day-labeled vs meal-labeled)")
    print("=" * 78)
    print(f"{'Metric':<18} | {'Day v3 (prod)':>14} | {'Meal v3 (new)':>14}")
    print("-" * 78)
    for key, label in [
        ("auc_roc", "AUC-ROC"),
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 macro"),
        ("f1_weighted", "F1 weighted"),
        ("high_sensitivity", "HIGH sens"),
        ("mod_sensitivity", "MOD sens"),
    ]:
        print(f"{label:<18} | {day[key]:>14.4f} | {meal_metrics[key]:>14.4f}")
    print("=" * 78)
    print("Proposal floors: AUC > 0.90, sensitivity ≥ 0.85")
    meets = (
        meal_metrics["auc_roc"] > 0.90
        and meal_metrics["high_sensitivity"] >= 0.85
        and meal_metrics["mod_sensitivity"] >= 0.85
    )
    print(f"Meal model meets proposal floors on meal test: {meets}")
    print("Production remains on xgboost_v3.pkl until explicit promotion.")


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    assert_day_v3_untouched("before")

    df = build_meal_dataset()
    export_cols = [
        "SEQN",
        "day",
        "occasion",
        "ckd_stage",
        "potassium",
        "phosphorus",
        "protein_per_kg",
        "sodium",
        "clinical_score",
        "risk_label",
        "rule_baseline",
    ]
    df[export_cols].to_csv(MEAL_DATASET_PATH, index=False)
    print(f"Saved meal labels: {MEAL_DATASET_PATH}")

    metrics, _model = train_and_eval(df)
    pd.DataFrame([metrics]).to_csv(MEAL_METRICS_PATH, index=False)
    print(f"Saved meal metrics: {MEAL_METRICS_PATH}")

    print_comparison(metrics)
    assert_day_v3_untouched("after")
    print("\nDONE — meal artifact ready; day v3 protected.")


if __name__ == "__main__":
    main()
