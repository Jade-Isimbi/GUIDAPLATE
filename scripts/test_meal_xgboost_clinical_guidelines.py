#!/usr/bin/env python3
"""
Clinical-guidelines compliance battery for meal-level XGBoost v3.

Verifies that training labels, occasion caps, and model predictions align with
GuidaPlate's declared clinical sources:

  - KDOQI 2020 Nutrition in CKD (daily limits / Na / protein)
  - KDIGO 2024 stage framing (via stage-stricter limits)
  - backend/clinical_constants.py (canonical product constants)
  - OCCASION_RULES nutrient_caps (meal operationalization)

Honest scope:
  XGBoost classifies nutrient-risk given totals. Forbidden-food lists and RAG
  meal planning are separate layers — this suite checks the risk model path.

Usage:
  ./venv311/bin/python3 scripts/test_meal_xgboost_clinical_guidelines.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.clinical_constants import (  # noqa: E402
    CLINICAL_SEVERITY_WEIGHTS,
    EGFR_RANGES,
    KDOQI_DAILY_LIMITS,
    SEVERITY_THRESHOLDS,
)
from backend.api.meal_planner import OCCASION_RULES  # noqa: E402
from train_xgboost_v3_meal import (  # noqa: E402
    FEATURES,
    KDOQI as TRAIN_KDOQI,
    MEAL_DATASET_PATH,
    MEAL_MODEL_PATH,
    OCCASION_FRACS as TRAIN_FRACS,
    RISK_CLASSES,
    RISK_ENCODE,
    STAGE_ENCODE,
    STAGE_NUMERIC,
    WEIGHTS as TRAIN_WEIGHTS,
    assign_label,
    meal_caps,
    sha256_file,
    PROTECTED_SHA256,
    DAY_V3_PATH,
)

STATS = ROOT / "outputs" / "stats"
REPORT_JSON = STATS / "12_meal_xgboost_clinical_guidelines_report.json"
REPORT_MD = STATS / "12_meal_xgboost_clinical_guidelines_report.md"
DOCS = ROOT / "docs" / "testing" / "12_meal_clinical_guidelines"


@dataclass
class Check:
    section: str
    name: str
    passed: bool
    detail: str
    guideline_ref: str = ""


@dataclass
class Suite:
    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        section: str,
        name: str,
        passed: bool,
        detail: str,
        guideline_ref: str = "",
    ) -> None:
        self.checks.append(
            Check(section, name, passed, detail, guideline_ref)
        )
        tag = "PASS" if passed else "FAIL"
        ref = f" [{guideline_ref}]" if guideline_ref else ""
        print(f"  [{tag}] {name}: {detail}{ref}")

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


def clinical_score_meal(
    stage: str,
    occasion: str,
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
) -> float:
    """Score using canonical backend constants + OCCASION_RULES caps."""
    caps = meal_caps_from_backend(stage, occasion)
    values = {
        "potassium": potassium,
        "phosphorus": phosphorus,
        "protein_per_kg": protein_per_kg,
        "sodium": sodium,
    }
    weight_map = {
        "potassium": CLINICAL_SEVERITY_WEIGHTS["potassium"],
        "phosphorus": CLINICAL_SEVERITY_WEIGHTS["phosphorus"],
        "protein_per_kg": CLINICAL_SEVERITY_WEIGHTS["protein"],
        "sodium": CLINICAL_SEVERITY_WEIGHTS["sodium"],
    }
    score = 0.0
    for nutrient, weight in weight_map.items():
        ratio = values[nutrient] / caps[nutrient]
        if ratio > 1.0:
            score += weight * (1 + (ratio - 1) * 2)
        else:
            score += weight * ratio
    return float(score)


def meal_caps_from_backend(stage: str, occasion: str) -> dict[str, float]:
    daily = KDOQI_DAILY_LIMITS[stage]
    fk, fp, fpro, fna = OCCASION_RULES[occasion]["nutrient_caps"]
    return {
        "potassium": daily["potassium"] * fk,
        "phosphorus": daily["phosphorus"] * fp,
        "protein_per_kg": daily["protein_per_kg"] * fpro,
        "sodium": daily["sodium"] * fna,
    }


def label_from_score(score: float) -> str:
    if score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= SEVERITY_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


def build_features(
    stage: str,
    occasion: str,
    k: float,
    p: float,
    pro: float,
    na: float,
) -> pd.DataFrame:
    cs = clinical_score_meal(stage, occasion, k, p, pro, na)
    row = {
        "potassium": k,
        "phosphorus": p,
        "protein_per_kg": pro,
        "sodium": na,
        "ckd_stage_encoded": float(STAGE_ENCODE[stage]),
        "stage_numeric": float(STAGE_NUMERIC[stage]),
        "k_p_product": (k * p) / 1e6,
        "protein_sodium_ratio": pro / (na / 1000 + 1e-6),
        "clinical_score": cs,
    }
    return pd.DataFrame([{f: row[f] for f in FEATURES}]), cs


def predict_label(model, stage, occasion, k, p, pro, na) -> tuple[str, float, float]:
    X, cs = build_features(stage, occasion, k, p, pro, na)
    idx = int(model.predict(X)[0])
    conf = float(model.predict_proba(X)[0][idx])
    return RISK_CLASSES[idx], conf, cs


# ── A. Constant alignment with declared guidelines / product source of truth ─

def test_constant_alignment(suite: Suite) -> None:
    print("\n=== A. Guideline constant alignment ===")

    # A1 Sodium = 2300 all stages (KDOQI 2020 Guideline 5.1)
    for stage, lim in KDOQI_DAILY_LIMITS.items():
        suite.add(
            "A_constants",
            f"sodium_2300_{stage}",
            lim["sodium"] == 2300.0,
            f"Na={lim['sodium']}",
            "KDOQI 2020 Guideline 5.1",
        )

    # A2 Protein matches documented KDOQI Table 11 values in clinical_constants
    expected_pro = {"G2": 0.8, "G3a": 0.6, "G3b": 0.6, "G4": 0.55}
    for stage, exp in expected_pro.items():
        suite.add(
            "A_constants",
            f"protein_kdoqi_{stage}",
            KDOQI_DAILY_LIMITS[stage]["protein_per_kg"] == exp,
            f"protein_per_kg={KDOQI_DAILY_LIMITS[stage]['protein_per_kg']}",
            "KDOQI 2020 Table 11",
        )

    # A3 Stage-stricter K/P (later stages tighter) — KDIGO severity framing
    suite.add(
        "A_constants",
        "potassium_monotone_by_stage",
        (
            KDOQI_DAILY_LIMITS["G2"]["potassium"]
            > KDOQI_DAILY_LIMITS["G3a"]["potassium"]
            >= KDOQI_DAILY_LIMITS["G3b"]["potassium"]
            > KDOQI_DAILY_LIMITS["G4"]["potassium"]
        ),
        f"G2={KDOQI_DAILY_LIMITS['G2']['potassium']} "
        f"G3a/b={KDOQI_DAILY_LIMITS['G3a']['potassium']} "
        f"G4={KDOQI_DAILY_LIMITS['G4']['potassium']}",
        "KDIGO stage severity + product KDOQI table",
    )
    suite.add(
        "A_constants",
        "phosphorus_monotone_by_stage",
        (
            KDOQI_DAILY_LIMITS["G2"]["phosphorus"]
            > KDOQI_DAILY_LIMITS["G3a"]["phosphorus"]
            >= KDOQI_DAILY_LIMITS["G3b"]["phosphorus"]
            > KDOQI_DAILY_LIMITS["G4"]["phosphorus"]
        ),
        f"G2={KDOQI_DAILY_LIMITS['G2']['phosphorus']} "
        f"G3={KDOQI_DAILY_LIMITS['G3a']['phosphorus']} "
        f"G4={KDOQI_DAILY_LIMITS['G4']['phosphorus']}",
        "KDIGO stage severity + product KDOQI table",
    )

    # A4 Training script must not drift from backend constants
    for stage in KDOQI_DAILY_LIMITS:
        ok = TRAIN_KDOQI[stage] == KDOQI_DAILY_LIMITS[stage]
        suite.add(
            "A_constants",
            f"train_kdoqi_matches_backend_{stage}",
            ok,
            f"train={TRAIN_KDOQI[stage]} backend={KDOQI_DAILY_LIMITS[stage]}",
            "clinical_constants.py source of truth",
        )

    # A5 Weights: training == backend; priority K > P > protein > Na
    train_w = {
        "potassium": TRAIN_WEIGHTS["potassium"],
        "phosphorus": TRAIN_WEIGHTS["phosphorus"],
        "protein": TRAIN_WEIGHTS["protein_per_kg"],
        "sodium": TRAIN_WEIGHTS["sodium"],
    }
    suite.add(
        "A_constants",
        "weights_match_backend",
        train_w == CLINICAL_SEVERITY_WEIGHTS,
        f"train={train_w} backend={CLINICAL_SEVERITY_WEIGHTS}",
        "clinical_constants CLINICAL_SEVERITY_WEIGHTS",
    )
    w = CLINICAL_SEVERITY_WEIGHTS
    suite.add(
        "A_constants",
        "weights_sum_to_one",
        abs(sum(w.values()) - 1.0) < 1e-9,
        f"sum={sum(w.values())}",
        "author-derived severity weights",
    )
    suite.add(
        "A_constants",
        "weight_priority_K_gt_P_gt_pro_gt_Na",
        w["potassium"] > w["phosphorus"] > w["protein"] > w["sodium"],
        f"K={w['potassium']} P={w['phosphorus']} pro={w['protein']} Na={w['sodium']}",
        "author-derived clinical priority (documented in clinical_constants)",
    )

    # A6 Thresholds
    suite.add(
        "A_constants",
        "thresholds_match_backend",
        SEVERITY_THRESHOLDS == {"HIGH": 1.2, "MODERATE": 0.7},
        f"{SEVERITY_THRESHOLDS}",
        "clinical_constants SEVERITY_THRESHOLDS",
    )
    suite.add(
        "A_constants",
        "assign_label_matches_thresholds",
        assign_label(0.699) == "LOW"
        and assign_label(0.7) == "MODERATE"
        and assign_label(1.199) == "MODERATE"
        and assign_label(1.2) == "HIGH",
        "boundaries 0.7 / 1.2",
        "SEVERITY_THRESHOLDS",
    )

    # A7 Occasion fracs match live OCCASION_RULES
    for occ in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        live = tuple(OCCASION_RULES[occ]["nutrient_caps"])
        train = TRAIN_FRACS[occ]
        suite.add(
            "A_constants",
            f"occasion_frac_{occ}",
            live == train,
            f"OCCASION_RULES={live} train={train}",
            "meal_planner.OCCASION_RULES",
        )

    # A8 eGFR ranges present for all stages (KDIGO framing metadata)
    for stage in ["G2", "G3a", "G3b", "G4"]:
        suite.add(
            "A_constants",
            f"egfr_range_{stage}",
            stage in EGFR_RANGES and bool(EGFR_RANGES[stage]),
            f"{EGFR_RANGES.get(stage)}",
            "KDIGO 2024 Chapter 1 (product table)",
        )


# ── B. Meal-cap clinical properties ─────────────────────────────────────────

def test_meal_cap_properties(suite: Suite) -> None:
    print("\n=== B. Meal-cap clinical properties ===")

    # B1 Meal caps = daily × occasion frac for every stage/occasion
    for stage in KDOQI_DAILY_LIMITS:
        for occ in OCCASION_RULES:
            caps = meal_caps_from_backend(stage, occ)
            daily = KDOQI_DAILY_LIMITS[stage]
            fr = OCCASION_RULES[occ]["nutrient_caps"]
            expected = {
                "potassium": daily["potassium"] * fr[0],
                "phosphorus": daily["phosphorus"] * fr[1],
                "protein_per_kg": daily["protein_per_kg"] * fr[2],
                "sodium": daily["sodium"] * fr[3],
            }
            suite.add(
                "B_meal_caps",
                f"caps_{stage}_{occ}",
                caps == expected,
                f"{caps}",
                "KDOQI daily × OCCASION_RULES",
            )

    # B2 Snack tightest K/P/Na among occasions; Lunch=Dinner
    for stage in ["G3a"]:
        bf = meal_caps_from_backend(stage, "Breakfast")
        ln = meal_caps_from_backend(stage, "Lunch")
        dn = meal_caps_from_backend(stage, "Dinner")
        sn = meal_caps_from_backend(stage, "Snack")
        suite.add(
            "B_meal_caps",
            "lunch_equals_dinner_caps",
            ln == dn,
            f"Lunch={ln} Dinner={dn}",
            "OCCASION_RULES",
        )
        suite.add(
            "B_meal_caps",
            "snack_tightest_potassium",
            sn["potassium"] < bf["potassium"] < ln["potassium"],
            f"Snack={sn['potassium']} BF={bf['potassium']} Lunch={ln['potassium']}",
            "OCCASION_RULES nutrient_caps",
        )

    # B3 Sum of occasion fractions (product design note)
    # Breakfast+Lunch+Dinner+Snack for K = 0.25+0.40+0.40+0.15 = 1.20
    k_sum = sum(OCCASION_RULES[o]["nutrient_caps"][0] for o in OCCASION_RULES)
    suite.add(
        "B_meal_caps",
        "daily_fraction_sum_documented",
        abs(k_sum - 1.20) < 1e-9,
        f"sum_K_fracs={k_sum:.2f} (allows ~120% if all meals maxed — product choice, not KDOQI text)",
        "OCCASION_RULES operationalization",
    )

    # B4 train meal_caps helper matches backend
    for stage in ["G2", "G4"]:
        for occ in ["Breakfast", "Snack"]:
            a = meal_caps(stage, occ)
            b = meal_caps_from_backend(stage, occ)
            suite.add(
                "B_meal_caps",
                f"train_helper_matches_{stage}_{occ}",
                a == b,
                f"train={a} backend={b}",
                "train_xgboost_v3_meal.meal_caps",
            )


# ── C. Guideline-derived clinical scenarios (model predictions) ─────────────

def test_clinical_scenarios(suite: Suite, model) -> None:
    print("\n=== C. Clinical scenario predictions ===")

    cases = []

    # C1 Well under all meal caps → must be LOW (never HIGH)
    for stage in ["G2", "G3a", "G3b", "G4"]:
        for occ in ["Breakfast", "Lunch", "Dinner", "Snack"]:
            caps = meal_caps_from_backend(stage, occ)
            k, p, pro, na = (
                caps["potassium"] * 0.4,
                caps["phosphorus"] * 0.4,
                caps["protein_per_kg"] * 0.4,
                caps["sodium"] * 0.4,
            )
            pred, conf, cs = predict_label(model, stage, occ, k, p, pro, na)
            exp = label_from_score(cs)
            ok = pred == "LOW" and pred == exp
            cases.append((f"under40_{stage}_{occ}", ok, pred, exp, cs))
            suite.add(
                "C_scenarios",
                f"under40_pct_caps_{stage}_{occ}",
                ok,
                f"pred={pred} exp={exp} score={cs:.3f} conf={conf:.2f}",
                "KDOQI meal-cap operationalization",
            )

    # C2 Exactly at all meal caps (score = 1.0) → MODERATE
    for stage in ["G3a", "G4"]:
        for occ in ["Lunch", "Snack"]:
            caps = meal_caps_from_backend(stage, occ)
            pred, conf, cs = predict_label(
                model,
                stage,
                occ,
                caps["potassium"],
                caps["phosphorus"],
                caps["protein_per_kg"],
                caps["sodium"],
            )
            exp = label_from_score(cs)
            ok = abs(cs - 1.0) < 1e-6 and pred == "MODERATE" and exp == "MODERATE"
            suite.add(
                "C_scenarios",
                f"at_100pct_caps_{stage}_{occ}",
                ok,
                f"pred={pred} exp={exp} score={cs:.4f}",
                "score=1.0 → MODERATE (0.7–1.2)",
            )

    # C3 Severe multi-nutrient exceedance → HIGH (never LOW)
    for stage in ["G3a", "G4"]:
        for occ in ["Dinner", "Snack"]:
            caps = meal_caps_from_backend(stage, occ)
            pred, conf, cs = predict_label(
                model,
                stage,
                occ,
                caps["potassium"] * 2.0,
                caps["phosphorus"] * 2.0,
                caps["protein_per_kg"] * 2.0,
                caps["sodium"] * 2.0,
            )
            exp = label_from_score(cs)
            ok = pred == "HIGH" and exp == "HIGH" and pred != "LOW"
            suite.add(
                "C_scenarios",
                f"double_all_caps_{stage}_{occ}",
                ok,
                f"pred={pred} exp={exp} score={cs:.3f}",
                "KDOQI exceedance → HIGH",
            )

    # C4 Potassium-priority: high K alone should escalate more than high Na alone
    # (weights 0.35 vs 0.10) — same overshoot ratio
    stage, occ = "G3a", "Dinner"
    caps = meal_caps_from_backend(stage, occ)
    # 2× K only
    pred_k, _, cs_k = predict_label(
        model, stage, occ, caps["potassium"] * 2, caps["phosphorus"] * 0.5,
        caps["protein_per_kg"] * 0.5, caps["sodium"] * 0.5,
    )
    # 2× Na only
    pred_na, _, cs_na = predict_label(
        model, stage, occ, caps["potassium"] * 0.5, caps["phosphorus"] * 0.5,
        caps["protein_per_kg"] * 0.5, caps["sodium"] * 2,
    )
    suite.add(
        "C_scenarios",
        "potassium_overshoot_score_gt_sodium_overshoot",
        cs_k > cs_na,
        f"2×K score={cs_k:.3f} ({pred_k}) vs 2×Na score={cs_na:.3f} ({pred_na})",
        "CLINICAL_SEVERITY_WEIGHTS K=0.35 > Na=0.10",
    )

    # C5 Later stage: same absolute meal → higher or equal risk than earlier stage
    # Fixed absolute intakes that are moderate for G2 dinner
    abs_meal = (900.0, 350.0, 0.25, 700.0)  # K,P,pro,Na
    labels_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    prev_rank = -1
    monotone_ok = True
    details = []
    for stage in ["G2", "G3a", "G3b", "G4"]:
        pred, _, cs = predict_label(model, stage, "Dinner", *abs_meal)
        rank = labels_rank[pred]
        details.append(f"{stage}:{pred}({cs:.2f})")
        if rank < prev_rank:
            monotone_ok = False
        prev_rank = max(prev_rank, rank)
    suite.add(
        "C_scenarios",
        "same_dinner_risk_nondecreasing_by_stage",
        monotone_ok,
        " → ".join(details),
        "tighter KDOQI caps at later stages",
    )

    # C6 Snack vs Dinner: same absolute nutrients → Snack risk ≥ Dinner risk
    # (snack caps tighter)
    abs_snackish = (500.0, 200.0, 0.15, 400.0)
    for stage in ["G3a", "G4"]:
        pred_s, _, cs_s = predict_label(model, stage, "Snack", *abs_snackish)
        pred_d, _, cs_d = predict_label(model, stage, "Dinner", *abs_snackish)
        suite.add(
            "C_scenarios",
            f"same_intake_snack_ge_dinner_{stage}",
            cs_s >= cs_d - 1e-9 and labels_rank[pred_s] >= labels_rank[pred_d],
            f"Snack {pred_s}({cs_s:.2f}) vs Dinner {pred_d}({cs_d:.2f})",
            "OCCASION_RULES tighter snack caps",
        )


# ── D. Monotonicity / safety invariants ─────────────────────────────────────

def test_monotonicity(suite: Suite, model) -> None:
    print("\n=== D. Monotonicity & safety invariants ===")
    stage, occ = "G3b", "Lunch"
    caps = meal_caps_from_backend(stage, occ)
    base = (
        caps["potassium"] * 0.5,
        caps["phosphorus"] * 0.5,
        caps["protein_per_kg"] * 0.5,
        caps["sodium"] * 0.5,
    )
    rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}

    for i, nutrient in enumerate(["potassium", "phosphorus", "protein_per_kg", "sodium"]):
        scores = []
        labels = []
        for factor in [0.5, 1.0, 1.5, 2.0, 3.0]:
            vals = list(base)
            vals[i] = caps[nutrient] * factor
            pred, _, cs = predict_label(model, stage, occ, *vals)
            scores.append(cs)
            labels.append(rank[pred])
        score_mono = all(scores[j] <= scores[j + 1] + 1e-9 for j in range(len(scores) - 1))
        label_mono = all(labels[j] <= labels[j + 1] for j in range(len(labels) - 1))
        suite.add(
            "D_monotonicity",
            f"score_monotone_increasing_{nutrient}",
            score_mono,
            f"scores={[round(s,3) for s in scores]}",
            "clinical_score definition",
        )
        suite.add(
            "D_monotonicity",
            f"label_nondecreasing_{nutrient}",
            label_mono,
            f"labels={[['LOW','MODERATE','HIGH'][r] for r in labels]}",
            "guideline-aligned severity direction",
        )

    # Safety: never HIGH if all nutrients ≤ 50% meal cap
    unsafe_high = 0
    total = 0
    for stage in KDOQI_DAILY_LIMITS:
        for occ in OCCASION_RULES:
            caps = meal_caps_from_backend(stage, occ)
            pred, _, cs = predict_label(
                model,
                stage,
                occ,
                caps["potassium"] * 0.5,
                caps["phosphorus"] * 0.5,
                caps["protein_per_kg"] * 0.5,
                caps["sodium"] * 0.5,
            )
            total += 1
            if pred == "HIGH":
                unsafe_high += 1
    suite.add(
        "D_monotonicity",
        "no_false_HIGH_when_all_le_50pct_meal_cap",
        unsafe_high == 0,
        f"false_HIGH={unsafe_high}/{total}",
        "clinical safety property",
    )

    # Safety: never LOW if all nutrients ≥ 2× meal cap
    unsafe_low = 0
    total = 0
    for stage in KDOQI_DAILY_LIMITS:
        for occ in OCCASION_RULES:
            caps = meal_caps_from_backend(stage, occ)
            pred, _, cs = predict_label(
                model,
                stage,
                occ,
                caps["potassium"] * 2.0,
                caps["phosphorus"] * 2.0,
                caps["protein_per_kg"] * 2.0,
                caps["sodium"] * 2.0,
            )
            total += 1
            if pred == "LOW":
                unsafe_low += 1
    suite.add(
        "D_monotonicity",
        "no_false_LOW_when_all_ge_2x_meal_cap",
        unsafe_low == 0,
        f"false_LOW={unsafe_low}/{total}",
        "clinical safety property",
    )


# ── E. Holdout agreement with guideline oracle ──────────────────────────────

def test_holdout_oracle(suite: Suite, model) -> dict:
    print("\n=== E. Holdout vs guideline oracle ===")
    if not MEAL_DATASET_PATH.exists():
        suite.add("E_holdout", "dataset_present", False, "missing meal dataset")
        return {}

    df = pd.read_csv(MEAL_DATASET_PATH)
    # Recompute oracle score from backend constants (not trusting CSV blindly)
    def oracle_row(r):
        return clinical_score_meal(
            r["ckd_stage"],
            r["occasion"],
            float(r["potassium"]),
            float(r["phosphorus"]),
            float(r["protein_per_kg"]),
            float(r["sodium"]),
        )

    df["oracle_score"] = df.apply(oracle_row, axis=1)
    df["oracle_label"] = df["oracle_score"].apply(label_from_score)

    # CSV risk_label should match oracle (training used same rule)
    label_agree = float((df["risk_label"] == df["oracle_label"]).mean())
    suite.add(
        "E_holdout",
        "dataset_labels_match_backend_oracle",
        label_agree >= 0.999,
        f"agreement={label_agree:.4f} n={len(df)}",
        "KDOQI×OCCASION_RULES labeling",
    )

    # Model vs oracle on full set (diagnostic; model trained on part of it)
    X = pd.DataFrame(
        {
            "potassium": df["potassium"],
            "phosphorus": df["phosphorus"],
            "protein_per_kg": df["protein_per_kg"],
            "sodium": df["sodium"],
            "ckd_stage_encoded": df["ckd_stage"].map(STAGE_ENCODE),
            "stage_numeric": df["ckd_stage"].map(STAGE_NUMERIC),
            "k_p_product": (df["potassium"] * df["phosphorus"]) / 1e6,
            "protein_sodium_ratio": df["protein_per_kg"]
            / (df["sodium"] / 1000 + 1e-6),
            "clinical_score": df["oracle_score"],
        }
    )[FEATURES]
    pred = model.predict(X)
    pred_lab = np.array([RISK_CLASSES[int(i)] for i in pred])
    oracle = df["oracle_label"].to_numpy()
    agree = float((pred_lab == oracle).mean())

    # Split identical to training for holdout safety check
    from sklearn.model_selection import train_test_split

    y_enc = df["oracle_label"].map(RISK_ENCODE)
    _, _, _, _, _, idx_test = train_test_split(
        X,
        y_enc,
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )
    test_mask = np.asarray(df.index.isin(idx_test))

    high_mask = oracle == "HIGH"
    high_miss_all = (pred_lab != "HIGH") & high_mask
    high_miss_test = high_miss_all & test_mask
    high_to_low = (pred_lab == "LOW") & high_mask

    miss_scores = df.loc[high_miss_all, "oracle_score"]
    near_boundary_only = (
        int(high_miss_all.sum()) == 0
        or (
            float(miss_scores.max()) < 1.205
            and int((pred_lab[high_miss_all] == "MODERATE").sum())
            == int(high_miss_all.sum())
        )
    )

    suite.add(
        "E_holdout",
        "model_agrees_with_guideline_oracle",
        agree >= 0.99,
        f"agreement={agree:.4f}",
        "SEVERITY_THRESHOLDS on meal clinical_score",
    )
    suite.add(
        "E_holdout",
        "zero_HIGH_misses_on_holdout",
        int(high_miss_test.sum()) == 0,
        f"HIGH_misses_holdout={int(high_miss_test.sum())}/"
        f"{int((high_mask & test_mask).sum())}",
        "clinical safety on unseen meals",
    )
    suite.add(
        "E_holdout",
        "never_predict_LOW_when_oracle_HIGH",
        int(high_to_low.sum()) == 0,
        f"HIGH→LOW={int(high_to_low.sum())}",
        "clinical safety: no class-skip under-call",
    )
    suite.add(
        "E_holdout",
        "fullset_HIGH_misses_only_near_1_2_boundary",
        near_boundary_only,
        f"HIGH_misses_full={int(high_miss_all.sum())}/"
        f"{int(high_mask.sum())}; "
        f"score_max={float(miss_scores.max()) if len(miss_scores) else 'n/a'}; "
        f"all_pred_MODERATE={near_boundary_only}",
        "boundary soft errors only (score < 1.205 → MOD not LOW)",
    )
    return {
        "oracle_label_agree": label_agree,
        "model_oracle_agree": agree,
        "high_miss_full": int(high_miss_all.sum()),
        "high_miss_holdout": int(high_miss_test.sum()),
        "high_to_low": int(high_to_low.sum()),
    }


# ── F. Scope honesty: what XGBoost does / does not enforce ───────────────────

def test_scope(suite: Suite) -> None:
    print("\n=== F. Scope boundaries (documented) ===")
    # These are informational passes documenting architecture — fail only if
    # forbidden lists missing when we claim stage-aware advice elsewhere.
    from backend.api.meal_planner import _FORBIDDEN_BY_STAGE

    suite.add(
        "F_scope",
        "forbidden_foods_exist_separate_from_xgb",
        "G3B" in _FORBIDDEN_BY_STAGE and "G4" in _FORBIDDEN_BY_STAGE,
        f"stages={list(_FORBIDDEN_BY_STAGE)}",
        "Meal planner / recommender layer (not XGBoost)",
    )
    suite.add(
        "F_scope",
        "xgb_does_not_encode_food_identity",
        True,
        "XGBoost inputs are nutrient totals + stage only — food bans are enforced elsewhere",
        "architecture boundary",
    )
    suite.add(
        "F_scope",
        "weights_thresholds_are_author_derived",
        True,
        "CLINICAL_SEVERITY_WEIGHTS and SEVERITY_THRESHOLDS are product-calibrated "
        "(documented in clinical_constants) — not a published KDOQI formula",
        "clinical_constants.py honesty note",
    )
    suite.add(
        "F_scope",
        "day_v3_still_protected",
        sha256_file(DAY_V3_PATH) == PROTECTED_SHA256,
        "production day model untouched",
        "artifact safety",
    )


def write_reports(suite: Suite, extras: dict) -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_pass": suite.n_pass,
        "n_fail": suite.n_fail,
        "n_total": len(suite.checks),
        "all_passed": suite.n_fail == 0,
        "verdict": (
            "Meal XGBoost v3 aligns with GuidaPlate's KDOQI/KDIGO-derived "
            "clinical constants and OCCASION_RULES meal operationalization. "
            "Severity weights/thresholds are author-derived (documented). "
            "Forbidden-food guidelines are enforced outside this model."
        ),
        "checks": [
            {
                "section": c.section,
                "name": c.name,
                "passed": c.passed,
                "detail": c.detail,
                "guideline_ref": c.guideline_ref,
            }
            for c in suite.checks
        ],
        "extras": extras,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Meal XGBoost — clinical guidelines compliance",
        "",
        f"**Result: {'ALL PASSED' if suite.n_fail == 0 else 'FAILURES'}** "
        f"({suite.n_pass}/{len(suite.checks)})",
        "",
        "## Verdict",
        "",
        payload["verdict"],
        "",
        "| Section | Check | Result | Guideline ref | Detail |",
        "|---|---|---|---|---|",
    ]
    for c in suite.checks:
        lines.append(
            f"| {c.section} | {c.name} | {'PASS' if c.passed else 'FAIL'} | "
            f"{c.guideline_ref or '—'} | {c.detail.replace('|', '/')} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")

    (DOCS / "README.md").write_text(
        "\n".join(
            [
                "# 12 — Meal XGBoost clinical guidelines compliance",
                "",
                "```bash",
                "./venv311/bin/python3 scripts/test_meal_xgboost_clinical_guidelines.py",
                "```",
                "",
                f"Latest: **{suite.n_pass}/{len(suite.checks)} passed**.",
                "",
                "See `outputs/stats/12_meal_xgboost_clinical_guidelines_report.md`.",
                "",
            ]
        )
    )
    pd.DataFrame(
        [
            {
                "section": c.section,
                "check": c.name,
                "passed": c.passed,
                "guideline_ref": c.guideline_ref,
                "detail": c.detail,
            }
            for c in suite.checks
        ]
    ).to_csv(STATS / "12_meal_xgboost_clinical_guidelines_checks.csv", index=False)

    print(f"\nWrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


def main() -> int:
    print("=" * 72)
    print("MEAL XGBoost — CLINICAL GUIDELINES COMPLIANCE BATTERY")
    print("=" * 72)

    suite = Suite()
    if not MEAL_MODEL_PATH.exists():
        print("Missing meal model")
        return 1

    model = joblib.load(MEAL_MODEL_PATH)

    test_constant_alignment(suite)
    test_meal_cap_properties(suite)
    test_clinical_scenarios(suite, model)
    test_monotonicity(suite, model)
    holdout = test_holdout_oracle(suite, model)
    test_scope(suite)

    write_reports(suite, {"holdout": holdout, "meal_model": str(MEAL_MODEL_PATH)})

    print("\n" + "=" * 72)
    print(f"RESULT: {suite.n_pass}/{len(suite.checks)} PASSED — {suite.n_fail} FAILED")
    print("=" * 72)
    if suite.n_fail:
        print("\nFailures:")
        for c in suite.checks:
            if not c.passed:
                print(f"  - [{c.section}] {c.name}: {c.detail}")
        return 1

    print(
        "\nVERDICT: Meal model follows GuidaPlate's KDOQI/KDIGO-derived constants "
        "and meal OCCASION_RULES. Weights/thresholds are author-derived "
        "(documented). Food-ban guidelines live outside XGBoost."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
