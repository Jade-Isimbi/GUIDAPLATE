"""
weekly_trend.py
GuidaPlate — Weekly nutrient trend and LSTM pattern summary
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import json
import logging

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.daily_budget import normalize_ckd_stage
from backend.auth.security import get_current_user_id
from backend.clinical_constants import (
    CLINICAL_SEVERITY_WEIGHTS,
    KDOQI_DAILY_LIMITS,
    SEVERITY_THRESHOLDS,
)
from backend.config import MODELS_DIR
from backend.database.db import FoodLog, Patient, get_db
from backend.models.lstm_model import get_analyzer
from backend.models.xgboost_model import get_predictor

router = APIRouter(tags=["Weekly Trend"])
logger = logging.getLogger(__name__)

MAX_LSTM_STEPS = 6

# ── Tier 3: Weekly RF ──────────────────
_weekly_rf: object | None = None
_weekly_config: dict | None = None
_RF_PATH = MODELS_DIR / "weekly_rf.pkl"
_CFG_PATH = MODELS_DIR / "weekly_rf_config.json"
WEEKLY_NEUTRAL = [1 / 3, 1 / 3, 1 / 3]
WEEKLY_LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def _get_weekly_rf():
    global _weekly_rf, _weekly_config
    if _weekly_rf is None:
        if _RF_PATH.exists():
            _weekly_rf = joblib.load(_RF_PATH)
            if _CFG_PATH.exists():
                with open(_CFG_PATH) as f:
                    _weekly_config = json.load(f)
            logger.info("Weekly RF loaded ✓")
        else:
            logger.warning("weekly_rf.pkl not found — rule fallback active")
    return _weekly_rf


def _rule_fallback(sequence: list[list[float]]) -> str:
    """
    Conservative rule fallback if RF unavailable.
    HIGH if any real day P(HIGH) > 0.5
    MODERATE if any real day P(MOD) > 0.5
    LOW otherwise
    """
    for day in sequence:
        if abs(day[0] - 1 / 3) < 0.01:
            continue
        if day[2] > 0.5:
            return "HIGH"
    for day in sequence:
        if abs(day[0] - 1 / 3) < 0.01:
            continue
        if day[1] > 0.5:
            return "MODERATE"
    return "LOW"


def _predict_weekly_tier3(daily_probas: list[list[float]]) -> dict:
    """
    Predict weekly risk from 7-day XGBoost probability sequences.

    daily_probas: list of [P(LOW), P(MOD), P(HIGH)] per logged day.
    Missing days padded with neutral prior [0.33, 0.33, 0.33].
    """
    rf = _get_weekly_rf()

    sequence = list(daily_probas)
    while len(sequence) < 7:
        sequence.append(WEEKLY_NEUTRAL.copy())
    sequence = sequence[:7]

    X = np.array(sequence).flatten().reshape(1, -1)

    if rf is not None:
        pred = rf.predict(X)[0]
        proba = rf.predict_proba(X)[0]
        label = WEEKLY_LABEL_MAP[int(pred)]
        conf = float(proba[int(pred)])
        method = "random_forest"
        model_name = (
            _weekly_config.get("winner", "RF + CW MOD=3") if _weekly_config else "RF + CW MOD=3"
        )
        mod_recall = _weekly_config.get("mod_recall", 0.9026) if _weekly_config else 0.9026
    else:
        label = _rule_fallback(sequence)
        conf = 0.70
        method = "rule_fallback"
        model_name = "Rule baseline"
        mod_recall = 0.351

    days_with_data = sum(1 for d in daily_probas if abs(d[0] - 1 / 3) > 0.01)

    return {
        "risk_label": label,
        "confidence": round(conf, 4),
        "method": method,
        "days_analyzed": days_with_data,
        "model_name": model_name,
        "mod_recall": mod_recall,
    }


class DayNutrients(BaseModel):
    potassium: float
    phosphorus: float
    protein_per_kg: float
    sodium: float


class DayPercentUsed(BaseModel):
    potassium: float
    phosphorus: float
    protein: float
    sodium: float


class WeeklyDaySummary(BaseModel):
    date: str
    meals_count: int
    nutrients: DayNutrients
    budget_label: str
    percent_used: DayPercentUsed


class LstmPatternSummary(BaseModel):
    risk_label: str
    confidence: float
    trend: str
    days_analyzed: int


class WeeklySummary(BaseModel):
    risk_label: str
    confidence: float
    method: str
    days_analyzed: int
    model_name: str
    mod_recall: float


class WeeklyTrendResponse(BaseModel):
    days: list[WeeklyDaySummary]
    lstm_pattern: LstmPatternSummary | None
    weekly_summary: WeeklySummary | None
    ckd_stage: str
    weight_kg: float


def _meals_count(day_logs: list[FoodLog]) -> int:
    occasions = {log.meal_occasion for log in day_logs if log.meal_occasion}
    if occasions:
        return len(occasions)
    return len(day_logs)


def _aggregate_day(day_logs: list[FoodLog]) -> dict[str, float]:
    return {
        "potassium": sum(float(log.potassium_mg or 0) for log in day_logs),
        "phosphorus": sum(float(log.phosphorus_mg or 0) for log in day_logs),
        "protein_g": sum(float(log.protein_g or 0) for log in day_logs),
        "sodium": sum(float(log.sodium_mg or 0) for log in day_logs),
    }


def _budget_label(
    totals: dict[str, float],
    limits: dict[str, float],
    weight_kg: float,
) -> str:
    """
    v3 weighted clinical severity score.
    Matches XGBoost v3 training labels.
    K(35%) P(30%) Protein(25%) Na(10%)
    HIGH >= 1.2  MODERATE >= 0.7  LOW < 0.7
    """
    k_ratio = totals["potassium"] / limits["potassium"] if limits["potassium"] > 0 else 0.0

    p_ratio = totals["phosphorus"] / limits["phosphorus"] if limits["phosphorus"] > 0 else 0.0

    protein_per_kg = totals["protein_g"] / weight_kg if weight_kg > 0 else 0.0

    pro_ratio = protein_per_kg / limits["protein_per_kg"] if limits["protein_per_kg"] > 0 else 0.0

    na_ratio = totals["sodium"] / limits["sodium"] if limits["sodium"] > 0 else 0.0

    score = (
        k_ratio * CLINICAL_SEVERITY_WEIGHTS["potassium"]
        + p_ratio * CLINICAL_SEVERITY_WEIGHTS["phosphorus"]
        + pro_ratio * CLINICAL_SEVERITY_WEIGHTS["protein"]
        + na_ratio * CLINICAL_SEVERITY_WEIGHTS["sodium"]
    )

    if score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= SEVERITY_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


def _percent_used(
    totals: dict[str, float],
    limits: dict[str, float],
    weight_kg: float,
) -> DayPercentUsed:
    protein_per_kg = totals["protein_g"] / weight_kg if weight_kg > 0 else 0.0

    def pct(consumed: float, limit: float) -> float:
        if limit <= 0:
            return 0.0
        return round(consumed / limit * 100.0, 1)

    return DayPercentUsed(
        potassium=pct(totals["potassium"], limits["potassium"]),
        phosphorus=pct(totals["phosphorus"], limits["phosphorus"]),
        protein=pct(protein_per_kg, limits["protein_per_kg"]),
        sodium=pct(totals["sodium"], limits["sodium"]),
    )


def _fetch_logs_for_window(
    db: Session,
    user_id: str,
    days: int,
) -> list[FoodLog]:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days - 1)
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)
    return (
        db.query(FoodLog)
        .filter(
            FoodLog.patient_id == user_id,
            FoodLog.logged_at >= start_dt,
            FoodLog.logged_at <= end_dt,
        )
        .order_by(FoodLog.logged_at.asc())
        .all()
    )


def _group_logs_by_day(logs: list[FoodLog]) -> dict[date, list[FoodLog]]:
    grouped: dict[date, list[FoodLog]] = defaultdict(list)
    for log in logs:
        if log.logged_at is None:
            continue
        grouped[log.logged_at.date()].append(log)
    return grouped


@router.get("/patient/weekly-trend", response_model=WeeklyTrendResponse)
def get_weekly_trend(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    days: int = Query(default=7, ge=1, le=7),
) -> WeeklyTrendResponse:
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient or not patient.ckd_stage:
        raise HTTPException(status_code=400, detail="Patient profile or CKD stage not set.")

    ckd_stage = normalize_ckd_stage(patient.ckd_stage)
    if ckd_stage not in KDOQI_DAILY_LIMITS:
        raise HTTPException(status_code=400, detail=f"Unsupported CKD stage: {ckd_stage!r}")

    weight_kg = float(patient.body_weight_kg or 70.0)
    limits = KDOQI_DAILY_LIMITS[ckd_stage]

    logs = _fetch_logs_for_window(db, user_id, days)
    grouped = _group_logs_by_day(logs)
    sorted_days = sorted(grouped.keys())[:days]

    day_summaries: list[WeeklyDaySummary] = []
    meal_sequence: list[list[float]] = []
    daily_probas: list[list[float]] = []

    for day in sorted_days:
        day_logs = grouped[day]
        totals = _aggregate_day(day_logs)
        protein_per_kg = totals["protein_g"] / weight_kg if weight_kg > 0 else 0.0

        try:
            predictor = get_predictor()
            feature_vector, _ = predictor._build_features(
                potassium=totals["potassium"],
                phosphorus=totals["phosphorus"],
                protein_per_kg=protein_per_kg,
                sodium=totals["sodium"],
                ckd_stage=ckd_stage,
            )
            day_proba = predictor.model.predict_proba(feature_vector)[0].tolist()
            daily_probas.append(day_proba)
        except Exception:
            logger.warning(
                "XGBoost per-day prediction failed for day %s — using "
                "neutral prior. Weekly RF result may understate risk.",
                day,
            )
            daily_probas.append(WEEKLY_NEUTRAL.copy())

        day_summaries.append(
            WeeklyDaySummary(
                date=day.isoformat(),
                meals_count=_meals_count(day_logs),
                nutrients=DayNutrients(
                    potassium=round(totals["potassium"], 1),
                    phosphorus=round(totals["phosphorus"], 1),
                    protein_per_kg=round(protein_per_kg, 4),
                    sodium=round(totals["sodium"], 1),
                ),
                budget_label=_budget_label(totals, limits, weight_kg),
                percent_used=_percent_used(totals, limits, weight_kg),
            )
        )
        meal_sequence.append([
            totals["potassium"],
            totals["phosphorus"],
            protein_per_kg,
            totals["sodium"],
            0.5,  # neutral occasion for daily aggregate
        ])

    lstm_sequence = meal_sequence[-MAX_LSTM_STEPS:] if len(meal_sequence) > MAX_LSTM_STEPS else meal_sequence
    lstm_pattern: LstmPatternSummary | None = None
    if lstm_sequence:
        try:
            lstm_result = get_analyzer().analyze(lstm_sequence)
            lstm_pattern = LstmPatternSummary(
                risk_label=lstm_result["risk_label"],
                confidence=lstm_result["confidence"],
                trend=lstm_result["trend"],
                days_analyzed=len(lstm_sequence),
            )
        except Exception as e:
            logger.error(
                "LSTM weekly pattern failed: %s",
                e,
                exc_info=True,
            )
            lstm_pattern = None

    weekly_summary: WeeklySummary | None = None
    try:
        tier3 = _predict_weekly_tier3(daily_probas)
        weekly_summary = WeeklySummary(
            risk_label=tier3["risk_label"],
            confidence=tier3["confidence"],
            method=tier3["method"],
            days_analyzed=tier3["days_analyzed"],
            model_name=tier3["model_name"],
            mod_recall=tier3["mod_recall"],
        )
    except Exception as exc:
        logger.warning("Tier 3 weekly summary failed: %s", exc)
        weekly_summary = None

    return WeeklyTrendResponse(
        days=day_summaries,
        lstm_pattern=lstm_pattern,
        weekly_summary=weekly_summary,
        ckd_stage=ckd_stage,
        weight_kg=weight_kg,
    )
