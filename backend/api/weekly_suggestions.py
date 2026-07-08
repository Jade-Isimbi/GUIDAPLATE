"""
weekly_suggestions.py
GuidaPlate — LSTM-driven food suggestions
for the remaining days of the week.

Uses the LSTM pattern analysis on the
patient's last 6 logged meals to identify
which nutrients have trended high, then
returns stage-safe, nutrient-appropriate
food suggestions organized by meal
occasion (breakfast/lunch/dinner/snack).

This is a curated suggestion list, not
a generated multi-day meal schedule.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.daily_budget import normalize_ckd_stage
from backend.auth.security import get_current_user_id
from backend.clinical_constants import KDOQI_DAILY_LIMITS
from backend.database.db import Food, FoodLog, Patient, User, get_db
from backend.models.lstm_model import get_analyzer
from backend.models.recommender import EXCLUDE_CATEGORIES, get_recommender
from backend.utils.meal_aggregation import (
    group_logs_into_meals,
    meal_vector_from_totals,
    sum_meal_nutrients,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Weekly Suggestions"])

MAX_MEALS = 6
WINDOW_DAYS = 7
SUGGESTIONS_PER_OCCASION = 5

NUTRIENT_LABELS = {
    "potassium": "potassium",
    "phosphorus": "phosphorus",
    "protein_per_kg": "protein",
    "sodium": "sodium",
}

NUTRIENT_COLUMN_MAP = {
    "potassium": "potassium_mg",
    "phosphorus": "phosphorus_mg",
    "protein_per_kg": "protein_g",
    "sodium": "sodium_mg",
}

OCCASION_KEYWORDS = {
    "breakfast": ["porridge", "bread", "egg", "tea", "fruit", "milk"],
    "lunch": ["rice", "beans", "vegetable", "meat", "fish", "potato"],
    "dinner": ["rice", "beans", "vegetable", "meat", "fish", "ugali", "cassava"],
    "snack": ["fruit", "nut", "biscuit"],
}


class FoodSuggestion(BaseModel):
    english: str
    french: str | None = None
    kinyarwanda: str | None = None
    category: str | None = None
    potassium_mg: float | None = None
    phosphorus_mg: float | None = None
    protein_g: float | None = None
    sodium_mg: float | None = None
    reason: str


class MealOccasionSuggestions(BaseModel):
    occasion: str
    suggestions: list[FoodSuggestion]


class WeeklySuggestionsResponse(BaseModel):
    trajectory_risk: str
    trajectory_confidence: float
    flagged_nutrient: str | None
    flagged_reason: str
    remaining_days: int
    suggestions_by_meal: list[MealOccasionSuggestions]
    clinical_note: str
    analysis_available: bool = True


def _fetch_recent_logs(db: Session, user_id: str, days: int = WINDOW_DAYS) -> list[FoodLog]:
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


def _get_recent_meal_sequence(
    db: Session,
    user_id: str,
    body_weight_kg: float,
) -> list[list[float]]:
    """
    Last 6 logged meals in LSTM format, using the same aggregation
    pipeline as next_meal.py (group_logs_into_meals → sum_meal_nutrients
    → meal_vector_from_totals), extended across a 7-day window.
    """
    logs = _fetch_recent_logs(db, user_id)
    grouped = _group_logs_by_day(logs)

    meal_sequence: list[list[float]] = []
    for day in sorted(grouped.keys()):
        meal_groups = group_logs_into_meals(grouped[day])
        for group in meal_groups:
            totals = sum_meal_nutrients(group)
            meal_sequence.append(
                meal_vector_from_totals(totals, body_weight_kg=body_weight_kg or 0)
            )

    return meal_sequence[-MAX_MEALS:] if len(meal_sequence) > MAX_MEALS else meal_sequence


def _meal_vectors_to_dicts(meal_sequence: list[list[float]]) -> list[dict[str, float]]:
    return [
        {
            "potassium": vec[0],
            "phosphorus": vec[1],
            "protein_per_kg": vec[2],
            "sodium": vec[3],
        }
        for vec in meal_sequence
    ]


def _identify_flagged_nutrient(meal_sequence: list[dict[str, float]]) -> tuple[str | None, str]:
    if not meal_sequence:
        return None, "No recent meals logged — showing general safe foods."

    totals = {
        "potassium": 0.0,
        "phosphorus": 0.0,
        "protein_per_kg": 0.0,
        "sodium": 0.0,
    }
    n = len(meal_sequence)
    for meal in meal_sequence:
        for key in totals:
            totals[key] += meal.get(key, 0.0)
    averages = {k: v / n for k, v in totals.items()}

    rough_limits = {
        "potassium": 1000.0,
        "phosphorus": 270.0,
        "protein_per_kg": 0.2,
        "sodium": 770.0,
    }

    ratios = {k: averages[k] / rough_limits[k] for k in averages}
    flagged = max(ratios, key=ratios.get)

    if ratios[flagged] < 0.8:
        return None, "Your recent meals have been within safe nutrient ranges."

    label = NUTRIENT_LABELS[flagged]
    reason = (
        f"Your recent meals have trended high in {label}. "
        f"Suggestions below are lower in {label} to help balance the rest of your week."
    )
    return flagged, reason


def _is_stage_safe(food: Food, ckd_stage: str) -> bool:
    recommender = get_recommender()
    stage_number = recommender._stage_to_number(ckd_stage)
    return recommender._parse_stage_safe(food.stage_safe_range, stage_number)


def _get_safe_foods_for_occasion(
    db: Session,
    ckd_stage: str,
    flagged_nutrient: str | None,
    occasion: str,
    limit: int = SUGGESTIONS_PER_OCCASION,
) -> list[Food]:
    foods = db.query(Food).filter(~Food.category.in_(EXCLUDE_CATEGORIES)).all()
    safe_foods = [f for f in foods if _is_stage_safe(f, ckd_stage)]

    if flagged_nutrient:
        col = NUTRIENT_COLUMN_MAP[flagged_nutrient]
        safe_foods.sort(key=lambda f: getattr(f, col, 9999) or 9999)

    keywords = OCCASION_KEYWORDS.get(occasion, [])
    matched = [
        f
        for f in safe_foods
        if any(
            kw in (f.category or "").lower() or kw in (f.english or "").lower()
            for kw in keywords
        )
    ]

    result = matched[:limit]
    if len(result) < limit:
        remaining = [f for f in safe_foods if f not in result]
        result += remaining[: limit - len(result)]

    return result[:limit]


def _remaining_days_in_week() -> int:
    today = datetime.utcnow().date()
    remaining = 7 - today.isoweekday()
    return 7 if remaining == 0 else remaining


@router.get("/next-meal/weekly-suggestions", response_model=WeeklySuggestionsResponse)
def get_weekly_suggestions(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> WeeklySuggestionsResponse:
    user = db.query(User).filter(User.user_id == user_id).first()
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    raw_stage = (
        patient.ckd_stage if patient and patient.ckd_stage else (user.ckd_stage if user else None)
    )
    if not raw_stage:
        raise HTTPException(status_code=404, detail="Patient not found")

    ckd_stage = normalize_ckd_stage(raw_stage)
    if ckd_stage not in KDOQI_DAILY_LIMITS:
        raise HTTPException(status_code=400, detail=f"Unsupported CKD stage: {ckd_stage!r}")

    weight_source = (
        patient.body_weight_kg
        if patient and patient.body_weight_kg is not None
        else (user.weight_kg if user else None)
    )
    body_weight = float(weight_source or 70.0)
    meal_sequence = _get_recent_meal_sequence(db, user_id, body_weight)

    if meal_sequence:
        try:
            lstm_result = get_analyzer().analyze(meal_sequence)
        except Exception as exc:
            logger.warning(
                "LSTM failed in weekly suggestions: %s", exc
            )
            lstm_result = {
                "risk_label": "UNAVAILABLE",
                "confidence": 0.0,
                "unavailable": True,
                "reason": "analysis_failed",
            }
    else:
        lstm_result = {
            "risk_label": "UNAVAILABLE",
            "confidence": 0.0,
            "unavailable": True,
            "reason": "no_meals_logged",
        }

    analysis_available = not lstm_result.get("unavailable", False)

    nutrient_dicts = _meal_vectors_to_dicts(meal_sequence)
    flagged, reason = _identify_flagged_nutrient(nutrient_dicts)

    occasions = ["breakfast", "lunch", "dinner", "snack"]
    suggestions_by_meal: list[MealOccasionSuggestions] = []
    for occ in occasions:
        foods = _get_safe_foods_for_occasion(
            db,
            ckd_stage=ckd_stage,
            flagged_nutrient=flagged,
            occasion=occ,
            limit=SUGGESTIONS_PER_OCCASION,
        )
        suggestion_list = [
            FoodSuggestion(
                english=food.english,
                french=food.french,
                kinyarwanda=food.kinyarwanda,
                category=food.category,
                potassium_mg=food.potassium_mg,
                phosphorus_mg=food.phosphorus_mg,
                protein_g=food.protein_g,
                sodium_mg=food.sodium_mg,
                reason=(
                    f"Low in {NUTRIENT_LABELS.get(flagged, 'key nutrients')}"
                    if flagged
                    else "Safe for your CKD stage"
                ),
            )
            for food in foods
        ]
        suggestions_by_meal.append(
            MealOccasionSuggestions(occasion=occ, suggestions=suggestion_list)
        )

    meal_count = len(meal_sequence)
    clinical_note = (
        f"This is a suggestion list based on your eating pattern over the "
        f"last {meal_count} meal{'s' if meal_count != 1 else ''}, "
        f"not a fixed meal schedule. Choose foods that fit your daily nutrient budget."
    )

    return WeeklySuggestionsResponse(
        trajectory_risk=lstm_result["risk_label"],
        trajectory_confidence=lstm_result["confidence"],
        flagged_nutrient=flagged,
        flagged_reason=reason,
        remaining_days=_remaining_days_in_week(),
        suggestions_by_meal=suggestions_by_meal,
        clinical_note=clinical_note,
        analysis_available=analysis_available,
    )
