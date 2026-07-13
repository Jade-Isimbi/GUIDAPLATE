"""
lstm_sequence.py
GuidaPlate — Canonical LSTM meal-sequence builder

All product call sites (What to eat next, weekly suggestions, weekly trend,
Meal Check pattern endpoint, next-meal forecast) must use build_lstm_sequence
so the same patient state yields the same LSTM inputs and outputs.

Training note (notebooks 05 / 05b / 05c):
  Sequences were NHANES 2-day Breakfast/Lunch/Dinner slots with encodings
  0.00 / 0.33 / 0.67. Snack=0.50 is a product extension and is
  out-of-training-distribution (OOD) relative to the trained model.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

from backend.database.db import FoodLog
from backend.utils.meal_aggregation import (
    group_logs_into_meals,
    sum_meal_nutrients,
)

LSTM_MAX_MEALS = 6
LSTM_LOOKBACK_DAYS = 7

# Breakfast/Lunch/Dinner match training (05b OCCASION_BY_SLOT).
# Snack=0.50 is OOD vs NHANES B/L/D-only training — documented intentionally.
OCCASION_TO_ENCODING: dict[str, float] = {
    "Breakfast": 0.00,
    "Lunch": 0.33,
    "Dinner": 0.67,
    "Snack": 0.50,
}

_DEFAULT_OCCASION_ENCODING = 0.50  # unknown / blank → neutral (same as Snack OOD)


def occasion_encoding(meal_occasion: str | None) -> float:
    """Map meal_occasion string → LSTM occasion feature."""
    if not meal_occasion:
        return _DEFAULT_OCCASION_ENCODING
    key = meal_occasion.strip()
    if key in OCCASION_TO_ENCODING:
        return OCCASION_TO_ENCODING[key]
    # Case-insensitive fallback
    for name, value in OCCASION_TO_ENCODING.items():
        if name.lower() == key.lower():
            return value
    return _DEFAULT_OCCASION_ENCODING


def meal_vector_5(
    totals: dict[str, float],
    body_weight_kg: float,
    meal_occasion: str | None,
) -> list[float]:
    """
    One LSTM timestep:
      [potassium_mg, phosphorus_mg, protein_per_kg, sodium_mg, occasion_encoded]
    """
    weight = body_weight_kg if body_weight_kg and body_weight_kg > 0 else 70.0
    return [
        float(totals.get("potassium_mg") or 0.0),
        float(totals.get("phosphorus_mg") or 0.0),
        float(totals.get("protein_g") or 0.0) / weight,
        float(totals.get("sodium_mg") or 0.0),
        occasion_encoding(meal_occasion),
    ]


def _fetch_logs_for_lookback(
    db: Session,
    user_id: str,
    *,
    lookback_days: int,
    as_of: date,
) -> list[FoodLog]:
    start_date = as_of - timedelta(days=lookback_days - 1)
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(as_of, time.max)
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


def _occasion_for_meal_group(group: list[FoodLog]) -> str | None:
    """Prefer an explicit meal_occasion on any log in the group."""
    for log in group:
        occ = (log.meal_occasion or "").strip()
        if occ and not occ.startswith("__ungrouped_"):
            return occ
    return None


def build_lstm_sequence(
    db: Session,
    user_id: str,
    body_weight_kg: float,
    *,
    lookback_days: int = LSTM_LOOKBACK_DAYS,
    max_meals: int = LSTM_MAX_MEALS,
    as_of: date | None = None,
) -> list[list[float]]:
    """
    Canonical LSTM input for all product call sites.

    1. Fetch FoodLogs in [as_of - (lookback_days-1), as_of]
    2. Group by calendar day (ascending)
    3. Within each day: group_logs_into_meals (existing order)
    4. For each meal group: sum nutrients → meal_vector_5 (real occasion)
    5. Return last max_meals vectors (chronological)

    Returns list of length 0..max_meals; each step is length 5.
    Padding to 6 timesteps remains inside LSTMPatternAnalyzer.analyze().
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    if max_meals < 1:
        raise ValueError("max_meals must be >= 1")

    end = as_of or datetime.utcnow().date()
    logs = _fetch_logs_for_lookback(
        db, user_id, lookback_days=lookback_days, as_of=end
    )
    if not logs:
        return []

    grouped: dict[date, list[FoodLog]] = defaultdict(list)
    for log in logs:
        if log.logged_at is None:
            continue
        grouped[log.logged_at.date()].append(log)

    meal_sequence: list[list[float]] = []
    for day in sorted(grouped.keys()):
        for group in group_logs_into_meals(grouped[day]):
            totals = sum_meal_nutrients(group)
            occasion = _occasion_for_meal_group(group)
            meal_sequence.append(
                meal_vector_5(totals, body_weight_kg, occasion)
            )

    if len(meal_sequence) > max_meals:
        return meal_sequence[-max_meals:]
    return meal_sequence


def analyze_patient_lstm(
    db: Session,
    user_id: str,
    body_weight_kg: float,
    *,
    lookback_days: int = LSTM_LOOKBACK_DAYS,
    max_meals: int = LSTM_MAX_MEALS,
    as_of: date | None = None,
) -> dict | None:
    """
    Build the canonical sequence and run LSTM analyze().
    Returns None if there are no meals in the window.
    """
    from backend.models.lstm_model import get_analyzer

    sequence = build_lstm_sequence(
        db,
        user_id,
        body_weight_kg,
        lookback_days=lookback_days,
        max_meals=max_meals,
        as_of=as_of,
    )
    if not sequence:
        return None
    return get_analyzer().analyze(sequence)
