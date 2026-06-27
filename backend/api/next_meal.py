"""
next_meal.py
GuidaPlate - Next-meal risk forecasting

Combines two complementary signals:
1. LSTM trajectory classification: feeds meals logged so far today
   (zero-padded to 6 slots) through the trained LSTM to classify
   the overall risk trajectory IF the current pattern continued.
2. Transition matrix forecast: uses empirical meal-to-meal transition
   probabilities (notebook 08) for the NEXT single meal, given the
   most recent meal's risk state (from today's risk_assessments when
   available, otherwise computed from logged nutrients).
"""

from __future__ import annotations

import json
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import get_current_user_id
from backend.config import MODELS_DIR
from backend.database.db import FoodLog, Patient, RiskAssessmentLog, get_db
from backend.models.lstm_model import get_analyzer
from backend.utils.meal_aggregation import (
    compute_meal_risk_state,
    day_bounds,
    fetch_food_logs_for_date,
    group_logs_into_meals,
    meal_vector_from_totals,
    sum_meal_nutrients,
)

router = APIRouter(tags=["Next Meal Forecasting"])

_TRANSITION_PATH = MODELS_DIR / "transition_matrix.json"
if not _TRANSITION_PATH.exists():
    raise FileNotFoundError(
        f"Transition matrix not found at {_TRANSITION_PATH}. "
        "Run notebook 08 transition-matrix cell to generate transition_matrix.json."
    )

with open(_TRANSITION_PATH) as f:
    TRANSITION_MATRIX: dict[str, dict[str, float]] = json.load(f)

RISK_LEVELS = {"LOW", "MODERATE", "HIGH"}
MAX_MEALS = 6


class NextMealForecastRequest(BaseModel):
    date: str | None = Field(
        default=None,
        description="Calendar date for meals (YYYY-MM-DD). Defaults to today (UTC).",
    )
    last_meal_risk_level: str | None = Field(
        default=None,
        description="Optional override for transition-matrix signal. "
        "When omitted, uses the latest risk_assessment today or computed meal risk.",
    )


class NextMealForecastResponse(BaseModel):
    trajectory_risk: str
    trajectory_confidence: float
    next_meal_probabilities: dict[str, float]
    most_likely_next: str
    recommendation_tier: str
    meals_in_sequence: int
    last_meal_risk_level: str
    last_meal_risk_source: str


def _parse_target_date(raw: str | None) -> date:
    if raw is None:
        return datetime.utcnow().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc


def _fetch_today_food_logs(db: Session, user_id: str, target: date) -> list[FoodLog]:
    return fetch_food_logs_for_date(db, user_id, target)


def _latest_risk_assessment_today(
    db: Session, user_id: str, target: date
) -> RiskAssessmentLog | None:
    start, end = day_bounds(target)
    return (
        db.query(RiskAssessmentLog)
        .filter(
            RiskAssessmentLog.patient_id == user_id,
            RiskAssessmentLog.assessed_at >= start,
            RiskAssessmentLog.assessed_at <= end,
        )
        .order_by(RiskAssessmentLog.assessed_at.desc())
        .first()
    )


def _resolve_last_meal_risk(
    request_override: str | None,
    latest_assessment: RiskAssessmentLog | None,
    last_meal_vector: list[float] | None,
) -> tuple[str, str]:
    if request_override is not None:
        if request_override not in RISK_LEVELS:
            raise HTTPException(
                status_code=400,
                detail=f"last_meal_risk_level must be one of {sorted(RISK_LEVELS)}",
            )
        return request_override, "request_override"

    if latest_assessment and latest_assessment.risk_label in RISK_LEVELS:
        return latest_assessment.risk_label, "risk_assessment"

    if last_meal_vector is not None:
        k, p, protein_per_kg, na = last_meal_vector
        computed = compute_meal_risk_state(k, p, protein_per_kg, na)
        return computed, "computed"

    raise HTTPException(
        status_code=400,
        detail="No meals logged today and no last_meal_risk_level provided.",
    )


@router.post("/next-meal/forecast", response_model=NextMealForecastResponse)
def forecast_next_meal(
    request: NextMealForecastRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
) -> NextMealForecastResponse:
    target = _parse_target_date(request.date)

    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    body_weight = patient.body_weight_kg if patient else None

    logs = _fetch_today_food_logs(db, user_id, target)
    meal_groups = group_logs_into_meals(logs)

    meal_sequence: list[list[float]] = []
    for group in meal_groups[:MAX_MEALS]:
        totals = sum_meal_nutrients(group)
        meal_sequence.append(meal_vector_from_totals(totals, body_weight_kg=body_weight or 0))

    if not meal_sequence and request.last_meal_risk_level is None:
        raise HTTPException(
            status_code=400,
            detail="No meals logged for this date. Log meals or provide last_meal_risk_level.",
        )

    last_meal_vector = meal_sequence[-1] if meal_sequence else None
    latest_assessment = _latest_risk_assessment_today(db, user_id, target)
    last_meal_risk, risk_source = _resolve_last_meal_risk(
        request.last_meal_risk_level,
        latest_assessment,
        last_meal_vector,
    )

    # Signal 1: LSTM trajectory — zero-pads remaining slots inside analyze()
    if meal_sequence:
        try:
            lstm_result = get_analyzer().analyze(meal_sequence)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        trajectory_label = lstm_result["risk_label"]
        trajectory_confidence = lstm_result["confidence"]
    else:
        trajectory_label = "LOW"
        trajectory_confidence = 0.0

    # Signal 2: transition matrix P(next state | last meal state)
    next_meal_probs = TRANSITION_MATRIX.get(last_meal_risk, TRANSITION_MATRIX["LOW"])
    most_likely_next = max(next_meal_probs, key=next_meal_probs.get)

    risk_priority = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    worse_signal = max(
        trajectory_label,
        most_likely_next,
        key=lambda x: risk_priority.get(x, 0),
    )

    return NextMealForecastResponse(
        trajectory_risk=trajectory_label,
        trajectory_confidence=trajectory_confidence,
        next_meal_probabilities=next_meal_probs,
        most_likely_next=most_likely_next,
        recommendation_tier=worse_signal,
        meals_in_sequence=len(meal_sequence),
        last_meal_risk_level=last_meal_risk,
        last_meal_risk_source=risk_source,
    )
