"""
what_to_eat_next.py
GuidaPlate — remaining-budget + LSTM-aware next-meal suggestions
for the Meal Check "What to eat next" panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.daily_budget import (
    MEAL_ORDER,
    _determine_next_meal_occasion,
    compute_daily_budget,
    normalize_ckd_stage,
)
from backend.auth.security import get_current_user_id
from backend.clinical_constants import KDOQI_DAILY_LIMITS
from backend.database.db import FoodLog, Patient, User, get_db
from backend.models.lstm_model import get_analyzer
from backend.utils.lstm_sequence import build_lstm_sequence
from backend.utils.meal_aggregation import fetch_food_logs_for_date

logger = logging.getLogger(__name__)
router = APIRouter(tags=["What to Eat Next"])

ESCALATING_SAFETY_MARGIN = 0.85


@dataclass
class NextMealBudget:
    occasion: str
    meals_remaining_today: int
    remaining_today: dict[str, float]
    meal_budget: dict[str, float]
    trend: str | None
    trajectory_risk: str | None
    safety_margin_applied: bool
    safety_margin: float
    budget_exhausted: bool = False


class NutrientAmounts(BaseModel):
    potassium_mg: float
    phosphorus_mg: float
    protein_g: float
    sodium_mg: float


class MealFoodItem(BaseModel):
    english: str
    kinyarwanda: str | None = None
    category: str | None = None
    portion_grams: float
    potassium_mg: float
    phosphorus_mg: float
    protein_g: float
    sodium_mg: float
    preparation_method: str | None = None


class WhatToEatNextResponse(BaseModel):
    occasion: str
    meals_remaining_today: int
    meal_budget: NutrientAmounts
    remaining_today: NutrientAmounts
    trend: str | None
    trajectory_risk: str | None
    safety_margin_applied: bool
    safety_margin: float
    meal_options: list[list[MealFoodItem]]
    reason: str
    budget_exhausted: bool = False


EPS_MG = 1.0
EPS_PRO_G = 0.1
BUDGET_EXHAUSTED_REASON = (
    "You've reached or exceeded today's nutrient limits. "
    "Here's the lowest-risk option for your next meal."
)

NUTRIENT_CAP_KEYS = (
    ("potassium_mg", 0),
    ("phosphorus_mg", 1),
    ("protein_g", 2),
    ("sodium_mg", 3),
)


def _logged_occasions_today(logs: list[FoodLog]) -> set[str]:
    found: set[str] = set()
    for log in logs:
        occ = (log.meal_occasion or "").strip()
        if occ in MEAL_ORDER:
            found.add(occ)
    return found


def _relevant_occasions(logs: list[FoodLog], requested: str) -> list[str]:
    """Unlogged occasions plus the requested one (always). Preserve MEAL_ORDER."""
    logged = _logged_occasions_today(logs)
    relevant = [o for o in MEAL_ORDER if o not in logged or o == requested]
    if requested not in relevant:
        relevant.append(requested)
    return relevant


def _weighted_meal_budget(
    rem: dict[str, float],
    requested: str,
    relevant: list[str],
) -> dict[str, float]:
    from backend.api.meal_planner import OCCASION_RULES

    out: dict[str, float] = {}
    for key, idx in NUTRIENT_CAP_KEYS:
        w_req = float(OCCASION_RULES[requested]["nutrient_caps"][idx])
        w_sum = sum(
            float(OCCASION_RULES[o]["nutrient_caps"][idx]) for o in relevant
        )
        if w_sum <= 0:
            out[key] = 0.0
        else:
            out[key] = float(rem[key]) * (w_req / w_sum)
    return out


def compute_next_meal_budget(
    *,
    logs: list[FoodLog],
    ckd_stage: str,
    weight_kg: float,
    db: Session,
    meal_sequence: list[list[float]],
    occasion: str | None = None,
) -> NextMealBudget:
    """Remaining-budget meal caps: weighted share among still-relevant occasions."""
    from backend.api.meal_planner import OCCASION_RULES

    weight = weight_kg if weight_kg and weight_kg > 0 else 70.0
    daily = compute_daily_budget(logs, ckd_stage, weight, db)

    rem = {
        "potassium_mg": max(0.0, float(daily["nutrients"]["potassium"].remaining)),
        "phosphorus_mg": max(0.0, float(daily["nutrients"]["phosphorus"].remaining)),
        "protein_g": max(
            0.0, float(daily["nutrients"]["protein_per_kg"].remaining) * weight
        ),
        "sodium_mg": max(0.0, float(daily["nutrients"]["sodium"].remaining)),
    }

    if occasion is not None:
        resolved = occasion
    else:
        display = _determine_next_meal_occasion(logs)
        resolved = display if display is not None else "Snack"

    relevant = _relevant_occasions(logs, resolved)
    meals_remaining_today = len(relevant)
    per_meal = _weighted_meal_budget(rem, resolved, relevant)

    # Snack absolute ceil vs full daily (defense-in-depth; after weighted share)
    if resolved == "Snack":
        k_f, p_f, pro_f, na_f = OCCASION_RULES["Snack"]["nutrient_caps"]
        full = KDOQI_DAILY_LIMITS[ckd_stage]
        snack_ceil = {
            "potassium_mg": float(full["potassium"]) * k_f,
            "phosphorus_mg": float(full["phosphorus"]) * p_f,
            "protein_g": float(full["protein_per_kg"]) * weight * pro_f,
            "sodium_mg": float(full["sodium"]) * na_f,
        }
        per_meal = {k: min(per_meal[k], snack_ceil[k]) for k in per_meal}

    trend: str | None = None
    trajectory_risk: str | None = None
    safety_margin = 1.0
    safety_margin_applied = False

    if meal_sequence:
        try:
            lstm = get_analyzer().analyze(meal_sequence)
            trend = lstm.get("trend")
            trajectory_risk = lstm.get("risk_label")
            if trend == "escalating":
                safety_margin = ESCALATING_SAFETY_MARGIN
                safety_margin_applied = True
                per_meal = {k: v * safety_margin for k, v in per_meal.items()}
        except Exception as exc:
            logger.warning("LSTM failed in compute_next_meal_budget: %s", exc)

    budget_exhausted = (
        rem["potassium_mg"] <= EPS_MG or rem["phosphorus_mg"] <= EPS_MG
    )

    return NextMealBudget(
        occasion=resolved,
        meals_remaining_today=meals_remaining_today,
        remaining_today=rem,
        meal_budget=per_meal,
        trend=trend,
        trajectory_risk=trajectory_risk,
        safety_margin_applied=safety_margin_applied,
        safety_margin=safety_margin,
        budget_exhausted=budget_exhausted,
    )


def _limits_from_meal_budget(meal_budget: dict[str, float]) -> tuple[dict[str, float], float]:
    """Map absolute meal budget → meal_planner limits + protein_limit_g."""
    limits = {
        "potassium": float(meal_budget["potassium_mg"]),
        "phosphorus": float(meal_budget["phosphorus_mg"]),
        "sodium": float(meal_budget["sodium_mg"]),
        # Unused for caps when absolute_meal_caps=True; kept for dict shape.
        "protein_per_kg": 0.0,
    }
    return limits, float(meal_budget["protein_g"])


def _serialize_food(item: dict[str, Any]) -> MealFoodItem:
    return MealFoodItem(
        english=str(item.get("english") or ""),
        kinyarwanda=str(item.get("kinyarwanda") or "") or None,
        category=str(item.get("category") or "") or None,
        portion_grams=float(item.get("portion_grams") or 0),
        potassium_mg=float(item.get("potassium_mg") or 0),
        phosphorus_mg=float(item.get("phosphorus_mg") or 0),
        protein_g=float(item.get("protein_g") or 0),
        sodium_mg=float(item.get("sodium_mg") or 0),
        preparation_method=str(item.get("preparation_method") or "") or None,
    )


def _build_reason(budget: NextMealBudget) -> str:
    if budget.budget_exhausted:
        return BUDGET_EXHAUSTED_REASON
    base = f"Fits your remaining budget for {budget.occasion}"
    if budget.safety_margin_applied:
        return f"{base} (tightened: recent intake looks up)"
    return base


def _load_rwandan_safe_foods(ckd_stage: str):
    from backend.api.meal_planner import (
        _FORBIDDEN_BY_STAGE,
        _filter_rwandan_with_bypass,
        _forbidden_english_mask,
        _is_stage_safe,
        _load_food_db,
        _stage_num,
    )

    food_db = _load_food_db()
    stage_number = _stage_num(ckd_stage)
    safe = food_db[
        food_db["ckd_stage_safe"].apply(lambda x: _is_stage_safe(str(x), stage_number))
    ]
    forbidden = _FORBIDDEN_BY_STAGE.get(ckd_stage.upper(), [])
    if forbidden:
        safe = safe[~_forbidden_english_mask(safe["english"], forbidden)]
    if "is_rwandan" in safe.columns:
        safe = _filter_rwandan_with_bypass(safe)
    return safe


@router.get("/next-meal/what-to-eat-next", response_model=WhatToEatNextResponse)
def get_what_to_eat_next(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    occasion: str | None = Query(
        default=None,
        description="Optional meal occasion override (Breakfast|Lunch|Dinner|Snack).",
    ),
) -> WhatToEatNextResponse:
    from backend.api.meal_planner import (
        build_lowest_risk_occasion_plate,
        build_meal_plan,
    )

    target = datetime.utcnow().date()
    logs = fetch_food_logs_for_date(db, user_id, target)

    user = db.query(User).filter(User.user_id == user_id).first()
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()

    raw_stage = (
        patient.ckd_stage if patient and patient.ckd_stage else (user.ckd_stage if user else None)
    )
    if not raw_stage:
        raise HTTPException(status_code=400, detail="Patient profile or CKD stage not set.")

    ckd_stage = normalize_ckd_stage(raw_stage)
    if ckd_stage not in KDOQI_DAILY_LIMITS:
        raise HTTPException(status_code=400, detail=f"Unsupported CKD stage: {ckd_stage!r}")

    weight_source = (
        patient.body_weight_kg
        if patient and patient.body_weight_kg is not None
        else (user.weight_kg if user else None)
    )
    weight_kg = float(weight_source or 70.0)

    resolved_occasion: str | None = None
    if occasion is not None:
        occ = occasion.strip()
        if occ not in MEAL_ORDER:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid occasion {occasion!r}. Must be one of: {MEAL_ORDER}.",
            )
        resolved_occasion = occ

    meal_sequence = build_lstm_sequence(db, user_id, weight_kg)
    budget = compute_next_meal_budget(
        logs=logs,
        ckd_stage=ckd_stage,
        weight_kg=weight_kg,
        db=db,
        meal_sequence=meal_sequence,
        occasion=resolved_occasion,
    )

    safe_foods = _load_rwandan_safe_foods(ckd_stage)
    meal_options: list[list[dict[str, Any]]] = []

    if budget.budget_exhausted:
        plan_seed = f"{user_id}:{target.isoformat()}:{budget.occasion}:damage"
        plate = build_lowest_risk_occasion_plate(
            safe_foods,
            budget.occasion,
            seed=plan_seed,
        )
        if plate:
            meal_options = [list(plate)]
    else:
        limits, protein_limit_g = _limits_from_meal_budget(budget.meal_budget)
        plan_seed = f"{user_id}:{target.isoformat()}:{budget.occasion}"
        full_plan = build_meal_plan(
            safe_foods,
            limits,
            protein_limit_g,
            absolute_meal_caps=True,
            seed=plan_seed,
        )
        packed = full_plan.get(budget.occasion) or []
        if packed:
            meal_options = [list(packed)]

    return WhatToEatNextResponse(
        occasion=budget.occasion,
        meals_remaining_today=budget.meals_remaining_today,
        meal_budget=NutrientAmounts(**budget.meal_budget),
        remaining_today=NutrientAmounts(**budget.remaining_today),
        trend=budget.trend,
        trajectory_risk=budget.trajectory_risk,
        safety_margin_applied=budget.safety_margin_applied,
        safety_margin=budget.safety_margin,
        meal_options=[[_serialize_food(f) for f in opt] for opt in meal_options],
        reason=_build_reason(budget),
        budget_exhausted=budget.budget_exhausted,
    )
