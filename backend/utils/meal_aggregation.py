"""
meal_aggregation.py
GuidaPlate — Meal nutrient aggregation (mirrors RiskAssessment.tsx sumMealNutrients)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time

from backend.database.db import Food, FoodLog, SessionLocal
from backend.database.food_queries import find_food_by_name, get_food_nutrients

MEAL_OCCASION_ORDER = ["Breakfast", "Lunch", "Dinner", "Snack"]

# Per-meal exceedance thresholds (notebook 08 transition matrix cell, G2 daily / 3)
G2_MEAL_THRESHOLDS = {
    "potassium": 3500 / 3,
    "phosphorus": 1000 / 3,
    "protein_per_kg": 0.8 / 3,
    "sodium": 2300 / 3,
}

RISK_INDEX_TO_LABEL = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def scaled_nutrients_from_food(food: Food, portion_grams: float) -> dict[str, float]:
    """Same scaling as RiskAssessment.tsx: nutrients per 100g × (grams / 100)."""
    scale = portion_grams / 100.0
    return {
        "potassium_mg": float(food.potassium_mg) * scale,
        "phosphorus_mg": float(food.phosphorus_mg) * scale,
        "protein_g": float(food.protein_g) * scale,
        "sodium_mg": float(food.sodium_mg) * scale,
    }


def nutrients_for_food_name(food_name: str, portion_grams: float) -> dict[str, float]:
    db = SessionLocal()
    try:
        nutrients = get_food_nutrients(food_name, portion_grams, db)
        if nutrients is None:
            raise ValueError(f"Food not found in database: {food_name!r}")
        return {
            "potassium_mg": nutrients["potassium_mg"],
            "phosphorus_mg": nutrients["phosphorus_mg"],
            "protein_g": nutrients["protein_g"],
            "sodium_mg": nutrients["sodium_mg"],
        }
    finally:
        db.close()


def nutrients_for_food_log(log: FoodLog) -> dict[str, float]:
    """Return stored nutrients or compute from foods table lookup."""
    has_stored = any(
        getattr(log, col) not in (None, 0)
        for col in ("potassium_mg", "phosphorus_mg", "protein_g", "sodium_mg")
    )
    if has_stored:
        return {
            "potassium_mg": float(log.potassium_mg or 0),
            "phosphorus_mg": float(log.phosphorus_mg or 0),
            "protein_g": float(log.protein_g or 0),
            "sodium_mg": float(log.sodium_mg or 0),
        }
    grams = float(log.portion_grams or 100.0)
    try:
        return nutrients_for_food_name(log.food_name, grams)
    except ValueError:
        return {"potassium_mg": 0.0, "phosphorus_mg": 0.0, "protein_g": 0.0, "sodium_mg": 0.0}


def sum_meal_nutrients(logs: list[FoodLog]) -> dict[str, float]:
    """Aggregate multiple food-log rows into one meal total (frontend sumMealNutrients)."""
    totals = {"potassium_mg": 0.0, "phosphorus_mg": 0.0, "protein_g": 0.0, "sodium_mg": 0.0}
    for log in logs:
        item = nutrients_for_food_log(log)
        for key in totals:
            totals[key] += item[key]
    return totals


def meal_vector_from_totals(totals: dict[str, float], body_weight_kg: float) -> list[float]:
    weight = body_weight_kg if body_weight_kg and body_weight_kg > 0 else 70.0
    return [
        totals["potassium_mg"],
        totals["phosphorus_mg"],
        totals["protein_g"] / weight,
        totals["sodium_mg"],
    ]


def compute_meal_risk_state(potassium_mg: float, phosphorus_mg: float, protein_per_kg: float, sodium_mg: float) -> str:
    """Notebook 08 meal_risk_state logic → LOW / MODERATE / HIGH."""
    t = G2_MEAL_THRESHOLDS
    exceed_count = sum([
        potassium_mg > t["potassium"],
        phosphorus_mg > t["phosphorus"],
        protein_per_kg > t["protein_per_kg"],
        sodium_mg > t["sodium"],
    ])
    if exceed_count >= 2:
        return "HIGH"
    if exceed_count == 1:
        return "MODERATE"
    return "LOW"


def group_logs_into_meals(logs: list[FoodLog]) -> list[list[FoodLog]]:
    """Group today's food logs into ordered meals by meal_occasion, then chronology."""
    if not logs:
        return []

    groups: dict[str, list[FoodLog]] = defaultdict(list)
    for log in logs:
        occasion = (log.meal_occasion or "").strip()
        if occasion:
            groups[occasion].append(log)
        else:
            groups[f"__ungrouped_{log.log_id}"].append(log)

    def group_sort_key(item: tuple[str, list[FoodLog]]) -> tuple:
        occasion, items = item
        timestamps = [l.logged_at for l in items if l.logged_at]
        earliest = min(timestamps) if timestamps else datetime.min
        if occasion in MEAL_OCCASION_ORDER:
            return (0, MEAL_OCCASION_ORDER.index(occasion), earliest)
        return (1, earliest, occasion)

    ordered = sorted(groups.items(), key=group_sort_key)
    return [items for _, items in ordered]


def day_bounds(target: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time.min)
    end = datetime.combine(target, time.max)
    return start, end


def fetch_food_logs_for_date(db, patient_id: str, target: date) -> list[FoodLog]:
    """Today's (or target date's) food logs — same filter as history + next-meal forecast."""
    start, end = day_bounds(target)
    return (
        db.query(FoodLog)
        .filter(
            FoodLog.patient_id == patient_id,
            FoodLog.logged_at >= start,
            FoodLog.logged_at <= end,
        )
        .order_by(FoodLog.logged_at.asc())
        .all()
    )


def energy_for_food_name(food_name: str, portion_grams: float) -> float:
    db = SessionLocal()
    try:
        food = find_food_by_name(db, food_name)
        if food is None:
            return 0.0
        scale = portion_grams / 100.0
        return float(food.energy_kcal or 0) * scale
    finally:
        db.close()


def energy_for_food_log(log: FoodLog) -> float:
    grams = float(log.portion_grams or 100.0)
    return energy_for_food_name(log.food_name, grams)
