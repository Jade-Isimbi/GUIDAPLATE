"""
daily_budget.py
GuidaPlate — Daily nutrient budget tracker and balanced food suggestions
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import get_current_user_id
from backend.clinical_constants import KDOQI_DAILY_LIMITS
from backend.database.db import Food, FoodLog, Patient, get_db
from backend.models.recommender import get_recommender
from backend.utils.meal_aggregation import (
    energy_for_food_log,
    fetch_food_logs_for_date,
    group_logs_into_meals,
    sum_meal_nutrients,
)

router = APIRouter(tags=["Daily Budget"])

SUPPORTED_STAGES = set(KDOQI_DAILY_LIMITS)
REFERENCE_PORTION_G = 100.0
NUTRIENT_KEYS = ("potassium", "phosphorus", "protein_per_kg", "sodium")

EXCLUDED_CATEGORIES = {"Fat/Oil", "Other"}

FALLBACK_EXCLUDED_CATEGORIES = {
    "Fat/Oil",
    "Other",
    "Beverage",
    "Condiment",
    "Sugar/Sweetener",
    "Spice/Herb",
}

FALLBACK_SLOT_MIN_ENERGY_KCAL: dict[str, float] = {
    "Vegetable": 20.0,
    "Starch": 30.0,
    "Protein": 30.0,
}
FALLBACK_DEFAULT_MIN_ENERGY_KCAL = 30.0

# Budget nutrient key → food_database.csv column + per-100g cap when that nutrient is exceeded.
NUTRIENT_FOOD_COLUMN: dict[str, str] = {
    "potassium": "potassium_mg",
    "phosphorus": "phosphorus_mg",
    "protein_per_kg": "protein_g",
    "sodium": "sodium_mg",
}

FALLBACK_TIER_CAPS: dict[int, dict[str, float]] = {
    1: {
        "potassium": 100.0,
        "phosphorus": 50.0,
        "protein_per_kg": 1.5,
        "sodium": 80.0,
    },
    2: {
        "potassium": 200.0,
        "phosphorus": 100.0,
        "protein_per_kg": 3.0,
        "sodium": 150.0,
    },
}

# CSV categories: Starch, Grain, Vegetable, Meat, Fish, Egg, Legume (+ others excluded from fallback)
PLATE_ROLE_CATEGORIES: dict[str, list[str]] = {
    "Starch": ["Starch", "Grain"],
    "Vegetable": ["Vegetable"],
    "Protein": ["Meat", "Fish", "Egg", "Legume"],
}

PLATE_ROLE_MAP: dict[str, str] = {
    "Starch/Grain": "Starch",
    "Bread": "Starch",
    "Root Vegetable": "Starch",
    "Starch": "Starch",
    "Grain": "Starch",
    "Vegetable": "Vegetable",
    "Fruit": "Vegetable",
    "Meat": "Protein",
    "Fish": "Protein",
    "Egg": "Protein",
    "Legume": "Protein",
    "Dairy": "Protein",
}


def plate_role_for_food(food: dict) -> str:
    return PLATE_ROLE_MAP.get(str(food.get("category") or "Other"), "Side")

MEAL_ORDER = ["Breakfast", "Lunch", "Dinner", "Snack"]

NEXT_MEAL_OCCASION: dict[str, str | None] = {
    "Breakfast": "Lunch",
    "Lunch": "Dinner",
    "Dinner": "Snack",
    "Snack": None,
}

# Category strings match food_database.csv (no Bread/Cereal/Poultry columns).
MEAL_APPROPRIATE_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "Breakfast": {
        "Starch": ["Grain"],
        "Vegetable": ["Vegetable"],
        "Protein": ["Egg", "Dairy"],
    },
    "Lunch": {
        "Starch": ["Starch", "Grain"],
        "Vegetable": ["Vegetable"],
        "Protein": ["Meat", "Fish", "Egg", "Legume"],
    },
    "Dinner": {
        "Starch": ["Starch", "Grain"],
        "Vegetable": ["Vegetable"],
        "Protein": ["Meat", "Fish", "Egg", "Legume"],
    },
    "Snack": {
        "Starch": ["Fruit", "Grain"],
        "Vegetable": ["Vegetable", "Fruit"],
        "Protein": ["Egg", "Dairy", "Legume"],
    },
}

# Condiment-like vegetables — exact english name match (case-insensitive), Vegetable slot only.
VEGETABLE_EXCLUDE: set[str] = {
    "onion",
    "onions",
    "garlic",
    "ginger",
    "spring onion",
    "green onion",
    "scallion",
    "shallot",
    "leek",
    "chili",
    "chilli",
    "pepper",
}

PROTEIN_SLOT_TIER2_PROTEIN_CAP_G = 11.0

FALLBACK_BASE_WEIGHTS: dict[str, float] = {
    "potassium_mg": 1.0,
    "phosphorus_mg": 1.0,
    "protein_g": 4.0,
    "sodium_mg": 1.0,
}

FALLBACK_EXCEEDED_MULTIPLIER = 5.0


def has_meaningful_nutrients(food: dict) -> bool:
    """Exclude oils and near-zero foods that trivially pass the safety filter."""
    return any([
        float(food.get("potassium_mg") or 0) > 0,
        float(food.get("phosphorus_mg") or 0) > 0,
        float(food.get("protein_g") or 0) > 0,
        float(food.get("sodium_mg") or 0) > 0,
    ])

# Legacy patient profiles may store numeric stage labels from early signup flows.
CKD_STAGE_ALIASES: dict[str, str] = {
    "1": "G2",
    "2": "G2",
    "3": "G3a",
    "4": "G4",
    "5": "G4",
    "G3": "G3a",
}


def normalize_ckd_stage(stage: str) -> str:
    cleaned = stage.strip()
    return CKD_STAGE_ALIASES.get(cleaned, cleaned)


class NutrientBudget(BaseModel):
    consumed: float
    limit: float
    percent_used: float
    remaining: float


class BalancedSuggestion(BaseModel):
    food_id: int
    english: str
    french: str
    kinyarwanda: str
    category: str
    potassium_mg: float
    phosphorus_mg: float
    protein_g: float
    sodium_mg: float
    energy_kcal: float
    ckd_stage_safe: str
    variety_score: float
    energy_rank_score: float
    fallback: bool = False
    tier: int | None = None
    plate_role: str | None = None
    option_index: int = 1


class SuggestionContext(BaseModel):
    constraint_level: str
    message: str


class DailyBudgetResponse(BaseModel):
    ckd_stage: str
    nutrients: dict[str, NutrientBudget]
    warning_level: str
    meals_logged_today: int
    energy_kcal_today: float
    categories_logged_today: dict[str, int]
    suggestion_context: SuggestionContext
    balanced_suggestions: list[BalancedSuggestion]
    next_meal_occasion: str | None = None
    date: str


def _parse_target_date(raw: str | None) -> date:
    if raw is None:
        return datetime.utcnow().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc


def _nutrient_budget(consumed: float, limit: float) -> NutrientBudget:
    if limit <= 0:
        percent = 0.0
    else:
        percent = round(consumed / limit * 100.0, 1)
    remaining = round(limit - consumed, 4 if limit < 10 else 1)
    consumed_round = round(consumed, 4 if limit < 10 else 1)
    return NutrientBudget(
        consumed=consumed_round,
        limit=limit,
        percent_used=percent,
        remaining=remaining,
    )


def _warning_level(nutrients: dict[str, NutrientBudget]) -> str:
    max_pct = max(n.percent_used for n in nutrients.values())
    if max_pct >= 90.0:
        return "HIGH"
    if max_pct >= 70.0:
        return "MODERATE"
    return "LOW"


def _count_categories(logs: list[FoodLog]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for log in logs:
        cat = (log.category or "").strip()
        if cat:
            counts[cat] += 1
    return dict(counts)


def _determine_next_meal_occasion(logs: list[FoodLog]) -> str | None:
    """Infer the next meal occasion from today's most recently logged meal."""
    if not logs:
        return "Breakfast"

    dated_logs = [log for log in logs if log.logged_at is not None]
    if not dated_logs:
        return "Breakfast"

    last_log = max(dated_logs, key=lambda log: log.logged_at)
    last_meal = (last_log.meal_occasion or "").strip()
    if not last_meal:
        return "Lunch"

    return NEXT_MEAL_OCCASION.get(last_meal, "Lunch")


def _food_contributions_at_portion(food: dict, portion_g: float, weight_kg: float) -> dict[str, float]:
    scale = portion_g / 100.0
    protein_g = float(food["protein_g"]) * scale
    weight = weight_kg if weight_kg > 0 else 70.0
    return {
        "potassium": float(food["potassium_mg"]) * scale,
        "phosphorus": float(food["phosphorus_mg"]) * scale,
        "protein_per_kg": protein_g / weight,
        "sodium": float(food["sodium_mg"]) * scale,
    }


def _passes_safety_filter(
    contributions: dict[str, float],
    remaining_headroom: dict[str, float],
) -> bool:
    for key in NUTRIENT_KEYS:
        contrib = contributions.get(key, 0.0)
        if contrib <= 0:
            continue
        headroom = remaining_headroom.get(key, 0.0)
        if headroom <= 0 or contrib > headroom:
            return False
    return True


def energy_adequacy_score(energy_kcal: float, target_min: float = 100, target_max: float = 400) -> float:
    """
    Score how well a food's energy density fits a reasonable meal-contribution range.
    Foods within the target range score highest. Foods far below (nutritionally thin)
    or far above (excessively calorie-dense, like pure oils/fats) score lower.
    """
    if target_min <= energy_kcal <= target_max:
        return 1.0
    if energy_kcal < target_min:
        return max(0.0, energy_kcal / target_min)
    excess_ratio = (energy_kcal - target_max) / target_max
    return max(0.0, 1.0 - min(excess_ratio, 1.0))


def classify_suggestion_context(safety_passing_count: int, total_foods: int) -> dict:
    """
    Determine how constrained the suggestion pool is, and return
    an appropriate message to accompany the suggestions.
    """
    passing_ratio = safety_passing_count / total_foods if total_foods > 0 else 0

    if safety_passing_count == 0:
        return {
            "constraint_level": "NONE_SAFE",
            "message": (
                "You're over budget on several nutrients. The options below "
                "have the lowest nutrient impact — very small portions are "
                "recommended. Always consult your care provider."
            ),
        }
    if passing_ratio < 0.05:
        return {
            "constraint_level": "SEVERELY_LIMITED",
            "message": (
                f"Very limited safe options remain today ({safety_passing_count} foods fit your "
                "remaining budget). Consider a smaller portion than usual, or wait until tomorrow "
                "for more variety."
            ),
        }
    if passing_ratio < 0.20:
        return {
            "constraint_level": "LIMITED",
            "message": (
                "Your options are narrowing for today. The suggestions below fit your remaining "
                "budget, but consider smaller portions to leave room for your next meal."
            ),
        }
    return {
        "constraint_level": "NORMAL",
        "message": "Here are some balanced options that fit your remaining budget today.",
    }


def _passes_fallback_meal_guard(food: dict, slot: str = "Starch") -> bool:
    """Fallback pool only: require caloric foods, not beverages or near-water items."""
    energy = float(food.get("energy_kcal") or 0)
    protein = float(food.get("protein_g") or 0)
    min_energy = FALLBACK_SLOT_MIN_ENERGY_KCAL.get(slot, FALLBACK_DEFAULT_MIN_ENERGY_KCAL)
    if slot == "Vegetable":
        return energy >= min_energy or protein >= 1.0
    if energy < min_energy:
        return False
    # energy_kcal is per 100g in food_database.csv; proxy for substantive foods
    # when carb/fat columns are unavailable: protein >= 1 g or energy >= 50 kcal.
    return protein >= 1.0 or energy >= 50.0


def _passes_fallback_exceeded_caps(
    food: dict,
    exceeded_nutrients: list[str],
    tier: int = 1,
    plate_role: str | None = None,
) -> bool:
    """Per-100g caps for nutrients already over the patient's daily limit."""
    caps = dict(FALLBACK_TIER_CAPS.get(tier, FALLBACK_TIER_CAPS[1]))
    if (
        tier == 2
        and plate_role == "Protein"
        and "protein_per_kg" in exceeded_nutrients
    ):
        caps["protein_per_kg"] = PROTEIN_SLOT_TIER2_PROTEIN_CAP_G
    for nutrient in exceeded_nutrients:
        column = NUTRIENT_FOOD_COLUMN.get(nutrient)
        cap = caps.get(nutrient)
        if column is None or cap is None:
            continue
        if float(food.get(column) or 0) > cap:
            return False
    return True


def _is_vegetable_slot_excluded(food: dict) -> bool:
    name = str(food.get("english") or "").strip().lower()
    return name in VEGETABLE_EXCLUDE


def _fallback_weighted_score(food: dict, exceeded_nutrients: list[str]) -> float:
    exceeded_columns = {
        NUTRIENT_FOOD_COLUMN[n]
        for n in exceeded_nutrients
        if n in NUTRIENT_FOOD_COLUMN
    }
    score = 0.0
    for column, base_weight in FALLBACK_BASE_WEIGHTS.items():
        weight = base_weight
        if column in exceeded_columns:
            weight *= FALLBACK_EXCEEDED_MULTIPLIER
        score += float(food.get(column) or 0) * weight
    return score


def _fallback_base_candidates(
    food_database: list[dict],
    stage_number: int | None,
    recommender,
) -> list[dict]:
    """Stage-safe fallback pool: category, meal, and CKD filters — no nutrient caps."""
    candidates: list[dict] = []
    for food in food_database:
        category = str(food.get("category") or "")
        if category in FALLBACK_EXCLUDED_CATEGORIES or not has_meaningful_nutrients(food):
            continue
        if stage_number is not None:
            safe = food.get("ckd_stage_safe", "")
            if not recommender._parse_stage_safe(str(safe), stage_number):
                continue
        candidates.append(food)
    return candidates


def _food_to_fallback_dict(food: dict, tier: int, plate_role: str | None) -> dict:
    return {
        **food,
        "variety_score": 0.0,
        "energy_rank_score": 0.0,
        "fallback": True,
        "tier": tier,
        "plate_role": plate_role,
    }


def _pick_plate_slot(
    candidates: list[dict],
    role_categories: list[str],
    exceeded_nutrients: list[str],
    plate_role: str,
    next_meal_occasion: str | None = None,
) -> list[dict]:
    """Top 2 lowest weighted-score foods for one plate role, trying Tier 1 then Tier 2 caps."""
    if (
        next_meal_occasion
        and next_meal_occasion in MEAL_APPROPRIATE_CATEGORIES
    ):
        allowed_categories = MEAL_APPROPRIATE_CATEGORIES[next_meal_occasion][plate_role]
    else:
        allowed_categories = role_categories

    pool = [f for f in candidates if str(f.get("category") or "") in allowed_categories]
    if not pool:
        pool = [f for f in candidates if str(f.get("category") or "") in role_categories]

    if plate_role == "Vegetable":
        pool = [f for f in pool if not _is_vegetable_slot_excluded(f)]
    pool = [f for f in pool if _passes_fallback_meal_guard(f, plate_role)]
    for tier in (1, 2):
        eligible = [
            f for f in pool
            if _passes_fallback_exceeded_caps(f, exceeded_nutrients, tier, plate_role)
        ]
        if eligible:
            ranked = sorted(eligible, key=lambda f: _fallback_weighted_score(f, exceeded_nutrients))
            return [_food_to_fallback_dict(f, tier, plate_role) for f in ranked[:2]]
    return []


def _fallback_flat_last_resort(
    candidates: list[dict],
    exceeded_nutrients: list[str],
    top_n: int = 3,
) -> list[dict]:
    """Flat lowest-score ranking across any category when plate slots all fail."""
    guarded = [f for f in candidates if _passes_fallback_meal_guard(f)]
    for tier in (1, 2):
        eligible = [
            f for f in guarded
            if _passes_fallback_exceeded_caps(f, exceeded_nutrients, tier)
        ]
        if eligible:
            ranked = sorted(eligible, key=lambda f: _fallback_weighted_score(f, exceeded_nutrients))
            return [
                _food_to_fallback_dict(
                    f,
                    tier,
                    plate_role_for_food(f),
                )
                for f in ranked[:top_n]
            ]
    ranked = sorted(guarded, key=lambda f: _fallback_weighted_score(f, exceeded_nutrients))
    if not ranked:
        return []
    return [
        _food_to_fallback_dict(f, 2, plate_role_for_food(f))
        for f in ranked[:top_n]
    ]


def _fallback_lowest_impact_foods(
    food_database: list[dict],
    stage_number: int | None,
    recommender,
    exceeded_nutrients: list[str],
    top_n: int = 3,
    next_meal_occasion: str | None = None,
) -> list[dict]:
    """Balanced-plate fallback: up to two foods per starch / vegetable / protein role."""
    base = _fallback_base_candidates(food_database, stage_number, recommender)
    starch_options = _pick_plate_slot(
        base, PLATE_ROLE_CATEGORIES["Starch"], exceeded_nutrients, "Starch", next_meal_occasion
    )
    vegetable_options = _pick_plate_slot(
        base, PLATE_ROLE_CATEGORIES["Vegetable"], exceeded_nutrients, "Vegetable", next_meal_occasion
    )
    protein_options = _pick_plate_slot(
        base, PLATE_ROLE_CATEGORIES["Protein"], exceeded_nutrients, "Protein", next_meal_occasion
    )

    result: list[dict] = []
    used_ids: set[object] = set()
    for role, options in [
        ("Starch", starch_options),
        ("Vegetable", vegetable_options),
        ("Protein", protein_options),
    ]:
        for i, food in enumerate(options):
            food_id = food.get("food_id")
            if food_id in used_ids:
                continue
            entry = {
                **food,
                "plate_role": role,
                "fallback": True,
                "option_index": i + 1,
            }
            result.append(entry)
            used_ids.add(food_id)

    if result:
        return result
    return _fallback_flat_last_resort(base, exceeded_nutrients, top_n=top_n)


def suggest_balanced_foods(
    remaining_headroom: dict[str, float],
    categories_logged_today: dict[str, int],
    food_database: list[dict],
    top_n: int = 5,
    weight_kg: float = 70.0,
    ckd_stage: str | None = None,
    exceeded_nutrients: list[str] | None = None,
    next_meal_occasion: str | None = None,
) -> dict:
    """
    Three-stage filtering and ranking for CKD-safe, varied, energy-adequate suggestions.

    Returns suggestions (top_n) and safety_passing_count (Stage 1 passers before truncation).
    """
    if not food_database:
        return {"suggestions": [], "safety_passing_count": 0}

    if exceeded_nutrients is None:
        exceeded_nutrients = []

    recommender = get_recommender()
    stage_number = None
    if ckd_stage:
        try:
            stage_number = recommender._stage_to_number(ckd_stage)
        except ValueError:
            stage_number = None

    safe_foods: list[dict] = []
    for food in food_database:
        category = str(food.get("category") or "")
        if category in EXCLUDED_CATEGORIES or not has_meaningful_nutrients(food):
            continue

        if stage_number is not None:
            safe = food.get("ckd_stage_safe", "")
            if not recommender._parse_stage_safe(str(safe), stage_number):
                continue

        contrib = _food_contributions_at_portion(food, REFERENCE_PORTION_G, weight_kg)
        if not _passes_safety_filter(contrib, remaining_headroom):
            continue

        cat_count = categories_logged_today.get(category, 0)
        variety_score = float(-cat_count)

        energy = float(food.get("energy_kcal") or 0)
        energy_rank_score = energy_adequacy_score(energy)

        safe_foods.append({
            **food,
            "variety_score": variety_score,
            "energy_rank_score": energy_rank_score,
            "fallback": False,
            "plate_role": plate_role_for_food(food),
        })

    safe_foods.sort(
        key=lambda f: (f["variety_score"], f["energy_rank_score"]),
        reverse=True,
    )
    suggestions = _pick_diverse(safe_foods, top_n)
    if not suggestions:
        suggestions = _fallback_lowest_impact_foods(
            food_database,
            stage_number,
            recommender,
            exceeded_nutrients,
            top_n=3,
            next_meal_occasion=next_meal_occasion,
        )
    else:
        for item in suggestions:
            item["fallback"] = False

    return {
        "suggestions": suggestions,
        "safety_passing_count": len(safe_foods),
    }


def _pick_diverse(ranked: list[dict], top_n: int) -> list[dict]:
    """Prefer one suggestion per category before repeating food groups."""
    selected: list[dict] = []
    seen_categories: set[str] = set()
    for food in ranked:
        cat = str(food.get("category") or "")
        if cat in seen_categories:
            continue
        selected.append(food)
        seen_categories.add(cat)
        if len(selected) >= top_n:
            return selected
    for food in ranked:
        if food in selected:
            continue
        selected.append(food)
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def compute_daily_budget(
    logs: list[FoodLog],
    ckd_stage: str,
    weight_kg: float,
    db: Session,
    food_database: list[dict] | None = None,
) -> dict:
    if ckd_stage not in KDOQI_DAILY_LIMITS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ckd_stage {ckd_stage!r}. Must be one of: {', '.join(sorted(SUPPORTED_STAGES))}",
        )

    limits = KDOQI_DAILY_LIMITS[ckd_stage]
    weight = weight_kg if weight_kg and weight_kg > 0 else 70.0

    totals = sum_meal_nutrients(logs)
    energy_kcal = sum(energy_for_food_log(log) for log in logs)
    categories = _count_categories(logs)
    meals_logged = len(group_logs_into_meals(logs))

    consumed = {
        "potassium": totals["potassium_mg"],
        "phosphorus": totals["phosphorus_mg"],
        "protein_per_kg": totals["protein_g"] / weight,
        "sodium": totals["sodium_mg"],
    }

    nutrients: dict[str, NutrientBudget] = {}
    remaining_headroom: dict[str, float] = {}
    for key in NUTRIENT_KEYS:
        limit = limits[key]
        nutrients[key] = _nutrient_budget(consumed[key], limit)
        remaining_headroom[key] = nutrients[key].remaining

    if food_database is None:
        food_database = get_recommender().get_all_foods(db=db, stage=ckd_stage)

    total_foods = db.query(Food).count()

    exceeded_nutrients = [
        key for key in NUTRIENT_KEYS if nutrients[key].percent_used >= 100.0
    ]

    next_meal_occasion = _determine_next_meal_occasion(logs)

    suggestion_result = suggest_balanced_foods(
        remaining_headroom=remaining_headroom,
        categories_logged_today=categories,
        food_database=food_database,
        top_n=5,
        weight_kg=weight,
        ckd_stage=ckd_stage,
        exceeded_nutrients=exceeded_nutrients,
        next_meal_occasion=next_meal_occasion,
    )
    suggestions_raw = suggestion_result["suggestions"]
    suggestion_context = classify_suggestion_context(
        suggestion_result["safety_passing_count"],
        total_foods,
    )

    suggestions = [
        BalancedSuggestion(
            food_id=int(s["food_id"]) if s.get("food_id") is not None else 0,
            english=str(s.get("english") or ""),
            french=str(s.get("french") or ""),
            kinyarwanda=str(s.get("kinyarwanda") or ""),
            category=str(s.get("category") or ""),
            potassium_mg=float(s.get("potassium_mg") or 0),
            phosphorus_mg=float(s.get("phosphorus_mg") or 0),
            protein_g=float(s.get("protein_g") or 0),
            sodium_mg=float(s.get("sodium_mg") or 0),
            energy_kcal=float(s.get("energy_kcal") or 0),
            ckd_stage_safe=str(s.get("ckd_stage_safe") or ""),
            variety_score=float(s["variety_score"]),
            energy_rank_score=float(s["energy_rank_score"]),
            fallback=bool(s.get("fallback", False)),
            tier=s.get("tier"),
            plate_role=s.get("plate_role"),
            option_index=int(s.get("option_index") or 1),
        )
        for s in suggestions_raw
    ]

    return {
        "ckd_stage": ckd_stage,
        "nutrients": nutrients,
        "warning_level": _warning_level(nutrients),
        "meals_logged_today": meals_logged,
        "energy_kcal_today": round(energy_kcal, 1),
        "categories_logged_today": categories,
        "suggestion_context": SuggestionContext(**suggestion_context),
        "balanced_suggestions": suggestions,
        "next_meal_occasion": next_meal_occasion,
    }


@router.get("/patient/daily-budget", response_model=DailyBudgetResponse)
def get_daily_budget(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    log_date: str | None = Query(
        default=None,
        alias="date",
        description="Calendar date (YYYY-MM-DD, UTC logged_at). Defaults to today.",
    ),
) -> DailyBudgetResponse:
    target = _parse_target_date(log_date)
    logs = fetch_food_logs_for_date(db, user_id, target)

    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient or not patient.ckd_stage:
        raise HTTPException(status_code=400, detail="Patient profile or CKD stage not set.")

    ckd_stage = normalize_ckd_stage(patient.ckd_stage)
    weight_kg = float(patient.body_weight_kg or 70.0)
    result = compute_daily_budget(logs, ckd_stage, weight_kg, db)
    return DailyBudgetResponse(date=target.isoformat(), **result)
