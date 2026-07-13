"""
recommender.py
GuidaPlate — Food recommendation engine backed by the SQLAlchemy foods table
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.clinical_constants import CLINICAL_SEVERITY_WEIGHTS, KDOQI_DAILY_LIMITS, STAGE_NUMERIC
from backend.database.db import Food
from backend.database.food_queries import find_food_by_name

_recommender: FoodRecommender | None = None

# G5 not in STAGE_NUMERIC (app supports G2–G4); keep for food DB range checks.
_FOOD_STAGE_NUMERIC = {**STAGE_NUMERIC, "G5": 5}

EXCLUDE_CATEGORIES: frozenset[str] = frozenset({
    "Fat/Oil",
    "Beverage",
    "Sugar/Sweetener",
    "Condiment",
    "Spice/Herb",
})

# Candidate category must be in the queried food's group.
# Meat/Fish/Egg interchange as protein sources; Dairy only with Dairy.
# Never cross into Fruit/Vegetable/Starch/Grain/Legume (or vice versa).
SUBSTITUTE_CATEGORY_GROUPS: dict[str, frozenset[str]] = {
    "Meat": frozenset({"Meat", "Fish", "Egg"}),
    "Fish": frozenset({"Meat", "Fish", "Egg"}),
    "Egg": frozenset({"Meat", "Fish", "Egg"}),
    "Dairy": frozenset({"Dairy"}),
    "Fruit": frozenset({"Fruit"}),
    "Vegetable": frozenset({"Vegetable"}),
    "Starch": frozenset({"Starch"}),
    "Grain": frozenset({"Grain"}),
    "Legume": frozenset({"Legume"}),
    "Other": frozenset({"Other"}),
}

INAPPROPRIATE_SUBSTITUTE_KEYWORDS: tuple[str, ...] = (
    # Oils and fats
    "oil",
    "butter",
    "lard",
    "shortening",
    "margarine",
    "ghee",
    "fat",
    # Pure sugars and sweeteners
    "sugar",
    "syrup",
    "honey",
    "sweetener",
    "glucose",
    "fructose",
    "sucrose",
    "molasses",
    "candy",
    "confection",
    # Condiments and seasonings
    "salt",
    "sauce",
    "vinegar",
    "mustard",
    "ketchup",
    "mayonnaise",
    "dressing",
    "seasoning",
    "spice",
    "herb",
    # Beverages
    "juice",
    "drink",
    "beverage",
    "soda",
    "cola",
    "coffee",
    "tea",
    "wine",
    "beer",
    "alcohol",
    # Other non-meal items
    "supplement",
    "vitamin",
    "powder",
    "formula",
    "infant",
)

DEFAULT_BODY_WEIGHT_KG = 65.0
SUBSTITUTE_LIMIT = 3

NUTRIENT_COLUMNS: dict[str, str] = {
    "potassium": "potassium_mg",
    "phosphorus": "phosphorus_mg",
    "protein": "protein_g",
    "protein_per_kg": "protein_g",
    "sodium": "sodium_mg",
}

SUBSTITUTE_OUTPUT_KEYS = [
    "english",
    "french",
    "kinyarwanda",
    "category",
    "potassium_mg",
    "phosphorus_mg",
    "protein_g",
    "sodium_mg",
    "ckd_stage_safe",
    "notes",
    "reason",
]

_RWANDAN_FOOD_IDS: set[str] | None = None


def _rwandan_food_ids() -> set[str]:
    """CSV-backed set of food_id values with is_rwandan == 1 (cached)."""
    global _RWANDAN_FOOD_IDS
    if _RWANDAN_FOOD_IDS is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "food_database.csv"
        csv_df = pd.read_csv(csv_path)
        if "is_rwandan" not in csv_df.columns or "food_id" not in csv_df.columns:
            _RWANDAN_FOOD_IDS = set()
        else:
            mask = (
                pd.to_numeric(csv_df["is_rwandan"], errors="coerce")
                .fillna(0)
                .astype(int)
                == 1
            )
            _RWANDAN_FOOD_IDS = set(
                csv_df.loc[mask, "food_id"].astype(str).str.strip()
            )
    return _RWANDAN_FOOD_IDS


def _is_rwandan_candidate(food: Food) -> bool:
    """True if food is flagged is_rwandan==1 or matches meal_planner oats bypass."""
    from backend.api.meal_planner import _english_matches_rwandan_bypass

    if str(food.food_id).strip() in _rwandan_food_ids():
        return True
    return _english_matches_rwandan_bypass(food.english or "")


def _is_appropriate_substitute(food_name: str) -> bool:
    name_lower = str(food_name).lower()
    return not any(kw in name_lower for kw in INAPPROPRIATE_SUBSTITUTE_KEYWORDS)


class FoodRecommender:
    """Food substitute and lookup engine backed by the GuidaPlate foods table."""

    @staticmethod
    def _stage_to_number(ckd_stage: str) -> int:
        if ckd_stage not in _FOOD_STAGE_NUMERIC:
            raise ValueError(f"Unknown CKD stage: {ckd_stage!r}")
        return _FOOD_STAGE_NUMERIC[ckd_stage]

    @staticmethod
    def _parse_stage_safe(ckd_stage_safe: str | None, stage_number: int) -> bool:
        if ckd_stage_safe is None or not str(ckd_stage_safe).strip():
            return False
        text = str(ckd_stage_safe).strip()
        if "-" in text:
            parts = text.split("-", 1)
            low, high = int(parts[0]), int(parts[1])
            return low <= stage_number <= high
        return stage_number == int(text)

    @staticmethod
    def _sanitize_value(value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, (int,)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return value

    @staticmethod
    def _food_to_dict(food: Food) -> dict:
        return {
            "food_id": int(food.food_id) if str(food.food_id).isdigit() else food.food_id,
            "english": food.english,
            "french": food.french or "",
            "kinyarwanda": food.kinyarwanda or "",
            "category": food.category,
            "potassium_mg": food.potassium_mg,
            "phosphorus_mg": food.phosphorus_mg,
            "protein_g": food.protein_g,
            "sodium_mg": food.sodium_mg,
            "energy_kcal": food.energy_kcal or 0.0,
            "ckd_stage_safe": food.stage_safe_range or "",
            "notes": "",
        }

    @staticmethod
    def _nutrient_value(food: Food, nutrient_col: str) -> float:
        return float(getattr(food, nutrient_col))

    def _nutrient_column(self, nutrient: str) -> str | None:
        return NUTRIENT_COLUMNS.get(nutrient.lower())

    @staticmethod
    def _primary_exceeded_nutrient(exceeded_nutrients: list[str]) -> str:
        return max(exceeded_nutrients, key=lambda n: CLINICAL_SEVERITY_WEIGHTS.get(n, 0))

    @staticmethod
    def _threshold_for_nutrient(nutrient: str, limits: dict[str, float]) -> float:
        if nutrient == "potassium":
            return limits["potassium"] * 0.15
        if nutrient == "phosphorus":
            return limits["phosphorus"] * 0.20
        if nutrient == "protein":
            return limits["protein_per_kg"] * 0.25 * DEFAULT_BODY_WEIGHT_KG
        if nutrient == "sodium":
            return limits["sodium"] * 0.10
        raise ValueError(f"Unknown nutrient for substitute threshold: {nutrient!r}")

    def _build_reason(
        self,
        queried: Food | None,
        candidate: Food,
        primary: str,
    ) -> str:
        col = self._nutrient_column(primary)
        if col is None:
            return "Safer alternative for your CKD stage"
        candidate_val = self._nutrient_value(candidate, col)
        label = primary.replace("_", " ")
        unit = "mg" if col.endswith("_mg") else "g"
        if queried is not None:
            queried_val = self._nutrient_value(queried, col)
            return (
                f"Lower {label} ({candidate_val:g}{unit} vs "
                f"{queried_val:g}{unit}) — safer alternative"
            )
        return f"Lower {label} ({candidate_val:g}{unit}) — within safe limit"

    def get_substitutes(
        self,
        food_name: str,
        ckd_stage: str,
        risk_label: str,
        exceeded_nutrients: list[str],
        db: Session,
        limit: int = SUBSTITUTE_LIMIT,
    ) -> list[dict]:
        if risk_label == "LOW":
            return []

        if not exceeded_nutrients:
            return []

        primary = self._primary_exceeded_nutrient(exceeded_nutrients)
        nutrient_col = self._nutrient_column(primary)
        if nutrient_col is None:
            return []

        limits = KDOQI_DAILY_LIMITS.get(ckd_stage, KDOQI_DAILY_LIMITS["G3b"])
        threshold = self._threshold_for_nutrient(primary, limits)
        nutrient_attr = getattr(Food, nutrient_col)
        stage_number = self._stage_to_number(ckd_stage)

        queried = find_food_by_name(db, food_name)
        queried_food_id = queried.food_id if queried is not None else None
        queried_category = queried.category if queried is not None else None
        allowed_categories = SUBSTITUTE_CATEGORY_GROUPS.get(
            queried_category or "", frozenset()
        )
        if not allowed_categories:
            return []

        candidates = (
            db.query(Food)
            .filter(
                nutrient_attr <= threshold,
                ~Food.category.in_(EXCLUDE_CATEGORIES),
            )
            .all()
        )

        candidates = [
            food
            for food in candidates
            if food.category in allowed_categories
            and _is_appropriate_substitute(food.english)
            and (queried_food_id is None or food.food_id != queried_food_id)
            and self._parse_stage_safe(food.stage_safe_range, stage_number)
            and _is_rwandan_candidate(food)
        ]

        if not candidates:
            return []

        candidates.sort(key=lambda food: self._nutrient_value(food, nutrient_col))
        candidates = candidates[:limit]

        results: list[dict] = []
        for food in candidates:
            reason = self._build_reason(queried, food, primary)
            item = {
                key: self._sanitize_value(self._food_to_dict(food).get(key, ""))
                for key in SUBSTITUTE_OUTPUT_KEYS
                if key != "reason"
            }
            item["reason"] = reason
            results.append(item)
        return results

    def get_all_foods(
        self,
        db: Session,
        stage: str | None = None,
        category: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        foods = db.query(Food).all()

        if stage is not None:
            stage_number = self._stage_to_number(stage)
            foods = [
                food
                for food in foods
                if self._parse_stage_safe(food.stage_safe_range, stage_number)
            ]

        if category is not None:
            foods = [food for food in foods if food.category == category]

        if search is not None and search.strip():
            term = search.strip().lower()
            foods = [
                food
                for food in foods
                if term in (food.english or "").lower()
                or term in (food.french or "").lower()
                or term in (food.kinyarwanda or "").lower()
            ]

        return [self._food_to_dict(food) for food in foods]

    def get_food_by_name(self, db: Session, name: str) -> dict | None:
        term = name.strip().lower()
        if not term:
            return None

        for food in db.query(Food).all():
            for col in (food.english, food.french, food.kinyarwanda):
                if col and col.lower() == term:
                    return self._food_to_dict(food)

        for food in db.query(Food).all():
            for col in (food.english, food.french, food.kinyarwanda):
                if col and term in col.lower():
                    return self._food_to_dict(food)

        return None


def get_recommender() -> FoodRecommender:
    global _recommender
    if _recommender is None:
        _recommender = FoodRecommender()
    return _recommender
