from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends
from groq import Groq
from pydantic import BaseModel, Field

from backend.auth.security import get_current_user_id
from backend.clinical_constants import KDOQI_DAILY_LIMITS, STAGE_NUMERIC as _CLINICAL_STAGE_NUMERIC
from backend.database.db import Food, SessionLocal
from backend.rag.retriever import get_retriever

_mp_logger = logging.getLogger(__name__)


router = APIRouter(prefix="/meal-planner", tags=["Meal Planner"])

CLINICAL_SAFETY_RULES = """

STRICT CLINICAL SAFETY RULES —
NEVER VIOLATE THESE:

1. NEVER recommend or mention
   these high-risk foods for
   Stage 3b or Stage 4 patients:
   beans, ibishyimbo, haricots,
   peas, amashaza, lentils,
   soybeans, soya, groundnuts,
   arachides, ubunyobwa,
   avocado, banana, umuneke,
   potato, ibirayi, spinach,
   tomato sauce, orange juice,
   nuts of any kind.

2. NEVER suggest a food that
   has potassium above 300mg
   per 100g serving.

3. NEVER suggest a food that
   has phosphorus above 200mg
   per 100g serving.

4. ALWAYS recommend only foods
   you have seen in the provided
   context that are explicitly
   marked as safe for the
   patient's CKD stage.

5. If you are not certain a
   food is safe, do not mention
   it. Recommend only foods
   from the provided clinical
   context.

6. NEVER contradict the
   structured food table
   in your response. If a food
   appears in the table it was
   verified safe. If it does not
   appear in the table do not
   recommend it in your text.
"""

HIGH_RISK_FOODS = [
    "beans",
    "ibishyimbo",
    "haricots",
    "peas",
    "amashaza",
    "lentils",
    "soybeans",
    "soya",
    "groundnuts",
    "arachides",
    "ubunyobwa",
    "avocado",
    "banana",
    "umuneke",
    "spinach",
    "irish potato",
    "ibirayi",
    "kidney beans",
    "black beans",
]


def get_rwanda_food_context(ckd_stage: str) -> str:
    """
    Returns stage-filtered Rwandan food context. Never lists
    forbidden foods as normal foods for the patient's stage.
    """
    # Foods safe for ALL stages
    always_safe = (
        "rice, sorghum, cassava, "
        "cabbage, carrots, pumpkin, "
        "eggplant, apples, mangoes, "
        "pineapples, watermelon, "
        "papaya, sugarcane, "
        "ikivuguto (small portions), "
        "eggs, tea, bread, oats, "
        "igikoma (maize porridge), "
        "tilapia (small portions), "
        "chicken (small portions)"
    )

    # Additional foods by stage
    stage_extras = {
        "G2": (
            ", sweet potatoes, "
            "banana, avocado, "
            "beans (ibishyimbo), "
            "peas (amashaza), "
            "irish potatoes, "
            "matoke, isombe, "
            "ugali (ubugali), "
            "tomatoes, onions, "
            "beef, goat meat"
        ),
        "G3A": (
            ", sweet potatoes, "
            "ugali (ubugali), "
            "tomatoes, onions, "
            "beef (small portions), "
            "goat meat (small portions)"
        ),
        "G3B": (
            ", ugali (ubugali), "
            "tomatoes (small amounts), "
            "onions"
        ),
        "G4": (
            ", ugali (small portions)"
        ),
    }

    stage_key = ckd_stage.upper()

    extras = stage_extras.get(stage_key, stage_extras["G3B"])

    safe_foods = always_safe + extras

    # Stage-specific forbidden list
    forbidden = {
        "G2": "None — all portions moderate",
        "G3A": (
            "groundnuts, soybeans, "
            "very large portions of "
            "beans or banana"
        ),
        "G3B": (
            "beans (ibishyimbo), "
            "peas (amashaza), "
            "banana, matoke, "
            "avocado, irish potatoes, "
            "ibirayi, isombe, "
            "cassava leaves, "
            "groundnuts, soybeans, "
            "spinach, large tomatoes"
        ),
        "G4": (
            "beans, peas, banana, "
            "matoke, avocado, "
            "irish potatoes, isombe, "
            "groundnuts, soybeans, "
            "spinach, sweet potatoes, "
            "milk (large amounts), "
            "oranges, tomatoes"
        ),
    }

    forbidden_str = forbidden.get(stage_key, forbidden["G3B"])

    return (
        f"Common Rwandan foods safe "
        f"for Stage {ckd_stage} CKD:\n"
        f"{safe_foods}\n\n"
        f"NEVER recommend these for "
        f"Stage {ckd_stage}:\n"
        f"{forbidden_str}\n\n"
        f"Do NOT suggest: almond milk, "
        f"quinoa, kale smoothies, tofu, "
        f"sushi, or non-Rwandan foods."
    )


class MealPlannerRequest(BaseModel):
    message: str = Field(min_length=1)
    ckd_stage: str = Field(default="G3b")
    weight_kg: float = Field(default=65.0, gt=0)
    uploaded_text: str | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


def _stage_num(stage: str) -> int:
    """
    Food safety stage encoding. G2=1, G3a=2, G3b=3, G4=4.

    Note: differs from ML feature encoding in xgboost_model.py (G2=2, G3a=3,
    G3b=3, G4=4) which is fixed to match trained model feature space.
    """
    return _CLINICAL_STAGE_NUMERIC.get(stage, 3)


def _is_stage_safe(ckd_stage_safe: str, stage_number: int) -> bool:
    if not ckd_stage_safe or str(ckd_stage_safe).lower() == "nan":
        return False
    raw = str(ckd_stage_safe).strip()
    if "-" in raw:
        parts = raw.split("-")
        try:
            low, high = int(parts[0]), int(parts[1])
            return low <= stage_number <= high
        except Exception:
            return False
    try:
        return int(raw) == stage_number
    except Exception:
        return False


def _load_food_db() -> pd.DataFrame:
    """
    Load foods from SQLite (seeded from CSV). Falls back to CSV if DB is empty/unavailable.
    """
    required_cols = [
        "english",
        "kinyarwanda",
        "category",
        "ckd_stage_safe",
        "potassium_mg",
        "phosphorus_mg",
        "protein_g",
        "sodium_mg",
    ]

    try:
        db = SessionLocal()
        try:
            foods = db.query(Food).all()
            if foods:
                df = pd.DataFrame([f.to_dict() for f in foods])
                for c in required_cols:
                    if c not in df.columns:
                        df[c] = None
                csv_path = Path(__file__).resolve().parent.parent / "data" / "food_database.csv"
                csv_df = pd.read_csv(csv_path)
                if "food_id" in df.columns:
                    df["food_id"] = df["food_id"].astype(str).str.strip()
                    csv_df["food_id"] = csv_df["food_id"].astype(str).str.strip()
                    merge_cols = ["food_id"]
                    if "is_rwandan" in csv_df.columns:
                        merge_cols.append("is_rwandan")
                    if "meal_type" in csv_df.columns:
                        merge_cols.append("meal_type")
                    df = df.merge(
                        csv_df[merge_cols],
                        on="food_id",
                        how="left",
                    )
                    if "is_rwandan" in df.columns:
                        df["is_rwandan"] = df["is_rwandan"].fillna(0).astype(int)
                    if "meal_type" in df.columns:
                        df["meal_type"] = df["meal_type"].fillna("Any")
                return df
        finally:
            db.close()
    except Exception as e:
        _mp_logger.warning("DB food load failed, falling back to CSV: %s", e)

    path = Path(__file__).resolve().parent.parent / "data" / "food_database.csv"
    df = pd.read_csv(path)
    for c in required_cols:
        if c not in df.columns:
            df[c] = None
    if "is_rwandan" not in df.columns:
        df["is_rwandan"] = 0
    return df


def _contains_nutrient_values(text: str) -> bool:
    """
    Returns True if text contains specific mg or g values that could
    be hallucinated nutrient data.
    """
    nutrient_pattern = re.compile(r"\d+\.?\d*\s*(?:mg|g)\b", re.IGNORECASE)
    return bool(nutrient_pattern.search(text))


def _is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    name = type(exc).__name__.lower().replace("_", "")
    if "ratelimit" in name:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def query_llm(
    prompt: str,
    history: list[dict] | None = None,
    max_tokens: int = 600,
    raise_on_rate_limit: bool = False,
) -> str | None:
    token = os.getenv("GROQ_API_KEY")
    if not token:
        _mp_logger.warning(
            "GROQ_API_KEY not set — "
            "LLM unavailable"
        )
        return None

    try:
        client = Groq(api_key=token)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a CKD dietary advisor for Rwanda. "
                    "Give concise advice. "
                    "Never mention specific mg or g nutrient values — "
                    "those come from the clinical database."
                    f"{CLINICAL_SAFETY_RULES}"
                ),
            },
        ]

        if history:
            for turn in history[-6:]:
                role = turn.get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                content = turn.get("content") or turn.get("text") or ""
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            _mp_logger.warning(
                "LLM response truncated (finish_reason=length, max_tokens=%s)",
                max_tokens,
            )

        result = (choice.message.content or "").strip()
        return result if result else None

    except Exception as e:
        if raise_on_rate_limit and _is_rate_limit_error(e):
            raise
        _mp_logger.warning("LLM failed (using template): %s", e)
        return None


def _format_food_list_rows(combined: pd.DataFrame) -> str:
    return "\n".join(
        [
            f"- {row['english']} "
            f"(K:{row['potassium_mg']:.0f}mg, "
            f"P:{row['phosphorus_mg']:.0f}mg, "
            f"Pro:{row['protein_g']:.1f}g per 100g)"
            for _, row in combined.iterrows()
        ]
    )


def _max_grams_for_meal_budget(
    english: str,
    protein_per_100: float,
    phosphorus_per_100: float,
    protein_budget_g: float,
    phosphorus_budget_mg: float,
    *,
    category: str = "",
) -> int:
    """
    Prompt-hint max grams so this food leaves room for other plate items.
    Protein share uses MEAL_RULES_GLOBAL['protein_pick_allowance_frac'] for
    Meat/Fish only (same distinction as build_meal_plan pick_one). Phosphorus
    uses that same frac for Meat/Fish (they burn both budgets); other foods
    still see the full phosphorus budget as their solo ceiling.
    """
    candidates: list[float] = [float(_DEFAULT_MAX_GRAMS)]
    _min_g, portion_max = _portion_bounds_for_name(str(english or ""))
    candidates.append(float(portion_max))

    protein_cats = MEAL_RULES_GLOBAL["protein_pick_categories"]
    is_protein_pick = str(category or "").strip() in protein_cats
    frac = float(MEAL_RULES_GLOBAL["protein_pick_allowance_frac"])
    pro_budget = (
        protein_budget_g * frac if is_protein_pick else protein_budget_g
    )
    phos_budget = (
        phosphorus_budget_mg * frac if is_protein_pick else phosphorus_budget_mg
    )

    pro = float(protein_per_100 or 0)
    phos = float(phosphorus_per_100 or 0)
    if pro > 0:
        candidates.append((pro_budget / pro) * 100.0)
    if phos > 0:
        candidates.append((phos_budget / phos) * 100.0)
    max_g = min(candidates)
    # Round down to nearest 5g for a clean prompt hint (min 1g).
    if max_g >= 5:
        return max(5, int(max_g // 5 * 5))
    return max(1, int(max_g))


def _format_structured_food_list_rows(
    combined: pd.DataFrame,
    *,
    protein_budget_g: float,
    phosphorus_budget_mg: float,
) -> str:
    """
    Structured-prompt food list with pre-computed max grams per food.
    Free-text paths keep using _format_food_list_rows (unchanged).
    """
    if combined is None or combined.empty:
        return ""
    lines: list[str] = []
    for _, row in combined.iterrows():
        english = str(row.get("english") or "").strip()
        category = str(row.get("category") or "").strip()
        k = float(row.get("potassium_mg") or 0)
        p = float(row.get("phosphorus_mg") or 0)
        pro = float(row.get("protein_g") or 0)
        max_g = _max_grams_for_meal_budget(
            english,
            pro,
            p,
            protein_budget_g,
            phosphorus_budget_mg,
            category=category,
        )
        lines.append(
            f"- {english} (max ~{max_g}g for this meal; "
            f"K:{k:.0f}mg, P:{p:.0f}mg, Pro:{pro:.1f}g per 100g)"
        )
    return "\n".join(lines)


def _rank_foods_for_prompt(
    foods: pd.DataFrame,
    limits: dict[str, float],
    max_rows: int = 30,
    day: str | None = None,
) -> pd.DataFrame:
    foods = foods.copy()
    if foods.empty:
        return foods
    for col in ("potassium_mg", "phosphorus_mg", "protein_g"):
        foods[col] = pd.to_numeric(foods[col], errors="coerce").fillna(0)

    safe_food_list = (
        foods[
            (foods["potassium_mg"] <= limits["potassium"] * 0.20)
            & (foods["phosphorus_mg"] <= limits["phosphorus"] * 0.25)
        ]
        .sort_values("potassium_mg")
        .head(25)
    )

    protein_options = (
        foods[
            (foods["protein_g"] >= 5)
            & (foods["potassium_mg"] <= limits["potassium"] * 0.25)
        ]
        .sort_values("potassium_mg")
        .head(10)
    )

    combined = (
        pd.concat([safe_food_list, protein_options])
        .drop_duplicates(subset=["english"])
    )

    # Day-seeded shuffle AFTER nutrient filtering only — never bypasses K/P/protein gates.
    if day and len(combined) > 1:
        rng = random.Random(day)
        order = list(range(len(combined)))
        rng.shuffle(order)
        combined = combined.iloc[order].reset_index(drop=True)

    return combined.head(max_rows)


def _build_prompt_food_list(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    max_rows: int = 30,
) -> str:
    # Non-weekly path: day=None → deterministic low-K order (unchanged).
    combined = _rank_foods_for_prompt(safe_foods, limits, max_rows=max_rows)
    return _format_food_list_rows(combined)


def _build_occasion_prompt_food_lists(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    max_rows: int = 10,
    day: str | None = None,
) -> dict[str, str]:
    """Split safe_foods by meal_type into breakfast / lunch-dinner / snack lists."""
    foods = safe_foods.copy()
    if "meal_type" not in foods.columns:
        foods["meal_type"] = "Any"
    foods["meal_type"] = foods["meal_type"].fillna("Any").astype(str)
    foods["category"] = foods.get("category", pd.Series(dtype=str))
    if "category" not in foods.columns:
        foods["category"] = "Other"
    foods["category"] = foods["category"].fillna("Other").astype(str)

    breakfast_pool = foods[foods["meal_type"].isin(["Any", "Breakfast/Any"])]
    lunch_dinner_pool = foods[foods["meal_type"].isin(["Any", "Lunch/Dinner"])]
    fat_name = (
        breakfast_pool["english"]
        .fillna("")
        .str.lower()
        .str.contains(
            r"\b(?:oil|ghee|butter|lard|honey)\b",
            regex=True,
            na=False,
        )
    )
    snack_pool = breakfast_pool[
        (breakfast_pool["category"] != "Fat/Oil")
        & (breakfast_pool["meal_type"] != "Cooking")
        & ~fat_name
    ]

    return {
        "breakfast": _format_food_list_rows(
            _rank_foods_for_prompt(
                breakfast_pool, limits, max_rows=max_rows, day=day
            )
        ),
        "lunch_dinner": _format_food_list_rows(
            _rank_foods_for_prompt(
                lunch_dinner_pool, limits, max_rows=max_rows, day=day
            )
        ),
        "snack": _format_food_list_rows(
            _rank_foods_for_prompt(
                snack_pool, limits, max_rows=max_rows, day=day
            )
        ),
    }


# Kenya FCT 2018 code 01007 Bread, White — per 100g edible portion
_BREAD_PROXY: dict[str, Any] = {
    "english": "bread",
    "kinyarwanda": "Umugati",
    "category": "Grain",
    "preparation_method": "baked",
    "potassium_mg": 120.0,
    "phosphorus_mg": 95.0,
    "protein_g": 7.7,
    "sodium_mg": 466.0,
}

# Cultural aliases with no DB rows.
# "proxy" is the sole nutrient-source key: "maize" | "bread".
_CULTURAL_ALIASES: dict[str, dict[str, Any]] = {
    "igikoma": {
        "canonical": "igikoma",
        "occasions": {"Breakfast"},
        "proxy": "maize",
    },
    "maize porridge": {
        "canonical": "igikoma",
        "occasions": {"Breakfast"},
        "proxy": "maize",
    },
    "ugali": {
        "canonical": "ugali",
        "occasions": {"Lunch", "Dinner"},
        "proxy": "maize",
    },
    "ubugali": {
        "canonical": "ugali",
        "occasions": {"Lunch", "Dinner"},
        "proxy": "maize",
    },
    "bread": {
        "canonical": "bread",
        "occasions": {"Breakfast"},
        "proxy": "bread",
    },
    "white bread": {
        "canonical": "bread",
        "occasions": {"Breakfast"},
        "proxy": "bread",
    },
    "umugati": {
        "canonical": "bread",
        "occasions": {"Breakfast"},
        "proxy": "bread",
    },
}

# Max edible grams when listed as its own plate item (cooking-amount realism).
# Keys: normalized english names (_normalize_food_key). Not an exclusion list.
_FLAVORING_MAX_GRAMS: dict[str, float] = {
    "onion": 25.0,
    "onions": 25.0,
    "tomato": 100.0,
    "tomatoes": 100.0,
}

_DEFAULT_MIN_GRAMS = 25.0
_DEFAULT_MAX_GRAMS = 300.0
_FLAVORING_MIN_GRAMS = 10.0

# Display-only renames for plate cards / structured output.
# Keys: normalized DB english (_normalize_food_key). Nutrients/filters unchanged.
_DISPLAY_ENGLISH_NAMES: dict[str, str] = {
    "maize": "maize porridge",
    "sorghum": "sorghum porridge",
    "millet": "millet porridge",
    "wheat": "wheat porridge",
    "sour milk": "ikivuguto",
}

# Snack-only english names: excluded from Breakfast/Lunch/Dinner;
# Snack Starch carve-out still uses this set.
_LUNCH_DINNER_EXCLUDED_ENGLISH = frozenset({"sugar cane"})

# Heavier/savory items not typical for Breakfast (Any meal_type in CSV).
_BREAKFAST_EXCLUDED_ENGLISH = frozenset({
    "banana",
    "irish potatoes",
    "sweet potatoes",
    "wheat",  # flour row; bread synthetic covers wheat-as-food
})

# Shared meal-planner rules (occasion pool + future readers). Values mirror
# tonight's live behavior — refactor source of truth, not a rule change.
MEAL_RULES_GLOBAL: dict[str, Any] = {
    "fat_name_re": r"\b(?:oil|ghee|butter|lard|honey|lemons?)\b",
    "resolve_exclude_category": "Fat/Oil",
    "resolve_exclude_meal_type": "Cooking",
    "default_portion_grams": (_DEFAULT_MIN_GRAMS, _DEFAULT_MAX_GRAMS),
    "flavoring_min_grams": _FLAVORING_MIN_GRAMS,
    "portion_caps": _FLAVORING_MAX_GRAMS,
    "display_names": _DISPLAY_ENGLISH_NAMES,
    "cultural_aliases": _CULTURAL_ALIASES,
    "bread_proxy": _BREAD_PROXY,
    "item_k_gate_frac": 0.20,
    "item_p_gate_frac": 0.25,
    "snack_only_english": _LUNCH_DINNER_EXCLUDED_ENGLISH,
    "foods_count_range": (1, 4),
    "allowed_list_max_rows": 20,
    "protein_pick_categories": frozenset({"Meat", "Fish", "Egg", "Dairy"}),
    "protein_pick_allowance_frac": 0.65,
    "tea_name_match": "exact_or_prefix_or_word",
    # Non-Rwandan english that may pass the is_rwandan filter (stage safety unchanged).
    # \boats\b matches "Oats, whole grain…" / "steel cut oats"; not goat or "oat milk".
    "rwandan_bypass_english_re": r"\boats\b",
    # Breakfast liquid/semi-liquid items for max_drink_like validation.
    "breakfast_drink_like_english": frozenset(
        {"igikoma", "ikivuguto", "sour milk", "tea", "coffee", "milk"}
    ),
    # Allowed-list synthetic injections (quota tokens __bread__/__igikoma__/__ugali__)
    "synthetic_specs": {
        "bread": {
            "id": "bread",
            "english": "bread",
            "meal_type": "Breakfast/Any",
            "proxy": "bread",
        },
        "igikoma": {
            "id": "igikoma",
            "english": "igikoma",
            "category": "Grain",
            "preparation_method": "boiled porridge",
            "meal_type": "Breakfast/Any",
            "proxy": "maize",
        },
        "ugali": {
            "id": "ugali",
            "english": "ugali",
            "category": "Starch",
            "preparation_method": "boiled",
            "meal_type": "Lunch/Dinner",
            "proxy": "maize",
        },
    },
}

OCCASION_RULES: dict[str, dict[str, Any]] = {
    "Breakfast": {
        "meal_types": ["Any", "Breakfast/Any"],
        "exclude_categories": [],
        "exclude_categories_unless_english": {},
        "exclude_meal_types": [],
        # Breakfast-only english bans (sugar cane via MEAL_RULES_GLOBAL.snack_only_english)
        "exclude_english": _BREAKFAST_EXCLUDED_ENGLISH,
        "starch_allow_english": frozenset(),
        "quota_grain_exclude_english": frozenset({"maize", "wheat"}),
        "quotas": [
            (("Egg", "Dairy"), 3),
            (("Grain",), 4),
            (("__bread__",), 1),
            (("__igikoma__",), 1),
            (("Fruit", "Beverage", "Vegetable", "Starch"), 6),
        ],
        "synthetics": ["bread", "igikoma"],
        "groups": [["Egg", "Dairy"], ["Grain"]],
        "default_portion_g": 150,
        "nutrient_caps": (0.25, 0.25, 0.30, 0.25),
        "max_drink_like": 2,
        "require_protein_exactly_one_if_gated": False,
        "pairings": {
            "forbid_together": [("igikoma", "tea")],
            "require_if": [
                {
                    "if_english": ["ikivuguto", "sour milk"],
                    "require_category": "Fruit",
                    "fail_reason": "ikivuguto at breakfast requires a fruit pairing",
                }
            ],
        },
        "fallback_repair": ["ikivuguto_fruit"],
        "prompt_rules": [
            "Prefer including at least one staple (igikoma or bread or grain).",
            "If you include ikivuguto (sour milk), you must also include a Fruit item.",
            "grams must be between 25 and 300 (except onion/onions: 10–25g max).",
            "Do not list oils, ghee, butter, lard, or honey as foods.",
            "Do not serve tea and igikoma in the same meal.",
        ],
    },
    "Lunch": {
        "meal_types": ["Any", "Lunch/Dinner"],
        "exclude_categories": [],
        "exclude_categories_unless_english": {},
        "exclude_meal_types": [],
        "exclude_english": frozenset(),
        "starch_allow_english": frozenset(),
        "quota_grain_exclude_english": frozenset(),
        "quotas": [
            (("Meat", "Fish"), 3),
            (("Starch", "Grain"), 3),
            (("Vegetable",), 4),
            (("__ugali__",), 1),
            (("Fruit",), 2),
        ],
        "synthetics": ["ugali"],
        "groups": [["Meat", "Fish"], ["Starch"], ["Vegetable"]],
        "default_portion_g": 200,
        "nutrient_caps": (0.40, 0.40, 0.40, 0.40),
        "require_protein_exactly_one_if_gated": True,
        "pairings": {
            "forbid_together": [("igikoma", "tea")],
            "require_if": [],
        },
        "fallback_repair": [],
        "prompt_rules": [
            "You MUST include exactly one Meat or Fish item from the list.",
            "Also include ugali or another starch when listed.",
            "grams must be between 25 and 300 (except onion/onions: 10–25g max).",
            "Do not list oils, ghee, butter, lard, or honey as foods.",
            "Do not serve tea and igikoma in the same meal.",
        ],
    },
    "Dinner": {
        "meal_types": ["Any", "Lunch/Dinner"],
        "exclude_categories": [],
        "exclude_categories_unless_english": {},
        "exclude_meal_types": [],
        "exclude_english": frozenset(),
        "starch_allow_english": frozenset(),
        "quota_grain_exclude_english": frozenset(),
        "quotas": [
            (("Meat", "Fish"), 3),
            (("Starch", "Grain"), 3),
            (("Vegetable",), 4),
            (("__ugali__",), 1),
            (("Fruit",), 2),
        ],
        "synthetics": ["ugali"],
        "groups": [["Meat", "Fish"], ["Starch"], ["Vegetable"]],
        "default_portion_g": 200,
        "nutrient_caps": (0.40, 0.40, 0.40, 0.40),
        "require_protein_exactly_one_if_gated": True,
        "pairings": {
            "forbid_together": [("igikoma", "tea")],
            "require_if": [],
        },
        "fallback_repair": [],
        "prompt_rules": [
            "You MUST include exactly one Meat or Fish item from the list.",
            "Also include ugali or another starch when listed.",
            "grams must be between 25 and 300 (except onion/onions: 10–25g max).",
            "Do not list oils, ghee, butter, lard, or honey as foods.",
            "Do not serve tea and igikoma in the same meal.",
        ],
    },
    "Snack": {
        "meal_types": ["Any", "Breakfast/Any"],
        "exclude_categories": ["Fat/Oil", "Grain", "Egg", "Vegetable"],
        "exclude_categories_unless_english": {
            "Starch": _LUNCH_DINNER_EXCLUDED_ENGLISH,  # sugar cane carve-out
        },
        "exclude_meal_types": ["Cooking"],
        "exclude_english": frozenset(),
        "starch_allow_english": _LUNCH_DINNER_EXCLUDED_ENGLISH,
        "quota_grain_exclude_english": frozenset(),
        "quotas": None,
        "synthetics": [],
        "groups": [["Fruit"], ["Dairy"]],
        "default_portion_g": 100,
        "nutrient_caps": (0.15, 0.15, 0.10, 0.15),
        "require_protein_exactly_one_if_gated": False,
        "pairings": {
            "forbid_together": [("igikoma", "tea")],
            "require_if": [],
        },
        "fallback_repair": [],
        "prompt_rules": [
            "Prefer including at least one staple (ugali or starch) when listed.",
            "grams must be between 25 and 300 (except onion/onions: 10–25g max).",
            "Do not list oils, ghee, butter, lard, or honey as foods.",
            "Do not serve tea and igikoma in the same meal.",
        ],
    },
}


def _detect_single_occasion(
    wants_breakfast: bool,
    wants_lunch: bool,
    wants_dinner: bool,
    wants_snack: bool,
) -> str | None:
    flags: list[str] = []
    if wants_breakfast:
        flags.append("Breakfast")
    if wants_lunch:
        flags.append("Lunch")
    if wants_dinner:
        flags.append("Dinner")
    if wants_snack:
        flags.append("Snack")
    if len(flags) == 1:
        return flags[0]
    return None


def _normalize_food_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _english_matches_rwandan_bypass(english: str) -> bool:
    """True if english matches MEAL_RULES_GLOBAL rwandan_bypass_english_re."""
    pattern = MEAL_RULES_GLOBAL.get("rwandan_bypass_english_re") or ""
    if not pattern:
        return False
    key = _normalize_food_key(english)
    return bool(re.search(pattern, key))


def _filter_rwandan_with_bypass(foods: pd.DataFrame) -> pd.DataFrame:
    """
    Keep is_rwandan==1 rows plus allowlisted non-Rwandan exceptions.
    Does not touch ckd_stage_safe / stage filtering — caller must already
    have applied stage safety.
    """
    if foods is None or foods.empty or "is_rwandan" not in foods.columns:
        return foods
    is_rw = (
        pd.to_numeric(foods["is_rwandan"], errors="coerce")
        .fillna(0)
        .astype(int)
        == 1
    )
    bypass = (
        foods["english"]
        .fillna("")
        .astype(str)
        .apply(_english_matches_rwandan_bypass)
    )
    return foods[is_rw | bypass].copy()


def _breakfast_recommended_staple(day: str | None) -> str | None:
    """
    Day-seeded soft staple steer: 'bread' or 'igikoma'.
    None when day is absent (single-occasion chat — no day-based steering).
    """
    if not day:
        return None
    rng = random.Random(f"{day}:Breakfast:staple")
    return "bread" if rng.random() < 0.5 else "igikoma"


def _forbidden_english_mask(
    english: pd.Series,
    forbidden_terms: list[str],
) -> pd.Series:
    """True where normalized english equals a forbidden term (not substring)."""
    if not forbidden_terms:
        return pd.Series(False, index=english.index)
    forbidden = {t.strip().lower() for t in forbidden_terms}
    keys = (
        english.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return keys.isin(forbidden)


def _portion_bounds_for_name(name: str) -> tuple[float, float]:
    """Return (min_grams, max_grams) for a food name (flavoring caps when listed)."""
    key = _normalize_food_key(name)
    max_g = _FLAVORING_MAX_GRAMS.get(key)
    if max_g is None:
        return _DEFAULT_MIN_GRAMS, _DEFAULT_MAX_GRAMS
    return _FLAVORING_MIN_GRAMS, max_g


def _display_english_name(english: str) -> str:
    key = _normalize_food_key(english)
    return _DISPLAY_ENGLISH_NAMES.get(key, (english or "").strip())


def _occasion_food_pool(
    safe_foods: pd.DataFrame,
    occasion: str,
) -> pd.DataFrame:
    """Same meal_type / snack fat rules as _build_occasion_prompt_food_lists."""
    rules = OCCASION_RULES[occasion]
    meal_types = rules["meal_types"]
    exclude_categories = rules["exclude_categories"]
    exclude_categories_unless_english = rules["exclude_categories_unless_english"]
    exclude_meal_types = rules["exclude_meal_types"]
    exclude_english = rules["exclude_english"]
    starch_allow_english = rules["starch_allow_english"]
    fat_name_re = MEAL_RULES_GLOBAL["fat_name_re"]
    snack_only_english = MEAL_RULES_GLOBAL["snack_only_english"]

    foods = safe_foods.copy()
    if "meal_type" not in foods.columns:
        foods["meal_type"] = "Any"
    foods["meal_type"] = foods["meal_type"].fillna("Any").astype(str)
    if "category" not in foods.columns:
        foods["category"] = "Other"
    foods["category"] = foods["category"].fillna("Other").astype(str)

    pool = foods[foods["meal_type"].isin(meal_types)]

    excluded_name = (
        pool["english"]
        .fillna("")
        .str.lower()
        .str.contains(fat_name_re, regex=True, na=False)
    )
    eng_norm = pool["english"].fillna("").str.lower().str.strip()

    if occasion == "Snack":
        keep = pd.Series(True, index=pool.index)
        for cat in exclude_categories:
            keep &= pool["category"] != cat
        for cat, allowed_eng in exclude_categories_unless_english.items():
            # Starch carve-out: allow-list is starch_allow_english (same set today)
            if cat == "Starch":
                allowed_eng = starch_allow_english
            keep &= (pool["category"] != cat) | eng_norm.isin(allowed_eng)
        for mt in exclude_meal_types:
            keep &= pool["meal_type"] != mt
        keep &= ~excluded_name
        pool = pool[keep]
    else:
        # Structured suggestions: never offer honey (or cooking fats) as eaten items
        pool = pool[~excluded_name]
        # Sugar cane (and any future snack-only english) off Breakfast/Lunch/Dinner
        if occasion in ("Breakfast", "Lunch", "Dinner"):
            eng_norm = pool["english"].fillna("").str.lower().str.strip()
            pool = pool[~eng_norm.isin(snack_only_english)]
        if exclude_english:
            eng_norm = pool["english"].fillna("").str.lower().str.strip()
            pool = pool[~eng_norm.isin(exclude_english)]
    return pool


def _lookup_maize_proxy(safe_foods: pd.DataFrame) -> dict[str, Any]:
    """Rwandan boiled maize row for igikoma/ugali nutrient math."""
    if safe_foods is None or safe_foods.empty or "english" not in safe_foods.columns:
        return {
            "english": "maize",
            "kinyarwanda": "Ibigori",
            "category": "Grain",
            "preparation_method": "boiled",
            "potassium_mg": 226.0,
            "phosphorus_mg": 73.0,
            "protein_g": 2.9,
            "sodium_mg": 2.0,
        }
    matches = safe_foods[
        safe_foods["english"].fillna("").str.lower().str.strip() == "maize"
    ]
    if matches.empty:
        return {
            "english": "maize",
            "kinyarwanda": "Ibigori",
            "category": "Grain",
            "preparation_method": "boiled",
            "potassium_mg": 226.0,
            "phosphorus_mg": 73.0,
            "protein_g": 2.9,
            "sodium_mg": 2.0,
        }
    row = matches.iloc[0]
    return {
        "english": str(row.get("english") or "maize").strip(),
        "kinyarwanda": str(row.get("kinyarwanda") or "Ibigori").strip(),
        "category": str(row.get("category") or "Grain").strip(),
        "preparation_method": str(row.get("preparation_method") or "boiled").strip(),
        "potassium_mg": float(row.get("potassium_mg") or 226.0),
        "phosphorus_mg": float(row.get("phosphorus_mg") or 73.0),
        "protein_g": float(row.get("protein_g") or 2.9),
        "sodium_mg": float(row.get("sodium_mg") or 2.0),
    }


def _scale_food_item(
    *,
    english: str,
    kinyarwanda: str,
    category: str,
    preparation_method: str,
    k_per_100: float,
    p_per_100: float,
    pro_per_100: float,
    na_per_100: float,
    grams: float,
    occasion: str,
) -> dict[str, Any]:
    factor = grams / 100.0
    return {
        "english": _display_english_name(english),
        "kinyarwanda": kinyarwanda,
        "preparation_method": preparation_method,
        "portion_grams": round(grams, 1),
        "meal_occasion": occasion,
        "potassium_mg": round(k_per_100 * factor, 1),
        "phosphorus_mg": round(p_per_100 * factor, 1),
        "protein_g": round(pro_per_100 * factor, 1),
        "sodium_mg": round(na_per_100 * factor, 1),
        "category": category,
    }


def _synthetic_row(meta: dict[str, Any], *, english: str, meal_type: str) -> dict[str, Any]:
    return {
        "english": english,
        "kinyarwanda": str(meta.get("kinyarwanda") or ""),
        "category": str(meta.get("category") or "Other"),
        "meal_type": meal_type,
        "preparation_method": str(meta.get("preparation_method") or ""),
        "potassium_mg": float(meta["potassium_mg"]),
        "phosphorus_mg": float(meta["phosphorus_mg"]),
        "protein_g": float(meta["protein_g"]),
        "sodium_mg": float(meta.get("sodium_mg") or 0),
    }


# Quota special-tokens → MEAL_RULES_GLOBAL["synthetic_specs"] ids
_QUOTA_SYNTHETIC_TOKEN: dict[tuple[str, ...], str] = {
    ("__bread__",): "bread",
    ("__igikoma__",): "igikoma",
    ("__ugali__",): "ugali",
}


def _shuffle_dataframe_tea_biased(
    cands: pd.DataFrame,
    rng: random.Random,
    *,
    tea_weight: float = 3.0,
    coffee_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Seeded weighted shuffle (Efraimidis–Spirakis).
    Tea weight 3 vs coffee 1 ≈ 75/25 ordering preference; others weight 1.
    Coffee is never excluded.
    """
    if len(cands) <= 1:
        return cands
    names = cands["english"].fillna("").str.lower().str.strip()
    weights: list[float] = []
    for name in names:
        if name == "tea" or name.startswith("tea "):
            weights.append(tea_weight)
        elif name == "coffee" or name.startswith("coffee "):
            weights.append(coffee_weight)
        else:
            weights.append(1.0)
    keyed: list[tuple[float, int]] = []
    for i, w in enumerate(weights):
        u = rng.random()
        keyed.append((u ** (1.0 / max(w, 1e-9)), i))
    keyed.sort(key=lambda t: t[0], reverse=True)
    order = [i for _, i in keyed]
    return cands.iloc[order]


def _build_structured_allowed_foods(
    safe_foods: pd.DataFrame,
    occasion: str,
    limits: dict[str, float],
    max_rows: int | None = None,
    day: str | None = None,
) -> pd.DataFrame:
    """
    Category-quota allowed list for structured JSON prompts.
    Guarantees bread + igikoma (Breakfast) and ugali (Lunch/Dinner).
    """
    rules = OCCASION_RULES[occasion]
    quotas = rules["quotas"]
    synthetics = rules["synthetics"]
    quota_grain_exclude_english = rules["quota_grain_exclude_english"]
    bread_proxy = MEAL_RULES_GLOBAL["bread_proxy"]
    synthetic_specs = MEAL_RULES_GLOBAL["synthetic_specs"]
    k_gate_frac = float(MEAL_RULES_GLOBAL["item_k_gate_frac"])
    p_gate_frac = float(MEAL_RULES_GLOBAL["item_p_gate_frac"])
    if max_rows is None:
        max_rows = int(MEAL_RULES_GLOBAL["allowed_list_max_rows"])

    pool = _occasion_food_pool(safe_foods, occasion)
    # Snack (and any occasion with quotas: None): flat rank, no quota fill
    if quotas is None:
        return _rank_foods_for_prompt(pool, limits, max_rows=max_rows, day=day)

    k_gate = limits["potassium"] * k_gate_frac
    p_gate = limits["phosphorus"] * p_gate_frac
    maize = _lookup_maize_proxy(safe_foods)

    foods = pool.copy()
    for col in ("potassium_mg", "phosphorus_mg", "protein_g", "sodium_mg"):
        if col in foods.columns:
            foods[col] = pd.to_numeric(foods[col], errors="coerce").fillna(0)
    if "category" not in foods.columns:
        foods["category"] = "Other"
    if "english" not in foods.columns:
        foods["english"] = ""

    used: set[str] = set()
    picked_rows: list[dict[str, Any]] = []

    def _take_categories(cats: tuple[str, ...], n: int) -> None:
        if foods.empty or n <= 0:
            return
        cands = foods[
            foods["category"].isin(cats)
            & ~foods["english"].fillna("").str.lower().str.strip().isin(used)
            & (foods["potassium_mg"] <= k_gate)
            & (foods["phosphorus_mg"] <= p_gate)
        ]
        # Raw maize → igikoma; wheat → bread synthetic — don't also list flour/grain rows.
        if cats == ("Grain",) and quota_grain_exclude_english:
            cands = cands[
                ~cands["english"].fillna("").str.lower().str.strip().isin(
                    quota_grain_exclude_english
                )
            ]
        # Day-seeded shuffle AFTER K/P gates — never bypasses nutrient filters.
        # Single-occasion (day=None) keeps deterministic low-K order.
        if day and len(cands) > 1:
            rng = random.Random(f"{day}:{occasion}:{','.join(cats)}")
            if occasion == "Breakfast" and "Beverage" in cats:
                cands = _shuffle_dataframe_tea_biased(cands, rng)
            else:
                order = list(range(len(cands)))
                rng.shuffle(order)
                cands = cands.iloc[order]
        else:
            cands = cands.sort_values("potassium_mg")
        for _, row in cands.head(n).iterrows():
            en = str(row.get("english") or "").strip().lower()
            if not en or en in used:
                continue
            used.add(en)
            picked_rows.append(row.to_dict())

    def _inject_synthetic(synth_id: str) -> None:
        if synth_id not in synthetics:
            return
        spec = synthetic_specs[synth_id]
        proxy_key = spec["proxy"]
        if proxy_key == "bread":
            meta: dict[str, Any] = {**bread_proxy}
        elif proxy_key == "maize":
            meta = {**maize}
        else:
            return
        if "category" in spec:
            meta["category"] = spec["category"]
        if "preparation_method" in spec:
            meta["preparation_method"] = spec["preparation_method"]
        picked_rows.append(
            _synthetic_row(
                meta,
                english=str(spec["english"]),
                meal_type=str(spec["meal_type"]),
            )
        )
        used.add(str(spec["english"]))

    for cats, n in quotas:
        synth_id = _QUOTA_SYNTHETIC_TOKEN.get(cats)
        if synth_id is not None:
            _inject_synthetic(synth_id)
            continue
        _take_categories(cats, n)

    if not picked_rows:
        return _rank_foods_for_prompt(pool, limits, max_rows=max_rows, day=day)

    out = pd.DataFrame(picked_rows)
    return out.head(max_rows)


def _occasion_nutrient_budgets(
    occasion: str,
    limits: dict[str, float],
    protein_limit_g: float,
) -> tuple[float, float, float, float]:
    """Absolute K/P/protein/Na meal caps matching _validate_occasion_suggestion."""
    k_cap, p_cap, pro_cap, na_cap = OCCASION_RULES[occasion]["nutrient_caps"]
    return (
        limits["potassium"] * k_cap,
        limits["phosphorus"] * p_cap,
        protein_limit_g * pro_cap,
        limits["sodium"] * na_cap,
    )


def _occasion_nutrient_budget_line(
    occasion: str,
    limits: dict[str, float],
    protein_limit_g: float,
) -> str:
    """Absolute per-meal caps matching _validate_occasion_suggestion."""
    k_bud, p_bud, pro_bud, na_bud = _occasion_nutrient_budgets(
        occasion, limits, protein_limit_g
    )
    return (
        f"FOR THIS MEAL ONLY, stay under: "
        f"Potassium ≤{k_bud:.0f}mg, "
        f"Phosphorus ≤{p_bud:.0f}mg, "
        f"Protein ≤{pro_bud:.1f}g, "
        f"Sodium ≤{na_bud:.0f}mg"
    )


def _build_structured_occasion_prompt(
    occasion: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    pool: pd.DataFrame,
) -> str:
    # Caller passes _build_structured_allowed_foods(...) as pool
    _k_bud, p_bud, pro_bud, _na_bud = _occasion_nutrient_budgets(
        occasion, limits, protein_limit_g
    )
    food_lines = _format_structured_food_list_rows(
        pool,
        protein_budget_g=pro_bud,
        phosphorus_budget_mg=p_bud,
    )
    meal_budget = _occasion_nutrient_budget_line(
        occasion, limits, protein_limit_g
    )

    has_meat_fish = (
        not pool.empty
        and "category" in pool.columns
        and bool(pool["category"].isin(["Meat", "Fish"]).any())
    )

    if occasion in ("Lunch", "Dinner") and has_meat_fish:
        staple_rule = (
            "- You MUST include exactly one Meat or Fish item from the list.\n"
            "- Also include ugali or another starch when listed.\n"
        )
        diversity_rule = (
            "- Every option must still follow all rules above, including the "
            "Meat/Fish requirement. Create diversity through different protein "
            "choices (e.g. chicken vs beef vs fish) and different "
            "starches/vegetables — never by omitting a required food category.\n"
        )
    elif occasion == "Breakfast":
        staple_rule = (
            "- Prefer including at least one staple (igikoma or bread or grain).\n"
            "- If you include ikivuguto (sour milk), you must also include a Fruit item.\n"
        )
        diversity_rule = (
            "- Every option must still follow all rules above. Create diversity "
            "through different staples and sides — never by omitting a required "
            "pairing (e.g. fruit with ikivuguto).\n"
        )
    else:
        staple_rule = (
            "- Prefer including at least one staple (ugali or starch) when listed.\n"
        )
        diversity_rule = (
            "- Every option must still follow all rules above. Create diversity "
            "through different foods from the allowed list — never by omitting "
            "a required food category.\n"
        )

    return (
        f"You are a CKD dietary advisor for Rwanda. "
        f"Respond with a single JSON object only (no markdown).\n\n"
        f"Patient stage: {ckd_stage}\n"
        f"Daily limits: K={limits['potassium']}mg, "
        f"P={limits['phosphorus']}mg, "
        f"Protein={protein_limit_g:.0f}g, "
        f"Na={limits['sodium']}mg\n"
        f"{meal_budget}\n\n"
        f"Occasion: {occasion}\n\n"
        f"ALLOWED FOODS (pick only from this list; "
        f"use grams at or under each food's max):\n"
        f"{food_lines}\n\n"
        f"Rules:\n"
        f"- Return 2 or 3 distinct meal options for {occasion} only.\n"
        f"- Each option is an independent meal with 1 to 4 foods "
        f"and must follow every rule below on its own "
        f"(including the FOR THIS MEAL ONLY budget above).\n"
        f"{diversity_rule}"
        f"{staple_rule}"
        f"- grams must be between 25 and the max shown for each food "
        f"(except onion/onions: 10–25g max).\n"
        f"- Do not list oils, ghee, butter, lard, or honey as foods.\n"
        f"- Do not serve tea and igikoma in the same meal.\n"
        f"- JSON schema exactly:\n"
        f'{{"occasion": "{occasion}", '
        f'"options": ['
        f'{{"foods": [{{"name": "<food>", "grams": <number>}}]}}, '
        f'{{"foods": [{{"name": "<food>", "grams": <number>}}]}}'
        f']}}\n'
    )


def query_llm_json(
    prompt: str,
    max_tokens: int = 300,
    raise_on_rate_limit: bool = False,
) -> str | None:
    """Groq call with JSON object mode. Returns raw JSON string or None."""
    token = os.getenv("GROQ_API_KEY")
    if not token:
        _mp_logger.warning("GROQ_API_KEY not set — LLM unavailable")
        return None

    try:
        client = Groq(api_key=token)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output only valid JSON objects. "
                        "No markdown, no commentary."
                        f"{CLINICAL_SAFETY_RULES}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.2,
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            _mp_logger.warning(
                "LLM JSON response truncated (finish_reason=length, max_tokens=%s)",
                max_tokens,
            )
        result = (choice.message.content or "").strip()
        return result if result else None
    except Exception as e:
        if raise_on_rate_limit and _is_rate_limit_error(e):
            raise
        _mp_logger.warning("LLM JSON call failed: %s", e)
        return None


def _generate_structured_occasion_suggestion(
    occasion: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    safe_foods: pd.DataFrame,
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    """Ask Groq for JSON foods for one occasion. Returns (parsed dict or None, allowed list)."""
    allowed = _build_structured_allowed_foods(
        safe_foods, occasion, limits, max_rows=20
    )
    if allowed.empty:
        _mp_logger.warning(
            "Structured occasion suggestion: empty pool for %s", occasion
        )
        return None, allowed

    prompt = _build_structured_occasion_prompt(
        occasion=occasion,
        ckd_stage=ckd_stage,
        limits=limits,
        protein_limit_g=protein_limit_g,
        pool=allowed,
    )
    raw = query_llm_json(prompt, max_tokens=600)
    if not raw:
        return None, allowed
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        _mp_logger.warning(
            "Structured occasion JSON parse failed for %s: %s | snippet=%r",
            occasion,
            e,
            raw[:200],
        )
        return None, allowed
    if not isinstance(data, dict):
        return None, allowed
    data["_raw_snippet"] = raw[:300]
    return data, allowed


def _resolve_name_to_item(
    name: str,
    grams: float,
    occasion: str,
    pool: pd.DataFrame,
    maize_proxy: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve AI name to a build_meal_plan-shaped item, or (None, reason)."""
    cultural_aliases = MEAL_RULES_GLOBAL["cultural_aliases"]
    bread_proxy = MEAL_RULES_GLOBAL["bread_proxy"]
    fat_name_re = re.compile(MEAL_RULES_GLOBAL["fat_name_re"], re.IGNORECASE)
    resolve_exclude_category = MEAL_RULES_GLOBAL["resolve_exclude_category"]
    resolve_exclude_meal_type = MEAL_RULES_GLOBAL["resolve_exclude_meal_type"]

    key = _normalize_food_key(name)
    if not key:
        return None, "empty food name"

    alias = cultural_aliases.get(key)
    if alias:
        if occasion not in alias["occasions"]:
            return None, f"alias {key!r} not allowed for {occasion}"

        proxy_key = alias["proxy"]
        if proxy_key == "bread":
            nutrient_proxy = bread_proxy
        elif proxy_key == "maize":
            nutrient_proxy = maize_proxy
        else:
            return None, f"unknown alias proxy {proxy_key!r} for {key!r}"

        return (
            _scale_food_item(
                english=alias["canonical"],
                kinyarwanda=str(nutrient_proxy.get("kinyarwanda") or ""),
                category=str(nutrient_proxy.get("category") or "Grain"),
                preparation_method=str(
                    nutrient_proxy.get("preparation_method") or "boiled"
                ),
                k_per_100=float(nutrient_proxy["potassium_mg"]),
                p_per_100=float(nutrient_proxy["phosphorus_mg"]),
                pro_per_100=float(nutrient_proxy["protein_g"]),
                na_per_100=float(nutrient_proxy["sodium_mg"]),
                grams=grams,
                occasion=occasion,
            ),
            None,
        )

    if pool.empty:
        return None, f"food {name!r} not in pool"

    en = pool["english"].fillna("").str.lower().str.strip()
    kin = (
        pool["kinyarwanda"].fillna("").str.lower().str.strip()
        if "kinyarwanda" in pool.columns
        else pd.Series([""] * len(pool), index=pool.index)
    )
    matches = pool[(en == key) | (kin == key)]
    if matches.empty:
        return None, f"food {name!r} not in occasion pool (hallucinated or wrong occasion)"

    row = matches.iloc[0]
    category = str(row.get("category") or "Other")
    meal_type = str(row.get("meal_type") or "Any")
    english = str(row.get("english") or "").strip()
    if (
        category == resolve_exclude_category
        or meal_type == resolve_exclude_meal_type
        or fat_name_re.search(english)
    ):
        return None, f"cooking fat/oil not allowed as eaten food: {english!r}"

    return (
        _scale_food_item(
            english=english,
            kinyarwanda=str(row.get("kinyarwanda") or "").strip(),
            category=category,
            preparation_method=str(row.get("preparation_method") or "").strip(),
            k_per_100=float(row.get("potassium_mg") or 0),
            p_per_100=float(row.get("phosphorus_mg") or 0),
            pro_per_100=float(row.get("protein_g") or 0),
            na_per_100=float(row.get("sodium_mg") or 0),
            grams=grams,
            occasion=occasion,
        ),
        None,
    )


def _tea_name_matches(name: str, mode: str) -> bool:
    """Apply MEAL_RULES_GLOBAL['tea_name_match'] heuristic to a normalized name."""
    if mode == "exact_or_prefix_or_word":
        return name == "tea" or name.startswith("tea ") or " tea" in f" {name} "
    # Unknown modes must not silently loosen matching
    raise ValueError(f"unsupported tea_name_match mode: {mode!r}")


def _validate_occasion_suggestion(
    data: dict[str, Any] | None,
    occasion: str,
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    protein_limit_g: float,
    allowed_foods: pd.DataFrame | None = None,
) -> tuple[bool, list[dict[str, Any]] | str]:
    """
    Validate AI JSON for one occasion against the prompt allowed list.
    Returns (True, food_items) or (False, reason).
    """
    rules = OCCASION_RULES[occasion]
    nutrient_caps = rules["nutrient_caps"]
    require_protein_exactly_one_if_gated = rules["require_protein_exactly_one_if_gated"]
    pairings = rules["pairings"]
    # foods_count_range is global (identical for every occasion today)
    foods_lo, foods_hi = MEAL_RULES_GLOBAL["foods_count_range"]
    k_gate_frac = float(MEAL_RULES_GLOBAL["item_k_gate_frac"])
    p_gate_frac = float(MEAL_RULES_GLOBAL["item_p_gate_frac"])
    tea_name_match = MEAL_RULES_GLOBAL["tea_name_match"]

    if not data or not isinstance(data, dict):
        return False, "missing or invalid JSON object"

    if str(data.get("occasion") or "").strip() != occasion:
        return False, f"occasion mismatch: got {data.get('occasion')!r}, expected {occasion!r}"

    foods_raw = data.get("foods")
    if not isinstance(foods_raw, list) or not (foods_lo <= len(foods_raw) <= foods_hi):
        return False, "foods must be a list of 1–4 items"

    # Same inventory the model was shown (quota list). Fallback to occasion pool
    # only if caller did not pass allowed_foods (should not happen on structured path).
    if allowed_foods is not None and not allowed_foods.empty:
        pool = allowed_foods
    else:
        pool = _occasion_food_pool(safe_foods, occasion)

    maize_proxy = _lookup_maize_proxy(safe_foods)
    resolved: list[dict[str, Any]] = []

    for item in foods_raw:
        if not isinstance(item, dict):
            return False, "food item is not an object"
        name = item.get("name")
        grams = item.get("grams")
        try:
            grams_f = float(grams)
        except (TypeError, ValueError):
            return False, f"invalid grams for {name!r}"

        food_item, err = _resolve_name_to_item(
            str(name or ""),
            grams_f,
            occasion,
            pool,
            maize_proxy,
        )
        if err or food_item is None:
            return False, err or "resolve failed"

        min_g, max_g = _portion_bounds_for_name(food_item["english"])
        if not (min_g <= grams_f <= max_g):
            return False, f"grams out of range for {name!r}: {grams_f}"

        resolved.append(food_item)

    names_lower = {_normalize_food_key(f["english"]) for f in resolved}

    def _pairing_name_present(target: str) -> bool:
        if target == "tea":
            return any(_tea_name_matches(n, tea_name_match) for n in names_lower)
        return target in names_lower

    for left, right in pairings.get("forbid_together") or []:
        if _pairing_name_present(left) and _pairing_name_present(right):
            # Same reason string as today's tea+igikoma hard reject
            return False, "tea and igikoma must not appear in the same meal"

    for req in pairings.get("require_if") or []:
        if_english = {_normalize_food_key(x) for x in req.get("if_english") or []}
        if names_lower & if_english:
            need_cat = req.get("require_category")
            if need_cat and not any(f.get("category") == need_cat for f in resolved):
                return False, str(
                    req.get("fail_reason")
                    or f"missing required {need_cat} pairing"
                )

    # Breakfast: hard cap on drink-like / liquid items (config-driven).
    max_drink_like = rules.get("max_drink_like")
    if max_drink_like is not None and occasion == "Breakfast":
        drink_like = MEAL_RULES_GLOBAL["breakfast_drink_like_english"]
        n_drinks = 0
        for f in resolved:
            key = _normalize_food_key(f["english"])
            if key in drink_like or key.startswith("tea ") or key.startswith("coffee "):
                n_drinks += 1
            elif _tea_name_matches(key, tea_name_match):
                n_drinks += 1
        if n_drinks > int(max_drink_like):
            return False, (
                f"expected at most {int(max_drink_like)} drink-like "
                f"breakfast items, got {n_drinks}"
            )

    # Protein: re-gate Meat/Fish on the allowed/pool list (K/P gates).
    # Intentionally NOT the prompt's ungated has_meat_fish check — preserve that quirk.
    if (
        require_protein_exactly_one_if_gated
        and not pool.empty
        and "category" in pool.columns
    ):
        k_gate = limits["potassium"] * k_gate_frac
        p_gate = limits["phosphorus"] * p_gate_frac
        protein_pool = pool[
            pool["category"].isin(["Meat", "Fish"])
        ].copy()
        for col in ("potassium_mg", "phosphorus_mg"):
            if col in protein_pool.columns:
                protein_pool[col] = pd.to_numeric(
                    protein_pool[col], errors="coerce"
                ).fillna(0)
        protein_available = not protein_pool[
            (protein_pool["potassium_mg"] <= k_gate)
            & (protein_pool["phosphorus_mg"] <= p_gate)
        ].empty
        if protein_available:
            n_protein = sum(
                1 for f in resolved if f.get("category") in ("Meat", "Fish")
            )
            if n_protein != 1:
                return False, (
                    f"expected exactly one Meat/Fish item, got {n_protein}"
                )

    k_cap, p_cap, pro_cap, na_cap = nutrient_caps
    total_k = sum(f["potassium_mg"] for f in resolved)
    total_p = sum(f["phosphorus_mg"] for f in resolved)
    total_pro = sum(f["protein_g"] for f in resolved)
    total_na = sum(f["sodium_mg"] for f in resolved)
    if total_k > limits["potassium"] * k_cap:
        return False, f"potassium {total_k:.0f}mg exceeds {occasion} cap"
    if total_p > limits["phosphorus"] * p_cap:
        return False, f"phosphorus {total_p:.0f}mg exceeds {occasion} cap"
    if total_pro > protein_limit_g * pro_cap:
        return False, f"protein {total_pro:.1f}g exceeds {occasion} cap"
    if total_na > limits["sodium"] * na_cap:
        return False, f"sodium {total_na:.0f}mg exceeds {occasion} cap"

    return True, resolved


def _validate_multi_option_occasion_suggestion(
    data: dict[str, Any] | None,
    occasion: str,
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    protein_limit_g: float,
    allowed_foods: pd.DataFrame | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Validate multi-option AI JSON for one occasion.
    Each option is checked independently via _validate_occasion_suggestion.
    Surviving plates are kept; failing options are logged and dropped.
    """
    if not data or not isinstance(data, dict):
        return []

    if str(data.get("occasion") or "").strip() != occasion:
        _mp_logger.warning(
            "Multi-option occasion mismatch: got %r, expected %r",
            data.get("occasion"),
            occasion,
        )
        return []

    options_raw = data.get("options")
    # Backward-compat: old single-plate schema {"occasion", "foods": [...]}
    if not isinstance(options_raw, list):
        foods_raw = data.get("foods")
        if isinstance(foods_raw, list):
            options_raw = [{"foods": foods_raw}]
        else:
            return []

    if not options_raw:
        return []

    surviving: list[list[dict[str, Any]]] = []
    for idx, option in enumerate(options_raw[:3]):
        if isinstance(option, dict):
            foods_raw = option.get("foods")
        elif isinstance(option, list):
            # tolerate bare food arrays inside options
            foods_raw = option
        else:
            _mp_logger.warning(
                "Multi-option %s option[%s] is not an object/list — dropped",
                occasion,
                idx,
            )
            continue

        ok, payload = _validate_occasion_suggestion(
            {"occasion": occasion, "foods": foods_raw},
            occasion,
            safe_foods,
            limits,
            protein_limit_g,
            allowed_foods=allowed_foods,
        )
        if ok and isinstance(payload, list) and payload:
            surviving.append(payload)
        else:
            reason = payload if isinstance(payload, str) else "unknown"
            _mp_logger.warning(
                "Multi-option %s option[%s] failed validation: %s",
                occasion,
                idx,
                reason,
            )

    return surviving


_MEAL_OCCASION_RULES = (
    "MEAL RULES:\n"
    "- igikoma (maize porridge) is a BREAKFAST food only.\n"
    "- ugali (ubugali) is a LUNCH or DINNER food only, never breakfast.\n"
    "- Do not serve tea and igikoma in the same meal.\n"
    "- Ghee is a cooking ingredient, not a direct pairing — never list ghee "
    "as a side item alongside a meal; only mention it if used IN cooking "
    "a dish.\n"
    "- Sugar (granulated or otherwise) must never be listed as a standalone "
    "snack item on its own. It may only appear as an addition to another "
    "food or beverage (e.g. 'tea with 5g sugar'), never as its own bullet "
    "point.\n\n"
)


def build_flan_prompt(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    retrieved_chunks: list[dict],
    safe_foods: pd.DataFrame | None = None,
    day: str | None = None,
    use_occasion_food_lists: bool = False,
) -> str:
    context = ""
    if retrieved_chunks:
        context = "\n".join([c["text"][:300] for c in retrieved_chunks[:2]])

    food_section = ""
    meal_rules = ""
    if safe_foods is not None and len(safe_foods) > 0:
        if use_occasion_food_lists:
            lists = _build_occasion_prompt_food_lists(
                safe_foods, limits, max_rows=10, day=day
            )
            food_section = (
                f"BREAKFAST-APPROPRIATE FOODS:\n{lists['breakfast']}\n\n"
                f"LUNCH/DINNER-APPROPRIATE FOODS:\n{lists['lunch_dinner']}\n\n"
                f"SNACK-APPROPRIATE FOODS:\n{lists['snack']}"
            )
            meal_rules = _MEAL_OCCASION_RULES
        else:
            food_section = _build_prompt_food_list(
                safe_foods, limits, max_rows=30
            )

    day_instruction = ""
    if day:
        day_instruction = (
            f"Give ONLY {day}'s meals (breakfast, lunch, dinner, snack) "
            f"for this plan. Do not include other days.\n\n"
        )

    foods_label = (
        "SAFE FOODS FROM DATABASE (by meal occasion):\n"
        if use_occasion_food_lists
        else "SAFE FOODS FROM DATABASE:\n"
    )

    def _assemble_prompt(food_str: str) -> str:
        return (
            f"You are a CKD dietary advisor for Rwanda.\n\n"
            f"Patient: Stage {ckd_stage} CKD\n"
            f"Daily limits: "
            f"K={limits['potassium']}mg, "
            f"P={limits['phosphorus']}mg, "
            f"Protein={protein_limit_g:.0f}g, "
            f"Na={limits['sodium']}mg\n\n"
            f"RWANDA FOOD CONTEXT:\n"
            f"{get_rwanda_food_context(ckd_stage)}\n\n"
            f"{foods_label}"
            f"{food_str}\n\n"
            f"Clinical context:\n"
            f"{context}\n\n"
            f"Question: {message}\n\n"
            f"{day_instruction}"
            f"{meal_rules}"
            f"Only recommend Rwandan foods. "
            f"Give portions in grams. Answer:"
        )

    prompt = _assemble_prompt(food_section)
    if len(prompt) > 1800 and food_section:
        if use_occasion_food_lists:
            # Trim each occasion block independently so lunch/snack are not dropped.
            lists = _build_occasion_prompt_food_lists(
                safe_foods, limits, max_rows=8, day=day
            )
            food_section = (
                f"BREAKFAST-APPROPRIATE FOODS:\n{lists['breakfast']}\n\n"
                f"LUNCH/DINNER-APPROPRIATE FOODS:\n{lists['lunch_dinner']}\n\n"
                f"SNACK-APPROPRIATE FOODS:\n{lists['snack']}"
            )
        else:
            food_section = "\n".join(food_section.split("\n")[:15])
        prompt = _assemble_prompt(food_section)

    return prompt


_WEEK_DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_OCCASIONS = ("Breakfast", "Lunch", "Dinner", "Snack")
_OCCASION_JSON_KEY = {
    "Breakfast": "breakfast",
    "Lunch": "lunch",
    "Dinner": "dinner",
    "Snack": "snack",
}


def _build_structured_day_prompt(
    day: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    allowed_by_occasion: dict[str, pd.DataFrame],
) -> str:
    sections: list[str] = []
    for occasion in _OCCASIONS:
        pool = allowed_by_occasion[occasion]
        _k_bud, p_bud, pro_bud, _na_bud = _occasion_nutrient_budgets(
            occasion, limits, protein_limit_g
        )
        food_lines = _format_structured_food_list_rows(
            pool,
            protein_budget_g=pro_bud,
            phosphorus_budget_mg=p_bud,
        )
        rules = OCCASION_RULES[occasion]
        # Swap flat 25–300 grams rule for per-food max hint (prompt only).
        rule_parts: list[str] = []
        for r in rules["prompt_rules"]:
            if r.startswith("grams must be between"):
                rule_parts.append(
                    "grams must be between 25 and the max shown for each food "
                    "(except onion/onions: 10–25g max)."
                )
            else:
                rule_parts.append(r)
        rule_lines = "\n".join(f"- {r}" for r in rule_parts)
        meal_budget = _occasion_nutrient_budget_line(
            occasion, limits, protein_limit_g
        )
        # Preserve Lunch/Dinner "exactly one Meat/Fish" only when list has any
        # (same ungated has_meat_fish quirk as single-occasion prompt — validate still re-gates)
        if occasion in ("Lunch", "Dinner"):
            has_mf = (
                not pool.empty
                and "category" in pool.columns
                and bool(pool["category"].isin(["Meat", "Fish"]).any())
            )
            if not has_mf:
                rule_lines = "\n".join(
                    f"- {r}"
                    for r in rule_parts
                    if "Meat or Fish" not in r
                )
            else:
                # Diversity without dropping required protein (weekly = one plate,
                # but model still sometimes omits Meat/Fish for "variety").
                rule_lines = (
                    rule_lines
                    + "\n- Do not omit Meat/Fish when required — vary protein "
                    "choice (chicken vs beef vs fish) and sides instead."
                )
        elif occasion == "Breakfast":
            staple = _breakfast_recommended_staple(day)
            if staple == "bread":
                rule_lines = (
                    rule_lines
                    + "\n- Today's recommended staple is bread — prefer it "
                    "for the Grain/staple choice (igikoma remains allowed)."
                )
            elif staple == "igikoma":
                rule_lines = (
                    rule_lines
                    + "\n- Today's recommended staple is igikoma — prefer it "
                    "for the Grain/staple choice (bread remains allowed)."
                )
        sections.append(
            f"### {occasion}\n"
            f"{meal_budget}\n"
            f"ALLOWED FOODS (pick only from this list; "
            f"use grams at or under each food's max):\n{food_lines}\n"
            f"Rules for {occasion}:\n{rule_lines}"
        )

    schema = (
        '{"day": "' + day + '", '
        '"breakfast": {"foods": [{"name": "<food>", "grams": <number>}]}, '
        '"lunch": {"foods": [...]}, '
        '"dinner": {"foods": [...]}, '
        '"snack": {"foods": [...]}}'
    )
    return (
        "You are a CKD dietary advisor for Rwanda. "
        "Respond with a single JSON object only (no markdown).\n\n"
        f"Patient stage: {ckd_stage}\n"
        f"Daily limits: K={limits['potassium']}mg, "
        f"P={limits['phosphorus']}mg, "
        f"Protein={protein_limit_g:.0f}g, Na={limits['sodium']}mg\n\n"
        f"Plan ONLY for {day}. Include all four occasions.\n"
        "Each occasion: 1–4 foods, names must appear on that occasion's "
        "allowed list, stay under that occasion's FOR THIS MEAL ONLY budget, "
        "and use grams at or under each food's listed max.\n\n"
        + "\n\n".join(sections)
        + f"\n\nJSON schema exactly:\n{schema}\n"
    )


def _format_occasion_block(occasion: str, items: list[dict[str, Any]]) -> str:
    lines = [f"{occasion}:"]
    for f in items:
        name = str(f.get("english") or "").strip()
        grams = f.get("portion_grams")
        if name and grams is not None:
            lines.append(f"- {name} ({grams:g}g)")
        elif name:
            lines.append(f"- {name}")
    return "\n".join(lines)


def _format_day_block(day: str, plan: dict[str, list[dict[str, Any]]]) -> str:
    parts = [f"**{day}**"]
    for occasion in _OCCASIONS:
        items = plan.get(occasion) or []
        if items:
            parts.append(_format_occasion_block(occasion, items))
    return "\n".join(parts)


def _generate_structured_day_plan(
    day: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    safe_foods: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """
    One Groq JSON call for all 4 occasions; validate each with
    _validate_occasion_suggestion; per-occasion build_meal_plan fallback.
    """
    allowed_by_occasion: dict[str, pd.DataFrame] = {
        occ: _build_structured_allowed_foods(
            safe_foods, occ, limits, max_rows=20, day=day
        )
        for occ in _OCCASIONS
    }

    fallback_full = build_meal_plan(
        safe_foods, limits, protein_limit_g, day=day
    )

    prompt = _build_structured_day_prompt(
        day=day,
        ckd_stage=ckd_stage,
        limits=limits,
        protein_limit_g=protein_limit_g,
        allowed_by_occasion=allowed_by_occasion,
    )

    raw: str | None = None
    try:
        try:
            raw = query_llm_json(
                prompt, max_tokens=700, raise_on_rate_limit=True
            )
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            _mp_logger.warning(
                "Rate limited on structured day %s, retrying once after 1.5s: %s",
                day,
                e,
            )
            time.sleep(1.5)
            raw = query_llm_json(
                prompt, max_tokens=700, raise_on_rate_limit=False
            )
    except Exception as e:
        _mp_logger.warning("Structured weekly day LLM failed for %s: %s", day, e)
        raw = None

    data: dict[str, Any] | None = None
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError as e:
            _mp_logger.warning(
                "Structured day JSON parse failed for %s: %s | snippet=%r",
                day,
                e,
                raw[:200],
            )

    # Total parse/LLM failure → all 4 occasions from build_meal_plan
    if data is None:
        return {occ: list(fallback_full.get(occ) or []) for occ in _OCCASIONS}

    day_plan: dict[str, list[dict[str, Any]]] = {}
    for occasion in _OCCASIONS:
        key = _OCCASION_JSON_KEY[occasion]
        block = data.get(key)
        foods_raw = None
        if isinstance(block, dict):
            foods_raw = block.get("foods")
        elif isinstance(block, list):
            # tolerate model omitting the {"foods": ...} wrapper
            foods_raw = block

        ai_payload = (
            {"occasion": occasion, "foods": foods_raw}
            if isinstance(foods_raw, list)
            else None
        )
        ok, payload = _validate_occasion_suggestion(
            ai_payload,
            occasion,
            safe_foods,
            limits,
            protein_limit_g,
            allowed_foods=allowed_by_occasion[occasion],
        )
        if ok and isinstance(payload, list) and payload:
            day_plan[occasion] = payload
        else:
            reason = payload if isinstance(payload, str) else "missing/invalid"
            _mp_logger.warning(
                "Structured day %s / %s validation failed: %s",
                day,
                occasion,
                reason,
            )
            day_plan[occasion] = list(fallback_full.get(occasion) or [])

    return day_plan


def _generate_weekly_llm_plan(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    retrieved_chunks: list[dict],
    safe_foods: pd.DataFrame,
) -> str:
    """Seven day-scoped structured JSON calls stitched into weekly answer text."""
    day_sections: list[str] = []
    for day in _WEEK_DAYS:
        day_plan = _generate_structured_day_plan(
            day=day,
            ckd_stage=ckd_stage,
            limits=limits,
            protein_limit_g=protein_limit_g,
            safe_foods=safe_foods,
        )
        day_sections.append(_format_day_block(day, day_plan))
    return "\n\n".join(day_sections)


def _apply_flan_enhancement(template_answer: str, llm_answer: str | None) -> str:
    if not llm_answer or len(llm_answer) < 20:
        return template_answer

    if _contains_nutrient_values(llm_answer):
        sentences = llm_answer.split(".")
        intro = ". ".join(s.strip() for s in sentences[:2] if s.strip())
        if intro:
            intro = intro.rstrip(".") + "."
        return f"{intro}\n\n{template_answer}" if intro else template_answer

    return llm_answer


def _clean_llm_response(text: str) -> str:
    """
    Remove empty numbered list items like '2.' or '3. '
    that appear without content, typically when LLM switches
    from list to table format.
    """
    text = re.sub(
        r"^\d+\.\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )
    return text.strip()


def _filter_high_risk_mentions(text: str, ckd_stage: str) -> str:
    """
    For Stage 3b and Stage 4,
    remove sentences that mention
    high-risk foods from the
    LLM conversational text.
    Only applies to stages where
    these foods are restricted.
    """
    restricted_stages = ["G3b", "G3a", "G4"]
    if ckd_stage not in restricted_stages:
        return text

    original = text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    safe_sentences = []
    for sentence in sentences:
        s_lower = sentence.lower()
        contains_risk = any(food in s_lower for food in HIGH_RISK_FOODS)
        if not contains_risk:
            safe_sentences.append(sentence)
    filtered = " ".join(safe_sentences)

    if original.strip() and not filtered.strip():
        _mp_logger.warning(
            "_filter_high_risk_mentions removed entire response for stage %s "
            "— falling back to unfiltered text",
            ckd_stage,
        )
        return original

    return filtered


def build_meal_plan(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    protein_limit_g: float,
    day: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build a balanced meal plan from Rwandan safe foods.

    Priority order per occasion:
    Breakfast: protein first (Egg/Dairy) then grain
    Lunch/Dinner: protein first (Meat/Fish), then starch, then vegetable
    Snack: fruit first, then dairy

    Optional day: day-seeded Breakfast staple steer (bread vs igikoma)
    for weekly fallbacks. day=None keeps prior Grain DB-only behavior.
    """
    import random

    fat_name_re = re.compile(MEAL_RULES_GLOBAL["fat_name_re"], re.IGNORECASE)
    resolve_exclude_category = MEAL_RULES_GLOBAL["resolve_exclude_category"]
    resolve_exclude_meal_type = MEAL_RULES_GLOBAL["resolve_exclude_meal_type"]
    protein_pick_categories = MEAL_RULES_GLOBAL["protein_pick_categories"]
    protein_pick_allowance_frac = float(
        MEAL_RULES_GLOBAL["protein_pick_allowance_frac"]
    )
    snack_only_english = MEAL_RULES_GLOBAL["snack_only_english"]
    bread_proxy = MEAL_RULES_GLOBAL["bread_proxy"]
    synthetic_specs = MEAL_RULES_GLOBAL["synthetic_specs"]
    recommended_staple = _breakfast_recommended_staple(day)

    # Budgets from per-occasion nutrient_caps (same fractions as structured path)
    K_BUDGET: dict[str, float] = {}
    PHOSPHORUS_BUDGET: dict[str, float] = {}
    PROTEIN_BUDGET: dict[str, float] = {}
    SODIUM_BUDGET: dict[str, float] = {}
    for _occ, _rules in OCCASION_RULES.items():
        k_cap, ph_cap, pro_cap, na_cap = _rules["nutrient_caps"]
        K_BUDGET[_occ] = limits["potassium"] * k_cap
        PHOSPHORUS_BUDGET[_occ] = limits["phosphorus"] * ph_cap
        PROTEIN_BUDGET[_occ] = protein_limit_g * pro_cap
        SODIUM_BUDGET[_occ] = limits["sodium"] * na_cap

    foods = safe_foods.copy()
    if "meal_type" not in foods.columns:
        foods["meal_type"] = "Any"
    foods["meal_type"] = foods["meal_type"].fillna("Any").astype(str)

    def pick_one(
        foods_df: pd.DataFrame,
        categories: list[str],
        k_remaining: float,
        protein_remaining: float,
        ph_remaining: float,
        na_remaining: float,
        portion: float,
        exclude_names: list[str],
        allowed_meal_types: list[str],
    ) -> dict | None:
        """Pick one food from categories that fits within K/protein/P/Na budgets."""
        candidates = foods_df[
            foods_df["category"].isin(categories)
            & foods_df["meal_type"].isin(allowed_meal_types)
            & ~foods_df["english"].isin(exclude_names)
        ].copy()

        if candidates.empty:
            return None

        # Same cooking-fat / honey ban as structured path
        fat_name = candidates["english"].fillna("").astype(str).apply(
            lambda s: bool(fat_name_re.search(s))
        )
        candidates = candidates[~fat_name]
        if "category" in candidates.columns:
            candidates = candidates[
                candidates["category"] != resolve_exclude_category
            ]
        if "meal_type" in candidates.columns:
            candidates = candidates[
                candidates["meal_type"] != resolve_exclude_meal_type
            ]

        if candidates.empty:
            return None

        candidates = candidates.sample(frac=1, random_state=random.randint(0, 9999))

        is_protein_pick = bool(set(categories) & protein_pick_categories)
        protein_allowance = protein_remaining * (
            protein_pick_allowance_frac if is_protein_pick else 1.0
        )

        for _, food in candidates.iterrows():
            raw_english = str(food.get("english") or "").strip()
            min_g, max_g = _portion_bounds_for_name(raw_english)
            k_per_100 = float(food.get("potassium_mg", 0) or 0)
            protein_per_100 = float(food.get("protein_g", 0) or 0)
            ph_per_100 = float(food.get("phosphorus_mg", 0) or 0)
            na_per_100 = float(food.get("sodium_mg", 0) or 0)

            fit_portion = portion
            if k_per_100 > 0:
                fit_portion = min(fit_portion, k_remaining / k_per_100 * 100)
            if protein_per_100 > 0:
                fit_portion = min(
                    fit_portion, protein_allowance / protein_per_100 * 100
                )
            if ph_per_100 > 0:
                fit_portion = min(fit_portion, ph_remaining / ph_per_100 * 100)
            if na_per_100 > 0:
                fit_portion = min(fit_portion, na_remaining / na_per_100 * 100)
            fit_portion = min(fit_portion, max_g)
            if fit_portion < min_g:
                continue

            food_k = k_per_100 * fit_portion / 100
            food_protein = protein_per_100 * fit_portion / 100
            food_ph = ph_per_100 * fit_portion / 100
            food_na = na_per_100 * fit_portion / 100

            if food_protein > protein_remaining:
                continue
            if food_ph > ph_remaining:
                continue
            if food_na > na_remaining:
                continue

            exclude_names.append(raw_english)
            return {
                "english": _display_english_name(raw_english),
                "kinyarwanda": str(food.get("kinyarwanda") or "").strip(),
                "preparation_method": str(food.get("preparation_method") or "").strip(),
                "portion_grams": round(fit_portion, 1),
                "meal_occasion": "",
                "potassium_mg": round(food_k, 1),
                "phosphorus_mg": round(food_ph, 1),
                "protein_g": round(food_protein, 1),
                "sodium_mg": round(food_na, 1),
                "category": str(food.get("category") or "Other"),
            }
        return None

    plan: dict[str, list[dict[str, Any]]] = {}

    for occasion in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        rules = OCCASION_RULES[occasion]
        portion = float(rules["default_portion_g"])
        k_budget = float(K_BUDGET[occasion])
        protein_budget = float(PROTEIN_BUDGET[occasion])
        ph_budget = float(PHOSPHORUS_BUDGET[occasion])
        na_budget = float(SODIUM_BUDGET[occasion])
        groups = rules["groups"]
        allowed_meal_types = rules["meal_types"]
        fallback_repair = rules.get("fallback_repair") or []
        picked: list[dict] = []
        k_used = 0.0
        protein_used = 0.0
        ph_used = 0.0
        na_used = 0.0
        used_names: list[str] = []

        occasion_foods = foods
        if occasion in ("Breakfast", "Lunch", "Dinner"):
            # Sugar cane via snack_only_english; Breakfast extras via exclude_english
            eng_norm = foods["english"].fillna("").str.lower().str.strip()
            excluded = set(snack_only_english) | set(rules.get("exclude_english") or ())
            occasion_foods = foods[~eng_norm.isin(excluded)]

        for group_categories in groups:
            food = None
            # Weekly bread/igikoma-days: try matching synthetic for Grain first.
            if (
                occasion == "Breakfast"
                and group_categories == ["Grain"]
                and recommended_staple in ("bread", "igikoma")
            ):
                spec = synthetic_specs[recommended_staple]
                if recommended_staple == "bread":
                    meta = {**bread_proxy}
                else:
                    meta = {**_lookup_maize_proxy(safe_foods)}
                if "category" in spec:
                    meta["category"] = spec["category"]
                if "preparation_method" in spec:
                    meta["preparation_method"] = spec["preparation_method"]
                synth_row = _synthetic_row(
                    meta,
                    english=str(spec["english"]),
                    meal_type=str(spec["meal_type"]),
                )
                food = pick_one(
                    pd.DataFrame([synth_row]),
                    group_categories,
                    k_budget - k_used,
                    protein_budget - protein_used,
                    ph_budget - ph_used,
                    na_budget - na_used,
                    portion,
                    used_names,
                    allowed_meal_types,
                )
            if food is None:
                food = pick_one(
                    occasion_foods,
                    group_categories,
                    k_budget - k_used,
                    protein_budget - protein_used,
                    ph_budget - ph_used,
                    na_budget - na_used,
                    portion,
                    used_names,
                    allowed_meal_types,
                )
            if food:
                food["meal_occasion"] = occasion
                picked.append(food)
                k_used += food["potassium_mg"]
                protein_used += food["protein_g"]
                ph_used += food["phosphorus_mg"]
                na_used += food["sodium_mg"]

        # Breakfast-only when fallback_repair includes ikivuguto_fruit
        if "ikivuguto_fruit" in fallback_repair:
            has_ikivuguto = any(
                _normalize_food_key(f["english"]) == "ikivuguto" for f in picked
            )
            has_fruit = any(f.get("category") == "Fruit" for f in picked)
            if has_ikivuguto and not has_fruit:
                fruit = pick_one(
                    occasion_foods,
                    ["Fruit"],
                    k_budget - k_used,
                    protein_budget - protein_used,
                    ph_budget - ph_used,
                    na_budget - na_used,
                    portion,
                    used_names,
                    allowed_meal_types,
                )
                if fruit:
                    fruit["meal_occasion"] = occasion
                    picked.append(fruit)
                    k_used += fruit["potassium_mg"]
                    protein_used += fruit["protein_g"]
                    ph_used += fruit["phosphorus_mg"]
                    na_used += fruit["sodium_mg"]
                else:
                    for idx in range(len(picked) - 1, -1, -1):
                        if _normalize_food_key(picked[idx]["english"]) != "ikivuguto":
                            continue
                        removed = picked.pop(idx)
                        k_used -= removed["potassium_mg"]
                        protein_used -= removed["protein_g"]
                        ph_used -= removed["phosphorus_mg"]
                        na_used -= removed["sodium_mg"]

                    used_names[:] = [
                        n
                        for n in used_names
                        if _normalize_food_key(n) != "sour milk"
                    ]
                    if "sour milk" not in used_names:
                        used_names.append("sour milk")

                    replacement = pick_one(
                        occasion_foods,
                        ["Egg", "Dairy"],
                        k_budget - k_used,
                        protein_budget - protein_used,
                        ph_budget - ph_used,
                        na_budget - na_used,
                        portion,
                        used_names,
                        allowed_meal_types,
                    )
                    if replacement:
                        replacement["meal_occasion"] = occasion
                        picked.append(replacement)
                        k_used += replacement["potassium_mg"]
                        protein_used += replacement["protein_g"]
                        ph_used += replacement["phosphorus_mg"]
                        na_used += replacement["sodium_mg"]

        plan[occasion] = picked

    return plan


def get_safe_alternatives(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
) -> list[dict[str, Any]]:
    safe = safe_foods[
        (safe_foods["potassium_mg"] <= limits["potassium"] * 0.1)
        & (safe_foods["protein_g"] <= 3.0)
    ].nsmallest(8, "potassium_mg")

    return [
        {
            "english": str(r.get("english") or "").strip(),
            "portion_grams": 150,
            "meal_occasion": "Lunch",
            "potassium_mg": float(r.get("potassium_mg") or 0),
            "phosphorus_mg": float(r.get("phosphorus_mg") or 0),
            "protein_g": float(r.get("protein_g") or 0),
            "sodium_mg": float(r.get("sodium_mg") or 0),
            "category": str(r.get("category") or "Other"),
        }
        for _, r in safe.iterrows()
    ]


def find_food_in_message(message: str, foods: pd.DataFrame) -> pd.DataFrame | None:
    msg_lower = message.lower()
    msg_tokens = set(msg_lower.replace("?", " ").replace(",", " ").split())
    best_match: str | None = None
    best_score = -1

    for _, food in foods.iterrows():
        food_name = str(food.get("english") or "").lower().strip()
        if not food_name:
            continue

        score = -1
        if food_name in msg_lower:
            score = 1000 + len(food_name)
        elif food_name in msg_tokens:
            score = 900 + len(food_name)
        else:
            words = [w for w in food_name.split() if len(w) > 3]
            if len(words) <= 2 and words and all(w in msg_lower for w in words):
                score = 500 + len(food_name)

        if score > best_score:
            best_score = score
            best_match = food_name

    if best_match is None:
        return None

    return foods[foods["english"].str.lower().str.strip() == best_match]


def build_general_answer(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    all_foods: pd.DataFrame,
    retrieved_chunks: list[dict],
) -> str:
    msg_lower = message.lower()
    answer_parts: list[str] = []

    food_question = not any(
        w in msg_lower for w in ["what can i eat", "what should i eat", "meal plan", "foods to avoid"]
    ) and any(
        w in msg_lower
        for w in [
            "can i eat",
            "is it safe",
            "is safe",
            "can i have",
            "can i drink",
            "safe for",
            "safe for me",
            "drink",
        ]
    )

    if food_question:
        mentioned_food = find_food_in_message(message, all_foods)
        if mentioned_food is not None and not mentioned_food.empty:
            food = mentioned_food.iloc[0]
            potassium_mg = float(food.get("potassium_mg") or 0)
            k_pct = (potassium_mg / limits["potassium"]) * 100
            safe = k_pct < 15

            answer_parts.append(f"**{food['english']}** for Stage {ckd_stage}:")
            answer_parts.append(
                f"{'✓ Generally safe' if safe else '⚠ Use with caution'} in moderate portions."
            )
            answer_parts.append(
                f"\nPer 100g: Potassium {potassium_mg:.0f}mg ({k_pct:.0f}% of your daily limit), "
                f"Phosphorus {float(food.get('phosphorus_mg') or 0):.0f}mg, "
                f"Protein {float(food.get('protein_g') or 0):.1f}g"
            )
            if not safe:
                answer_parts.append(
                    f"\nKeep portions small — under 80g recommended for Stage {ckd_stage}."
                )
            return "\n".join(answer_parts)

    nutrient_topics = {
        "potassium": (
            f"**Potassium and CKD (Stage {ckd_stage}):**\n"
            f"Healthy kidneys remove excess potassium from the blood. When kidneys are damaged, "
            f"potassium builds up — this can cause dangerous heart rhythm problems "
            f"(hyperkalaemia).\n\n"
            f"Your daily limit: {limits['potassium']:.0f}mg\n"
            f"High-potassium foods to limit: bananas, potatoes, tomatoes, spinach, beans.\n"
            f"Low-potassium safe options: rice, cabbage, apples, white bread."
        ),
        "phosphorus": (
            f"**Phosphorus and CKD (Stage {ckd_stage}):**\n"
            f"Damaged kidneys cannot remove excess phosphorus. High phosphorus weakens bones "
            f"and can cause calcium deposits in blood vessels.\n\n"
            f"Your daily limit: {limits['phosphorus']:.0f}mg\n"
            f"High-phosphorus foods to limit: dairy, processed foods, cola drinks, nuts.\n"
            f"Safer options: rice, cassava, cabbage, egg whites."
        ),
        "protein": (
            f"**Protein and CKD (Stage {ckd_stage}):**\n"
            f"Too much protein creates waste products your kidneys struggle to filter, "
            f"accelerating kidney disease progression.\n\n"
            f"Your daily limit: {limits['protein_per_kg']}g per kg body weight\n"
            f"Choose lean proteins in small portions: egg whites, small fish portions.\n"
            f"Limit: large meat portions, legumes in excess."
        ),
        "sodium": (
            f"**Sodium and CKD (Stage {ckd_stage}):**\n"
            f"High sodium causes fluid retention and raises blood pressure, both harmful for "
            f"CKD progression.\n\n"
            f"Your daily limit: {limits['sodium']:.0f}mg\n"
            f"Avoid: added salt, processed foods, salty sauces, canned foods.\n"
            f"Use herbs and spices instead of salt."
        ),
    }

    for nutrient, explanation in nutrient_topics.items():
        if nutrient in msg_lower:
            return explanation

    if any(w in msg_lower for w in ["egfr", "gfr", "stage", "ckd", "kidney", "chronic"]):
        return (
            f"**CKD Stage {ckd_stage} Overview:**\n"
            f"CKD (Chronic Kidney Disease) is staged G1–G5 based on eGFR (kidney filtration "
            f"rate). Stage {ckd_stage} means your kidneys are working at reduced capacity.\n\n"
            f"At Stage {ckd_stage}, dietary management is essential to slow progression:\n"
            f"- Limit potassium to {limits['potassium']:.0f}mg/day\n"
            f"- Limit phosphorus to {limits['phosphorus']:.0f}mg/day\n"
            f"- Limit protein intake\n"
            f"- Reduce sodium\n\n"
            f"Always work with your nephrologist and dietitian for personalised guidance."
        )

    if retrieved_chunks:
        best_chunk = retrieved_chunks[0].get("text", "")
        clean = best_chunk.replace("\n\n", "\n").strip()
        return (
            f"Based on clinical guidelines for CKD Stage {ckd_stage}:\n\n"
            f"{clean[:500]}\n\n"
            f"For personalised advice, please consult your healthcare provider."
        )

    return (
        f"I'm your CKD dietary assistant for Stage {ckd_stage}. I can help with:\n"
        f"- Safe food choices for your stage\n"
        f"- Meal plans within your limits\n"
        f"- Information about potassium, phosphorus, protein and sodium\n"
        f"- Foods to avoid\n\n"
        f"Try asking: 'What can I eat for breakfast?' or 'Is banana safe for me?'"
    )


def generate_meal_recommendation(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    retrieved_chunks: list[dict],
    safe_foods: pd.DataFrame,
    all_foods: pd.DataFrame,
    uploaded_text: str | None,
    weight_kg: float,
) -> dict[str, Any]:
    msg_lower = message.lower()

    wants_meal_plan = any(
        w in msg_lower
        for w in [
            "meal plan",
            "week",
            "plan",
            "schedule",
            "what should i eat",
            "suggest meals",
            "daily menu",
            "menu",
        ]
    )
    wants_breakfast = "breakfast" in msg_lower
    wants_lunch = "lunch" in msg_lower
    wants_dinner = "dinner" in msg_lower
    wants_snack = "snack" in msg_lower
    wants_safe = any(
        w in msg_lower
        for w in [
            "what can i eat",
            "suggest a safe",
            "give me a full day",
            "safe lunch",
            "safe breakfast",
            "safe dinner",
        ]
    )
    wants_avoid = any(
        w in msg_lower
        for w in [
            "avoid",
            "cannot eat",
            "can't eat",
            "not eat",
            "bad for",
            "should i avoid",
            "foods to avoid",
        ]
    ) or (
        "dangerous" in msg_lower
        and not any(
            w in msg_lower
            for w in ["why is", "why are", "what is", "what are", "how does", "how do", "explain"]
        )
    )

    is_general_question = not any(
        [
            wants_meal_plan,
            wants_breakfast,
            wants_lunch,
            wants_dinner,
            wants_snack,
            wants_avoid,
            wants_safe,
            bool(uploaded_text),
        ]
    )

    if is_general_question:
        answer_parts = [
            build_general_answer(
                message=message,
                ckd_stage=ckd_stage,
                limits=limits,
                all_foods=all_foods,
                retrieved_chunks=retrieved_chunks,
            ),
            "\n\n⚕ *Always consult your healthcare provider or dietitian before making dietary changes.*",
        ]

        return {
            "answer": "\n".join(answer_parts),
            "sources": [
                {
                    "source": c.get("source", ""),
                    "excerpt": (c.get("text", "")[:150] + "...") if c.get("text") else "",
                }
                for c in retrieved_chunks[:2]
            ],
            "suggested_foods": [],
            "meal_plan": None,
            "meal_options": [],
        }

    answer_parts: list[str] = []
    suggested_foods: list[dict[str, Any]] = []
    meal_plan: dict[str, list[dict[str, Any]]] | None = None
    meal_options: list[list[dict[str, Any]]] = []
    structured_meal = False
    skip_free_text_enhancement = False

    # Use Rwandan foods (+ allowlisted non-Rwandan bypass) for meal plan suggestions
    rwandan_foods = (
        _filter_rwandan_with_bypass(safe_foods)
        if "is_rwandan" in safe_foods.columns
        else safe_foods
    )

    wants_food_suggestions = (
        wants_meal_plan
        or wants_breakfast
        or wants_lunch
        or wants_dinner
        or wants_snack
        or wants_safe
    )

    if wants_food_suggestions and not wants_avoid:
        single_occasion = _detect_single_occasion(
            wants_breakfast, wants_lunch, wants_dinner, wants_snack
        )

        if single_occasion:
            # Success or build_meal_plan fallback — never free-text-overwrite answer/table.
            skip_free_text_enhancement = True
            ai_data, allowed_foods = _generate_structured_occasion_suggestion(
                occasion=single_occasion,
                ckd_stage=ckd_stage,
                limits=limits,
                protein_limit_g=protein_limit_g,
                safe_foods=rwandan_foods,
            )
            passed = _validate_multi_option_occasion_suggestion(
                ai_data,
                single_occasion,
                rwandan_foods,
                limits,
                protein_limit_g,
                allowed_foods=allowed_foods,
            )
            if passed:
                meal_options = passed
                meal_plan = {single_occasion: list(passed[0])}
                suggested_foods = list(passed[0])
                structured_meal = True
                answer_parts = [
                    f"Here are {single_occasion.lower()} ideas for Stage {ckd_stage}, "
                    f"keeping you within your daily nutrient limits:"
                ]
            else:
                snippet = ""
                if isinstance(ai_data, dict):
                    snippet = str(ai_data.get("_raw_snippet") or "")[:200]
                _mp_logger.warning(
                    "Structured occasion multi-option validation failed for %s "
                    "(no surviving options) | snippet=%r",
                    single_occasion,
                    snippet,
                )
                full_plan = build_meal_plan(rwandan_foods, limits, protein_limit_g)
                occasion_foods = full_plan.get(single_occasion) or []
                meal_plan = {single_occasion: occasion_foods}
                suggested_foods = list(occasion_foods)
                meal_options = [list(occasion_foods)] if occasion_foods else []
                if occasion_foods:
                    answer_parts = [
                        f"Here are {single_occasion.lower()} ideas for Stage {ckd_stage}, "
                        f"keeping you within your daily nutrient limits:"
                    ]
                else:
                    if retrieved_chunks:
                        answer_parts.append(
                            f"Based on local clinical guideline documents for CKD Stage {ckd_stage}:"
                        )
                    answer_parts.append(
                        "\n**Your daily limits:**\n"
                        f"- Potassium: {limits['potassium']:.0f} mg/day\n"
                        f"- Phosphorus: {limits['phosphorus']:.0f} mg/day\n"
                        f"- Protein: {protein_limit_g:.1f} g/day "
                        f"({limits['protein_per_kg']} g/kg × {weight_kg:.0f} kg)\n"
                        f"- Sodium: {limits['sodium']:.0f} mg/day"
                    )
        else:
            # Vague / multi-occasion — unchanged deterministic full-day plan
            meal_plan = build_meal_plan(rwandan_foods, limits, protein_limit_g)
            for foods in meal_plan.values():
                for food in foods:
                    suggested_foods.append(food)

            if meal_plan and any(foods for foods in meal_plan.values()):
                answer_parts = [
                    f"Here's a balanced meal plan for Stage {ckd_stage}, "
                    f"keeping you within your daily nutrient limits:"
                ]
            else:
                if retrieved_chunks:
                    answer_parts.append(
                        f"Based on local clinical guideline documents for CKD Stage {ckd_stage}:"
                    )
                answer_parts.append(
                    "\n**Your daily limits:**\n"
                    f"- Potassium: {limits['potassium']:.0f} mg/day\n"
                    f"- Phosphorus: {limits['phosphorus']:.0f} mg/day\n"
                    f"- Protein: {protein_limit_g:.1f} g/day ({limits['protein_per_kg']} g/kg × {weight_kg:.0f} kg)\n"
                    f"- Sodium: {limits['sodium']:.0f} mg/day"
                )
    else:
        if retrieved_chunks:
            answer_parts.append(
                f"Based on local clinical guideline documents for CKD Stage {ckd_stage}:"
            )

        answer_parts.append(
            "\n**Your daily limits:**\n"
            f"- Potassium: {limits['potassium']:.0f} mg/day\n"
            f"- Phosphorus: {limits['phosphorus']:.0f} mg/day\n"
            f"- Protein: {protein_limit_g:.1f} g/day ({limits['protein_per_kg']} g/kg × {weight_kg:.0f} kg)\n"
            f"- Sodium: {limits['sodium']:.0f} mg/day"
        )

        high_k = safe_foods[safe_foods["potassium_mg"] > limits["potassium"] * 0.15].nlargest(
            10, "potassium_mg"
        )
        answer_parts.append(f"\n**Foods to limit (potassium-heavy) for Stage {ckd_stage}:**")
        for _, food in high_k.iterrows():
            answer_parts.append(
                f"- {food['english']}: {float(food.get('potassium_mg') or 0):.0f}mg potassium per 100g"
            )

    if uploaded_text:
        if meal_plan and any(foods for foods in meal_plan.values()):
            answer_parts.append(
                "\nI also reviewed the text you uploaded and matched CKD-safe options from the local food database."
            )
        else:
            answer_parts.append(
                "\n**Analysis of your uploaded list:**\n"
                "I reviewed the text you provided and pulled CKD-safe options from the local food database."
            )
            for alt in get_safe_alternatives(rwandan_foods, limits)[:5]:
                answer_parts.append(
                    f"- {alt['english']} ({alt['portion_grams']}g): safe candidate for Stage {ckd_stage}"
                )
                suggested_foods.append(alt)

    answer_parts.append(
        "\n\n⚕ *Always consult your healthcare provider or dietitian before making dietary changes.*"
    )

    sources = [
        {
            "source": c.get("source", ""),
            "excerpt": (c.get("text", "")[:150] + "...") if c.get("text") else "",
        }
        for c in retrieved_chunks[:3]
    ]

    return {
        "answer": "\n".join(answer_parts),
        "sources": sources,
        "suggested_foods": suggested_foods,
        "meal_plan": meal_plan,
        "meal_options": meal_options,
        "structured_meal": structured_meal,
        "skip_free_text_enhancement": skip_free_text_enhancement,
    }


@router.post("/chat")
async def meal_planner_chat(
    request: MealPlannerRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    _ = user_id
    retriever = get_retriever()

    limits = KDOQI_DAILY_LIMITS.get(request.ckd_stage, KDOQI_DAILY_LIMITS["G3b"])
    protein_limit_g = limits["protein_per_kg"] * request.weight_kg

    query = (
        f"{request.message}\n"
        f"CKD stage {request.ckd_stage}\n"
        f"potassium limit {limits['potassium']}mg\n"
        f"phosphorus limit {limits['phosphorus']}mg\n"
        f"protein limit {protein_limit_g:.1f}g\n"
        f"sodium limit {limits['sodium']}mg\n"
    )
    if request.uploaded_text:
        query += f"\n{request.uploaded_text}"

    chunks = retriever.retrieve(query, top_k=5, patient_stage=request.ckd_stage)

    food_db = _load_food_db()
    stage_number = _stage_num(request.ckd_stage)
    safe_foods = food_db[food_db["ckd_stage_safe"].apply(lambda x: _is_stage_safe(str(x), stage_number))]

    # Clinical forbidden foods per stage — removes foods that pass the stage_safe
    # range check but are clinically restricted due to high potassium/phosphorus
    _FORBIDDEN_BY_STAGE: dict[str, list[str]] = {
        "G3A": [
            "groundnuts", "soybeans", "soya",
            "spinach", "cassava leaves", "isombe",
        ],
        "G3B": [
            "beans", "peas", "banana", "matoke", "plantains",
            "avocado", "avocados", "irish potatoes",
            "cassava leaves", "isombe", "groundnuts", "soybeans",
            "soya", "spinach", "oranges", "passion fruit",
        ],
        "G4": [
            "beans", "peas", "banana", "matoke", "plantains",
            "avocado", "avocados", "irish potatoes",
            "cassava leaves", "isombe", "groundnuts", "soybeans",
            "soya", "spinach", "sweet potatoes", "yams",
            "oranges", "passion fruit", "milk", "pumpkin",
        ],
    }

    _stage_key = request.ckd_stage.upper()
    _forbidden = _FORBIDDEN_BY_STAGE.get(_stage_key, [])

    if _forbidden:
        safe_foods = safe_foods[
            ~_forbidden_english_mask(safe_foods["english"], _forbidden)
        ]

    if "is_rwandan" in safe_foods.columns:
        safe_foods = _filter_rwandan_with_bypass(safe_foods)

    result = generate_meal_recommendation(
        message=request.message,
        ckd_stage=request.ckd_stage,
        limits=limits,
        protein_limit_g=protein_limit_g,
        retrieved_chunks=chunks,
        safe_foods=safe_foods,
        all_foods=food_db,
        uploaded_text=request.uploaded_text,
        weight_kg=request.weight_kg,
    )

    # Groq LLM is called only on user chat requests — never at startup.
    template_answer = result.get("answer", "")

    is_weekly = any(
        w in request.message.lower()
        for w in [
            "week",
            "weekly",
            "7 day",
            "seven day",
            "meal plan for the week",
            "weekly meal plan",
        ]
    )

    if is_weekly:
        weekly_raw = _generate_weekly_llm_plan(
            message=request.message,
            ckd_stage=request.ckd_stage,
            limits=limits,
            protein_limit_g=protein_limit_g,
            retrieved_chunks=chunks,
            safe_foods=safe_foods,
        )
        if weekly_raw.strip():
            result["answer"] = _filter_high_risk_mentions(
                _clean_llm_response(weekly_raw),
                request.ckd_stage,
            )
        else:
            result["answer"] = template_answer
        suggested_foods_for_response: list[dict[str, Any]] = []
        result["meal_plan"] = None
    else:
        if result.get("skip_free_text_enhancement"):
            # Single-occasion structured success OR build_meal_plan fallback —
            # keep answer_parts / suggested_foods as-is (no free-text overwrite).
            suggested_foods_for_response = result.get("suggested_foods") or []
        else:
            flan_prompt = build_flan_prompt(
                message=request.message,
                ckd_stage=request.ckd_stage,
                limits=limits,
                protein_limit_g=protein_limit_g,
                retrieved_chunks=chunks,
                safe_foods=safe_foods,
            )
            llm_raw = query_llm(flan_prompt, history=request.conversation_history)
            result["answer"] = _filter_high_risk_mentions(
                _clean_llm_response(
                    _apply_flan_enhancement(template_answer, llm_raw)
                ),
                request.ckd_stage,
            )
            suggested_foods_for_response = result.get("suggested_foods") or []

    sources = result.get("sources") or []
    result["sources"] = list({s["source"]: s for s in sources}.values())
    result["suggested_foods"] = suggested_foods_for_response
    return result

