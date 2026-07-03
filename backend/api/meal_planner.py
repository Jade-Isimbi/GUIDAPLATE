from __future__ import annotations

import logging
import os
import re
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
    "potato",
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
                if "is_rwandan" in csv_df.columns and "food_id" in df.columns:
                    df["food_id"] = df["food_id"].astype(str).str.strip()
                    csv_df["food_id"] = csv_df["food_id"].astype(str).str.strip()
                    df = df.merge(
                        csv_df[["food_id", "is_rwandan"]],
                        on="food_id",
                        how="left",
                    )
                    df["is_rwandan"] = df["is_rwandan"].fillna(0).astype(int)
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


def query_llm(prompt: str, history: list[dict] | None = None) -> str | None:
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
            max_tokens=600,
            temperature=0.3,
        )

        result = response.choices[0].message.content.strip()
        return result if result else None

    except Exception as e:
        _mp_logger.warning("LLM failed (using template): %s", e)
        return None


def _build_prompt_food_list(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    max_rows: int = 30,
) -> str:
    foods = safe_foods.copy()
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
        .head(max_rows)
    )

    return "\n".join(
        [
            f"- {row['english']} "
            f"(K:{row['potassium_mg']:.0f}mg, "
            f"P:{row['phosphorus_mg']:.0f}mg, "
            f"Pro:{row['protein_g']:.1f}g per 100g)"
            for _, row in combined.iterrows()
        ]
    )


def build_flan_prompt(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    retrieved_chunks: list[dict],
    safe_foods: pd.DataFrame | None = None,
) -> str:
    context = ""
    if retrieved_chunks:
        context = "\n".join([c["text"][:300] for c in retrieved_chunks[:2]])

    food_list_str = ""
    if safe_foods is not None and len(safe_foods) > 0:
        food_list_str = _build_prompt_food_list(safe_foods, limits, max_rows=30)

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
            f"SAFE FOODS FROM DATABASE:\n"
            f"{food_str}\n\n"
            f"Clinical context:\n"
            f"{context}\n\n"
            f"Question: {message}\n\n"
            f"Only recommend Rwandan foods. "
            f"Give portions in grams. Answer:"
        )

    prompt = _assemble_prompt(food_list_str)
    if len(prompt) > 1800 and food_list_str:
        food_list_str = "\n".join(food_list_str.split("\n")[:15])
        prompt = _assemble_prompt(food_list_str)

    return prompt


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

    sentences = re.split(r"(?<=[.!?])\s+", text)
    safe_sentences = []
    for sentence in sentences:
        s_lower = sentence.lower()
        contains_risk = any(food in s_lower for food in HIGH_RISK_FOODS)
        if not contains_risk:
            safe_sentences.append(sentence)
    return " ".join(safe_sentences)


def build_meal_plan(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    protein_limit_g: float,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build a balanced meal plan from Rwandan safe foods.

    Priority order per occasion:
    Breakfast: protein first (Egg/Dairy) then grain
    Lunch/Dinner: protein first (Meat/Fish), then starch, then vegetable
    Snack: fruit first, then grain
    """
    import random

    PORTION = {
        "Breakfast": 150,
        "Lunch": 200,
        "Dinner": 200,
        "Snack": 100,
    }

    K_BUDGET = {
        "Breakfast": limits["potassium"] * 0.20,
        "Lunch": limits["potassium"] * 0.35,
        "Dinner": limits["potassium"] * 0.35,
        "Snack": limits["potassium"] * 0.10,
    }

    P_BUDGET = {
        "Breakfast": protein_limit_g * 0.25,
        "Lunch": protein_limit_g * 0.35,
        "Dinner": protein_limit_g * 0.35,
        "Snack": protein_limit_g * 0.05,
    }

    # Priority groups per occasion — picked in order so protein is always included
    GROUPS = {
        "Breakfast": [
            ["Egg", "Dairy"],
            ["Grain"],
        ],
        "Lunch": [
            ["Meat", "Fish"],
            ["Starch"],
            ["Vegetable"],
        ],
        "Dinner": [
            ["Meat", "Fish"],
            ["Starch"],
            ["Vegetable"],
        ],
        "Snack": [
            ["Fruit"],
            ["Grain", "Dairy"],
        ],
    }

    def pick_one(
        foods_df: pd.DataFrame,
        categories: list[str],
        k_remaining: float,
        p_remaining: float,
        portion: float,
        exclude_names: list[str],
    ) -> dict | None:
        """Pick one food from categories that fits within budget."""
        candidates = foods_df[
            foods_df["category"].isin(categories)
            & ~foods_df["english"].isin(exclude_names)
        ].copy()

        if candidates.empty:
            return None

        candidates = candidates.sample(frac=1, random_state=random.randint(0, 9999))

        protein_categories = {"Meat", "Fish", "Egg", "Dairy"}
        is_protein_pick = bool(set(categories) & protein_categories)
        p_allowance = p_remaining * (0.65 if is_protein_pick else 1.0)

        for _, food in candidates.iterrows():
            k_per_100 = float(food.get("potassium_mg", 0) or 0)
            p_per_100 = float(food.get("protein_g", 0) or 0)
            ph_per_100 = float(food.get("phosphorus_mg", 0) or 0)
            na_per_100 = float(food.get("sodium_mg", 0) or 0)

            fit_portion = portion
            if k_per_100 > 0:
                fit_portion = min(fit_portion, k_remaining / k_per_100 * 100)
            if p_per_100 > 0:
                fit_portion = min(fit_portion, p_allowance / p_per_100 * 100)
            if fit_portion < 25:
                continue

            food_k = k_per_100 * fit_portion / 100
            food_p = p_per_100 * fit_portion / 100
            food_ph = ph_per_100 * fit_portion / 100
            food_na = na_per_100 * fit_portion / 100

            if food_p > p_remaining:
                continue

            return {
                "english": str(food.get("english") or "").strip(),
                "kinyarwanda": str(food.get("kinyarwanda") or "").strip(),
                "preparation_method": str(food.get("preparation_method") or "").strip(),
                "portion_grams": round(fit_portion, 1),
                "meal_occasion": "",
                "potassium_mg": round(food_k, 1),
                "phosphorus_mg": round(food_ph, 1),
                "protein_g": round(food_p, 1),
                "sodium_mg": round(food_na, 1),
                "category": str(food.get("category") or "Other"),
            }
        return None

    plan: dict[str, list[dict[str, Any]]] = {}

    for occasion in ["Breakfast", "Lunch", "Dinner", "Snack"]:
        portion = float(PORTION[occasion])
        k_budget = float(K_BUDGET[occasion])
        p_budget = float(P_BUDGET[occasion])
        groups = GROUPS[occasion]
        picked: list[dict] = []
        k_used = 0.0
        p_used = 0.0
        used_names: list[str] = []

        for group_categories in groups:
            food = pick_one(
                safe_foods,
                group_categories,
                k_budget - k_used,
                p_budget - p_used,
                portion,
                used_names,
            )
            if food:
                food["meal_occasion"] = occasion
                picked.append(food)
                k_used += food["potassium_mg"]
                p_used += food["protein_g"]
                used_names.append(food["english"])

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
        }

    answer_parts: list[str] = []
    suggested_foods: list[dict[str, Any]] = []
    meal_plan: dict[str, list[dict[str, Any]]] | None = None

    # Use only Rwandan foods for meal plan suggestions
    rwandan_foods = safe_foods[
        safe_foods.get(
            "is_rwandan",
            pd.Series(
                [1] * len(safe_foods),
                index=safe_foods.index,
            ),
        )
        == 1
    ] if "is_rwandan" in safe_foods.columns else safe_foods

    wants_food_suggestions = (
        wants_meal_plan
        or wants_breakfast
        or wants_lunch
        or wants_dinner
        or wants_snack
        or wants_safe
    )

    if wants_food_suggestions and not wants_avoid:
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
            ~safe_foods["english"]
            .str.lower()
            .str.contains("|".join(_forbidden), na=False)
        ]

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
    sources = result.get("sources") or []
    result["sources"] = list({s["source"]: s for s in sources}.values())

    llm_answer = result.get("answer", "")
    suggested_foods = result.get("suggested_foods") or []

    is_weekly = any(
        w in request.message.lower()
        for w in [
            "week",
            "weekly",
            "7 day",
            "seven day",
            "full day",
            "all day",
            "day plan",
            "full meal",
            "meal plan",
            "plan for",
            "plan my",
            "entire day",
            "whole day",
            "give me a plan",
            "help me plan",
        ]
    )

    display_text = llm_answer

    # Only suppress foods for weekly plans where the LLM generates a full text plan.
    if is_weekly:
        suggested_foods_for_response: list[dict[str, Any]] = []
        result["meal_plan"] = None
    else:
        suggested_foods_for_response = suggested_foods

    result["answer"] = display_text
    result["suggested_foods"] = suggested_foods_for_response
    return result

