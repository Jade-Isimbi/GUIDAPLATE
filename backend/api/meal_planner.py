from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

from backend.auth.security import get_current_user_id
from backend.rag.retriever import get_retriever


router = APIRouter(prefix="/meal-planner", tags=["Meal Planner"])


KDOQI_LIMITS: dict[str, dict[str, float]] = {
    "G2": {"potassium": 3500, "phosphorus": 1000, "protein_per_kg": 0.8, "sodium": 2300},
    "G3a": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G3b": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G4": {"potassium": 2500, "phosphorus": 700, "protein_per_kg": 0.55, "sodium": 2300},
}

RWANDA_FOOD_CONTEXT = """
Common Rwandan foods by meal:

Breakfast: bread, milk, tea, oats,
  ikivuguto (fermented milk),
  sweet potatoes, cassava,
  maize porridge (uji), eggs,
  peanut butter, bananas

Lunch/Dinner: ugali (ubugali),
  isombe (cassava leaves),
  beans (ibishyimbo), rice,
  irish potatoes, sweet potatoes,
  cassava, matoke (cooking banana),
  tilapia, chicken, beef, goat meat,
  cabbage, tomatoes, onions,
  peas (amashaza), sorghum

Snacks/Fruits: banana, pineapple,
  avocado, mango, passion fruit,
  watermelon, sugarcane

Do NOT suggest: almond milk, quinoa,
  kale smoothies, tofu, sushi,
  or any non-Rwandan foods.
"""


class MealPlannerRequest(BaseModel):
    message: str = Field(min_length=1)
    ckd_stage: str = Field(default="G3b")
    weight_kg: float = Field(default=65.0, gt=0)
    uploaded_text: str | None = None


def _stage_num(ckd_stage: str) -> int:
    s = ckd_stage.strip()
    if s.startswith("G"):
        s = s[1:]
    s = s.replace("a", "").replace("b", "")
    try:
        return int(s)
    except ValueError:
        return 3


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
    path = Path("backend/data/food_database.csv")
    df = pd.read_csv(path)
    for c in [
        "english",
        "kinyarwanda",
        "category",
        "ckd_stage_safe",
        "potassium_mg",
        "phosphorus_mg",
        "protein_g",
        "sodium_mg",
    ]:
        if c not in df.columns:
            df[c] = None
    return df


def query_llm(prompt: str) -> str | None:
    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    try:
        client = InferenceClient(
            provider="auto",
            api_key=token,
        )

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a CKD dietary advisor "
                        "for patients in Rwanda. "
                        "You only recommend foods commonly "
                        "eaten in Rwanda. "
                        "Never suggest Western foods like "
                        "almond milk, quinoa, or tofu. "
                        "Always give portions in grams. "
                        "Be concise and practical."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"HuggingFace API error: {e}")
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
            f"{RWANDA_FOOD_CONTEXT}\n\n"
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


def _apply_flan_enhancement(
    message: str,
    ckd_stage: str,
    limits: dict[str, float],
    protein_limit_g: float,
    retrieved_chunks: list[dict],
    answer_parts: list[str],
    safe_foods: pd.DataFrame | None = None,
) -> str:
    template_answer = "\n".join(answer_parts)
    flan_prompt = build_flan_prompt(
        message=message,
        ckd_stage=ckd_stage,
        limits=limits,
        protein_limit_g=protein_limit_g,
        retrieved_chunks=retrieved_chunks,
        safe_foods=safe_foods,
    )
    flan_answer = query_llm(flan_prompt)

    if flan_answer and len(flan_answer) > 20:
        return flan_answer

    return template_answer


def build_meal_plan(
    safe_foods: pd.DataFrame,
    limits: dict[str, float],
    protein_limit_g: float,
) -> dict[str, list[dict[str, Any]]]:
    occasions = {
        "Breakfast": {
            "categories": ["Grain", "Egg", "Dairy"],
            "k_budget": limits["potassium"] * 0.2,
            "protein_budget": protein_limit_g * 0.25,
            "portion": 150,
        },
        "Lunch": {
            "categories": ["Starch", "Meat", "Fish", "Vegetable", "Legume"],
            "k_budget": limits["potassium"] * 0.35,
            "protein_budget": protein_limit_g * 0.35,
            "portion": 200,
        },
        "Dinner": {
            "categories": ["Starch", "Meat", "Fish", "Vegetable", "Legume"],
            "k_budget": limits["potassium"] * 0.35,
            "protein_budget": protein_limit_g * 0.35,
            "portion": 200,
        },
        "Snack": {
            "categories": ["Fruit", "Grain", "Dairy"],
            "k_budget": limits["potassium"] * 0.1,
            "protein_budget": protein_limit_g * 0.05,
            "portion": 100,
        },
    }

    plan: dict[str, list[dict[str, Any]]] = {}
    for occasion, cfg in occasions.items():
        foods = safe_foods[safe_foods["category"].isin(cfg["categories"])].copy()
        if foods.empty:
            plan[occasion] = []
            continue

        foods = foods.sort_values("potassium_mg")
        picked: list[dict[str, Any]] = []
        k_used = 0.0
        protein_used = 0.0
        for _, food in foods.iterrows():
            if len(picked) >= 3:
                break
            portion = float(cfg["portion"])
            food_k = float(food.get("potassium_mg", 0) or 0) * portion / 100
            food_p = float(food.get("phosphorus_mg", 0) or 0) * portion / 100
            food_protein = float(food.get("protein_g", 0) or 0) * portion / 100
            food_na = float(food.get("sodium_mg", 0) or 0) * portion / 100

            if k_used + food_k > float(cfg["k_budget"]):
                continue
            if protein_used + food_protein > float(cfg["protein_budget"]):
                continue

            picked.append(
                {
                    "english": str(food.get("english") or "").strip(),
                    "portion_grams": portion,
                    "meal_occasion": occasion,
                    "potassium_mg": round(food_k, 1),
                    "phosphorus_mg": round(food_p, 1),
                    "protein_g": round(food_protein, 1),
                    "sodium_mg": round(food_na, 1),
                }
            )
            k_used += food_k
            protein_used += food_protein

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

    wants_food_suggestions = (
        wants_meal_plan
        or wants_breakfast
        or wants_lunch
        or wants_dinner
        or wants_snack
        or wants_safe
    )

    if wants_food_suggestions and not wants_avoid:
        meal_plan = build_meal_plan(safe_foods, limits, protein_limit_g)
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
            for alt in get_safe_alternatives(safe_foods, limits)[:5]:
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

    limits = KDOQI_LIMITS.get(request.ckd_stage, KDOQI_LIMITS["G3b"])
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

    chunks = retriever.retrieve(query, top_k=5)

    food_db = _load_food_db()
    stage_number = _stage_num(request.ckd_stage)
    safe_foods = food_db[food_db["ckd_stage_safe"].apply(lambda x: _is_stage_safe(str(x), stage_number))]

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

    # HuggingFace LLM is called only on user chat requests — never at startup.
    template_answer = result.get("answer", "")
    result["answer"] = _apply_flan_enhancement(
        message=request.message,
        ckd_stage=request.ckd_stage,
        limits=limits,
        protein_limit_g=protein_limit_g,
        retrieved_chunks=chunks,
        answer_parts=[template_answer] if template_answer else [],
        safe_foods=safe_foods,
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

    if is_weekly or (llm_answer and len(llm_answer) > 100):
        suggested_foods_for_response: list[dict[str, Any]] = []
        result["meal_plan"] = None
    else:
        suggested_foods_for_response = suggested_foods

    result["answer"] = display_text
    result["suggested_foods"] = suggested_foods_for_response
    return result

