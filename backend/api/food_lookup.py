"""
food_lookup.py
GuidaPlate — API endpoint for Rwanda food database lookup
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.models.recommender import get_recommender
from backend.utils.unit_converter import format_unit_display, get_unit_info, units_to_grams

router = APIRouter(tags=["Food Database"])

SUPPORTED_STAGES = ["G2", "G3a", "G3b", "G4"]


class ConvertRequest(BaseModel):
    food_name: str
    category: str
    quantity: float = Field(default=1.0, gt=0)


def _attach_unit_info(food: dict) -> dict:
    english = str(food.get("english") or food.get("food_name") or "")
    category = str(food.get("category") or "Other")
    unit_info = get_unit_info(english, category)
    food["unit"] = unit_info["unit"]
    food["grams_per_unit"] = unit_info["grams_per_unit"]
    return food


@router.get("/foods")
def list_foods(
    stage: str | None = Query(default=None, description="CKD stage filter (e.g. G3a)"),
    category: str | None = Query(default=None, description="Food category filter"),
    search: str | None = Query(
        default=None, description="Search term for english, french, or kinyarwanda names"
    ),
) -> dict:
    """List foods with optional stage, category, and multilingual name filters."""
    recommender = get_recommender()
    foods = recommender.get_all_foods(stage=stage, category=category, search=search)
    enriched = [_attach_unit_info(food) for food in foods]
    return {"count": len(enriched), "foods": enriched}


@router.get("/foods/search/{query}")
def search_foods(query: str) -> dict:
    """Search foods by a term across english, french, and kinyarwanda name columns."""
    recommender = get_recommender()
    results = recommender.get_all_foods(search=query)
    enriched = [_attach_unit_info(food) for food in results]
    return {"count": len(enriched), "results": enriched}


@router.get("/foods/{food_id}")
def get_food_by_id(food_id: int) -> dict:
    """Return a single food record by its food_id."""
    recommender = get_recommender()
    matches = recommender.foods[recommender.foods["food_id"] == food_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
    return _attach_unit_info(recommender._row_to_dict(matches.iloc[0]))


@router.post("/foods/convert-units")
def convert_units(req: ConvertRequest) -> dict:
    grams = units_to_grams(req.food_name, req.category, req.quantity)
    unit_info = get_unit_info(req.food_name, req.category)
    return {
        "food_name": req.food_name,
        "quantity": req.quantity,
        "unit": unit_info["unit"],
        "grams": round(grams, 1),
        "display": format_unit_display(req.quantity, unit_info["unit"], grams),
    }


@router.get("/categories")
def list_categories() -> dict:
    """Return all unique food categories in the database."""
    recommender = get_recommender()
    categories = sorted(
        recommender.foods["category"].dropna().astype(str).unique().tolist()
    )
    return {"categories": categories}


@router.get("/stages")
def list_stages() -> dict:
    """Return the CKD stages supported by GuidaPlate."""
    return {"stages": SUPPORTED_STAGES}
