"""
food_lookup.py
GuidaPlate — API endpoint for Rwanda food database lookup
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.database.db import Food, get_db
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
    db: Session = Depends(get_db),
) -> dict:
    """List foods with optional stage, category, and multilingual name filters."""
    recommender = get_recommender()
    foods = recommender.get_all_foods(
        db=db,
        stage=stage,
        category=category,
        search=search,
    )
    enriched = [_attach_unit_info(food) for food in foods]
    return {"count": len(enriched), "foods": enriched}


@router.get("/foods/search/{query}")
def search_foods(
    query: str,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> dict:
    """Search foods by a term across english, french, and kinyarwanda names."""
    term = f"%{query.strip()}%"
    foods = (
        db.query(Food)
        .filter(
            or_(
                Food.english.ilike(term),
                Food.french.ilike(term),
                Food.kinyarwanda.ilike(term),
            )
        )
        .limit(limit)
        .all()
    )
    enriched = [_attach_unit_info(food.to_dict()) for food in foods]
    return {"count": len(enriched), "results": enriched}


@router.get("/foods/{food_id}")
def get_food_by_id(food_id: int, db: Session = Depends(get_db)) -> dict:
    """Return a single food record by its food_id."""
    food = db.query(Food).filter(Food.food_id == str(food_id)).first()
    if food is None:
        food = db.query(Food).filter(Food.id == food_id).first()
    if food is None:
        raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
    return _attach_unit_info(food.to_dict())


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
def list_categories(db: Session = Depends(get_db)) -> dict:
    """Return all unique food categories in the database."""
    rows = db.query(Food.category).distinct().all()
    categories = sorted(str(row[0]) for row in rows if row[0])
    return {"categories": categories}


@router.get("/stages")
def list_stages() -> dict:
    """Return the CKD stages supported by GuidaPlate."""
    return {"stages": SUPPORTED_STAGES}
