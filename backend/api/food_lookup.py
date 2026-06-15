"""
food_lookup.py
GuidaPlate — API endpoint for Rwanda food database lookup
"""

from fastapi import APIRouter, HTTPException, Query

from backend.models.recommender import get_recommender

router = APIRouter(tags=["Food Database"])

SUPPORTED_STAGES = ["G2", "G3a", "G3b", "G4"]


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
    return {"count": len(foods), "foods": foods}


@router.get("/foods/search/{query}")
def search_foods(query: str) -> dict:
    """Search foods by a term across english, french, and kinyarwanda name columns."""
    recommender = get_recommender()
    results = recommender.get_all_foods(search=query)
    return {"count": len(results), "results": results}


@router.get("/foods/{food_id}")
def get_food_by_id(food_id: int) -> dict:
    """Return a single food record by its food_id."""
    recommender = get_recommender()
    matches = recommender.foods[recommender.foods["food_id"] == food_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
    return recommender._row_to_dict(matches.iloc[0])


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
