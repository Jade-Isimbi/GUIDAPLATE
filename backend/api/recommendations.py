"""
recommendations.py
GuidaPlate — API endpoint for safer food recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.models.recommender import get_recommender

router = APIRouter(tags=["Recommendations"])

SUPPORTED_STAGES = {"G2", "G3a", "G3b", "G4"}


class SubstituteRequest(BaseModel):
    food_name: str
    ckd_stage: str
    exceeded_nutrients: list[str] = []


@router.post("/recommendations/substitutes")
def get_substitutes(
    request: SubstituteRequest,
    db: Session = Depends(get_db),
) -> dict:
    """Return safer food substitutes for a queried food."""
    if request.ckd_stage not in SUPPORTED_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ckd_stage {request.ckd_stage!r}. Must be one of: G2, G3a, G3b, G4.",
        )

    recommender = get_recommender()
    if recommender.get_food_by_name(db, request.food_name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Food {request.food_name!r} not found in database",
        )

    substitutes = recommender.get_substitutes(
        food_name=request.food_name,
        ckd_stage=request.ckd_stage,
        risk_label="MODERATE",
        exceeded_nutrients=request.exceeded_nutrients,
        db=db,
    )

    return {
        "food_queried": request.food_name,
        "ckd_stage": request.ckd_stage,
        "substitutes": substitutes,
        "count": len(substitutes),
    }


@router.get("/recommendations/safe-foods/{stage}")
def get_safe_foods(stage: str, db: Session = Depends(get_db)) -> dict:
    """Return all foods safe for the given CKD stage."""
    if stage not in SUPPORTED_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage {stage!r}. Must be one of: G2, G3a, G3b, G4.",
        )

    recommender = get_recommender()
    foods = recommender.get_all_foods(db=db, stage=stage)

    return {
        "stage": stage,
        "count": len(foods),
        "foods": foods,
    }
