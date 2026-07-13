"""
risk_prediction.py
GuidaPlate — API endpoint for dietary risk prediction
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.clinical_constants import (
    KDOQI_DAILY_LIMITS,
    NEAR_LIMIT_RATIO,
)
from backend.config import GUIDAPLATE_MEAL_XGB
from backend.database.db import get_db
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import (
    VALID_OCCASIONS,
    compute_exceeded_nutrients_meal,
    get_meal_predictor,
    get_predictor,
    meal_limits_for_occasion,
)

router = APIRouter(tags=["Risk Prediction"])

SUPPORTED_STAGES = set(KDOQI_DAILY_LIMITS)

MealOccasion = Literal["Breakfast", "Lunch", "Dinner", "Snack"]

CLINICAL_NOTES = {
    "HIGH": (
        "Immediate dietary adjustment recommended. Multiple "
        "nutrients exceed KDOQI 2020 stage limits."
    ),
    "MODERATE": (
        "Monitor intake carefully. At least one nutrient "
        "approaching or exceeding stage limit."
    ),
    "LOW": (
        "Dietary intake appears within safe limits for your "
        "CKD stage."
    ),
}


class RiskPredictionRequest(BaseModel):
    potassium: float
    phosphorus: float
    protein_per_kg: float
    sodium: float
    ckd_stage: str
    occasion: MealOccasion  # REQUIRED
    food_name: str | None = None


class RiskPredictionResponse(BaseModel):
    risk_label: str
    confidence: float
    probabilities: dict
    ckd_stage: str
    features_used: dict
    exceeded_nutrients: list[str]
    near_limit_nutrients: list[str]
    clinical_note: str
    substitutes: list[dict] = []
    shap_contributions: dict | None = None
    shap_explanation: str | None = None
    shap_dominant_nutrient: str | None = None
    scoring_scale: Literal["meal", "day"]
    meal_limits: dict | None = None
    occasion: str


def compute_exceeded_nutrients(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> tuple[list[str], list[str]]:
    """
    Compare intake against KDOQI daily limits (rollback / day path).
    Returns exceeded (over limit) and near_limit (80–99% of limit).
    """
    limits = KDOQI_DAILY_LIMITS.get(ckd_stage, KDOQI_DAILY_LIMITS["G3b"])

    nutrient_values = {
        "potassium": (potassium, limits["potassium"]),
        "phosphorus": (phosphorus, limits["phosphorus"]),
        "protein": (protein_per_kg, limits["protein_per_kg"]),
        "sodium": (sodium, limits["sodium"]),
    }

    exceeded: list[str] = []
    near_limit: list[str] = []

    for name, (value, limit) in nutrient_values.items():
        if limit <= 0:
            continue
        ratio = value / limit
        if ratio >= 1.0:
            exceeded.append(name)
        elif ratio >= NEAR_LIMIT_RATIO:
            near_limit.append(name)

    return exceeded, near_limit


def nutrient_limit_ratios(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    limits: dict[str, float],
) -> dict[str, float]:
    """value / limit for each nutrient (same keys as exceeded/near-limit lists)."""
    pairs = {
        "potassium": (potassium, limits["potassium"]),
        "phosphorus": (phosphorus, limits["phosphorus"]),
        "protein": (protein_per_kg, limits["protein_per_kg"]),
        "sodium": (sodium, limits["sodium"]),
    }
    ratios: dict[str, float] = {}
    for name, (value, limit) in pairs.items():
        ratios[name] = (value / limit) if limit > 0 else 0.0
    return ratios


def _headline_by_ratio(candidates: list[str], ratios: dict[str, float]) -> str:
    """Most severely over (or near) limit among flagged nutrients."""
    return max(candidates, key=lambda n: ratios.get(n, 0.0))


def generate_shap_explanation(
    shap_contributions: dict,
    exceeded_nutrients: list[str],
    near_limit_nutrients: list[str],
    risk_label: str,
    ckd_stage: str,
    scoring_scale: Literal["meal", "day"] = "day",
    shap_dominant_nutrient: str | None = None,
    nutrient_ratios: dict[str, float] | None = None,
) -> str:
    """
    Clinical severity leads the headline; SHAP is supporting context when it differs.
    """
    parts: list[str] = []
    limit_phrase = "this meal's limits" if scoring_scale == "meal" else "daily limits"
    ratios = nutrient_ratios or {}

    if risk_label == "LOW":
        if scoring_scale == "meal":
            return (
                "All four nutrients are well within this meal's safe limits for "
                f"Stage {ckd_stage}. This is a well-balanced meal."
            )
        return (
            "All four nutrients are well within your safe limits for "
            f"Stage {ckd_stage}. This is a well-balanced meal."
        )

    nutrient_names = {
        "potassium": "Potassium",
        "phosphorus": "Phosphorus",
        "protein": "Protein",
        "sodium": "Sodium",
    }
    advice = {
        "potassium": (
            "Choose lower-potassium options like rice, cabbage or apples."
        ),
        "phosphorus": (
            "Limit dairy and processed foods. Choose white rice over whole grains."
        ),
        "protein": (
            "Keep meat portions palm-sized. Consider eggs as a safer option."
        ),
        "sodium": (
            "Avoid added salt and processed foods. Use herbs and lemon instead."
        ),
    }

    def _append_shap_support(headline: str) -> None:
        if (
            shap_dominant_nutrient
            and shap_dominant_nutrient != headline
            and shap_dominant_nutrient in nutrient_names
        ):
            parts.append(
                f"{nutrient_names[shap_dominant_nutrient]} also contributed "
                f"significantly to this assessment."
            )

    if exceeded_nutrients:
        headline = _headline_by_ratio(exceeded_nutrients, ratios)
        parts.append(
            f"{nutrient_names[headline]} is the primary concern in this meal "
            f"for Stage {ckd_stage}. {advice[headline]}"
        )
        others = [
            nutrient_names[n] for n in exceeded_nutrients if n != headline
        ]
        if others:
            parts.append(
                f"{' and '.join(others)} also exceeded {limit_phrase}."
            )
        _append_shap_support(headline)

    elif near_limit_nutrients:
        headline = _headline_by_ratio(near_limit_nutrients, ratios)
        if scoring_scale == "meal":
            parts.append(
                f"This meal is approaching your {headline} limit for this occasion. "
                f"Monitor your remaining meals carefully today."
            )
        else:
            parts.append(
                f"This meal is approaching your {headline} limit. "
                f"Monitor your remaining meals carefully today."
            )
        _append_shap_support(headline)

    elif shap_contributions:
        dominant = (
            shap_dominant_nutrient
            if shap_dominant_nutrient in shap_contributions
            else max(shap_contributions, key=shap_contributions.get)
        )
        if shap_contributions.get(dominant, 0) > 0:
            parts.append(
                f"{nutrient_names.get(dominant, dominant.title())} contributed most "
                f"to the model's {risk_label.lower()} risk assessment "
                f"({shap_contributions[dominant]:.1f}%)."
            )

    return " ".join(parts) if parts else CLINICAL_NOTES.get(risk_label, "")


@router.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(
    request: RiskPredictionRequest,
    db: Session = Depends(get_db),
) -> RiskPredictionResponse:
    """Predict dietary risk from nutrient intake and optional food substitute lookup."""
    if request.ckd_stage not in SUPPORTED_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ckd_stage {request.ckd_stage!r}. Must be one of: G2, G3a, G3b, G4.",
        )
    if request.occasion not in VALID_OCCASIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid occasion {request.occasion!r}. "
                "Must be one of: Breakfast, Lunch, Dinner, Snack."
            ),
        )

    use_meal = GUIDAPLATE_MEAL_XGB
    scoring_scale: Literal["meal", "day"] = "meal" if use_meal else "day"

    try:
        if use_meal:
            predictor = get_meal_predictor()
            prediction = predictor.predict(
                request.potassium,
                request.phosphorus,
                request.protein_per_kg,
                request.sodium,
                request.ckd_stage,
                occasion=request.occasion,
            )
            exceeded_nutrients, near_limit_nutrients = compute_exceeded_nutrients_meal(
                request.potassium,
                request.phosphorus,
                request.protein_per_kg,
                request.sodium,
                request.ckd_stage,
                request.occasion,
            )
            meal_limits = meal_limits_for_occasion(request.ckd_stage, request.occasion)
            shap_occasion = request.occasion
            ratio_limits = meal_limits
        else:
            # Rollback: day path — occasion accepted but not used for scoring
            predictor = get_predictor()
            prediction = predictor.predict(
                request.potassium,
                request.phosphorus,
                request.protein_per_kg,
                request.sodium,
                request.ckd_stage,
            )
            exceeded_nutrients, near_limit_nutrients = compute_exceeded_nutrients(
                request.potassium,
                request.phosphorus,
                request.protein_per_kg,
                request.sodium,
                request.ckd_stage,
            )
            meal_limits = None
            shap_occasion = None
            ratio_limits = KDOQI_DAILY_LIMITS.get(
                request.ckd_stage, KDOQI_DAILY_LIMITS["G3b"]
            )

        ratios = nutrient_limit_ratios(
            request.potassium,
            request.phosphorus,
            request.protein_per_kg,
            request.sodium,
            ratio_limits,
        )

        risk_label = prediction["risk_label"]
        clinical_note = CLINICAL_NOTES.get(
            risk_label,
            "Unable to generate clinical guidance for this risk level.",
        )

        shap_contributions = None
        shap_explanation = None
        shap_dominant_nutrient = None
        try:
            shap_kwargs = dict(
                potassium=request.potassium,
                phosphorus=request.phosphorus,
                protein_per_kg=request.protein_per_kg,
                sodium=request.sodium,
                ckd_stage=request.ckd_stage,
            )
            if use_meal:
                shap_data = predictor.explain(**shap_kwargs, occasion=shap_occasion)
            else:
                shap_data = predictor.explain(**shap_kwargs)
            shap_contributions = shap_data["contributions"]
            shap_dominant_nutrient = shap_data["dominant_nutrient"]
        except Exception as e:
            print(f"SHAP computation failed: {e}")

        shap_explanation = generate_shap_explanation(
            shap_contributions or {},
            exceeded_nutrients,
            near_limit_nutrients,
            risk_label,
            request.ckd_stage,
            scoring_scale=scoring_scale,
            shap_dominant_nutrient=shap_dominant_nutrient,
            nutrient_ratios=ratios,
        )

        substitutes: list[dict] = []
        if (
            request.food_name
            and risk_label != "LOW"
            and exceeded_nutrients
        ):
            substitutes = get_recommender().get_substitutes(
                food_name=request.food_name,
                ckd_stage=request.ckd_stage,
                risk_label=risk_label,
                exceeded_nutrients=exceeded_nutrients,
                db=db,
            )

        return RiskPredictionResponse(
            risk_label=risk_label,
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            ckd_stage=request.ckd_stage,
            features_used=prediction["features_used"],
            exceeded_nutrients=exceeded_nutrients,
            near_limit_nutrients=near_limit_nutrients,
            clinical_note=clinical_note,
            substitutes=substitutes,
            shap_contributions=shap_contributions,
            shap_explanation=shap_explanation,
            shap_dominant_nutrient=shap_dominant_nutrient,
            scoring_scale=scoring_scale,
            meal_limits=meal_limits,
            occasion=request.occasion,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predict/thresholds/{stage}")
def get_thresholds(stage: str) -> dict:
    """Return KDOQI dietary thresholds for a given CKD stage."""
    if stage not in KDOQI_DAILY_LIMITS:
        raise HTTPException(
            status_code=404,
            detail=f"Thresholds not found for stage {stage!r}",
        )
    limits = KDOQI_DAILY_LIMITS[stage]
    return {
        "stage": stage,
        "thresholds": dict(limits),
    }
