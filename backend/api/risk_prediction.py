"""
risk_prediction.py
GuidaPlate — API endpoint for dietary risk prediction
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.clinical_constants import (
    CLINICAL_SEVERITY_WEIGHTS,
    KDOQI_DAILY_LIMITS,
    NEAR_LIMIT_RATIO,
    SEVERITY_THRESHOLDS,
)
from backend.database.db import get_db
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import get_predictor

router = APIRouter(tags=["Risk Prediction"])

SUPPORTED_STAGES = set(KDOQI_DAILY_LIMITS)

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


def compute_exceeded_nutrients(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> tuple[list[str], list[str]]:
    """
    Compare intake against KDOQI limits (matches XGBoost v3 training).
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


def generate_shap_explanation(
    shap_contributions: dict,
    exceeded_nutrients: list[str],
    near_limit_nutrients: list[str],
    risk_label: str,
    ckd_stage: str,
) -> str:
    """Hybrid explanation: KDOQI exceedance first, SHAP when informative."""
    parts: list[str] = []

    if risk_label == "LOW":
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
            "Keep meat portions palm-sized. Consider egg whites as a safer option."
        ),
        "sodium": (
            "Avoid added salt and processed foods. Use herbs and lemon instead."
        ),
    }

    if exceeded_nutrients:
        primary = exceeded_nutrients[0]

        parts.append(
            f"{nutrient_names[primary]} is the primary concern in this meal "
            f"for Stage {ckd_stage}. {advice[primary]}"
        )

        others = [nutrient_names[n] for n in exceeded_nutrients[1:]]
        if others:
            parts.append(
                f"{' and '.join(others)} also exceeded your daily limits in this meal."
            )

    elif near_limit_nutrients:
        near = near_limit_nutrients[0]
        parts.append(
            f"This meal is approaching your {near} limit. "
            f"Monitor your remaining meals carefully today."
        )
    elif shap_contributions:
        dominant = max(shap_contributions, key=shap_contributions.get)
        if shap_contributions[dominant] > 0:
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

    try:
        prediction = get_predictor().predict(
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

        risk_label = prediction["risk_label"]
        clinical_note = CLINICAL_NOTES.get(
            risk_label,
            "Unable to generate clinical guidance for this risk level.",
        )

        shap_contributions = None
        shap_explanation = None
        shap_dominant_nutrient = None
        try:
            shap_data = get_predictor().explain(
                request.potassium,
                request.phosphorus,
                request.protein_per_kg,
                request.sodium,
                request.ckd_stage,
            )
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
