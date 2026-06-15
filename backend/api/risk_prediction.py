"""
risk_prediction.py
GuidaPlate — API endpoint for dietary risk prediction
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import DIETARY_RISK_THRESHOLDS
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import get_predictor

router = APIRouter(tags=["Risk Prediction"])

SUPPORTED_STAGES = {"G2", "G3a", "G3b", "G4"}

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


def compute_exceeded_nutrients(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> tuple[list[str], list[str]]:
    """
    Compares each nutrient intake against its KDOQI stage threshold.
    Returns two lists: exceeded (ratio >= 1.0) and near_limit
    (0.8 <= ratio < 1.0).

    Nutrient name mapping for output:
      potassium -> "potassium"
      phosphorus -> "phosphorus"
      protein_per_kg -> "protein"
      sodium -> "sodium"
    """
    if ckd_stage not in DIETARY_RISK_THRESHOLDS:
        return [], []

    t = DIETARY_RISK_THRESHOLDS[ckd_stage]

    nutrient_values = {
        "potassium": (potassium, t["potassium"]),
        "phosphorus": (phosphorus, t["phosphorus"]),
        "protein": (protein_per_kg, t["protein"]),
        "sodium": (sodium, t["sodium"]),
    }

    exceeded: list[str] = []
    near_limit: list[str] = []

    for name, (value, limit) in nutrient_values.items():
        if limit == 0:
            continue
        ratio = value / limit
        if ratio >= 1.0:
            exceeded.append(name)
        elif ratio >= 0.8:
            near_limit.append(name)

    return exceeded, near_limit


@router.post("/predict/risk", response_model=RiskPredictionResponse)
def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
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
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predict/thresholds/{stage}")
def get_thresholds(stage: str) -> dict:
    """Return KDOQI dietary thresholds for a given CKD stage."""
    if stage not in DIETARY_RISK_THRESHOLDS:
        raise HTTPException(
            status_code=404,
            detail=f"Thresholds not found for stage {stage!r}",
        )
    return {"stage": stage, "thresholds": DIETARY_RISK_THRESHOLDS[stage]}
