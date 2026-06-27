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


def generate_shap_explanation(
    risk_label: str,
    contributions: dict,
    dominant_nutrient: str,
    dominant_pct: float,
    ckd_stage: str,
    food_name: str | None,
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
) -> str:
    """Generate a rich, specific explanation string."""
    STAGE_LIMITS = {
        "G2": {"potassium": 3500, "phosphorus": 1000, "protein": 0.8, "sodium": 2300},
        "G3a": {"potassium": 3000, "phosphorus": 800, "protein": 0.6, "sodium": 2300},
        "G3b": {"potassium": 3000, "phosphorus": 800, "protein": 0.6, "sodium": 2300},
        "G4": {"potassium": 2500, "phosphorus": 700, "protein": 0.55, "sodium": 2300},
    }
    limits = STAGE_LIMITS.get(ckd_stage, STAGE_LIMITS["G3b"])

    NUTRIENT_META = {
        "potassium": {
            "label": "Potassium",
            "unit": "mg",
            "value": potassium,
            "limit": limits["potassium"],
        },
        "phosphorus": {
            "label": "Phosphorus",
            "unit": "mg",
            "value": phosphorus,
            "limit": limits["phosphorus"],
        },
        "protein": {
            "label": "Protein",
            "unit": "g/kg",
            "value": round(protein_per_kg, 2),
            "limit": limits["protein"],
        },
        "sodium": {
            "label": "Sodium",
            "unit": "mg",
            "value": sodium,
            "limit": limits["sodium"],
        },
    }

    dom = NUTRIENT_META[dominant_nutrient]
    food_ref = f" from {food_name}" if food_name else ""

    sorted_nutrients = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    second = sorted_nutrients[1] if len(sorted_nutrients) > 1 else None

    if risk_label == "HIGH":
        explanation = (
            f"{dom['label']} is the primary driver "
            f"of this HIGH risk assessment{food_ref} "
            f"— it accounts for {dominant_pct}% of "
            f"the risk signal "
            f"({dom['value']}{dom['unit']} vs your "
            f"Stage {ckd_stage} limit of "
            f"{dom['limit']}{dom['unit']})."
        )
        if second and second[1] > 15:
            sec = NUTRIENT_META[second[0]]
            explanation += (
                f" {sec['label']} contributed an "
                f"additional {second[1]}%, pushing "
                f"this meal above safe thresholds."
            )
        safe = [
            NUTRIENT_META[n]["label"]
            for n, pct in contributions.items()
            if pct < 15
        ]
        if safe:
            explanation += (
                f" {' and '.join(safe)} were within "
                f"safe ranges and had minimal impact."
            )

    elif risk_label == "MODERATE":
        explanation = (
            f"This meal's risk is primarily driven "
            f"by {dom['label']} ({dominant_pct}%)"
            f"{food_ref} — approaching your Stage "
            f"{ckd_stage} limit of "
            f"{dom['limit']}{dom['unit']}."
        )
        if second and second[1] > 15:
            sec = NUTRIENT_META[second[0]]
            explanation += (
                f" {sec['label']} also contributed "
                f"{second[1]}% to the moderate risk "
                f"signal."
            )
        explanation += (
            " Small portion adjustments can bring "
            "this meal into the safe range."
        )

    else:
        explanation = (
            f"All four nutrients are well within "
            f"your safe limits for Stage {ckd_stage}."
        )
        explanation += (
            f" {dom['label']} had the highest "
            f"relative contribution ({dominant_pct}%) "
            f"but remained safely below your "
            f"{dom['limit']}{dom['unit']} threshold."
        )
        explanation += (
            " This is a well-balanced meal for "
            "your CKD stage."
        )

    return explanation


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
            shap_explanation = generate_shap_explanation(
                risk_label=risk_label,
                contributions=shap_data["contributions"],
                dominant_nutrient=shap_data["dominant_nutrient"],
                dominant_pct=shap_data["dominant_pct"],
                ckd_stage=request.ckd_stage,
                food_name=request.food_name,
                potassium=request.potassium,
                phosphorus=request.phosphorus,
                protein_per_kg=request.protein_per_kg,
                sodium=request.sodium,
            )
        except Exception as e:
            print(f"SHAP computation failed: {e}")

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
    if stage not in DIETARY_RISK_THRESHOLDS:
        raise HTTPException(
            status_code=404,
            detail=f"Thresholds not found for stage {stage!r}",
        )
    return {"stage": stage, "thresholds": DIETARY_RISK_THRESHOLDS[stage]}
