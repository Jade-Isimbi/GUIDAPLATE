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
from backend.database.db import get_db
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import (
    VALID_OCCASIONS,
    compute_exceeded_nutrients_meal,
    get_meal_predictor,
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
    scoring_scale: Literal["meal"]
    meal_limits: dict | None = None
    occasion: str
    meal_feature_set: Literal[
        "noscore_occasion_caps",
        "rule_fallback",
    ]
    prediction_source: Literal["xgboost", "rule_fallback"]


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


def meal_rule_fallback(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
    occasion: str,
) -> tuple[dict, list[str], list[str], dict[str, float]]:
    """
    Transparent same-scale safety fallback when meal-model inference is unavailable.

    Baseline rule used in evaluation: no exceeded nutrients = LOW, one = MODERATE,
    two or more = HIGH. Probabilities are one-hot schema placeholders, not model
    calibration. Confidence is 0 because the rule has no learned probability;
    prediction_source makes that explicit to clients.
    """
    exceeded, near_limit = compute_exceeded_nutrients_meal(
        potassium,
        phosphorus,
        protein_per_kg,
        sodium,
        ckd_stage,
        occasion,
    )
    if len(exceeded) >= 2:
        label = "HIGH"
    elif len(exceeded) == 1:
        label = "MODERATE"
    else:
        label = "LOW"

    limits = meal_limits_for_occasion(ckd_stage, occasion)
    probabilities = {name: float(name == label) for name in ("LOW", "MODERATE", "HIGH")}
    prediction = {
        "risk_label": label,
        "confidence": 0.0,
        "probabilities": probabilities,
        "features_used": {
            "potassium": float(potassium),
            "phosphorus": float(phosphorus),
            "protein_per_kg": float(protein_per_kg),
            "sodium": float(sodium),
            "occasion": occasion,
            "rule": "exceeded_count: 0=LOW, 1=MODERATE, >=2=HIGH",
        },
    }
    return prediction, exceeded, near_limit, limits


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

    try:
        predictor = None
        prediction_source: Literal["xgboost", "rule_fallback"] = "xgboost"
        try:
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
            meal_feature_set = "noscore_occasion_caps"
        except Exception as model_exc:
            print(
                "Meal XGBoost inference unavailable; using same-scale rule fallback: "
                f"{model_exc}"
            )
            prediction_source = "rule_fallback"
            meal_feature_set = "rule_fallback"
            prediction, exceeded_nutrients, near_limit_nutrients, meal_limits = (
                meal_rule_fallback(
                    request.potassium,
                    request.phosphorus,
                    request.protein_per_kg,
                    request.sodium,
                    request.ckd_stage,
                    request.occasion,
                )
            )

        ratios = nutrient_limit_ratios(
            request.potassium,
            request.phosphorus,
            request.protein_per_kg,
            request.sodium,
            meal_limits,
        )

        risk_label = prediction["risk_label"]
        clinical_note = CLINICAL_NOTES.get(
            risk_label,
            "Unable to generate clinical guidance for this risk level.",
        )

        shap_contributions = None
        shap_dominant_nutrient = None
        if predictor is not None:
            try:
                shap_data = predictor.explain(
                    potassium=request.potassium,
                    phosphorus=request.phosphorus,
                    protein_per_kg=request.protein_per_kg,
                    sodium=request.sodium,
                    ckd_stage=request.ckd_stage,
                    occasion=request.occasion,
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
            scoring_scale="meal",
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
            scoring_scale="meal",
            meal_limits=meal_limits,
            occasion=request.occasion,
            meal_feature_set=meal_feature_set,
            prediction_source=prediction_source,
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
