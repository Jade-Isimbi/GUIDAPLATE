"""
pattern_analysis.py
GuidaPlate — API endpoint for LSTM-based dietary pattern analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.lstm_model import get_analyzer

router = APIRouter(tags=["Pattern Analysis"])

MAX_SEQUENCE_STEPS = 6


class MealStep(BaseModel):
    potassium: float
    phosphorus: float
    protein_per_kg: float
    sodium: float
    occasion_encoded: float = 0.5  # default neutral / backward compatible


class MealSequenceRequest(BaseModel):
    # Accept both formats:
    # - legacy: list[list[float]] with 4 values per step
    # - v2: list[MealStep] (dicts)
    meal_sequence: list
    ckd_stage: str


class PatternAnalysisResponse(BaseModel):
    risk_label: str
    confidence: float
    probabilities: dict
    sequence_length: int
    trend: str
    clinical_note: str


def build_clinical_note(risk_label: str, trend: str) -> str:
    """Generate guidance from LSTM risk level and meal-sequence trend."""
    notes = {
        ("HIGH", "escalating"): (
            "Nutrient burden is increasing across recent meals with high overall "
            "dietary risk. Immediate intervention recommended."
        ),
        ("HIGH", "stable"): (
            "Consistently high nutrient intake detected across the meal sequence. "
            "Dietary adjustment recommended."
        ),
        ("MODERATE", "escalating"): (
            "Nutrient intake is trending upward across meals with moderate risk. "
            "Closer monitoring advised."
        ),
        ("MODERATE", "stable"): (
            "Moderate dietary risk detected with relatively stable intake across "
            "meals. Continue monitoring."
        ),
        ("LOW", "escalating"): (
            "Overall risk is low but nutrient intake is increasing across meals. "
            "Watch for continued escalation."
        ),
        ("LOW", "stable"): (
            "Meal sequence shows stable, low-risk dietary patterns for the "
            "analyzed period."
        ),
        ("HIGH", "improving"): (
            "Overall dietary risk remains high, but nutrient intake is trending "
            "downward across recent meals. Keep up the improving pattern."
        ),
        ("MODERATE", "improving"): (
            "Moderate dietary risk with nutrient intake trending downward across "
            "meals. Your recent choices are moving in the right direction."
        ),
        ("LOW", "improving"): (
            "Low dietary risk with nutrient intake trending downward across meals. "
            "Excellent progress — keep it up."
        ),
    }
    return notes.get(
        (risk_label, trend),
        "Pattern analysis complete. Review nutrient trends with your care team.",
    )


@router.post("/predict/pattern", response_model=PatternAnalysisResponse)
def predict_pattern(request: MealSequenceRequest) -> PatternAnalysisResponse:
    """Analyze a temporal meal sequence for escalating dietary risk patterns."""
    sequence_len = len(request.meal_sequence)
    if sequence_len < 1 or sequence_len > MAX_SEQUENCE_STEPS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"meal_sequence must contain between 1 and {MAX_SEQUENCE_STEPS} "
                f"steps; received {sequence_len}."
            ),
        )

    normalized: list[list[float]] = []
    for i, step in enumerate(request.meal_sequence):
        # New structured form (dict / pydantic model)
        if isinstance(step, dict):
            try:
                s = MealStep(**step)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid step {i + 1}: {exc}") from exc
            normalized.append([s.potassium, s.phosphorus, s.protein_per_kg, s.sodium, s.occasion_encoded])
            continue

        # Legacy list-of-floats form (4 or 5)
        if isinstance(step, list):
            if len(step) == 4:
                normalized.append([float(step[0]), float(step[1]), float(step[2]), float(step[3]), 0.5])
                continue
            if len(step) == 5:
                normalized.append([float(step[0]), float(step[1]), float(step[2]), float(step[3]), float(step[4])])
                continue
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Step {i + 1} must have 4 values "
                    f"[potassium, phosphorus, protein_per_kg, sodium] "
                    f"or 5 values (including occasion_encoded); received {len(step)}."
                ),
            )

        raise HTTPException(status_code=400, detail=f"Invalid step {i + 1}: expected object or list.")

    try:
        result = get_analyzer().analyze(normalized)
        risk_label = result["risk_label"]
        trend = result["trend"]

        return PatternAnalysisResponse(
            risk_label=risk_label,
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            sequence_length=result["sequence_length"],
            trend=trend,
            clinical_note=build_clinical_note(risk_label, trend),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
