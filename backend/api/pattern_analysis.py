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
    clinical_note: str


def build_clinical_note(risk_label: str) -> str:
    """Generate guidance from the trained LSTM sequence-risk label only."""
    notes = {
        "HIGH": (
            "High dietary risk detected across the recent meal sequence. "
            "Dietary adjustment and closer monitoring are recommended."
        ),
        "MODERATE": (
            "Moderate dietary risk detected across the recent meal sequence. "
            "Continue monitoring nutrient intake."
        ),
        "LOW": (
            "Low dietary risk across the analyzed meal sequence for this period."
        ),
    }
    return notes.get(
        risk_label,
        "Pattern analysis complete. Review nutrient intake with your care team.",
    )


@router.post("/predict/pattern", response_model=PatternAnalysisResponse)
def predict_pattern(request: MealSequenceRequest) -> PatternAnalysisResponse:
    """Run the trained LSTM sequence-risk classifier on a meal sequence."""
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

        return PatternAnalysisResponse(
            risk_label=risk_label,
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            sequence_length=result["sequence_length"],
            clinical_note=build_clinical_note(risk_label),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
