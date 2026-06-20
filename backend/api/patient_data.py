"""
patient_data.py
GuidaPlate - Persistence endpoints for patient profile, food logs,
and risk assessment history

JWT user_id is stored in patient_id columns (auth-era identifier).
"""
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.security import get_current_user_id
from backend.database.db import FoodLog, Patient, RiskAssessmentLog, get_db

router = APIRouter(tags=["Patient Data"])


class FoodLogRequest(BaseModel):
    food_name: str
    category: str
    stage_safe_range: str  # CKD stage safety range, e.g. "1-5" or "1-3" (not LOW/MODERATE/HIGH)


class RiskAssessmentRequest(BaseModel):
    risk_level: str
    risk_score: float
    nutrients_summary: str | None = None  # JSON string of nutrient totals


def _parse_nutrient_totals(raw: str | None) -> dict | list | None:
    if raw is None or raw.strip() == "":
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, (dict, list)):
        raise ValueError("nutrients_summary must be a JSON object or array")
    return parsed


@router.post("/patient/food-log")
def save_food_log(
    request: FoodLogRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    log = FoodLog(
        log_id=str(uuid.uuid4()),
        patient_id=user_id,
        food_name=request.food_name,
        category=request.category,
        stage_safe_range=request.stage_safe_range,
        logged_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    return {"status": "saved", "log_id": log.log_id}


@router.get("/patient/food-log/history")
def get_food_log_history(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    logs = (
        db.query(FoodLog)
        .filter(FoodLog.patient_id == user_id)
        .order_by(FoodLog.logged_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "log_id": l.log_id,
            "food_name": l.food_name,
            "category": l.category,
            "stage_safe_range": l.stage_safe_range,
            "logged_at": l.logged_at.isoformat() if l.logged_at else None,
        }
        for l in logs
    ]


@router.post("/patient/risk-assessment")
def save_risk_assessment(
    request: RiskAssessmentRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    nutrient_totals = None
    if request.nutrients_summary is not None:
        try:
            nutrient_totals = _parse_nutrient_totals(request.nutrients_summary)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    ckd_stage = patient.ckd_stage if patient else None

    assessment = RiskAssessmentLog(
        assessment_id=str(uuid.uuid4()),
        patient_id=user_id,
        ckd_stage=ckd_stage,
        risk_label=request.risk_level,
        confidence=request.risk_score,
        nutrient_totals=nutrient_totals,
        assessed_at=datetime.utcnow(),
    )
    db.add(assessment)
    db.commit()
    return {"status": "saved", "assessment_id": assessment.assessment_id}


@router.get("/patient/risk-assessment/history")
def get_risk_assessment_history(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    assessments = (
        db.query(RiskAssessmentLog)
        .filter(RiskAssessmentLog.patient_id == user_id)
        .order_by(RiskAssessmentLog.assessed_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "assessment_id": a.assessment_id,
            "risk_level": a.risk_label,
            "risk_score": a.confidence,
            "ckd_stage": a.ckd_stage,
            "assessed_at": a.assessed_at.isoformat() if a.assessed_at else None,
        }
        for a in assessments
    ]
