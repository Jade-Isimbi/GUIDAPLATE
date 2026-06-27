"""
patient_data.py
GuidaPlate - Persistence endpoints for patient profile, food logs,
and risk assessment history

JWT user_id is stored in patient_id columns (auth-era identifier).
"""
import json
import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import get_current_user_id
from backend.database.db import FoodLog, Patient, RiskAssessmentLog, User, get_db
from backend.utils.meal_aggregation import day_bounds, nutrients_for_food_name

router = APIRouter(tags=["Patient Data"])

SUPPORTED_PROFILE_STAGES = {"G2", "G3a", "G3b", "G4"}


class PatientProfileResponse(BaseModel):
    ckd_stage: str | None = None
    weight_kg: float | None = None
    name: str | None = None
    email: str | None = None


class PatientProfileUpdateRequest(BaseModel):
    ckd_stage: str
    weight_kg: float = Field(gt=0)


class FoodLogRequest(BaseModel):
    food_name: str
    category: str
    stage_safe_range: str  # CKD stage safety range, e.g. "1-5" or "1-3"
    portion_grams: float = Field(default=100.0, gt=0)
    meal_occasion: str | None = None  # Breakfast | Lunch | Dinner | Snack


class FoodLogHistoryItem(BaseModel):
    log_id: str
    food_name: str
    category: str | None
    stage_safe_range: str | None
    portion_grams: float | None
    meal_occasion: str | None
    potassium_mg: float | None
    phosphorus_mg: float | None
    protein_g: float | None
    sodium_mg: float | None
    logged_at: str | None


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


@router.get("/patient/profile", response_model=PatientProfileResponse)
def get_patient_profile(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    ckd_stage = patient.ckd_stage if patient and patient.ckd_stage else user.ckd_stage
    weight_kg = (
        patient.body_weight_kg
        if patient and patient.body_weight_kg is not None
        else user.weight_kg
    )

    return PatientProfileResponse(
        ckd_stage=ckd_stage,
        weight_kg=weight_kg,
        name=user.name,
        email=user.email,
    )


@router.patch("/patient/profile", response_model=PatientProfileResponse)
def update_patient_profile(
    request: PatientProfileUpdateRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    if request.ckd_stage not in SUPPORTED_PROFILE_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ckd_stage {request.ckd_stage!r}. Must be one of: G2, G3a, G3b, G4.",
        )

    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.ckd_stage = request.ckd_stage
    user.weight_kg = request.weight_kg

    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if patient is None:
        patient = Patient(
            patient_id=user_id,
            ckd_stage=request.ckd_stage,
            body_weight_kg=request.weight_kg,
        )
        db.add(patient)
    else:
        patient.ckd_stage = request.ckd_stage
        patient.body_weight_kg = request.weight_kg

    db.commit()
    db.refresh(user)

    return PatientProfileResponse(
        ckd_stage=request.ckd_stage,
        weight_kg=request.weight_kg,
        name=user.name,
        email=user.email,
    )


def _serialize_food_log(log: FoodLog) -> dict:
    return {
        "log_id": log.log_id,
        "food_name": log.food_name,
        "category": log.category,
        "stage_safe_range": log.stage_safe_range,
        "portion_grams": log.portion_grams,
        "meal_occasion": log.meal_occasion,
        "potassium_mg": log.potassium_mg,
        "phosphorus_mg": log.phosphorus_mg,
        "protein_g": log.protein_g,
        "sodium_mg": log.sodium_mg,
        "logged_at": log.logged_at.isoformat() if log.logged_at else None,
    }


@router.post("/patient/food-log")
def save_food_log(
    request: FoodLogRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    try:
        nutrients = nutrients_for_food_name(request.food_name, request.portion_grams)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    log = FoodLog(
        log_id=str(uuid.uuid4()),
        patient_id=user_id,
        food_name=request.food_name,
        category=request.category,
        stage_safe_range=request.stage_safe_range,
        portion_grams=request.portion_grams,
        meal_occasion=request.meal_occasion,
        potassium_mg=nutrients["potassium_mg"],
        phosphorus_mg=nutrients["phosphorus_mg"],
        protein_g=nutrients["protein_g"],
        sodium_mg=nutrients["sodium_mg"],
        logged_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    return {"status": "saved", "log_id": log.log_id}


@router.get("/patient/food-log/history", response_model=list[FoodLogHistoryItem])
def get_food_log_history(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    log_date: str | None = Query(
        default=None,
        alias="date",
        description="Filter to calendar date (YYYY-MM-DD, UTC logged_at)",
    ),
):
    query = db.query(FoodLog).filter(FoodLog.patient_id == user_id)

    if log_date is not None:
        try:
            target = date.fromisoformat(log_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="date must be YYYY-MM-DD",
            ) from exc
        start, end = day_bounds(target)
        query = query.filter(FoodLog.logged_at >= start, FoodLog.logged_at <= end)

    logs = query.order_by(FoodLog.logged_at.desc()).limit(50).all()
    return [_serialize_food_log(log) for log in logs]


def _parse_log_date(raw: str | None) -> date:
    if raw is None:
        return datetime.utcnow().date()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from exc


@router.delete("/patient/food-log/{log_id}")
def delete_food_log(
    log_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    log = (
        db.query(FoodLog)
        .filter(FoodLog.log_id == log_id, FoodLog.patient_id == user_id)
        .first()
    )
    if log is None:
        raise HTTPException(status_code=404, detail="Food log not found")
    db.delete(log)
    db.commit()
    return {"status": "deleted", "log_id": log_id}


@router.delete("/patient/food-log/day")
def clear_food_logs_for_day(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    log_date: str | None = Query(
        default=None,
        alias="date",
        description="Calendar date to clear (YYYY-MM-DD, UTC logged_at). Defaults to today.",
    ),
):
    target = _parse_log_date(log_date)
    start, end = day_bounds(target)
    deleted = (
        db.query(FoodLog)
        .filter(
            FoodLog.patient_id == user_id,
            FoodLog.logged_at >= start,
            FoodLog.logged_at <= end,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"status": "cleared", "deleted_count": deleted, "date": target.isoformat()}


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
