"""
auth.py
GuidaPlate - User registration and login endpoints
"""
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from backend.auth.security import create_access_token, hash_password, verify_password
from backend.database.db import Patient, User, get_db

router = APIRouter(tags=["Authentication"])


def _age_from_dob(dob: str | None) -> int | None:
    if not dob or not dob.strip():
        return None
    try:
        born = date.fromisoformat(dob.strip())
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str | None = None
    ckd_stage: str | None = None
    weight_kg: float | None = None
    dob: str | None = None
    sex: str | None = None
    language: str | None = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    name: str
    ckd_stage: str | None = None
    weight_kg: float | None = None

@router.post("/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        user_id=str(uuid.uuid4()),
        email=request.email,
        hashed_password=hash_password(request.password),
        name=request.name,
        phone=request.phone,
        ckd_stage=request.ckd_stage,
        weight_kg=request.weight_kg,
        dob=request.dob,
        sex=request.sex,
        language=request.language,
    )
    patient = Patient(
        patient_id=user.user_id,
        ckd_stage=user.ckd_stage,
        body_weight_kg=user.weight_kg,
        age=_age_from_dob(user.dob),
        sex=user.sex,
    )
    try:
        db.add(user)
        db.add(patient)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Registration failed. Please try again.",
        ) from None

    token = create_access_token({"sub": user.user_id, "email": user.email})
    return AuthResponse(
        access_token=token,
        user_id=user.user_id,
        name=user.name,
        ckd_stage=user.ckd_stage,
        weight_kg=user.weight_kg,
    )

@router.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.user_id, "email": user.email})
    return AuthResponse(
        access_token=token,
        user_id=user.user_id,
        name=user.name,
        ckd_stage=user.ckd_stage,
        weight_kg=user.weight_kg,
    )
