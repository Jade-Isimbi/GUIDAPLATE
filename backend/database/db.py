"""
db.py
GuidaPlate — SQLite database connection and session management
"""

import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import ROOT

DATABASE_URL = f"sqlite:///{ROOT}/guidaplate.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String, primary_key=True)
    ckd_stage = Column(String)
    body_weight_kg = Column(Float)
    age = Column(Integer)
    sex = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class FoodLog(Base):
    __tablename__ = "food_logs"

    log_id = Column(String, primary_key=True)
    patient_id = Column(String)
    food_name = Column(String)
    portion_grams = Column(Float)
    meal_occasion = Column(String, nullable=True)
    category = Column(String, nullable=True)
    stage_safe_range = Column(String, nullable=True)
    potassium_mg = Column(Float)
    phosphorus_mg = Column(Float)
    protein_g = Column(Float)
    sodium_mg = Column(Float)
    logged_at = Column(DateTime, default=datetime.utcnow)


class RiskAssessmentLog(Base):
    __tablename__ = "risk_assessments"

    assessment_id = Column(String, primary_key=True)
    patient_id = Column(String)
    ckd_stage = Column(String)
    risk_label = Column(String)
    confidence = Column(Float)
    nutrient_totals = Column(JSON)
    shap_values = Column(JSON, nullable=True)
    assessed_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String)
    phone = Column(String, nullable=True)
    ckd_stage = Column(String, nullable=True)
    weight_kg = Column(Float, nullable=True)
    dob = Column(String, nullable=True)
    sex = Column(String, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, nullable=False)
    token = Column(String, nullable=False, unique=True)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    title = Column(String, nullable=False)
    preview = Column(String, nullable=True)
    messages = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


create_tables()
