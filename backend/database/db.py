"""
db.py
GuidaPlate — SQLite database connection and session management
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
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
    meal_occasion = Column(String)
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


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


create_tables()
