"""
db.py
GuidaPlate — SQLite database connection and session management
"""

import os
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

_DB_FILE = Path(os.environ["DATABASE_PATH"]) if os.environ.get("DATABASE_PATH") else ROOT / "guidaplate.db"
DATABASE_URL = f"sqlite:///{_DB_FILE}"
print(f"[DB] Using database at: {DATABASE_URL}")

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


class Food(Base):
    __tablename__ = "foods"

    id = Column(Integer, primary_key=True, index=True)
    food_id = Column(String, unique=True, nullable=False, index=True)
    english = Column(String, nullable=False, index=True)
    french = Column(String, nullable=True)
    kinyarwanda = Column(String, nullable=True)
    category = Column(String, nullable=False, default="Other")
    potassium_mg = Column(Float, nullable=False, default=0.0)
    phosphorus_mg = Column(Float, nullable=False, default=0.0)
    protein_g = Column(Float, nullable=False, default=0.0)
    sodium_mg = Column(Float, nullable=False, default=0.0)
    energy_kcal = Column(Float, nullable=True)
    stage_safe_range = Column(String, nullable=True)
    preparation_method = Column(String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "food_id": self.food_id,
            "english": self.english,
            "french": self.french,
            "kinyarwanda": self.kinyarwanda,
            "category": self.category,
            "potassium_mg": self.potassium_mg,
            "phosphorus_mg": self.phosphorus_mg,
            "protein_g": self.protein_g,
            "sodium_mg": self.sodium_mg,
            "energy_kcal": self.energy_kcal,
            "stage_safe_range": self.stage_safe_range,
            "ckd_stage_safe": self.stage_safe_range,
            "preparation_method": self.preparation_method or "",
        }


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
    shap_explanation = Column(Text, nullable=True)
    feature_values = Column(JSON, nullable=True)
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
    _migrate_risk_assessment_columns()


def _migrate_risk_assessment_columns() -> None:
    """Add risk_assessments columns introduced after initial schema."""
    import sqlite3

    db_path = _DB_FILE
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(risk_assessments)")
        }
        if "shap_explanation" not in columns:
            conn.execute(
                "ALTER TABLE risk_assessments ADD COLUMN shap_explanation TEXT"
            )
        if "feature_values" not in columns:
            conn.execute(
                "ALTER TABLE risk_assessments ADD COLUMN feature_values JSON"
            )
        conn.commit()
    finally:
        conn.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


create_tables()
