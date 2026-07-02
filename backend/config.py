"""
GuidaPlate — central configuration (paths, thresholds, encodings).

All notebooks and app code should import from here instead of hardcoding paths.
"""

from __future__ import annotations

from pathlib import Path

from backend.clinical_constants import KDOQI_DAILY_LIMITS

# ---------------------------------------------------------------------------
# Project root (repository root; parent of backend/)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
BACKEND_DIR: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
NHANES_DIR: Path = RAW_DIR / "nhanes"
USDA_DIR: Path = RAW_DIR / "usda"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FOOD_DATABASE_CSV: Path = BACKEND_DIR / "data" / "food_database.csv"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS_DIR: Path = ROOT / "models"
# LSTM v3 — v3 clinical labels + occasion feature + proper masking + augmentation
# Accuracy: 91.80% | F1: 0.9191 | AUC: 0.9835
# HIGH sens: 88.44% | MOD sens: 90.82%
# Beats B2 on all metrics — deployed: 2026-06-23
LSTM_MODEL_PATH: Path = MODELS_DIR / "lstm_v3_final.keras"
LSTM_SCALER_PATH: Path = MODELS_DIR / "lstm_v3_scaler.pkl"
LSTM_LABEL_ENCODER_PATH: Path = MODELS_DIR / "lstm_v3_label_encoder.pkl"
# XGBoost v3 — weighted clinical score labels
# Raw features (no ratio leakage)
# Accuracy: 98.99% | F1 macro: 0.9853
# MOD sensitivity: 96.94% | McNemar p<0.0001
# Leakage resolved — deployed: 2026-06-23
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_v3.pkl"

# Reproducible splits / synthetic generation
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# NHANES (2017–2018, J cycle) — raw files
# ---------------------------------------------------------------------------
XPT_DR1TOT: Path = NHANES_DIR / "DR1TOT_J.xpt"
XPT_DR2TOT: Path = NHANES_DIR / "DR2TOT_J.xpt"
XPT_BIOPRO: Path = NHANES_DIR / "BIOPRO_J.xpt"
XPT_DEMO: Path = NHANES_DIR / "DEMO_J.xpt"

CSV_DR1TOT: Path = NHANES_DIR / "DR1TOT_J.csv"
CSV_DR2TOT: Path = NHANES_DIR / "DR2TOT_J.csv"
CSV_BIOPRO: Path = NHANES_DIR / "BIOPRO_J.csv"
CSV_DEMO: Path = NHANES_DIR / "DEMO_J.csv"

NHANES_XPT_FILES: list[Path] = [XPT_DR1TOT, XPT_DR2TOT, XPT_BIOPRO, XPT_DEMO]

# ---------------------------------------------------------------------------
# USDA FoodData Central — Foundation Foods (raw)
# ---------------------------------------------------------------------------
USDA_FOOD_CSV: Path = USDA_DIR / "food.csv"
USDA_FOOD_NUTRIENT_CSV: Path = USDA_DIR / "food_nutrient.csv"
USDA_NUTRIENT_CSV: Path = USDA_DIR / "nutrient.csv"

# ---------------------------------------------------------------------------
# CKD stage encoding (for models)
# ---------------------------------------------------------------------------
CKD_STAGE_ENCODING: dict[str, int] = {
    "G2": 1,
    "G3a": 2,
    "G3b": 3,
    "G4": 4,
    "G5": 5,
}

CKD_STAGE_DECODING: dict[int, str] = {v: k for k, v in CKD_STAGE_ENCODING.items()}

# LEGACY — DO NOT USE IN PRODUCTION
# These values were used in early EDA
# (notebook 01_data_exploration.ipynb only)
# Production limits are in:
# backend.clinical_constants.KDOQI_DAILY_LIMITS
LEGACY_EDA_THRESHOLDS: dict[str, dict[str, float]] = {
    "G2": {
        "potassium": 3500.0,
        "phosphorus": 1200.0,
        "protein": 90.0,
        "sodium": 2300.0,
    },
    "G3a": {
        "potassium": 3000.0,
        "phosphorus": 1000.0,
        "protein": 75.0,
        "sodium": 2300.0,
    },
    "G3b": {
        "potassium": 2500.0,  # legacy — production: 3000.0
        "phosphorus": 900.0,  # legacy — production: 800.0
        "protein": 65.0,  # legacy abs g/day — production: 0.6 g/kg
        "sodium": 2000.0,  # legacy — production: 2300.0
    },
    "G4": {
        "potassium": 2000.0,
        "phosphorus": 800.0,
        "protein": 56.0,
        "sodium": 2000.0,
    },
    "G5": {
        "potassium": 1500.0,
        "phosphorus": 800.0,
        "protein": 50.0,
        "sodium": 1800.0,
    },
    # Do not use — see clinical_constants.py
}

# g/kg/day — derived from production limits. Prefer KDOQI_DAILY_LIMITS in new code.
PROTEIN_PER_KG_THRESHOLDS: dict[str, float] = {
    stage: limits["protein_per_kg"] for stage, limits in KDOQI_DAILY_LIMITS.items()
} | {"G5": 0.55}

# ---------------------------------------------------------------------------
# Daily meal budget (fraction of daily nutrient target)
# ---------------------------------------------------------------------------
MEAL_BUDGET_BREAKFAST: float = 0.25
MEAL_BUDGET_LUNCH: float = 0.35
MEAL_BUDGET_DINNER: float = 0.40

# sanity check
assert abs(MEAL_BUDGET_BREAKFAST + MEAL_BUDGET_LUNCH + MEAL_BUDGET_DINNER - 1.0) < 1e-9
