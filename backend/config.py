"""
GuidaPlate — central configuration (paths, thresholds, encodings).

All notebooks and app code should import from here instead of hardcoding paths.
"""

from __future__ import annotations

from pathlib import Path

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
LSTM_MODEL_PATH: Path = MODELS_DIR / "lstm_final.keras"
LSTM_SCALER_PATH: Path = MODELS_DIR / "lstm_scaler.pkl"
LSTM_LABEL_ENCODER_PATH: Path = MODELS_DIR / "lstm_label_encoder.pkl"
XGBOOST_MODEL_PATH: Path = MODELS_DIR / "xgboost_v1.pkl"
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
RANDOM_FOREST_PATH: Path = MODELS_DIR / "random_forest.pkl"


# Saved by `03_lstm_training.ipynb` so `04_evaluation.ipynb` uses the same test split
EVAL_TEST_NPZ: Path = PROCESSED_DIR / "guidaplate_test_split.npz"

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
# Processed outputs
# ---------------------------------------------------------------------------
CKD_PATIENTS_CSV: Path = PROCESSED_DIR / "ckd_patients.csv"
CKD_PATIENTS_CLEAN_CSV: Path = PROCESSED_DIR / "ckd_patients_clean.csv"
FOOD_NUTRIENTS_CLEAN_CSV: Path = PROCESSED_DIR / "food_nutrients_clean.csv"

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

# ---------------------------------------------------------------------------
# Dietary risk thresholds (daily totals) — EDA / synthetic / app risk logic
# ---------------------------------------------------------------------------
# Evidence-aligned targets (KDOQI 2020, KDIGO 2024, NKF) — mg/day except protein (g/day).
DIETARY_RISK_THRESHOLDS: dict[str, dict[str, float]] = {
    "G2": {"potassium": 3500.0, "phosphorus": 1200.0, "protein": 90.0, "sodium": 2300.0},
    "G3a": {"potassium": 3000.0, "phosphorus": 1000.0, "protein": 75.0, "sodium": 2300.0},
    "G3b": {"potassium": 2500.0, "phosphorus": 900.0, "protein": 65.0, "sodium": 2000.0},
    "G4": {"potassium": 2000.0, "phosphorus": 800.0, "protein": 56.0, "sodium": 2000.0},
    "G5": {"potassium": 1500.0, "phosphorus": 800.0, "protein": 50.0, "sodium": 1800.0},
}

# Used for XGBoost protein_ratio feature — must match
# notebook 04 THRESHOLDS exactly (g/kg/day, not g/day)
PROTEIN_PER_KG_THRESHOLDS: dict[str, float] = {
    "G2": 0.8,
    "G3a": 0.6,
    "G3b": 0.6,
    "G4": 0.55,
    "G5": 0.55,
}

# ---------------------------------------------------------------------------
# Daily meal budget (fraction of daily nutrient target)
# ---------------------------------------------------------------------------
MEAL_BUDGET_BREAKFAST: float = 0.25
MEAL_BUDGET_LUNCH: float = 0.35
MEAL_BUDGET_DINNER: float = 0.40

# sanity check
assert abs(MEAL_BUDGET_BREAKFAST + MEAL_BUDGET_LUNCH + MEAL_BUDGET_DINNER - 1.0) < 1e-9
