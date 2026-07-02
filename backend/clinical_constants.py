"""
GuidaPlate Clinical Constants
Source: KDOQI 2020 Nutrition in CKD
(Am J Kidney Dis. 2020;76(3) Suppl 1)
and KDIGO 2024 CKD Guidelines
"""

# Daily nutrient limits per CKD stage
# Protein: KDOQI 2020 Table 11 exact match
# Sodium: KDOQI 2020 Guideline 5.1 exact
# Potassium/Phosphorus: conservative clinical
# practice derived from KDOQI/KDIGO guidance
KDOQI_DAILY_LIMITS = {
    "G2": {
        "potassium": 3500.0,
        "phosphorus": 1000.0,
        "protein_per_kg": 0.8,
        "sodium": 2300.0,
    },
    "G3a": {
        "potassium": 3000.0,
        "phosphorus": 800.0,
        "protein_per_kg": 0.6,
        "sodium": 2300.0,
    },
    "G3b": {
        "potassium": 3000.0,
        "phosphorus": 800.0,
        "protein_per_kg": 0.6,
        "sodium": 2300.0,
    },
    "G4": {
        "potassium": 2500.0,
        "phosphorus": 700.0,
        "protein_per_kg": 0.55,
        "sodium": 2300.0,
    },
}

# eGFR ranges per stage
# Source: KDIGO 2024 Chapter 1
EGFR_RANGES = {
    "G2": "60–89",
    "G3a": "45–59",
    "G3b": "30–44",
    "G4": "15–29",
}

# Clinical severity score weights
# Author-derived from KDOQI/KDIGO
# clinical priority — not a published
# formula. Requires prospective validation.
CLINICAL_SEVERITY_WEIGHTS = {
    "potassium": 0.35,
    "phosphorus": 0.30,
    "protein": 0.25,
    "sodium": 0.10,
}

# Risk label thresholds
# Author-derived. Calibrated on NHANES.
SEVERITY_THRESHOLDS = {
    "HIGH": 1.2,
    "MODERATE": 0.7,
}

# Near-limit warning threshold (80% of daily)
NEAR_LIMIT_RATIO = 0.8

# Stage numeric encoding for ML features
STAGE_NUMERIC = {
    "G2": 1,
    "G3a": 2,
    "G3b": 3,
    "G4": 4,
}
