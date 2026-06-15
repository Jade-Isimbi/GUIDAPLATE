"""
xgboost_model.py
GuidaPlate — XGBoost classifier for dietary risk prediction (HIGH/MODERATE/LOW)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    CKD_STAGE_ENCODING,
    DIETARY_RISK_THRESHOLDS,
    PROTEIN_PER_KG_THRESHOLDS,
    XGBOOST_MODEL_PATH,
)

_predictor: XGBoostRiskPredictor | None = None

NUTRIENT_THRESHOLD_KEYS = {
    "potassium": "potassium",
    "phosphorus": "phosphorus",
    "protein_per_kg": "protein_per_kg",
    "sodium": "sodium",
}

FEATURE_ORDER = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "potassium_ratio",
    "phosphorus_ratio",
    "protein_ratio",
    "sodium_ratio",
    "ckd_stage_encoded",
]


class XGBoostRiskPredictor:
    """Wrapper for the trained XGBoost 3-class dietary risk classifier."""

    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    def __init__(self) -> None:
        try:
            if not Path(XGBOOST_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"XGBoost model not found at {XGBOOST_MODEL_PATH}. "
                    "Run notebooks/04_xgboost_training.ipynb to generate xgboost_v1.pkl."
                )
            self.model = joblib.load(XGBOOST_MODEL_PATH)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise

    def _threshold(self, ckd_stage: str, nutrient: str) -> float:
        if ckd_stage not in DIETARY_RISK_THRESHOLDS:
            raise KeyError(f"Unknown CKD stage: {ckd_stage!r}")
        thresholds = DIETARY_RISK_THRESHOLDS[ckd_stage]
        key = NUTRIENT_THRESHOLD_KEYS[nutrient]
        if key in thresholds:
            return float(thresholds[key])
        if nutrient == "protein_per_kg" and "protein" in thresholds:
            return float(thresholds["protein"])
        raise KeyError(f"Threshold for {nutrient!r} not found for stage {ckd_stage!r}")

    def predict(
        self,
        potassium: float,
        phosphorus: float,
        protein_per_kg: float,
        sodium: float,
        ckd_stage: str,
    ) -> dict:
        if ckd_stage not in CKD_STAGE_ENCODING:
            raise KeyError(f"Unknown CKD stage: {ckd_stage!r}")

        potassium_ratio = potassium / self._threshold(ckd_stage, "potassium")
        phosphorus_ratio = phosphorus / self._threshold(ckd_stage, "phosphorus")
        protein_ratio = protein_per_kg / PROTEIN_PER_KG_THRESHOLDS[ckd_stage]
        sodium_ratio = sodium / self._threshold(ckd_stage, "sodium")
        ckd_stage_encoded = float(CKD_STAGE_ENCODING[ckd_stage])

        features_used = {
            "potassium": float(potassium),
            "phosphorus": float(phosphorus),
            "protein_per_kg": float(protein_per_kg),
            "sodium": float(sodium),
            "potassium_ratio": float(potassium_ratio),
            "phosphorus_ratio": float(phosphorus_ratio),
            "protein_ratio": float(protein_ratio),
            "sodium_ratio": float(sodium_ratio),
            "ckd_stage_encoded": ckd_stage_encoded,
        }

        feature_vector = np.array(
            [[features_used[name] for name in FEATURE_ORDER]],
            dtype=float,
        )

        proba = self.model.predict_proba(feature_vector)[0]
        class_idx = int(np.argmax(proba))
        risk_label = self.LABEL_MAP[class_idx]
        confidence = float(proba[class_idx])

        probabilities = {
            self.LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))
        }

        return {
            "risk_label": risk_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "features_used": features_used,
        }


def get_predictor() -> XGBoostRiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = XGBoostRiskPredictor()
    return _predictor
