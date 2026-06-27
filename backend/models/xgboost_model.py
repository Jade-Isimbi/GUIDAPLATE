"""
xgboost_model.py
GuidaPlate — XGBoost classifier for dietary risk prediction (HIGH/MODERATE/LOW)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from backend.config import CKD_STAGE_ENCODING, XGBOOST_MODEL_PATH

_predictor: XGBoostRiskPredictor | None = None

# v3 feature order — must match notebooks/04c_xgboost_v3_raw_features.ipynb
FEATURE_ORDER = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "ckd_stage_encoded",
    "stage_numeric",
    "k_p_product",
    "protein_sodium_ratio",
    "clinical_score",
]

STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

WEIGHTS = {
    "potassium": 0.35,
    "phosphorus": 0.30,
    "protein_per_kg": 0.25,
    "sodium": 0.10,
}

KDOQI_LIMITS = {
    "G2": {
        "potassium": 3500,
        "phosphorus": 1000,
        "protein_per_kg": 0.8,
        "sodium": 2300,
    },
    "G3a": {
        "potassium": 3000,
        "phosphorus": 800,
        "protein_per_kg": 0.6,
        "sodium": 2300,
    },
    "G3b": {
        "potassium": 3000,
        "phosphorus": 800,
        "protein_per_kg": 0.6,
        "sodium": 2300,
    },
    "G4": {
        "potassium": 2500,
        "phosphorus": 700,
        "protein_per_kg": 0.55,
        "sodium": 2300,
    },
}


def compute_clinical_score(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> float:
    limits = KDOQI_LIMITS[ckd_stage]
    score = 0.0
    values = {
        "potassium": potassium,
        "phosphorus": phosphorus,
        "protein_per_kg": protein_per_kg,
        "sodium": sodium,
    }
    for nutrient, weight in WEIGHTS.items():
        ratio = values[nutrient] / limits[nutrient]
        if ratio > 1.0:
            score += weight * (1 + (ratio - 1) * 2)
        else:
            score += weight * ratio
    return score


class XGBoostRiskPredictor:
    """Wrapper for the trained XGBoost 3-class dietary risk classifier."""

    # RISK_ENCODE from notebook 04c: LOW=0, MODERATE=1, HIGH=2
    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    def __init__(self) -> None:
        try:
            if not Path(XGBOOST_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"XGBoost model not found at {XGBOOST_MODEL_PATH}. "
                    "Run notebooks/04c_xgboost_v3_raw_features.ipynb to generate xgboost_v3.pkl."
                )
            self.model = joblib.load(XGBOOST_MODEL_PATH)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise

    def _build_features(
        self,
        potassium: float,
        phosphorus: float,
        protein_per_kg: float,
        sodium: float,
        ckd_stage: str,
    ) -> tuple[np.ndarray, dict]:
        if ckd_stage not in CKD_STAGE_ENCODING:
            raise KeyError(f"Unknown CKD stage: {ckd_stage!r}")
        if ckd_stage not in STAGE_NUMERIC:
            raise KeyError(f"Unknown CKD stage for v3 features: {ckd_stage!r}")

        ckd_stage_encoded = float(CKD_STAGE_ENCODING[ckd_stage])
        stage_numeric = float(STAGE_NUMERIC[ckd_stage])
        k_p_product = (potassium * phosphorus) / 1e6
        protein_sodium_ratio = protein_per_kg / (sodium / 1000 + 1e-6)
        clinical_score = compute_clinical_score(
            potassium, phosphorus, protein_per_kg, sodium, ckd_stage
        )

        features_used = {
            "potassium": float(potassium),
            "phosphorus": float(phosphorus),
            "protein_per_kg": float(protein_per_kg),
            "sodium": float(sodium),
            "ckd_stage_encoded": ckd_stage_encoded,
            "stage_numeric": stage_numeric,
            "k_p_product": float(k_p_product),
            "protein_sodium_ratio": float(protein_sodium_ratio),
            "clinical_score": float(clinical_score),
        }

        return np.array(
            [[features_used[name] for name in FEATURE_ORDER]],
            dtype=float,
        ), features_used

    def predict(
        self,
        potassium: float,
        phosphorus: float,
        protein_per_kg: float,
        sodium: float,
        ckd_stage: str,
    ) -> dict:
        feature_vector, features_used = self._build_features(
            potassium, phosphorus, protein_per_kg, sodium, ckd_stage
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

    def explain(
        self,
        potassium: float,
        phosphorus: float,
        protein_per_kg: float,
        sodium: float,
        ckd_stage: str,
    ) -> dict:
        """Compute SHAP values for a single prediction. Returns nutrient contributions as percentages."""
        import shap

        features, _ = self._build_features(
            potassium, phosphorus, protein_per_kg, sodium, ckd_stage
        )

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)

        pred_class = int(self.model.predict(features)[0])

        if isinstance(shap_values, list):
            class_shap = shap_values[pred_class][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_shap = shap_values[0, :, pred_class]
        else:
            class_shap = shap_values[0]

        nutrient_indices = {
            "potassium": FEATURE_ORDER.index("potassium"),
            "phosphorus": FEATURE_ORDER.index("phosphorus"),
            "protein": FEATURE_ORDER.index("protein_per_kg"),
            "sodium": FEATURE_ORDER.index("sodium"),
        }

        raw = {
            k: abs(float(class_shap[v]))
            for k, v in nutrient_indices.items()
        }
        total = sum(raw.values()) or 1.0

        contributions = {
            k: round((v / total) * 100, 1)
            for k, v in raw.items()
        }

        dominant = max(contributions, key=contributions.get)

        return {
            "contributions": contributions,
            "dominant_nutrient": dominant,
            "dominant_pct": contributions[dominant],
        }


def get_predictor() -> XGBoostRiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = XGBoostRiskPredictor()
    return _predictor
