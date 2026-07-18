"""
xgboost_model.py
GuidaPlate — XGBoost classifier for dietary risk prediction (HIGH/MODERATE/LOW)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from backend.config import (
    CKD_STAGE_ENCODING,
    XGBOOST_MEAL_NOSCORE_MODEL_PATH,
    XGBOOST_MODEL_PATH,
)
from backend.clinical_constants import (
    CLINICAL_SEVERITY_WEIGHTS,
    KDOQI_DAILY_LIMITS,
    NEAR_LIMIT_RATIO,
)

_predictor: XGBoostRiskPredictor | None = None
_meal_predictor: XGBoostRiskPredictor | None = None

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

# Default live meal model — no clinical_score; occasion + meal caps instead.
FEATURE_ORDER_MEAL_NOSCORE = [
    "potassium",
    "phosphorus",
    "protein_per_kg",
    "sodium",
    "ckd_stage_encoded",
    "stage_numeric",
    "k_p_product",
    "protein_sodium_ratio",
    "occasion_encoded",
    "meal_cap_potassium",
    "meal_cap_phosphorus",
    "meal_cap_protein_per_kg",
    "meal_cap_sodium",
]

OCCASION_ENCODE = {"Breakfast": 0, "Lunch": 1, "Dinner": 2, "Snack": 3}

STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

_CLINICAL_SCORE_NUTRIENTS = (
    ("potassium", "potassium"),
    ("phosphorus", "phosphorus"),
    ("protein_per_kg", "protein"),
    ("sodium", "sodium"),
)

VALID_OCCASIONS = frozenset({"Breakfast", "Lunch", "Dinner", "Snack"})


def compute_clinical_score(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
) -> float:
    """Day-scale clinical score (production day v3). UNCHANGED formula."""
    limits = KDOQI_DAILY_LIMITS[ckd_stage]
    score = 0.0
    values = {
        "potassium": potassium,
        "phosphorus": phosphorus,
        "protein_per_kg": protein_per_kg,
        "sodium": sodium,
    }
    for nutrient, weight_key in _CLINICAL_SCORE_NUTRIENTS:
        weight = CLINICAL_SEVERITY_WEIGHTS[weight_key]
        ratio = values[nutrient] / limits[nutrient]
        if ratio > 1.0:
            score += weight * (1 + (ratio - 1) * 2)
        else:
            score += weight * ratio
    return score


def meal_limits_for_occasion(ckd_stage: str, occasion: str) -> dict[str, float]:
    """
    Occasion caps = KDOQI_DAILY_LIMITS × OCCASION_RULES[occasion]["nutrient_caps"].
    Single source of truth: meal_planner.OCCASION_RULES (no hardcoded fractions).
    """
    if ckd_stage not in KDOQI_DAILY_LIMITS:
        raise KeyError(f"Unknown CKD stage: {ckd_stage!r}")
    if occasion not in VALID_OCCASIONS:
        raise KeyError(f"Unknown occasion: {occasion!r}")

    from backend.api.meal_planner import OCCASION_RULES

    daily = KDOQI_DAILY_LIMITS[ckd_stage]
    fk, fp, fpro, fna = OCCASION_RULES[occasion]["nutrient_caps"]
    return {
        "potassium": float(daily["potassium"] * fk),
        "phosphorus": float(daily["phosphorus"] * fp),
        "protein_per_kg": float(daily["protein_per_kg"] * fpro),
        "sodium": float(daily["sodium"] * fna),
    }


def compute_clinical_score_meal(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
    occasion: str,
) -> float:
    """Meal-scale clinical score (matches xgboost_v3_meal training)."""
    limits = meal_limits_for_occasion(ckd_stage, occasion)
    score = 0.0
    values = {
        "potassium": potassium,
        "phosphorus": phosphorus,
        "protein_per_kg": protein_per_kg,
        "sodium": sodium,
    }
    for nutrient, weight_key in _CLINICAL_SCORE_NUTRIENTS:
        weight = CLINICAL_SEVERITY_WEIGHTS[weight_key]
        ratio = values[nutrient] / limits[nutrient]
        if ratio > 1.0:
            score += weight * (1 + (ratio - 1) * 2)
        else:
            score += weight * ratio
    return score


def compute_exceeded_nutrients_meal(
    potassium: float,
    phosphorus: float,
    protein_per_kg: float,
    sodium: float,
    ckd_stage: str,
    occasion: str,
) -> tuple[list[str], list[str]]:
    """
    Compare intake against meal caps (same NEAR_LIMIT_RATIO as day path).
    Returns exceeded (>=100%) and near_limit (80–99%).
    """
    limits = meal_limits_for_occasion(ckd_stage, occasion)
    nutrient_values = {
        "potassium": (potassium, limits["potassium"]),
        "phosphorus": (phosphorus, limits["phosphorus"]),
        "protein": (protein_per_kg, limits["protein_per_kg"]),
        "sodium": (sodium, limits["sodium"]),
    }
    exceeded: list[str] = []
    near_limit: list[str] = []
    for name, (value, limit) in nutrient_values.items():
        if limit <= 0:
            continue
        ratio = value / limit
        if ratio >= 1.0:
            exceeded.append(name)
        elif ratio >= NEAR_LIMIT_RATIO:
            near_limit.append(name)
    return exceeded, near_limit


class XGBoostRiskPredictor:
    """Wrapper for the trained XGBoost 3-class dietary risk classifier."""

    # RISK_ENCODE from notebook 04c: LOW=0, MODERATE=1, HIGH=2
    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    def __init__(
        self,
        model_path: Path | None = None,
        score_mode: Literal["day", "meal", "meal_noscore"] = "day",
    ) -> None:
        # Default args preserve historical day behavior for get_predictor().
        path = Path(model_path) if model_path is not None else Path(XGBOOST_MODEL_PATH)
        self.score_mode: Literal["day", "meal", "meal_noscore"] = score_mode
        self.model_path = path
        self.feature_order = (
            FEATURE_ORDER_MEAL_NOSCORE
            if score_mode == "meal_noscore"
            else FEATURE_ORDER
        )
        try:
            if not path.exists():
                raise FileNotFoundError(
                    f"XGBoost model not found at {path}. "
                    "Run notebooks/04c_xgboost_v3_raw_features.ipynb to generate xgboost_v3.pkl."
                )
            self.model = joblib.load(path)
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
        occasion: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        if ckd_stage not in CKD_STAGE_ENCODING:
            raise KeyError(f"Unknown CKD stage: {ckd_stage!r}")
        if ckd_stage not in STAGE_NUMERIC:
            raise KeyError(f"Unknown CKD stage for v3 features: {ckd_stage!r}")

        ckd_stage_encoded = float(CKD_STAGE_ENCODING[ckd_stage])
        stage_numeric = float(STAGE_NUMERIC[ckd_stage])
        k_p_product = (potassium * phosphorus) / 1e6
        protein_sodium_ratio = protein_per_kg / (sodium / 1000 + 1e-6)

        if self.score_mode == "meal_noscore":
            if occasion is None or occasion not in VALID_OCCASIONS:
                raise ValueError(
                    f"occasion is required for meal-scale scoring; got {occasion!r}"
                )
            caps = meal_limits_for_occasion(ckd_stage, occasion)
            features_used = {
                "potassium": float(potassium),
                "phosphorus": float(phosphorus),
                "protein_per_kg": float(protein_per_kg),
                "sodium": float(sodium),
                "ckd_stage_encoded": ckd_stage_encoded,
                "stage_numeric": stage_numeric,
                "k_p_product": float(k_p_product),
                "protein_sodium_ratio": float(protein_sodium_ratio),
                "occasion_encoded": float(OCCASION_ENCODE[occasion]),
                "meal_cap_potassium": float(caps["potassium"]),
                "meal_cap_phosphorus": float(caps["phosphorus"]),
                "meal_cap_protein_per_kg": float(caps["protein_per_kg"]),
                "meal_cap_sodium": float(caps["sodium"]),
            }
            return np.array(
                [[features_used[name] for name in self.feature_order]],
                dtype=float,
            ), features_used

        if self.score_mode == "meal":
            if occasion is None or occasion not in VALID_OCCASIONS:
                raise ValueError(
                    f"occasion is required for meal-scale scoring; got {occasion!r}"
                )
            clinical_score = compute_clinical_score_meal(
                potassium, phosphorus, protein_per_kg, sodium, ckd_stage, occasion
            )
        else:
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
            [[features_used[name] for name in self.feature_order]],
            dtype=float,
        ), features_used

    def predict(
        self,
        potassium: float,
        phosphorus: float,
        protein_per_kg: float,
        sodium: float,
        ckd_stage: str,
        occasion: str | None = None,
    ) -> dict:
        feature_vector, features_used = self._build_features(
            potassium, phosphorus, protein_per_kg, sodium, ckd_stage, occasion
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
        occasion: str | None = None,
    ) -> dict:
        """Compute SHAP values for a single prediction. Returns nutrient contributions as percentages."""
        import shap

        features, _ = self._build_features(
            potassium, phosphorus, protein_per_kg, sodium, ckd_stage, occasion
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
            "potassium": self.feature_order.index("potassium"),
            "phosphorus": self.feature_order.index("phosphorus"),
            "protein": self.feature_order.index("protein_per_kg"),
            "sodium": self.feature_order.index("sodium"),
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
    """Day-scale singleton — path/behavior unchanged (xgboost_v3.pkl)."""
    global _predictor
    if _predictor is None:
        _predictor = XGBoostRiskPredictor()  # defaults: XGBOOST_MODEL_PATH, score_mode="day"
    return _predictor


def get_meal_predictor() -> XGBoostRiskPredictor:
    """
    Live meal-scale predictor — noscore model only (occasion + meal caps).

    Legacy xgboost_v3_meal.pkl and day xgboost_v3.pkl are offline research
    artifacts and are never loaded here. Model failure is handled by the
    meal-scale rule fallback in /api/predict/risk.
    """
    global _meal_predictor
    if _meal_predictor is None:
        _meal_predictor = XGBoostRiskPredictor(
            model_path=XGBOOST_MEAL_NOSCORE_MODEL_PATH,
            score_mode="meal_noscore",
        )
    return _meal_predictor


def get_live_risk_predictor() -> tuple[XGBoostRiskPredictor, Literal["meal"]]:
    """Return the live meal predictor for /api/predict/risk and health checks."""
    return get_meal_predictor(), "meal"


def smoke_predict_live_risk_predictor() -> dict:
    """
    Run a fixed nutrient smoke prediction on the live-selected XGBoost model.
    Live scoring is always meal-scale and requires an occasion.
    """
    predictor, scoring_scale = get_live_risk_predictor()
    kwargs: dict = {
        "potassium": 2800.0,
        "phosphorus": 650.0,
        "protein_per_kg": 0.55,
        "sodium": 1800.0,
        "ckd_stage": "G3b",
    }
    kwargs["occasion"] = "Lunch"
    result = predictor.predict(**kwargs)
    return {
        "scoring_scale": scoring_scale,
        "risk_label": result["risk_label"],
        "score_mode": predictor.score_mode,
        "uses_clinical_score_feature": "clinical_score" in result["features_used"],
    }
