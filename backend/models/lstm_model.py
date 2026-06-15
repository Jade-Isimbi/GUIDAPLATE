"""
lstm_model.py
GuidaPlate — LSTM model for temporal dietary pattern detection
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from backend.config import (
    LSTM_LABEL_ENCODER_PATH,
    LSTM_MODEL_PATH,
    LSTM_SCALER_PATH,
)

_analyzer: LSTMPatternAnalyzer | None = None

SEQUENCE_STEPS = 6
FEATURES_PER_STEP = 4


class LSTMPatternAnalyzer:
    """Wrapper for the trained LSTM 3-class dietary pattern classifier."""

    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    def __init__(self) -> None:
        try:
            if not Path(LSTM_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM model not found at {LSTM_MODEL_PATH}. "
                    "Run notebooks/05_lstm_training.ipynb to generate lstm_final.keras."
                )
            if not Path(LSTM_SCALER_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM scaler not found at {LSTM_SCALER_PATH}. "
                    "Run notebooks/05_lstm_training.ipynb to generate lstm_scaler.pkl."
                )
            if not Path(LSTM_LABEL_ENCODER_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM label encoder not found at {LSTM_LABEL_ENCODER_PATH}. "
                    "Run notebooks/05_lstm_training.ipynb to generate lstm_label_encoder.pkl."
                )

            self.model = load_model(LSTM_MODEL_PATH)
            self.scaler = joblib.load(LSTM_SCALER_PATH)
            self.label_encoder = joblib.load(LSTM_LABEL_ENCODER_PATH)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise

    def analyze(self, meal_sequence: list[list[float]]) -> dict:
        sequence_length = len(meal_sequence)

        sequence = np.zeros((SEQUENCE_STEPS, FEATURES_PER_STEP), dtype=float)
        for i, step in enumerate(meal_sequence[:SEQUENCE_STEPS]):
            sequence[i] = np.asarray(step[:FEATURES_PER_STEP], dtype=float)

        scaled = self.scaler.transform(sequence.reshape(-1, FEATURES_PER_STEP))
        model_input = scaled.reshape(1, SEQUENCE_STEPS, FEATURES_PER_STEP)

        proba = self.model(model_input, training=False).numpy()[0]
        class_idx = int(np.argmax(proba))
        risk_label = self.LABEL_MAP[class_idx]
        confidence = float(proba[class_idx])

        probabilities = {
            self.LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))
        }

        first_total = float(np.sum(sequence[0]))
        last_idx = min(sequence_length, SEQUENCE_STEPS) - 1
        last_total = float(np.sum(sequence[last_idx])) if sequence_length > 0 else 0.0
        trend = "escalating" if last_total > first_total else "stable"

        return {
            "risk_label": risk_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "sequence_length": sequence_length,
            "trend": trend,
        }


def get_analyzer() -> LSTMPatternAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = LSTMPatternAnalyzer()
    return _analyzer


def warmup_lstm() -> None:
    """Eager-load the LSTM model at application startup."""
    get_analyzer()
