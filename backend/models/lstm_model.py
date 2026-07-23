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
FEATURES_PER_STEP = 5

# Slot encoding (matches notebooks/05b ablation B2)
OCCASION_ENCODING = [
    0.00,  # slot 0: Day1 Breakfast
    0.33,  # slot 1: Day1 Lunch
    0.67,  # slot 2: Day1 Dinner
    0.00,  # slot 3: Day2 Breakfast
    0.33,  # slot 4: Day2 Lunch
    0.67,  # slot 5: Day2 Dinner
]


class LSTMPatternAnalyzer:
    """Wrapper for the trained LSTM 3-class dietary pattern classifier."""

    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

    def __init__(self) -> None:
        try:
            if not Path(LSTM_MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM model not found at {LSTM_MODEL_PATH}. "
                    "Run notebooks/05c_lstm_v3_improved.ipynb to generate lstm_v3_final.keras."
                )
            if not Path(LSTM_SCALER_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM scaler not found at {LSTM_SCALER_PATH}. "
                    "Run notebooks/05c_lstm_v3_improved.ipynb to generate lstm_v3_scaler.pkl."
                )
            if not Path(LSTM_LABEL_ENCODER_PATH).exists():
                raise FileNotFoundError(
                    f"LSTM label encoder not found at {LSTM_LABEL_ENCODER_PATH}. "
                    "Run notebooks/05c_lstm_v3_improved.ipynb to generate lstm_v3_label_encoder.pkl."
                )

            self.model = load_model(LSTM_MODEL_PATH)
            self.scaler = joblib.load(LSTM_SCALER_PATH)
            self.label_encoder = joblib.load(LSTM_LABEL_ENCODER_PATH)

            # Warm up the graph so layer I/O tensors are defined (Keras 3).
            _dummy = np.zeros((1, SEQUENCE_STEPS, FEATURES_PER_STEP), dtype=float)
            self.model(_dummy, training=False)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            raise

    def analyze(self, meal_sequence: list[list[float]]) -> dict:
        """Run the trained LSTM sequence-risk classifier (LOW / MODERATE / HIGH)."""
        sequence_length = len(meal_sequence)

        # Build raw (6, 5) — occasion only on filled slots; empty slots stay all-zero
        # (matches notebook 05c training padding for Masking after scaling)
        raw = np.zeros((SEQUENCE_STEPS, FEATURES_PER_STEP), dtype=float)
        for i, step in enumerate(meal_sequence[:SEQUENCE_STEPS]):
            step_arr = np.asarray(step, dtype=float)
            raw[i, :4] = step_arr[:4]
            if len(step_arr) >= 5:
                raw[i, 4] = float(step_arr[4])
            else:
                raw[i, 4] = OCCASION_ENCODING[i]

        flat = self.scaler.transform(raw.reshape(-1, FEATURES_PER_STEP))
        scaled = flat.reshape(SEQUENCE_STEPS, FEATURES_PER_STEP)

        pad_mask = (raw == 0).all(axis=-1)
        scaled[pad_mask, :] = 0.0

        model_input = scaled.reshape(1, SEQUENCE_STEPS, FEATURES_PER_STEP)

        proba = self.model(model_input, training=False).numpy()[0]
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
            "sequence_length": sequence_length,
        }


def get_analyzer() -> LSTMPatternAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = LSTMPatternAnalyzer()
    return _analyzer


def warmup_lstm() -> None:
    """Eager-load the LSTM model at application startup."""
    get_analyzer()
