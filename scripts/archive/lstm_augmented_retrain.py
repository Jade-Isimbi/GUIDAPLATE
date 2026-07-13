"""
Retrain LSTM with truncated-sequence augmentation (notebook 05 extension).

Augments training data only; evaluates on original full test set plus
truncated test variants for before/after comparison.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_STEPS = 6
N_FEATURES = 4
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
BASELINE_ACC = 0.9071
ACC_TOLERANCE = 0.02

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"

BEST_CONFIG = {
    "lstm1_units": 64,
    "lstm2_units": 32,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "label": "higher dropout",
}


def augment_with_truncated_sequences(X: np.ndarray, y: np.ndarray, n_steps: int = 6, cutoffs: range | None = None):
    """Truncate each sequence at cutoffs, zero-pad remainder, keep label."""
    if cutoffs is None:
        cutoffs = range(1, n_steps)
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    for cutoff in cutoffs:
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    return X_final, y_final


def scale_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n = X.shape[0]
    flat = X.reshape(-1, N_FEATURES)
    scaled = scaler.transform(flat).reshape(n, N_STEPS, N_FEATURES)
    return scaled


def build_model(config: dict) -> Sequential:
    return Sequential(
        [
            LSTM(
                config["lstm1_units"],
                input_shape=(N_STEPS, N_FEATURES),
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                dropout=config["dropout"],
                recurrent_dropout=0.1,
                name="lstm_layer_1",
            ),
            LSTM(
                config["lstm2_units"],
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                dropout=config["dropout"],
                name="lstm_layer_2",
            ),
            Dense(16, activation="relu", name="dense_layer"),
            Dropout(0.3, name="dropout_layer"),
            Dense(3, activation="softmax", name="output_layer"),
        ],
        name="GuidaPlate_LSTM",
    )


def evaluate_full(model, X_test, idx_test) -> dict:
    proba = model.predict(X_test, verbose=0)
    pred = np.argmax(proba, axis=1)
    high_idx = RISK_ENCODE["HIGH"]
    high_tp = ((idx_test == high_idx) & (pred == high_idx)).sum()
    high_fn = ((idx_test == high_idx) & (pred != high_idx)).sum()
    high_recall = high_tp / (high_tp + high_fn) if (high_tp + high_fn) > 0 else 0.0
    return {
        "accuracy": accuracy_score(idx_test, pred),
        "precision": precision_score(idx_test, pred, average="weighted", zero_division=0),
        "recall": recall_score(idx_test, pred, average="weighted", zero_division=0),
        "f1_score": f1_score(idx_test, pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(idx_test, proba, multi_class="ovr", average="weighted"),
        "high_risk_sensitivity": high_recall,
    }


def evaluate_truncated_by_cutoff(model, X_test_raw, idx_test, scaler) -> list[dict]:
    rows = []
    for cutoff in range(1, N_STEPS):
        X_trunc_raw = X_test_raw.copy()
        X_trunc_raw[:, cutoff:, :] = 0
        X_trunc = scale_sequences(X_trunc_raw, scaler)
        proba = model.predict(X_trunc, verbose=0)
        pred = np.argmax(proba, axis=1)
        low_frac = (pred == RISK_ENCODE["LOW"]).mean()
        rows.append(
            {
                "meals_present": cutoff,
                "accuracy": accuracy_score(idx_test, pred),
                "f1_score": f1_score(idx_test, pred, average="weighted", zero_division=0),
                "pct_predicted_low": round(float(low_frac) * 100, 2),
            }
        )
    return rows


def class_balance(y_onehot: np.ndarray) -> dict[str, float]:
    idx = np.argmax(y_onehot, axis=1)
    counts = np.bincount(idx, minlength=3)
    total = len(idx)
    return {RISK_CLASSES[i]: counts[i] / total for i in range(3)}


def main() -> None:
    cache = np.load(MODEL_DIR / "lstm_sequences_cache.npz", allow_pickle=True)
    X_seq = cache["sequences"]
    y_seq = cache["labels"]
    print(f"Loaded cache: X_seq {X_seq.shape}, labels {len(y_seq)}")

    y_encoded = np.array([RISK_ENCODE[r] for r in y_seq])
    y_cat = to_categorical(y_encoded, num_classes=3)

    scaler = StandardScaler()
    X_flat = X_seq.reshape(-1, N_FEATURES)
    scaler.fit(X_flat)

    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_seq,
        y_cat,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(f"Original training set: {X_train_raw.shape}")
    print(f"Test set (unchanged):  {X_test_raw.shape}")

    X_train_aug_raw, y_train_aug = augment_with_truncated_sequences(
        X_train_raw, y_train, n_steps=N_STEPS, cutoffs=range(1, N_STEPS)
    )
    print(f"Augmented training set: {X_train_aug_raw.shape} (expect 6x rows)")

    bal_orig = class_balance(y_train)
    bal_aug = class_balance(y_train_aug)
    print("Class balance original train:", {k: f"{v:.3f}" for k, v in bal_orig.items()})
    print("Class balance augmented train:", {k: f"{v:.3f}" for k, v in bal_aug.items()})

    X_train_aug = scale_sequences(X_train_aug_raw, scaler)
    X_test = scale_sequences(X_test_raw, scaler)

    old_model_path = MODEL_DIR / "lstm_final.keras"
    print(f"\nEvaluating pre-augmentation model: {old_model_path}")
    old_model = load_model(old_model_path)
    old_full = evaluate_full(old_model, X_test, idx_test)
    old_trunc = evaluate_truncated_by_cutoff(old_model, X_test_raw, idx_test, scaler)
    print(f"Original model full-sequence accuracy: {old_full['accuracy']:.4f}")

    print("\nFine-tuning pretrained model on augmented dataset (weighted, lr=0.0001)...")
    tf.random.set_seed(RANDOM_STATE)
    model = load_model(old_model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    n_orig = len(X_train_raw)
    sample_weights = np.ones(len(X_train_aug), dtype=float)
    for block in range(1, N_STEPS):
        start = block * n_orig
        end = (block + 1) * n_orig
        sample_weights[start:end] = 0.15

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(MODEL_DIR / "lstm_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train_aug,
        y_train_aug,
        sample_weight=sample_weights,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    new_full = evaluate_full(model, X_test, idx_test)
    new_trunc = evaluate_truncated_by_cutoff(model, X_test_raw, idx_test, scaler)
    print(f"\nAugmented model full-sequence accuracy: {new_full['accuracy']:.4f}")

    acc_delta = new_full["accuracy"] - old_full["accuracy"]
    within_tolerance = new_full["accuracy"] >= (BASELINE_ACC - ACC_TOLERANCE)
    print(f"Delta vs saved baseline ({BASELINE_ACC:.4f}): {acc_delta:+.4f}")
    print(f"Within {ACC_TOLERANCE:.0%} tolerance: {within_tolerance}")

    comparison_rows = []
    comparison_rows.append(
        {
            "model": "original",
            "eval_type": "full_sequence",
            "meals_present": 6,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in old_full.items()},
            "pct_predicted_low": None,
        }
    )
    for row in old_trunc:
        comparison_rows.append({"model": "original", "eval_type": "truncated", **row})
    comparison_rows.append(
        {
            "model": "augmented",
            "eval_type": "full_sequence",
            "meals_present": 6,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in new_full.items()},
            "pct_predicted_low": None,
        }
    )
    for row in new_trunc:
        comparison_rows.append({"model": "augmented", "eval_type": "truncated", **row})

    comparison_df = pd.DataFrame(comparison_rows)
    out_path = STATS_DIR / "16_lstm_augmentation_comparison.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison: {out_path}")
    print(comparison_df.to_string(index=False))

    if within_tolerance:
        model.save(MODEL_DIR / "lstm_final.keras")
        print("\nSaved models/lstm_final.keras (augmented retrain)")
        print("Scaler and label encoder unchanged (same fit on full X_seq)")
    else:
        print(
            "\nNOT overwriting lstm_final.keras — full-sequence accuracy dropped "
            f"below {BASELINE_ACC - ACC_TOLERANCE:.4f}"
        )

    summary = {
        "train_raw_shape": list(X_train_raw.shape),
        "train_aug_shape": list(X_train_aug_raw.shape),
        "test_shape": list(X_test_raw.shape),
        "old_full_accuracy": old_full["accuracy"],
        "new_full_accuracy": new_full["accuracy"],
        "saved_model": within_tolerance,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
