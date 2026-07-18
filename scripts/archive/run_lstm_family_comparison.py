#!/usr/bin/env python3
"""
LSTM-family architecture comparison (pattern task, risk labels v3).

Trains GRU / Bidirectional LSTM / SimpleRNN / 1D-CNN on the same holdout +
truncated-sequence augmentation protocol as notebooks/05c_lstm_v3_improved.ipynb,
and scores models/lstm_v3_final.keras predict-only on the unaugmented test set.

Never writes to models/lstm_v3_final.keras or models/lstm_v3_scaler.pkl.
Reads models/archive/lstm_sequences_cache_v2.npz in place (no copy/move/restore).

Note on 1D-CNN: no Masking layer — documented architectural difference (CNNs do
not consume zero-pad masks the same way recurrent Masking does).

Usage (repo root):
  ./venv311/bin/python3 scripts/run_lstm_family_comparison.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    GRU,
    LSTM,
    Masking,
    SimpleRNN,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs" / "stats"
CACHE = ROOT / "models" / "archive" / "lstm_sequences_cache_v2.npz"
LABELS = STATS / "05_risk_labels_v3.csv"
LSTM_KERAS = ROOT / "models" / "lstm_v3_final.keras"
LSTM_SCALER = ROOT / "models" / "lstm_v3_scaler.pkl"
OUT_CSV = STATS / "14_lstm_family_comparison.csv"

PROTECTED_SHA256 = (
    "aba54112efac35ab8382cd225dba59798c0911c1c2ce5f80b21004f3154b2b26"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_STEPS = 6
N_FEATURES = 5
EXPECTED_N = 1830
EXPECTED_LABELS = {"LOW": 608, "MODERATE": 489, "HIGH": 733}
EXPECTED_TRAIN = 1464
EXPECTED_TEST = 366
EXPECTED_AUG = 8784

RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_protected(when: str) -> str:
    digest = sha256_file(LSTM_KERAS)
    if digest != PROTECTED_SHA256:
        raise RuntimeError(
            f"FATAL [{when}]: models/lstm_v3_final.keras SHA-256 mismatch!\n"
            f"  expected: {PROTECTED_SHA256}\n"
            f"  got:      {digest}\n"
            "Stopping — do not trust this run as a baseline."
        )
    print(f"[{when}] lstm_v3_final.keras SHA-256 OK: {digest}")
    return digest


def augment_with_truncated_sequences(X, y, n_steps=6):
    """Verbatim from notebooks/05c_lstm_v3_improved.ipynb."""
    X_augmented = [X.copy()]
    y_augmented = [y.copy()]
    for cutoff in range(1, n_steps):
        X_truncated = X.copy()
        X_truncated[:, cutoff:, :] = 0
        X_augmented.append(X_truncated)
        y_augmented.append(y.copy())
    return np.concatenate(X_augmented, axis=0), np.concatenate(y_augmented, axis=0)


def zero_padded_slots(X_scaled, X_raw):
    """Verbatim logic from 05c — zero padded slots after scaling."""
    X_out = X_scaled.copy()
    pad_mask = (X_raw == 0).all(axis=-1)
    for i in range(X_out.shape[0]):
        X_out[i, pad_mask[i], :] = 0.0
    return X_out


def scale_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    flat = X.reshape(-1, N_FEATURES)
    return scaler.transform(flat).reshape(n, N_STEPS, N_FEATURES)


def per_class_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    total = int(cm.sum())
    out: dict[str, float] = {}
    for i, label in enumerate(RISK_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        out[label] = float(tn / denom) if denom > 0 else float("nan")
    return out


def metrics_row(
    model_name: str,
    notes: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    epochs_trained: int | None,
    train_seconds: float | None,
) -> dict:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    spec = per_class_specificity(y_true, y_pred)
    row: dict = {
        "model": model_name,
        "notes": notes,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc_ovr_weighted": float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
        ),
        "epochs_trained": epochs_trained if epochs_trained is not None else "",
        "train_seconds": (
            round(float(train_seconds), 2) if train_seconds is not None else ""
        ),
    }
    for i, label in enumerate(RISK_CLASSES):
        row[f"{label}_precision"] = float(prec[i])
        row[f"{label}_recall"] = float(rec[i])
        row[f"{label}_specificity"] = spec[label]
    return row


def compile_model(model: Sequential) -> Sequential:
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru() -> Sequential:
    return compile_model(
        Sequential(
            [
                Masking(mask_value=0.0, input_shape=(N_STEPS, N_FEATURES)),
                GRU(64, return_sequences=True),
                Dropout(0.3),
                GRU(32),
                Dropout(0.3),
                Dense(3, activation="softmax"),
            ],
            name="gru",
        )
    )


def build_bilstm() -> Sequential:
    return compile_model(
        Sequential(
            [
                Masking(mask_value=0.0, input_shape=(N_STEPS, N_FEATURES)),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dropout(0.3),
                Dense(3, activation="softmax"),
            ],
            name="bilstm",
        )
    )


def build_simplernn() -> Sequential:
    return compile_model(
        Sequential(
            [
                Masking(mask_value=0.0, input_shape=(N_STEPS, N_FEATURES)),
                SimpleRNN(64, return_sequences=True),
                Dropout(0.3),
                SimpleRNN(32),
                Dropout(0.3),
                Dense(3, activation="softmax"),
            ],
            name="simplernn",
        )
    )


def build_cnn1d() -> Sequential:
    # No Masking — documented architectural difference vs recurrent candidates / LSTM v3.
    return compile_model(
        Sequential(
            [
                Conv1D(
                    64,
                    kernel_size=3,
                    activation="relu",
                    padding="same",
                    input_shape=(N_STEPS, N_FEATURES),
                ),
                Conv1D(32, kernel_size=3, activation="relu", padding="same"),
                GlobalMaxPooling1D(),
                Dense(3, activation="softmax"),
            ],
            name="cnn1d",
        )
    )


def make_callbacks() -> list:
    # Exact EarlyStopping / ReduceLROnPlateau from 05c
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_candidate(
    name: str,
    builder,
    X_train_aug: np.ndarray,
    y_train_aug: np.ndarray,
    class_weight: dict,
) -> tuple[Sequential, int, float]:
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = builder()
    callbacks = make_callbacks()
    early = callbacks[0]
    print(f"\nTraining {name}...")
    t0 = time.time()
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )
    elapsed = time.time() - t0
    epochs_ran = len(history.history["loss"])
    print(
        f"  {name}: epochs={epochs_ran} "
        f"(early_stop patience={early.patience}) "
        f"wall={elapsed:.1f}s"
    )
    return model, epochs_ran, elapsed


def main() -> int:
    STATS.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists():
        print(f"STOP: refusing to overwrite existing {OUT_CSV}")
        return 1

    assert_protected("before")
    scaler_before = sha256_file(LSTM_SCALER)
    print(f"[before] lstm_v3_scaler.pkl SHA-256 (read-only check): {scaler_before}")

    # 1. Load archive cache (read-only path) + remap v3 labels
    if not CACHE.exists():
        print(f"STOP: missing cache {CACHE}")
        return 1

    cache = np.load(CACHE, allow_pickle=True)
    X_seq = cache["sequences"]
    y_seq_orig = cache["labels"]
    patient_ids = cache["patient_ids"]

    labels_v3 = pd.read_csv(LABELS)
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y_seq_v3 = np.array(
        [label_map.get(seqn, y_seq_orig[i]) for i, seqn in enumerate(patient_ids)]
    )

    if X_seq.shape != (EXPECTED_N, N_STEPS, N_FEATURES):
        print(
            f"STOP: sequence shape {X_seq.shape} != "
            f"({EXPECTED_N}, {N_STEPS}, {N_FEATURES})"
        )
        return 1
    if len(y_seq_v3) != EXPECTED_N:
        print(f"STOP: label length {len(y_seq_v3)} != {EXPECTED_N}")
        return 1

    counts = pd.Series(y_seq_v3).value_counts().reindex(RISK_CLASSES).fillna(0).astype(int)
    expected_series = pd.Series(EXPECTED_LABELS)
    if not counts.equals(expected_series):
        print("STOP: v3 label distribution mismatch vs expected LOW/MOD/HIGH.")
        print(f"  got:      {counts.to_dict()}")
        print(f"  expected: {EXPECTED_LABELS}")
        return 1
    print(f"[OK] label counts: {counts.to_dict()} (n={EXPECTED_N})")

    y_encoded = np.array([RISK_ENCODE[r] for r in y_seq_v3])

    # 2. Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_seq,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    if len(X_train_raw) != EXPECTED_TRAIN or len(X_test_raw) != EXPECTED_TEST:
        print(
            f"STOP: split train={len(X_train_raw)} test={len(X_test_raw)} "
            f"!= {EXPECTED_TRAIN}/{EXPECTED_TEST}"
        )
        return 1
    print(f"[OK] split: train={len(X_train_raw)} test={len(X_test_raw)}")

    # 3. NEW scaler for candidates only (never load production scaler here)
    candidate_scaler = StandardScaler()
    candidate_scaler.fit(X_train_raw.reshape(-1, N_FEATURES))
    X_train_scaled = zero_padded_slots(
        scale_sequences(candidate_scaler, X_train_raw), X_train_raw
    )
    X_test_scaled = zero_padded_slots(
        scale_sequences(candidate_scaler, X_test_raw), X_test_raw
    )
    print(
        f"[OK] candidate scaler fitted on train only; "
        f"scaled shapes train={X_train_scaled.shape} test={X_test_scaled.shape}"
    )

    # 4. Truncated-sequence augmentation (train only)
    X_train_aug, y_train_aug = augment_with_truncated_sequences(
        X_train_scaled, y_train, n_steps=N_STEPS
    )
    if X_train_aug.shape[0] != EXPECTED_AUG:
        print(
            f"STOP: augmented train size {X_train_aug.shape[0]} != {EXPECTED_AUG}"
        )
        return 1
    if len(X_test_scaled) != EXPECTED_TEST:
        print("STOP: test set size changed after scaling")
        return 1
    print(
        f"[OK] augmented train={X_train_aug.shape[0]} "
        f"(test remains {len(X_test_scaled)}, unaugmented)"
    )

    classes = np.unique(y_train_aug)
    class_weights_arr = compute_class_weight(
        "balanced", classes=classes, y=y_train_aug
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    print(f"[OK] class_weight={class_weight}")

    # 5. Train candidates
    candidates = [
        (
            "GRU",
            "Masking→GRU(64,rs=True)→Dropout(0.3)→GRU(32)→Dropout(0.3)→Dense(3); "
            "Adam lr=0.001; ES patience=10; ReduceLR patience=5; "
            "val_split=0.1; batch=32; epochs≤100; new train-only StandardScaler",
            build_gru,
        ),
        (
            "BidirectionalLSTM",
            "Masking→BiLSTM(64,rs=True)→Dropout(0.3)→BiLSTM(32)→Dropout(0.3)→Dense(3); "
            "same training config; new train-only StandardScaler",
            build_bilstm,
        ),
        (
            "SimpleRNN",
            "Masking→SimpleRNN(64,rs=True)→Dropout(0.3)→SimpleRNN(32)→Dropout(0.3)→Dense(3); "
            "same training config; new train-only StandardScaler",
            build_simplernn,
        ),
        (
            "CNN1D",
            "NO Masking (documented difference): "
            "Conv1D(64,k=3,same)→Conv1D(32,k=3,same)→GlobalMaxPooling1D→Dense(3); "
            "same training config; new train-only StandardScaler",
            build_cnn1d,
        ),
    ]

    rows: list[dict] = []
    y_test_np = np.asarray(y_test)

    for name, notes, builder in candidates:
        model, epochs_ran, elapsed = train_candidate(
            name, builder, X_train_aug, y_train_aug, class_weight
        )
        y_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        row = metrics_row(
            name,
            notes,
            y_test_np,
            y_pred,
            y_prob,
            epochs_trained=epochs_ran,
            train_seconds=elapsed,
        )
        rows.append(row)
        print(
            f"  {name} test: acc={row['accuracy']:.4f} "
            f"f1_macro={row['f1_macro']:.4f} "
            f"auc={row['auc_roc_ovr_weighted']:.4f}"
        )
        del model
        tf.keras.backend.clear_session()

    # 6. Production LSTM predict-only with production scaler only
    print("\nScoring production lstm_v3_final.keras (predict-only)...")
    prod_scaler = joblib.load(LSTM_SCALER)
    X_test_prod = zero_padded_slots(
        scale_sequences(prod_scaler, X_test_raw), X_test_raw
    )
    prod_model = load_model(LSTM_KERAS)
    y_prob_prod = prod_model.predict(X_test_prod, verbose=0)
    y_pred_prod = np.argmax(y_prob_prod, axis=1)
    rows.append(
        metrics_row(
            "LSTM_v3_production",
            "predict_only models/lstm_v3_final.keras; "
            "scaled with models/lstm_v3_scaler.pkl only (not used for candidates)",
            y_test_np,
            y_pred_prod,
            y_prob_prod,
            epochs_trained=None,
            train_seconds=None,
        )
    )
    print(
        f"  LSTM_v3_production test: acc={rows[-1]['accuracy']:.4f} "
        f"f1_macro={rows[-1]['f1_macro']:.4f} "
        f"auc={rows[-1]['auc_roc_ovr_weighted']:.4f}"
    )

    # 7–8. Save NEW CSV
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # 9. Hash confirmation
    after = assert_protected("after")
    scaler_after = sha256_file(LSTM_SCALER)
    if scaler_after != scaler_before:
        print(
            "FATAL: lstm_v3_scaler.pkl hash changed during run "
            f"({scaler_before} → {scaler_after})"
        )
        return 2
    print(f"[after] lstm_v3_scaler.pkl unchanged: {scaler_after}")
    print(f"HASH_CONFIRMATION={after}")
    print(f"HASH_MATCHES_EXPECTED={after == PROTECTED_SHA256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
