"""Run trend prediction experiment (notebook 09 logic) for result verification."""
from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
N_STEPS = 6
N_FEATURES = 4

THRESHOLDS = {
    "G2": {
        "potassium": 3500 / 3,
        "phosphorus": 1000 / 3,
        "protein": 0.8 / 3,
        "sodium": 2300 / 3,
    }
}
T = THRESHOLDS["G2"]


def is_empty_slot(vec: np.ndarray) -> bool:
    return not np.any(vec != 0)


def meal_risk_state(meal_vec: np.ndarray) -> int:
    k, p, pr, s = meal_vec
    exceed = sum(
        [
            k > T["potassium"],
            p > T["phosphorus"],
            pr > T["protein"],
            s > T["sodium"],
        ]
    )
    if exceed >= 2:
        return 2
    if exceed == 1:
        return 1
    return 0


def build_examples(sequences: np.ndarray, patient_ids: np.ndarray):
    X_list, len_list, y_list, pid_list, last_state_list, t_list = [], [], [], [], [], []
    excluded_target = excluded_input = 0

    for i in range(len(sequences)):
        seq = sequences[i]
        states = [meal_risk_state(seq[j]) for j in range(N_STEPS)]

        for t in range(N_STEPS - 1):
            if is_empty_slot(seq[t + 1]):
                excluded_target += 1
                continue
            if any(is_empty_slot(seq[k]) for k in range(t + 1)):
                excluded_input += 1
                continue

            padded = np.zeros((N_STEPS, N_FEATURES), dtype=float)
            padded[: t + 1] = seq[: t + 1]
            X_list.append(padded)
            len_list.append(t + 1)
            y_list.append(states[t + 1])
            pid_list.append(patient_ids[i])
            last_state_list.append(states[t])
            t_list.append(t)

    return {
        "X": np.array(X_list),
        "seq_len": np.array(len_list, dtype=float),
        "y": np.array(y_list),
        "patient_ids": np.array(pid_list),
        "last_state": np.array(last_state_list),
        "prefix_t": np.array(t_list),
        "excluded_target": excluded_target,
        "excluded_input": excluded_input,
    }


def scale_nutrients(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n = X.shape[0]
    flat = X.reshape(-1, N_FEATURES)
    scaled = scaler.transform(flat).reshape(n, N_STEPS, N_FEATURES)
    return scaled


def evaluate(y_true, y_pred, y_proba=None, name="model"):
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    row = {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_LOW": round(float(f1_per[0]), 4),
        "f1_MODERATE": round(float(f1_per[1]), 4),
        "f1_HIGH": round(float(f1_per[2]), 4),
    }
    if y_proba is not None:
        row["auc_roc"] = round(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"), 4
        )
    else:
        row["auc_roc"] = None
    return row


def main():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache.npz", allow_pickle=True)
    sequences = cache["sequences"]
    labels = cache["labels"]
    patient_ids = cache["patient_ids"]

    data = build_examples(sequences, patient_ids)
    print("STEP 1")
    print(f"Total examples: {len(data['y'])}")
    print(f"Excluded (empty target): {data['excluded_target']}")
    print(f"Excluded (empty in prefix): {data['excluded_input']}")
    vc = pd.Series([RISK_CLASSES[y] for y in data["y"]]).value_counts()
    print("Target distribution:")
    print(vc)

    # Patient-level split
    unique_pids = np.unique(patient_ids)
    pid_labels = []
    for pid in unique_pids:
        idx = np.where(patient_ids == pid)[0][0]
        pid_labels.append(RISK_ENCODE[labels[idx]])
    train_pids, test_pids = train_test_split(
        unique_pids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=pid_labels,
    )
    train_pids = set(train_pids)
    train_mask = np.array([p in train_pids for p in data["patient_ids"]])
    test_mask = ~train_mask

    X = data["X"]
    y = data["y"]
    seq_len = data["seq_len"]
    last_state = data["last_state"]

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    seq_len_train, seq_len_test = seq_len[train_mask], seq_len[test_mask]
    last_state_train, last_state_test = last_state[train_mask], last_state[test_mask]

    print("\nSTEP 2")
    print(f"Train examples: {len(y_train)}, Test examples: {len(y_test)}")
    print(f"Train patients: {len(train_pids)}, Test patients: {len(set(test_pids))}")

    # Transition matrix
    with open(MODEL_DIR / "archive" / "transition_matrix.json") as f:
        tm = json.load(f)
    tm_idx = {
        0: [tm["LOW"]["LOW"], tm["LOW"]["MODERATE"], tm["LOW"]["HIGH"]],
        1: [tm["MODERATE"]["LOW"], tm["MODERATE"]["MODERATE"], tm["MODERATE"]["HIGH"]],
        2: [tm["HIGH"]["LOW"], tm["HIGH"]["MODERATE"], tm["HIGH"]["HIGH"]],
    }

    # Baselines
    majority = int(np.bincount(y_train, minlength=3).argmax())
    pred_majority = np.full_like(y_test, majority)

    pred_repeat = last_state_test.copy()

    pred_trans = np.array([int(np.argmax(tm_idx[s])) for s in last_state_test])

    rows = []
    rows.append(evaluate(y_test, pred_majority, name="majority_class"))
    rows.append(evaluate(y_test, pred_repeat, name="repeat_last_meal"))
    rows.append(evaluate(y_test, pred_trans, name="transition_matrix"))

    print("\nSTEP 3 baselines")
    print(pd.DataFrame(rows).to_string(index=False))

    # Scale nutrients
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, N_FEATURES))
    X_train_s = scale_nutrients(X_train, scaler)
    X_test_s = scale_nutrients(X_test, scaler)

    # Normalize seq_len to [0,1]
    seq_len_train_n = (seq_len_train - 1) / (N_STEPS - 1)
    seq_len_test_n = (seq_len_test - 1) / (N_STEPS - 1)

    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)

    # GRU model - 6181 examples, ~4945 train -> GRU simpler than stacked LSTM
    nut_in = Input(shape=(N_STEPS, N_FEATURES), name="nutrients")
    len_in = Input(shape=(1,), name="seq_length")
    x = GRU(32, dropout=0.2, name="gru")(nut_in)
    x = Concatenate(name="concat_len")([x, len_in])
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(3, activation="softmax", name="next_meal")(x)
    model = Model(inputs=[nut_in, len_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    tf.random.set_seed(RANDOM_STATE)
    model.fit(
        [X_train_s, seq_len_train_n],
        y_train_cat,
        validation_split=0.15,
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        verbose=0,
    )

    proba = model.predict([X_test_s, seq_len_test_n], verbose=0)
    pred = np.argmax(proba, axis=1)
    rows.append(evaluate(y_test, pred, proba, name="trained_GRU"))

    print("\nSTEP 4-5 comparison")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nClassification report (trained GRU):")
    print(classification_report(y_test, pred, target_names=RISK_CLASSES, zero_division=0))

    df.to_csv(STATS_DIR / "17_trend_prediction_comparison.csv", index=False)
    print(f"\nSaved {STATS_DIR / '17_trend_prediction_comparison.csv'}")


if __name__ == "__main__":
    main()
