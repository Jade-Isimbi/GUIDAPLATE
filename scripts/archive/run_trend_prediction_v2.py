"""
Trend prediction v2 — 5-fold patient CV with sequential improvements.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_STEPS = 6
N_NUTRIENTS = 4
N_RISK_OH = 3
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}

T = {"potassium": 3500 / 3, "phosphorus": 1000 / 3, "protein": 0.8 / 3, "sodium": 2300 / 3}


def is_empty_slot(vec):
    return not np.any(vec != 0)


def meal_risk_state(meal_vec):
    k, p, pr, s = meal_vec
    ex = sum([k > T["potassium"], p > T["phosphorus"], pr > T["protein"], s > T["sodium"]])
    return 2 if ex >= 2 else 1 if ex == 1 else 0


def risk_onehot(state: int) -> np.ndarray:
    oh = np.zeros(N_RISK_OH)
    oh[state] = 1.0
    return oh


def build_examples(sequences, patient_ids):
    X_nut, X_risk, lens, ys, pids = [], [], [], [], []
    ex_t = ex_i = 0
    for i in range(len(sequences)):
        seq = sequences[i]
        states = [meal_risk_state(seq[j]) for j in range(N_STEPS)]
        for t in range(N_STEPS - 1):
            if is_empty_slot(seq[t + 1]):
                ex_t += 1
                continue
            if any(is_empty_slot(seq[k]) for k in range(t + 1)):
                ex_i += 1
                continue
            nut = np.zeros((N_STEPS, N_NUTRIENTS))
            nut[: t + 1] = seq[: t + 1]
            risk = np.zeros((N_STEPS, N_RISK_OH))
            for s in range(t + 1):
                risk[s] = risk_onehot(states[s])
            X_nut.append(nut)
            X_risk.append(risk)
            lens.append(t + 1)
            ys.append(states[t + 1])
            pids.append(patient_ids[i])
    return {
        "X_nut": np.array(X_nut),
        "X_risk": np.array(X_risk),
        "seq_len": np.array(lens, dtype=float),
        "y": np.array(ys),
        "patient_ids": np.array(pids),
        "excluded_target": ex_t,
        "excluded_input": ex_i,
    }


def prepare_X(X_nut, X_risk, use_risk: bool):
    if use_risk:
        return np.concatenate([X_nut, X_risk], axis=-1)
    return X_nut


def scale_nutrients(X_nut, scaler):
    n = X_nut.shape[0]
    return scaler.transform(X_nut.reshape(-1, N_NUTRIENTS)).reshape(n, N_STEPS, N_NUTRIENTS)


def featurize(X_nut, X_risk, scaler, use_risk: bool):
    nut_s = scale_nutrients(X_nut, scaler)
    if use_risk:
        return np.concatenate([nut_s, X_risk], axis=-1)
    return nut_s


def build_model(n_features, units, layers, dropout):
    seq_in = Input(shape=(N_STEPS, n_features))
    len_in = Input(shape=(1,))
    x = seq_in
    for i in range(layers):
        x = GRU(units, dropout=dropout, return_sequences=(i < layers - 1), name=f"gru_{i+1}")(x)
    x = Concatenate()([x, len_in])
    x = Dense(16, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(3, activation="softmax")(x)
    model = Model([seq_in, len_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def compute_metrics(y_true, y_pred, y_proba):
    f1p = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    hi = RISK_ENCODE["HIGH"]
    hs = ((y_true == hi) & (y_pred == hi)).sum() / max((y_true == hi).sum(), 1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_LOW": float(f1p[0]),
        "f1_MODERATE": float(f1p[1]),
        "f1_HIGH": float(f1p[2]),
        "high_sensitivity": float(hs),
        "auc_roc": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
    }


def train_eval(X_nut_tr, X_risk_tr, len_tr, y_tr, X_nut_va, X_risk_va, len_va, y_va, cfg, use_risk, class_weight):
    tf.random.set_seed(RANDOM_STATE)
    scaler = StandardScaler()
    scaler.fit(X_nut_tr.reshape(-1, N_NUTRIENTS))
    X_tr = featurize(X_nut_tr, X_risk_tr, scaler, use_risk)
    X_va = featurize(X_nut_va, X_risk_va, scaler, use_risk)
    n_feat = X_tr.shape[2]
    len_tr_n = (len_tr - 1) / (N_STEPS - 1)
    len_va_n = (len_va - 1) / (N_STEPS - 1)
    cw = None
    if class_weight:
        w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr)
        cw = {i: float(w[i]) for i in range(3)}
    model = build_model(n_feat, cfg["units"], cfg["layers"], cfg["dropout"])
    history = model.fit(
        [X_tr, len_tr_n], to_categorical(y_tr, 3),
        validation_data=([X_va, len_va_n], to_categorical(y_va, 3)),
        epochs=50, batch_size=64, class_weight=cw, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)],
    )
    proba = model.predict([X_va, len_va_n], verbose=0)
    pred = np.argmax(proba, axis=1)
    return compute_metrics(y_va, pred, proba), model, scaler, history


def patient_mask(pids, selected):
    selected = set(selected)
    return np.array([p in selected for p in pids])


def run_cv(data, patient_folds, cfg, use_risk, class_weight):
    folds = []
    for train_pids, val_pids in patient_folds:
        tr = patient_mask(data["patient_ids"], train_pids)
        va = patient_mask(data["patient_ids"], val_pids)
        m, _, _, _ = train_eval(
            data["X_nut"][tr], data["X_risk"][tr], data["seq_len"][tr], data["y"][tr],
            data["X_nut"][va], data["X_risk"][va], data["seq_len"][va], data["y"][va],
            cfg, use_risk, class_weight,
        )
        folds.append(m)
    return folds


def summarize(folds):
    df = pd.DataFrame(folds)
    means = {c: df[c].mean() for c in df.columns}
    stds = {f"std_{c}": df[c].std() for c in df.columns}
    return means, stds, df


def means_to_row(prefix, means, stds=None):
    row = {"stage": prefix}
    for k, v in means.items():
        row[k] = round(v, 4)
        if stds:
            row[f"std_{k}"] = round(stds[f"std_{k}"], 4)
    return row


def main():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache.npz", allow_pickle=True)
    sequences = cache["sequences"]
    full_labels = cache["labels"]
    patient_ids_seq = cache["patient_ids"]
    data = build_examples(sequences, patient_ids_seq)
    print(f"Examples: {len(data['y'])}")

    unique_pids = np.unique(patient_ids_seq)
    pid_label = [RISK_ENCODE[full_labels[np.where(patient_ids_seq == p)[0][0]]] for p in unique_pids]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    patient_folds = [(unique_pids[tr], unique_pids[va]) for tr, va in skf.split(unique_pids, pid_label)]

    base_cfg = {"units": 32, "layers": 1, "dropout": 0.2, "label": "baseline (current)"}

    print("Step 1...")
    s1_f = run_cv(data, patient_folds, base_cfg, use_risk=False, class_weight=False)
    s1_m, s1_s, s1_df = summarize(s1_f)

    print("Step 2...")
    s2_f = run_cv(data, patient_folds, base_cfg, use_risk=False, class_weight=True)
    s2_m, s2_s, _ = summarize(s2_f)

    print("Step 3...")
    s3_f = run_cv(data, patient_folds, base_cfg, use_risk=True, class_weight=True)
    s3_m, s3_s, _ = summarize(s3_f)

    configs = [
        {"units": 32, "layers": 1, "dropout": 0.2, "label": "baseline (current)"},
        {"units": 64, "layers": 1, "dropout": 0.2, "label": "larger single layer"},
        {"units": 32, "layers": 1, "dropout": 0.4, "label": "higher dropout"},
        {"units": 32, "layers": 2, "dropout": 0.3, "label": "stacked GRU"},
    ]
    step4_rows = []
    print("Step 4...")
    for cfg in configs:
        fm = run_cv(data, patient_folds, cfg, use_risk=True, class_weight=True)
        m, s, _ = summarize(fm)
        row = {"config": cfg["label"]}
        for k in m:
            row[k] = round(m[k], 4)
            row[f"std_{k}"] = round(s[f"std_{k}"], 4)
        step4_rows.append(row)
        print(f"  {cfg['label']}: f1_w={row['f1_weighted']}, mod={row['f1_MODERATE']}")

    step4_df = pd.DataFrame(step4_rows)
    winner_cfg = configs[int(step4_df["f1_weighted"].idxmax())]
    print("Winner:", winner_cfg["label"])

    nb09 = pd.read_csv(STATS_DIR / "17_trend_prediction_comparison.csv")
    r = nb09[nb09["model"] == "trained_GRU"].iloc[0]
    progression = [
        {"stage": "nb09_single_split", "accuracy": r["accuracy"], "f1_weighted": r["f1_weighted"],
         "f1_MODERATE": r["f1_MODERATE"], "high_sensitivity": r["high_sensitivity"], "auc_roc": r["auc_roc"]},
        means_to_row("step1_cv_baseline", s1_m, s1_s),
        means_to_row("step2_class_weights", s2_m, s2_s),
        means_to_row("step3_risk_features", s3_m, s3_s),
    ]
    w = step4_df[step4_df["config"] == winner_cfg["label"]].iloc[0]
    progression.append({
        "stage": f"step4_{winner_cfg['label']}",
        "accuracy": w["accuracy"], "f1_weighted": w["f1_weighted"], "f1_MODERATE": w["f1_MODERATE"],
        "high_sensitivity": w["high_sensitivity"], "auc_roc": w["auc_roc"],
        "std_accuracy": w["std_accuracy"], "std_f1_weighted": w["std_f1_weighted"],
    })
    prog_df = pd.DataFrame(progression)
    prog_df.to_csv(STATS_DIR / "18_trend_prediction_v2_comparison.csv", index=False)
    step4_df.to_csv(STATS_DIR / "18_trend_prediction_v2_arch_sweep.csv", index=False)

    # Step comparisons
    comp12 = pd.DataFrame([
        {"comparison": "step1_to_step2", "metric": k, "before": round(s1_m[k], 4), "after": round(s2_m[k], 4),
         "delta": round(s2_m[k] - s1_m[k], 4)} for k in s1_m
    ])
    comp23 = pd.DataFrame([
        {"comparison": "step2_to_step3", "metric": k, "before": round(s2_m[k], 4), "after": round(s3_m[k], 4),
         "delta": round(s3_m[k] - s2_m[k], 4)} for k in s2_m
    ])
    comp12.to_csv(STATS_DIR / "18_trend_v2_step1_vs_step2.csv", index=False)
    comp23.to_csv(STATS_DIR / "18_trend_v2_step2_vs_step3.csv", index=False)

    # Final model 80/20 (same patient split as nb09)
    train_pids, test_pids = train_test_split(unique_pids, test_size=0.2, random_state=RANDOM_STATE, stratify=pid_label)
    tr = patient_mask(data["patient_ids"], train_pids)
    va = patient_mask(data["patient_ids"], test_pids)
    metrics, model, scaler, history = train_eval(
        data["X_nut"][tr], data["X_risk"][tr], data["seq_len"][tr], data["y"][tr],
        data["X_nut"][va], data["X_risk"][va], data["seq_len"][va], data["y"][va],
        winner_cfg, use_risk=True, class_weight=True,
    )
    model.save(MODEL_DIR / "trend_gru_v2.keras")
    joblib.dump(scaler, MODEL_DIR / "trend_gru_v2_scaler.pkl")
    joblib.dump({"config": winner_cfg, "use_risk": True, "class_weight": True}, MODEL_DIR / "trend_gru_v2_meta.pkl")

    X_va = featurize(data["X_nut"][va], data["X_risk"][va], scaler, True)
    pred = np.argmax(model.predict([X_va, (data["seq_len"][va] - 1) / (N_STEPS - 1)], verbose=0), axis=1)
    cm = confusion_matrix(data["y"][va], pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=RISK_CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Trend GRU v2 — {winner_cfg['label']} (80/20 test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "28_trend_prediction_v2_confusion_matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Trend GRU v2 Training Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Trend GRU v2 Training Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "29_trend_prediction_v2_training_history.png", dpi=150)
    plt.close()

    print("\n=== PROGRESSION ===")
    print(prog_df.to_string(index=False))
    print("\n=== FINAL 80/20 TEST ===")
    print(metrics)
    print("Saved outputs.")


if __name__ == "__main__":
    main()
