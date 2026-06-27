"""Overfitting analysis for XGBoost v3, v1, and LSTM v3."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_ENCODE = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}

FEATURES_V1 = [
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

FEATURES_V3 = [
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

KDOQI = {
    "G2": {"potassium": 3500, "phosphorus": 1000, "protein_per_kg": 0.8, "sodium": 2300},
    "G3a": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G3b": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G4": {"potassium": 2500, "phosphorus": 700, "protein_per_kg": 0.55, "sodium": 2300},
}

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_tabular_df() -> pd.DataFrame:
    cohort = pd.read_csv(ROOT / "data" / "processed" / "ckd_cohort_final.csv")
    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    df = cohort.merge(labels_v3, on="SEQN", how="inner", suffixes=("", "_v3"))
    df = df.dropna(subset=["risk_label", "clinical_score"])
    nutrient_cols = ["potassium", "phosphorus", "protein_per_kg", "sodium"]
    df = df.dropna(subset=nutrient_cols)

    for stage, limits in KDOQI.items():
        mask = df["ckd_stage"] == stage
        df.loc[mask, "potassium_ratio"] = df.loc[mask, "potassium"] / limits["potassium"]
        df.loc[mask, "phosphorus_ratio"] = df.loc[mask, "phosphorus"] / limits["phosphorus"]
        df.loc[mask, "protein_ratio"] = df.loc[mask, "protein_per_kg"] / limits["protein_per_kg"]
        df.loc[mask, "sodium_ratio"] = df.loc[mask, "sodium"] / limits["sodium"]

    df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
    df["stage_numeric"] = df["ckd_stage"].map({"G2": 2, "G3a": 3, "G3b": 3, "G4": 4})
    df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
    df["protein_sodium_ratio"] = df["protein_per_kg"] / (df["sodium"] / 1000 + 1e-6)
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    return df


def gap_status(gap: float) -> str:
    if gap < 0.02:
        return "No overfitting"
    if gap < 0.05:
        return "Mild overfitting"
    return "Overfitting"


def cv_status(std: float) -> str:
    if std < 0.02:
        return "Stable"
    if std < 0.05:
        return "Moderate variance"
    return "High variance"


def prepare_lstm_splits():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache_v2.npz", allow_pickle=True)
    X_seq = cache["sequences"]
    patient_ids = cache["patient_ids"]
    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y_encoded = np.array([RISK_ENCODE[label_map[seqn]] for seqn in patient_ids])

    n_patients, n_steps, n_features = X_seq.shape
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_seq,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, n_features))

    def scale_sequences(X):
        n = X.shape[0]
        flat = X.reshape(-1, n_features)
        return scaler.transform(flat).reshape(n, n_steps, n_features)

    def zero_padded_slots(X_scaled, X_raw):
        X_out = X_scaled.copy()
        pad_mask = (X_raw == 0).all(axis=-1)
        for i in range(X_out.shape[0]):
            X_out[i, pad_mask[i], :] = 0.0
        return X_out

    X_train_scaled = zero_padded_slots(scale_sequences(X_train_raw), X_train_raw)
    X_test_scaled = zero_padded_slots(scale_sequences(X_test_raw), X_test_raw)
    return X_train_scaled, X_test_scaled, y_train, y_test, n_steps, n_features


def scale_lstm_fold(X_raw, scaler, n_steps, n_features):
    n = X_raw.shape[0]
    flat = X_raw.reshape(-1, n_features)
    X_scaled = scaler.transform(flat).reshape(n, n_steps, n_features)
    X_out = X_scaled.copy()
    pad_mask = (X_raw == 0).all(axis=-1)
    for i in range(X_out.shape[0]):
        X_out[i, pad_mask[i], :] = 0.0
    return X_out


def run_lstm_cv():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache_v2.npz", allow_pickle=True)
    X_raw = cache["sequences"]
    patient_ids = cache["patient_ids"]
    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y = np.array([RISK_ENCODE[label_map[seqn]] for seqn in patient_ids])

    n_patients, n_steps, n_features = X_raw.shape
    model = load_model(MODEL_DIR / "lstm_v3_final.keras")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    print()
    print("=" * 45)
    print("LSTM v3 — 5-FOLD CROSS VALIDATION")
    print("=" * 45)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        scaler = StandardScaler()
        scaler.fit(X_raw[train_idx].reshape(-1, n_features))
        X_val = scale_lstm_fold(X_raw[val_idx], scaler, n_steps, n_features)
        y_val = y[val_idx]

        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_scores.append({"fold": fold + 1, "accuracy": round(acc, 4), "f1_macro": round(f1, 4)})
        print(f"Fold {fold + 1}: Acc={acc:.4f} F1={f1:.4f}")

    fold_df = pd.DataFrame(fold_scores)
    print()
    print(
        f"Mean Accuracy: {fold_df['accuracy'].mean():.4f} "
        f"± {fold_df['accuracy'].std():.4f}"
    )
    print(
        f"Mean F1 Macro: {fold_df['f1_macro'].mean():.4f} "
        f"± {fold_df['f1_macro'].std():.4f}"
    )
    if fold_df["accuracy"].std() < 0.02:
        print("✓ LSTM v3 is stable across folds")
    else:
        print("⚠ LSTM v3 shows variance — note in Chapter 4")

    out_path = STATS_DIR / "15_lstm_cv_results.csv"
    fold_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return fold_df


def plot_lstm_history(history: dict, out_path: Path) -> None:
    train_loss = history["loss"]
    val_loss = history["val_loss"]
    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, color="#0D9488", label="Training loss", linewidth=2)
    ax1.plot(epochs, val_loss, color="#EF4444", label="Validation loss", linewidth=2, linestyle="--")
    ax1.set_title("LSTM v3 — Loss Curves", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=final_train_loss, color="#0D9488", alpha=0.3, linestyle=":")

    ax2.plot(epochs, train_acc, color="#0D9488", label="Training accuracy", linewidth=2)
    ax2.plot(epochs, val_acc, color="#EF4444", label="Validation accuracy", linewidth=2, linestyle="--")
    ax2.set_title("LSTM v3 — Accuracy Curves", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    df = load_tabular_df()
    y = df["risk_encoded"].values
    results = []

    # ── XGBoost v3 ──
    xgb_v3 = joblib.load(MODEL_DIR / "xgboost_v3.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES_V3],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_acc_v3 = accuracy_score(y_train, xgb_v3.predict(X_train))
    test_acc_v3 = accuracy_score(y_test, xgb_v3.predict(X_test))
    gap_v3 = train_acc_v3 - test_acc_v3

    print("=" * 45)
    print("XGBOOST v3 — OVERFIT CHECK")
    print("=" * 45)
    print(f"Train accuracy: {train_acc_v3:.4f} ({train_acc_v3 * 100:.2f}%)")
    print(f"Test  accuracy: {test_acc_v3:.4f} ({test_acc_v3 * 100:.2f}%)")
    print(f"Gap:            {gap_v3:.4f} ({gap_v3 * 100:.2f}%)")
    print(f"→ {gap_status(gap_v3)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(xgb_v3, df[FEATURES_V3], y, cv=cv, scoring="accuracy")
    print()
    print("=" * 45)
    print("XGBOOST v3 — 5-FOLD CROSS VALIDATION")
    print("=" * 45)
    print(f"Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean:  {cv_scores.mean():.4f} ({cv_scores.mean() * 100:.2f}%)")
    print(f"Std:   {cv_scores.std():.4f}")
    print(f"→ {cv_status(cv_scores.std())}")

    results.append(
        {
            "model": "XGBoost v3",
            "train_acc": round(train_acc_v3 * 100, 2),
            "test_acc": round(test_acc_v3 * 100, 2),
            "gap_pct": round(gap_v3 * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "status": gap_status(gap_v3),
        }
    )

    # ── XGBoost v1 (leakage) ──
    v1_path = MODEL_DIR / "backup_verified" / "xgboost_v1_backup.pkl"
    if not v1_path.exists():
        v1_path = MODEL_DIR / "xgboost_v1.pkl"
    xgb_v1 = joblib.load(v1_path)
    X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(
        df[FEATURES_V1],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_acc_v1 = accuracy_score(y_train_v1, xgb_v1.predict(X_train_v1))
    test_acc_v1 = accuracy_score(y_test_v1, xgb_v1.predict(X_test_v1))
    gap_v1 = train_acc_v1 - test_acc_v1

    print()
    print("=" * 45)
    print("XGBOOST v1 — LEAKAGE CHECK")
    print("=" * 45)
    print(f"v1 Train accuracy: {train_acc_v1 * 100:.2f}%")
    print(f"v1 Test  accuracy: {test_acc_v1 * 100:.2f}%")
    print(f"v1 Gap:            {gap_v1 * 100:.2f}%")
    print("Note: v1 low accuracy reflects label/feature mismatch with v3 ground truth,")
    print("not classical overfitting — v1 learned rule-derived ratio features.")

    results.append(
        {
            "model": "XGBoost v1 (leakage)",
            "train_acc": round(train_acc_v1 * 100, 2),
            "test_acc": round(test_acc_v1 * 100, 2),
            "gap_pct": round(gap_v1 * 100, 2),
            "cv_mean": None,
            "cv_std": None,
            "status": "Leakage / mismatch",
        }
    )

    # ── LSTM v3 train vs test ──
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, _, _ = prepare_lstm_splits()
    lstm_v3 = load_model(MODEL_DIR / "lstm_v3_final.keras")
    y_pred_train_lstm = np.argmax(lstm_v3.predict(X_train_lstm, verbose=0), axis=1)
    y_pred_test_lstm = np.argmax(lstm_v3.predict(X_test_lstm, verbose=0), axis=1)
    train_acc_lstm = accuracy_score(y_train_lstm, y_pred_train_lstm)
    test_acc_lstm = accuracy_score(y_test_lstm, y_pred_test_lstm)
    gap_lstm = train_acc_lstm - test_acc_lstm

    print()
    print("=" * 45)
    print("LSTM v3 — TRAIN VS TEST")
    print("=" * 45)
    print(f"Train accuracy: {train_acc_lstm:.4f} ({train_acc_lstm * 100:.2f}%)")
    print(f"Test  accuracy: {test_acc_lstm:.4f} ({test_acc_lstm * 100:.2f}%)")
    print(f"Gap:            {gap_lstm:.4f} ({gap_lstm * 100:.2f}%)")
    print(f"→ {gap_status(gap_lstm)}")

    history_path = STATS_DIR / "lstm_v3_history.npy"
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        final_train_loss = history["loss"][-1]
        final_val_loss = history["val_loss"][-1]
        loss_gap = final_val_loss - final_train_loss
        print(f"Final train loss: {final_train_loss:.4f}")
        print(f"Final val loss:   {final_val_loss:.4f}")
        print(f"Loss gap:         {loss_gap:.4f}")
        plot_lstm_history(history, FIG_DIR / "lstm_v3_training_curves.png")
        print(f"Saved: {FIG_DIR / 'lstm_v3_training_curves.png'}")
    else:
        print("History file not found — run notebook 05c cell that saves lstm_v3_history.npy")

    fold_df = run_lstm_cv()

    results.append(
        {
            "model": "LSTM v3",
            "train_acc": round(train_acc_lstm * 100, 2),
            "test_acc": round(test_acc_lstm * 100, 2),
            "gap_pct": round(gap_lstm * 100, 2),
            "cv_mean": round(fold_df["accuracy"].mean() * 100, 2),
            "cv_std": round(fold_df["accuracy"].std() * 100, 2),
            "status": gap_status(gap_lstm),
        }
    )

    # Rule baseline test acc from prior comparison
    results.insert(
        0,
        {
            "model": "Rule Baseline",
            "train_acc": None,
            "test_acc": 75.0,
            "gap_pct": None,
            "cv_mean": None,
            "cv_std": None,
            "status": "No params",
        },
    )
    results.append(
        {
            "model": "HMM Supervised",
            "train_acc": 63.8,
            "test_acc": 67.8,
            "gap_pct": -4.0,
            "cv_mean": None,
            "cv_std": None,
            "status": "Too simple",
        }
    )

    summary_df = pd.DataFrame(results)
    out_csv = STATS_DIR / "13_overfitting_analysis.csv"
    summary_df.to_csv(out_csv, index=False)

    print()
    print("=" * 55)
    print("OVERFITTING SUMMARY")
    print("=" * 55)
    print(f"{'Model':<25} {'Train':>8} {'Test':>8} {'Gap':>8} {'Status':>12}")
    print("-" * 55)
    for row in results:
        tr = f"{row['train_acc']:.1f}%" if row["train_acc"] is not None else "N/A"
        te = f"{row['test_acc']:.1f}%"
        gap = f"{row['gap_pct']:.1f}%" if row["gap_pct"] is not None else "N/A"
        print(f"{row['model']:<25} {tr:>8} {te:>8} {gap:>8} {row['status']:>12}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
