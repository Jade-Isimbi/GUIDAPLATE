"""Generate master model comparison metrics (used by notebook 06_model_comparison)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}

KDOQI = {
    "G2": {"potassium": 3500, "phosphorus": 1000, "protein_per_kg": 0.8, "sodium": 2300},
    "G3a": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G3b": {"potassium": 3000, "phosphorus": 800, "protein_per_kg": 0.6, "sodium": 2300},
    "G4": {"potassium": 2500, "phosphorus": 700, "protein_per_kg": 0.55, "sodium": 2300},
}

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


def project_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    return p


ROOT = project_root()
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def assign_rule_baseline_label(row) -> int | None:
    stage = row["ckd_stage"]
    if stage not in KDOQI:
        return None
    limits = KDOQI[stage]
    exceeded = 0
    for nutrient in ["potassium", "phosphorus", "protein_per_kg", "sodium"]:
        if pd.notna(row.get(nutrient)) and row[nutrient] > limits[nutrient]:
            exceeded += 1
    if exceeded >= 2:
        return 2
    if exceeded == 1:
        return 1
    return 0


def compute_all_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    classes = [0, 1, 2]
    class_names = RISK_CLASSES

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    specificity_per_class = []
    for i in range(len(classes)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc = None

    results = {
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "auc": round(auc, 4) if auc is not None else "N/A",
        "test_samples": len(y_true),
    }

    for i, cls in enumerate(class_names):
        results[f"precision_{cls}"] = round(float(precision_per_class[i]), 4)
        results[f"recall_{cls}"] = round(float(recall_per_class[i]), 4)
        results[f"f1_{cls}"] = round(float(f1_per_class[i]), 4)
        results[f"specificity_{cls}"] = round(float(specificity_per_class[i]), 4)

    return results


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "statistic": round(statistic, 4), "p_value": round(p_value, 6)}


def mcnemar_row(comparison: str, y_true, y_pred_a, y_pred_b) -> dict:
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    return {
        "comparison": comparison,
        **result,
        "significant": result["p_value"] < 0.05,
    }


def derive_rule_lstm_labels(patient_ids, idx_test) -> np.ndarray:
    cohort = pd.read_csv(ROOT / "data" / "processed" / "ckd_cohort_final.csv")
    cohort_idx = cohort.set_index("SEQN")
    test_seqn = patient_ids[idx_test]
    lstm_test_df = cohort_idx.loc[test_seqn].reset_index()
    return lstm_test_df.apply(assign_rule_baseline_label, axis=1).values


def scale_lstm_sequences(X_raw, scaler, n_steps, n_features):
    n = X_raw.shape[0]
    flat = X_raw.reshape(-1, n_features)
    X_scaled = scaler.transform(flat).reshape(n, n_steps, n_features)
    X_out = X_scaled.copy()
    pad_mask = (X_raw == 0).all(axis=-1)
    for i in range(X_out.shape[0]):
        X_out[i, pad_mask[i], :] = 0.0
    return X_out


def load_lstm_full_dataset():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache_v2.npz", allow_pickle=True)
    X_raw = cache["sequences"]
    patient_ids = cache["patient_ids"]
    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y = np.array([RISK_ENCODE[label_map[seqn]] for seqn in patient_ids])
    return X_raw, y, patient_ids


def run_lstm_cv(model_path=None):
    from sklearn.model_selection import StratifiedKFold

    model_path = model_path or MODEL_DIR / "lstm_v3_final.keras"
    X_raw, y, _ = load_lstm_full_dataset()
    n_patients, n_steps, n_features = X_raw.shape
    model = load_model(model_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        scaler = StandardScaler()
        scaler.fit(X_raw[train_idx].reshape(-1, n_features))
        X_val = scale_lstm_sequences(X_raw[val_idx], scaler, n_steps, n_features)
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


def run_per_stage_breakdown(test_df, y_true, y_xgb_v3):
    stage_labels = {1: "G2", 2: "G3a", 3: "G3b", 4: "G4"}
    print("═" * 55)
    print("PER-CKD-STAGE BREAKDOWN — XGBoost v3")
    print("═" * 55)
    print(
        f"{'Stage':<8} {'N':>5} {'Accuracy':>10} "
        f"{'F1 Macro':>10} {'MOD Recall':>12} {'HIGH Recall':>12}"
    )
    print("-" * 55)

    stage_results = []
    for stage_code in sorted(test_df["ckd_stage_encoded"].unique()):
        mask = test_df["ckd_stage_encoded"].values == stage_code
        n = int(mask.sum())
        if n < 5:
            print(f"Stage {stage_code}: too few samples (n={n}) — skip")
            continue

        y_stage_true = y_true[mask]
        y_stage_pred = y_xgb_v3[mask]
        acc = accuracy_score(y_stage_true, y_stage_pred)
        f1 = f1_score(y_stage_true, y_stage_pred, average="macro", zero_division=0)
        recalls = recall_score(
            y_stage_true, y_stage_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        mod_recall = recalls[1]
        high_recall = recalls[2]
        stage_name = stage_labels.get(stage_code, str(stage_code))

        print(
            f"{stage_name:<8} {n:>5} {acc * 100:>9.1f}% "
            f"{f1:>10.4f} {mod_recall:>12.4f} {high_recall:>12.4f}"
        )
        stage_results.append(
            {
                "stage": stage_name,
                "n": n,
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "mod_recall": round(mod_recall, 4),
                "high_recall": round(high_recall, 4),
            }
        )

    stage_df = pd.DataFrame(stage_results)
    out_path = STATS_DIR / "16_per_stage_breakdown.csv"
    stage_df.to_csv(out_path, index=False)
    print()
    print(f"Saved: {out_path}")

    for _, row in stage_df.iterrows():
        if row["mod_recall"] < 0.7:
            print(
                f"⚠ Stage {row['stage']}: MOD recall = {row['mod_recall']:.3f} "
                f"— document in Chapter 5"
            )
        if row["high_recall"] < 0.8:
            print(
                f"⚠ Stage {row['stage']}: HIGH recall = {row['high_recall']:.3f} "
                f"— document in Chapter 5"
            )
    return stage_df


def prepare_tabular_test():
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

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["risk_encoded"],
    )
    y_true = test_df["risk_encoded"].values
    return test_df, y_true


def prepare_sequence_test():
    cache = np.load(MODEL_DIR / "lstm_sequences_cache_v2.npz", allow_pickle=True)
    X_seq = cache["sequences"]
    patient_ids = cache["patient_ids"]

    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y_seq_v3 = np.array([label_map.get(seqn) for seqn in patient_ids])
    y_encoded = np.array([RISK_ENCODE[r] for r in y_seq_v3])

    n_patients, n_steps, n_features = X_seq.shape
    _, X_test_raw, _, y_test, _, idx_test = train_test_split(
        X_seq,
        y_encoded,
        np.arange(n_patients),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    train_mask = np.ones(n_patients, dtype=bool)
    train_mask[idx_test] = False
    scaler.fit(X_seq[train_mask].reshape(-1, n_features))

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

    X_test_scaled = zero_padded_slots(scale_sequences(X_test_raw), X_test_raw)
    return X_test_raw, X_test_scaled, y_test, idx_test, n_steps, n_features, patient_ids


def predict_hmm_supervised(X_train_scaled, y_train, X_test_scaled):
    N_HIDDEN = 3
    class_models = {}
    for class_idx, class_name in enumerate(RISK_CLASSES):
        class_seqs = X_train_scaled[y_train == class_idx]
        if len(class_seqs) < 5:
            continue
        model = hmm.GaussianHMM(
            n_components=N_HIDDEN,
            covariance_type="diag",
            n_iter=100,
            random_state=RANDOM_STATE,
        )
        lengths = [seq.shape[0] for seq in class_seqs]
        model.fit(np.vstack(class_seqs), lengths)
        class_models[class_idx] = model

    def predict_supervised(seq_scaled, models):
        scores = {
            cls: models[cls].score(seq_scaled, [seq_scaled.shape[0]])
            for cls in models
        }
        return max(scores, key=scores.get)

    y_pred = np.array(
        [predict_supervised(X_test_scaled[i], class_models) for i in range(len(X_test_scaled))]
    )

    def get_supervised_proba(seq_scaled, models, n_classes=3):
        scores = np.array(
            [models[c].score(seq_scaled, [seq_scaled.shape[0]]) for c in range(n_classes) if c in models]
        )
        exp = np.exp(scores - scores.max())
        proba = np.zeros(n_classes)
        valid = [c for c in range(n_classes) if c in models]
        proba[valid] = exp / exp.sum()
        return proba

    y_prob = np.array([get_supervised_proba(X_test_scaled[i], class_models) for i in range(len(X_test_scaled))])
    return y_pred, y_prob


def main():
    all_results = []

    # --- Tabular models (XGB + rule baseline) ---
    test_df, y_tabular = prepare_tabular_test()
    y_rule = test_df.apply(assign_rule_baseline_label, axis=1).values
    all_results.append(
        compute_all_metrics(y_tabular, y_rule, model_name="Rule-Based Baseline")
    )

    xgb_v1_path = MODEL_DIR / "backup_verified" / "xgboost_v1_backup.pkl"
    if not xgb_v1_path.exists():
        xgb_v1_path = MODEL_DIR / "xgboost_v1.pkl"
    xgb_v1 = joblib.load(xgb_v1_path)
    y_xgb_v1 = xgb_v1.predict(test_df[FEATURES_V1])
    y_prob_v1 = xgb_v1.predict_proba(test_df[FEATURES_V1])
    all_results.append(
        compute_all_metrics(
            y_tabular,
            y_xgb_v1,
            y_prob_v1,
            model_name="XGBoost v1 (leakage)",
        )
    )

    xgb_v3 = joblib.load(MODEL_DIR / "xgboost_v3.pkl")
    y_xgb_v3 = xgb_v3.predict(test_df[FEATURES_V3])
    y_prob_v3 = xgb_v3.predict_proba(test_df[FEATURES_V3])
    all_results.append(
        compute_all_metrics(
            y_tabular,
            y_xgb_v3,
            y_prob_v3,
            model_name="XGBoost v3 (production)",
        )
    )

    mcnemar_rows = [
        mcnemar_row("Rule-Based vs XGBoost v3", y_tabular, y_rule, y_xgb_v3),
        mcnemar_row("XGBoost v1 vs v3", y_tabular, y_xgb_v1, y_xgb_v3),
    ]

    print("McNemar: Baseline vs XGBoost v3")
    print(mcnemar_test(y_tabular, y_rule, y_xgb_v3))
    print("McNemar: XGBoost v1 vs v3")
    print(mcnemar_test(y_tabular, y_xgb_v1, y_xgb_v3))

    # --- Sequence models (LSTM + HMM) ---
    X_test_raw, X_test_scaled, y_seq, idx_test, n_steps, n_features, patient_ids = (
        prepare_sequence_test()
    )
    y_true_lstm = y_seq
    y_rule_lstm = derive_rule_lstm_labels(patient_ids, idx_test)

    v3_model = load_model(MODEL_DIR / "lstm_v3_final.keras")
    y_prob_lstm_v3 = v3_model.predict(X_test_scaled, verbose=0)
    y_lstm_v3 = np.argmax(y_prob_lstm_v3, axis=1)
    all_results.append(
        compute_all_metrics(
            y_seq,
            y_lstm_v3,
            y_prob_lstm_v3,
            model_name="LSTM v3 (production)",
        )
    )

    v1_model = load_model(MODEL_DIR / "lstm_final.keras")
    v1_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    X_v1 = np.load(MODEL_DIR / "lstm_sequences_cache.npz", allow_pickle=True)["sequences"]
    X_test_v1_raw = X_v1[idx_test]
    flat = X_test_v1_raw.reshape(-1, 4)
    X_test_v1 = v1_scaler.transform(flat).reshape(len(X_test_v1_raw), n_steps, 4)
    y_prob_lstm_v1 = v1_model.predict(X_test_v1, verbose=0)
    y_lstm_v1 = np.argmax(y_prob_lstm_v1, axis=1)
    all_results.append(
        compute_all_metrics(
            y_seq,
            y_lstm_v1,
            y_prob_lstm_v1,
            model_name="LSTM v1 (original)",
        )
    )

    # HMM supervised — train on train split matching 05c
    cache = np.load(MODEL_DIR / "lstm_sequences_cache_v2.npz", allow_pickle=True)
    X_seq = cache["sequences"]
    patient_ids = cache["patient_ids"]
    labels_v3 = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    label_map = labels_v3.set_index("SEQN")["risk_label"].to_dict()
    y_all = np.array([RISK_ENCODE[label_map[seqn]] for seqn in patient_ids])

    n_patients, n_steps, n_features = X_seq.shape
    X_train_raw, X_test_raw_hmm, y_train, y_test_hmm = train_test_split(
        X_seq,
        y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, n_features))

    def scale_seqs(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape)

    def zero_pad(X_scaled, X_raw):
        out = X_scaled.copy()
        pad_mask = (X_raw == 0).all(axis=-1)
        for i in range(out.shape[0]):
            out[i, pad_mask[i], :] = 0.0
        return out

    X_train_scaled = zero_pad(scale_seqs(X_train_raw), X_train_raw)
    X_test_scaled_hmm = zero_pad(scale_seqs(X_test_raw_hmm), X_test_raw_hmm)

    y_hmm, y_prob_hmm = predict_hmm_supervised(X_train_scaled, y_train, X_test_scaled_hmm)
    all_results.append(
        compute_all_metrics(
            y_test_hmm,
            y_hmm,
            y_prob_hmm,
            model_name="HMM Supervised",
        )
    )

    results_df = pd.DataFrame(all_results)
    out_path = STATS_DIR / "12_model_comparison.csv"
    results_df.to_csv(out_path, index=False)

    summary_cols = [
        "model",
        "accuracy",
        "f1_macro",
        "auc",
        "recall_LOW",
        "recall_MODERATE",
        "recall_HIGH",
        "precision_LOW",
        "precision_MODERATE",
        "precision_HIGH",
        "specificity_LOW",
        "specificity_MODERATE",
        "specificity_HIGH",
        "test_samples",
    ]
    print("\n" + results_df[summary_cols].to_string(index=False))
    print(f"\nSaved: {out_path}")

    print("\nMcNemar: Baseline vs LSTM v1")
    r = mcnemar_test(y_true_lstm, y_rule_lstm, y_lstm_v1)
    print(f"  b={r['b']}, c={r['c']}, p={r['p_value']}")
    mcnemar_rows.append(mcnemar_row("Baseline vs LSTM v1", y_true_lstm, y_rule_lstm, y_lstm_v1))

    print("\nMcNemar: Baseline vs LSTM v3")
    r = mcnemar_test(y_true_lstm, y_rule_lstm, y_lstm_v3)
    print(f"  b={r['b']}, c={r['c']}, p={r['p_value']}")
    mcnemar_rows.append(mcnemar_row("Baseline vs LSTM v3", y_true_lstm, y_rule_lstm, y_lstm_v3))

    print("\nMcNemar: LSTM v1 vs LSTM v3")
    r = mcnemar_test(y_true_lstm, y_lstm_v1, y_lstm_v3)
    print(f"  b={r['b']}, c={r['c']}, p={r['p_value']}")
    mcnemar_rows.append(mcnemar_row("LSTM v1 vs LSTM v3", y_true_lstm, y_lstm_v1, y_lstm_v3))

    print("\nMcNemar: XGBoost v3 vs LSTM v3")
    print("(on common test samples)")
    lstm_test_seqn = patient_ids[idx_test]
    xgb_pred_map = dict(zip(test_df["SEQN"].values, y_xgb_v3))
    lstm_pred_map = dict(zip(lstm_test_seqn, y_lstm_v3))
    common_seqn = sorted(set(test_df["SEQN"]) & set(lstm_test_seqn))
    if len(common_seqn) >= 5:
        y_common = np.array(
            [test_df.set_index("SEQN").loc[s, "risk_encoded"] for s in common_seqn]
        )
        y_xgb_common = np.array([xgb_pred_map[s] for s in common_seqn])
        y_lstm_common = np.array([lstm_pred_map[s] for s in common_seqn])
        r = mcnemar_test(y_common, y_xgb_common, y_lstm_common)
        print(f"  n={len(common_seqn)}, b={r['b']}, c={r['c']}, p={r['p_value']}")
        mcnemar_rows.append(
            mcnemar_row("XGBoost v3 vs LSTM v3 (common)", y_common, y_xgb_common, y_lstm_common)
        )
    else:
        print(f"  Skipped — only {len(common_seqn)} common samples")

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(STATS_DIR / "14_mcnemar_results.csv", index=False)
    mcnemar_df.to_csv(STATS_DIR / "12_model_comparison_mcnemar.csv", index=False)
    print(f"\nSaved: {STATS_DIR / '14_mcnemar_results.csv'}")

    print("\n" + "=" * 45)
    print("LSTM v3 — 5-FOLD CROSS VALIDATION")
    print("=" * 45)
    run_lstm_cv()

    run_per_stage_breakdown(test_df, y_tabular, y_xgb_v3)


if __name__ == "__main__":
    main()
