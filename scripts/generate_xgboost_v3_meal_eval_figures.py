#!/usr/bin/env python3
"""
ONE-OFF: generate meal XGBoost v3 eval figures from the EXISTING production
pickle (predict-only). Never writes models/xgboost_v3_meal.pkl.

Outputs:
  outputs/figures/xgb_v3_meal_06_confusion.png
  outputs/figures/xgb_v3_meal_07_roc.png
  outputs/figures/xgb_v3_meal_10_shap_high.png
  outputs/figures/xgb_v3_meal_11_shap_moderate.png

Usage (repo root):
  ./venv311/bin/python3 scripts/generate_xgboost_v3_meal_eval_figures.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_xgboost_v3_meal import (  # noqa: E402
    FEATURES,
    MEAL_DATASET_PATH,
    MEAL_MODEL_PATH,
    RANDOM_STATE,
    RISK_CLASSES,
    RISK_ENCODE,
    STAGE_ENCODE,
    STAGE_NUMERIC,
    TEST_SIZE,
    build_meal_dataset,
    rule_baseline_label,
)

FIG_DIR = ROOT / "outputs" / "figures"
STATS_DIR = ROOT / "outputs" / "stats"
DEEP_EVAL = STATS_DIR / "10_xgboost_v3_meal_deep_eval.json"

MEAL_PROTECTED_SHA256 = (
    "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meal_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def assert_meal_untouched(when: str, expected_mtime_ns: int) -> str:
    digest = sha256_file(MEAL_MODEL_PATH)
    mtime = meal_mtime_ns(MEAL_MODEL_PATH)
    if digest != MEAL_PROTECTED_SHA256:
        raise RuntimeError(
            f"PROTECTED meal hash changed ({when})!\n"
            f" expected {MEAL_PROTECTED_SHA256}\n"
            f" got      {digest}"
        )
    if mtime != expected_mtime_ns:
        raise RuntimeError(
            f"PROTECTED meal mtime changed ({when})!\n"
            f" expected_ns {expected_mtime_ns}\n"
            f" got_ns      {mtime}"
        )
    print(f"[OK] meal pkl untouched ({when}): sha={digest[:16]}... mtime_ns={mtime}")
    return digest


def load_meal_dataframe() -> pd.DataFrame:
    """Identical path to scripts/test_xgboost_v3_meal.py main() dataset load."""
    if MEAL_DATASET_PATH.exists():
        df = pd.read_csv(MEAL_DATASET_PATH).copy()
        df["ckd_stage_encoded"] = df["ckd_stage"].map(STAGE_ENCODE)
        df["stage_numeric"] = df["ckd_stage"].map(STAGE_NUMERIC)
        df["k_p_product"] = (df["potassium"] * df["phosphorus"]) / 1e6
        df["protein_sodium_ratio"] = df["protein_per_kg"] / (df["sodium"] / 1000 + 1e-6)
        if "rule_baseline" not in df.columns:
            df["rule_baseline"] = df.apply(rule_baseline_label, axis=1)
        return df
    return build_meal_dataset()


def prepare_holdout(df: pd.DataFrame):
    """Identical split to test_xgboost_v3_meal.split_data / prepare_xy."""
    df = df.dropna(subset=FEATURES + ["risk_label"]).copy()
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    X = df[FEATURES]
    y = df["risk_encoded"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def plot_confusion_from_json() -> None:
    payload = json.loads(DEEP_EVAL.read_text())
    cm = np.asarray(payload["meal_model"]["confusion"], dtype=int)
    # Reorder LOW/MOD/HIGH → HIGH/LOW/MODERATE to match day xgb_v3_06 style order
    plot_labels = ["HIGH", "LOW", "MODERATE"]
    idx = [RISK_ENCODE[c] for c in plot_labels]
    cm_plot = cm[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=plot_labels,
        yticklabels=plot_labels,
        linewidths=0.5,
    )
    ax.set_title("XGBoost v3 Meal Confusion Matrix (holdout)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out = FIG_DIR / "xgb_v3_meal_06_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  JSON cm (LOW/MOD/HIGH order): {cm.tolist()}")


def plot_roc(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_true = y_test.to_numpy()
    y_prob = model.predict_proba(X_test)
    classes = RISK_CLASSES  # LOW, MODERATE, HIGH
    y_bin = label_binarize(y_true, classes=[RISK_ENCODE[c] for c in classes])

    # Day style uses HIGH, LOW, MODERATE plotting order / colors
    plot_order = ["HIGH", "LOW", "MODERATE"]
    colors = {"HIGH": "#ef4444", "LOW": "#22c55e", "MODERATE": "#f59e0b"}

    fig, ax = plt.subplots(figsize=(8, 6))
    per_class_auc: dict[str, float] = {}
    for cls in plot_order:
        i = RISK_ENCODE[cls]
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        cls_auc = auc(fpr, tpr)
        per_class_auc[cls] = float(cls_auc)
        ax.plot(
            fpr,
            tpr,
            color=colors[cls],
            lw=2,
            label=f"{cls} (AUC = {cls_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("XGBoost v3 Meal ROC Curves by Class")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "xgb_v3_meal_07_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    auc_weighted = float(
        roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
    )
    auc_macro = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    print(f"Wrote {out}")
    print(f"  per-class AUC: {per_class_auc}")
    print(f"  weighted AUC: {auc_weighted:.6f}  macro AUC: {auc_macro:.6f}")

    # Sanity vs published meal deep-eval (~1.0)
    published = json.loads(DEEP_EVAL.read_text())["meal_model"]
    pub_w = float(published["auc_weighted"])
    pub_pc = published["auc_per_class"]
    assert auc_weighted >= 0.99, f"weighted AUC too low: {auc_weighted}"
    for cls in RISK_CLASSES:
        diff = abs(per_class_auc[cls] - float(pub_pc[cls]))
        assert diff < 1e-3, f"{cls} AUC mismatch live={per_class_auc[cls]} pub={pub_pc[cls]}"
    assert abs(auc_weighted - pub_w) < 1e-3, (
        f"weighted AUC mismatch live={auc_weighted} pub={pub_w}"
    )
    print(
        f"[OK] ROC AUC sanity: matches published deep-eval "
        f"(weighted {auc_weighted:.6f} ≈ {pub_w:.6f})"
    )

    # Also confirm confusion from live predict matches JSON
    y_pred = model.predict(X_test)
    cm_live = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pub = np.asarray(published["confusion"], dtype=int)
    assert np.array_equal(cm_live, cm_pub), (
        f"Live CM != published JSON\n live={cm_live.tolist()}\n pub={cm_pub.tolist()}"
    )
    print("[OK] Live holdout confusion matches meal_model.confusion in deep-eval JSON")
    return {"per_class_auc": per_class_auc, "auc_weighted": auc_weighted}


def _meal_pred_contribs(model, X_plot: pd.DataFrame) -> np.ndarray:
    """
    Native XGBoost pred_contribs (predict-only).

    shap.TreeExplainer is incompatible with this environment's xgboost 3.2
    multi-class output shape; pred_contribs is the supported workaround and
    does not write/modify the pickle.
    Returns array shape (n_samples, n_classes, n_features + 1) with bias last.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_plot, feature_names=list(X_plot.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    arr = np.asarray(contribs)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected pred_contribs shape: {arr.shape}")
    return arr


def plot_shap(model, X_test: pd.DataFrame) -> None:
    # Cap beeswarm sample for speed if holdout is large (still representative)
    n = len(X_test)
    if n > 800:
        X_plot = X_test.sample(n=800, random_state=RANDOM_STATE)
    else:
        X_plot = X_test

    contribs = _meal_pred_contribs(model, X_plot)
    # (n, n_classes, n_features+1)
    n_feat = len(FEATURES)
    assert contribs.shape[2] == n_feat + 1, contribs.shape

    for class_name, out_name in (
        ("HIGH", "xgb_v3_meal_10_shap_high.png"),
        ("MODERATE", "xgb_v3_meal_11_shap_moderate.png"),
    ):
        cls_idx = RISK_ENCODE[class_name]
        values = contribs[:, cls_idx, :n_feat]
        base = contribs[:, cls_idx, n_feat]
        explanation = shap.Explanation(
            values=values,
            base_values=base,
            data=X_plot.to_numpy(),
            feature_names=list(FEATURES),
        )
        mean_abs = float(np.abs(values).mean())
        assert mean_abs > 0, f"SHAP values empty/zero for {class_name}"

        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(explanation, show=False, max_display=15)
        plt.title(f"SHAP Beeswarm — {class_name} Class (v3 meal features)", fontsize=13)
        plt.tight_layout()
        out = FIG_DIR / out_name
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out} (n={len(X_plot)}, mean|shap|={mean_abs:.4f})")


def main() -> int:
    os.chdir(ROOT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not MEAL_MODEL_PATH.exists():
        raise FileNotFoundError(MEAL_MODEL_PATH)
    if not DEEP_EVAL.exists():
        raise FileNotFoundError(DEEP_EVAL)

    mtime0 = meal_mtime_ns(MEAL_MODEL_PATH)
    digest0 = assert_meal_untouched("before", mtime0)

    print("\n=== 1) Confusion matrix from saved JSON ===")
    plot_confusion_from_json()
    assert_meal_untouched("after confusion plot", mtime0)

    print("\n=== 2–3) Load production meal model (predict-only) + holdout ===")
    df = load_meal_dataframe()
    _, X_test, _, y_test, _, _ = prepare_holdout(df)
    print(
        f"Holdout: n_test={len(X_test)}  "
        f"test_size={TEST_SIZE}  random_state={RANDOM_STATE}  stratify=True"
    )
    print(f"Loading READ-ONLY: {MEAL_MODEL_PATH}")
    model = joblib.load(MEAL_MODEL_PATH)

    print("\n=== 2) ROC (predict_proba on identical holdout) ===")
    plot_roc(model, X_test, y_test)
    assert_meal_untouched("after ROC", mtime0)

    print("\n=== 3) SHAP beeswarm HIGH / MODERATE ===")
    plot_shap(model, X_test)
    digest1 = assert_meal_untouched("after SHAP", mtime0)

    print("\n=== FINAL VERIFICATION ===")
    print(f"  path:  {MEAL_MODEL_PATH}")
    print(f"  sha256: {digest1}")
    print(f"  matches protected: {digest1 == MEAL_PROTECTED_SHA256}")
    print(f"  mtime_ns unchanged: {meal_mtime_ns(MEAL_MODEL_PATH) == mtime0}")
    print(f"  started_sha == ended_sha: {digest0 == digest1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
