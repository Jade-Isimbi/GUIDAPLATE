#!/usr/bin/env python3
"""
Reproduce XGBoost v3 day/meal RandomizedSearchCV logs + per-round loss curves
+ dataset checksum manifest.

SAFETY: never writes to models/. Production pickles are hash-guarded before/after.
Diagnostic refits (if saved) go only under outputs/stats/*_diagnostic_refit.pkl.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
STATS_DIR = ROOT / "outputs" / "stats"
FIG_DIR = ROOT / "outputs" / "figures"
DATA = ROOT / "data"

DAY_MODEL = MODEL_DIR / "xgboost_v3.pkl"
MEAL_MODEL = MODEL_DIR / "xgboost_v3_meal.pkl"

DAY_HASH = "0c31b13c74fd49b63e7d4ce750fdcf897c850410438b99e8f27d364d17b679f5"
MEAL_HASH = "564c1cd5e4c735c41cbe03584cfb44812692e1ebe1e37baf06e3c58a6aa776db"

RANDOM_STATE = 42
TEST_SIZE = 0.2
RISK_CLASSES = ["LOW", "MODERATE", "HIGH"]
RISK_ENCODE = {c: i for i, c in enumerate(RISK_CLASSES)}
STAGE_ENCODE = {"G2": 1, "G3a": 2, "G3b": 3, "G4": 4}
STAGE_NUMERIC = {"G2": 2, "G3a": 3, "G3b": 3, "G4": 4}

FEATURES = [
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

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 1.5, 2.0],
}

PARAM_KEYS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
]

# Forbid any write under models/
FORBIDDEN_WRITE_PREFIX = str(MODEL_DIR.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_production_untouched(when: str) -> dict[str, str]:
    digests = {}
    for path, expected in ((DAY_MODEL, DAY_HASH), (MEAL_MODEL, MEAL_HASH)):
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        digests[path.name] = digest
        if digest != expected:
            raise RuntimeError(
                f"PROTECTED {path.name} hash mismatch ({when})!\n"
                f" expected {expected}\n"
                f" got      {digest}"
            )
    print(f"[OK] production hashes unchanged ({when})")
    for name, digest in digests.items():
        print(f"     {name}: {digest}")
    return digests


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def assert_no_models_write(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved.startswith(FORBIDDEN_WRITE_PREFIX + os.sep) or resolved == FORBIDDEN_WRITE_PREFIX:
        raise RuntimeError(f"Refusing to write under models/: {path}")


def safe_write_df(path: Path, df: pd.DataFrame) -> None:
    assert_no_models_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def production_params(path: Path) -> dict:
    model = joblib.load(path)
    if isinstance(model, dict):
        for value in model.values():
            if hasattr(value, "get_params"):
                model = value
                break
    params = model.get_params()
    return {k: params.get(k) for k in PARAM_KEYS}


def params_equal(a: dict, b: dict) -> bool:
    for key in PARAM_KEYS:
        av, bv = a[key], b[key]
        if isinstance(av, float) or isinstance(bv, float):
            if abs(float(av) - float(bv)) > 1e-12:
                return False
        else:
            if av != bv:
                return False
    return True


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ckd_stage_encoded"] = out["ckd_stage"].map(STAGE_ENCODE)
    out["stage_numeric"] = out["ckd_stage"].map(STAGE_NUMERIC)
    out["k_p_product"] = (out["potassium"] * out["phosphorus"]) / 1e6
    out["protein_sodium_ratio"] = out["protein_per_kg"] / (out["sodium"] / 1000 + 1e-6)
    return out


def load_day_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Match notebooks/04c + generate_v3_notebooks.py day training pipeline."""
    cohort = pd.read_csv(DATA / "processed" / "ckd_cohort_final.csv")
    labels = pd.read_csv(STATS_DIR / "05_risk_labels_v3.csv")
    df = cohort.merge(labels, on="SEQN", how="inner", suffixes=("", "_label"))
    df = df.dropna(subset=["risk_label", "clinical_score"])
    nutrient_cols = ["potassium", "phosphorus", "protein_per_kg", "sodium"]
    df = df.dropna(subset=nutrient_cols)
    # Prefer cohort nutrients (suffixes keep cohort columns without rename)
    df = add_engineered_features(df)
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    X = df[FEATURES]
    y = df["risk_encoded"]
    return X, y


def load_meal_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Use persisted meal labels + same engineered features as train script."""
    df = pd.read_csv(STATS_DIR / "05_risk_labels_v3_meal.csv")
    df = add_engineered_features(df)
    df = df.dropna(subset=FEATURES + ["risk_label"]).copy()
    df["risk_encoded"] = df["risk_label"].map(RISK_ENCODE)
    X = df[FEATURES]
    y = df["risk_encoded"]
    return X, y


def run_search(X: pd.DataFrame, y: pd.Series, label: str) -> tuple[RandomizedSearchCV, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    class_weight = {
        RISK_ENCODE["HIGH"]: 1.0,
        RISK_ENCODE["MODERATE"]: 4.0,
        RISK_ENCODE["LOW"]: 1.0,
    }
    sample_weight_train = compute_sample_weight(class_weight=class_weight, y=y_train)

    base = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=50,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    print(f"\n=== {label}: RandomizedSearchCV (50 iter, f1_macro) ===")
    print(f"Train={len(X_train)} Test={len(X_test)}")
    t0 = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weight_train)
    print(f"Done in {time.time() - t0:.1f}s | Best CV F1 macro: {search.best_score_:.4f}")
    print("Best params:", search.best_params_)
    split = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sample_weight_train": sample_weight_train,
    }
    return search, split


def cv_results_to_df(search: RandomizedSearchCV) -> pd.DataFrame:
    """Export every RandomizedSearchCV candidate.

    Note: sklearn may assign rank_test_score=1 to several tied scores.
    ``is_best_estimator`` marks the single ``search.best_params_`` chosen
    (same object used to build the production model after search).
    """
    raw = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in raw.columns if c.startswith("param_")]
    keep = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        *param_cols,
        "params",
    ]
    keep = [c for c in keep if c in raw.columns]
    out = raw[keep].copy()
    best = {k: search.best_params_[k] for k in PARAM_KEYS}

    def is_best(params: dict) -> bool:
        return params_equal({k: params[k] for k in PARAM_KEYS}, best)

    out["is_best_estimator"] = out["params"].apply(is_best)
    if int(out["is_best_estimator"].sum()) != 1:
        raise RuntimeError(
            f"Expected exactly one best_estimator row, found {out['is_best_estimator'].sum()}"
        )
    # Best estimator first, then rank, so ties don't hide the true winner at row 0
    out = out.sort_values(
        ["is_best_estimator", "rank_test_score", "mean_test_score"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    out.insert(0, "candidate_id", range(1, len(out) + 1))
    out["params_json"] = out["params"].apply(lambda d: json.dumps(d, sort_keys=True))
    out = out.drop(columns=["params"])
    # Put flag near the front for readability
    cols = ["candidate_id", "is_best_estimator"] + [
        c for c in out.columns if c not in {"candidate_id", "is_best_estimator"}
    ]
    return out[cols]


def task1_search_logs() -> dict[str, dict]:
    expected = {
        "day": production_params(DAY_MODEL),
        "meal": production_params(MEAL_MODEL),
    }
    print("\nExpected production hyperparams:")
    print("  day :", expected["day"])
    print("  meal:", expected["meal"])

    X_day, y_day = load_day_xy()
    search_day, split_day = run_search(X_day, y_day, "DAY")
    found_day = {k: search_day.best_params_[k] for k in PARAM_KEYS}

    if not params_equal(found_day, expected["day"]):
        print("\nSTOP — DAY search winner does NOT match production.")
        print("  expected:", expected["day"])
        print("  found:   ", found_day)
        raise SystemExit(2)

    X_meal, y_meal = load_meal_xy()
    search_meal, split_meal = run_search(X_meal, y_meal, "MEAL")
    found_meal = {k: search_meal.best_params_[k] for k in PARAM_KEYS}

    if not params_equal(found_meal, expected["meal"]):
        print("\nSTOP — MEAL search winner does NOT match production.")
        print("  expected:", expected["meal"])
        print("  found:   ", found_meal)
        raise SystemExit(3)

    print("\n[OK] Day and meal search winners match production hyperparams exactly.")

    day_log = STATS_DIR / "xgboost_v3_day_search_log.csv"
    meal_log = STATS_DIR / "xgboost_v3_meal_search_log.csv"
    safe_write_df(day_log, cv_results_to_df(search_day))
    safe_write_df(meal_log, cv_results_to_df(search_meal))

    return {
        "day": {"search": search_day, "split": split_day, "params": found_day},
        "meal": {"search": search_meal, "split": split_meal, "params": found_meal},
    }


def fit_diagnostic_eval_history(
    params: dict,
    split: dict,
    title: str,
    fig_path: Path,
    diagnostic_pkl: Path,
) -> None:
    assert_no_models_write(fig_path)
    assert_no_models_write(diagnostic_pkl)

    model = xgb.XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(
        split["X_train"],
        split["y_train"],
        sample_weight=split["sample_weight_train"],
        eval_set=[
            (split["X_train"], split["y_train"]),
            (split["X_test"], split["y_test"]),
        ],
        verbose=False,
    )
    history = model.evals_result()
    # Typical keys: validation_0 (train), validation_1 (test)
    train_key, test_key = list(history.keys())[:2]
    metric = list(history[train_key].keys())[0]
    train_loss = history[train_key][metric]
    test_loss = history[test_key][metric]
    rounds = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, train_loss, label="Train mlogloss", color="#0D9488", linewidth=2)
    ax.plot(rounds, test_loss, label="Test mlogloss", color="#EF4444", linewidth=2, linestyle="--")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Multiclass log loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {fig_path} ({len(train_loss)} rounds)")

    # Optional clearly-labeled diagnostic byproduct — NOT production path
    joblib.dump(
        {
            "note": "DIAGNOSTIC ONLY — do not deploy; production weights unchanged in models/",
            "params": params,
            "evals_result": history,
            "model": model,
        },
        diagnostic_pkl,
    )
    print(f"Wrote diagnostic refit (not production): {diagnostic_pkl}")


def task2_loss_curves(bundle: dict) -> None:
    print("\n=== TASK 2: per-round loss curves (diagnostic refit only) ===")
    print("Will write ONLY:")
    print("  outputs/figures/xgboost_v3_day_loss_curve.png")
    print("  outputs/figures/xgboost_v3_meal_loss_curve.png")
    print("  outputs/stats/xgboost_v3_day_diagnostic_refit.pkl")
    print("  outputs/stats/xgboost_v3_meal_diagnostic_refit.pkl")
    print("Will NOT write models/xgboost_v3.pkl or models/xgboost_v3_meal.pkl")

    fit_diagnostic_eval_history(
        bundle["day"]["params"],
        bundle["day"]["split"],
        "XGBoost v3 day — mlogloss per boosting round (diagnostic refit)",
        FIG_DIR / "xgboost_v3_day_loss_curve.png",
        STATS_DIR / "xgboost_v3_day_diagnostic_refit.pkl",
    )
    fit_diagnostic_eval_history(
        bundle["meal"]["params"],
        bundle["meal"]["split"],
        "XGBoost v3 meal — mlogloss per boosting round (diagnostic refit)",
        FIG_DIR / "xgboost_v3_meal_loss_curve.png",
        STATS_DIR / "xgboost_v3_meal_diagnostic_refit.pkl",
    )


def row_count(path: Path) -> int | str:
    if not path.exists():
        return "MISSING"
    if path.suffix.lower() in {".xpt", ".XPT"}:
        try:
            import pyreadstat

            _, meta = pyreadstat.read_xport(str(path), metadataonly=True)
            # metadataonly may not give rows; fall back
            raise RuntimeError("row count via full read")
        except Exception:
            try:
                import pyreadstat

                df, _ = pyreadstat.read_xport(str(path))
                return len(df)
            except Exception:
                try:
                    df = pd.read_sas(path, format="xport")
                    return len(df)
                except Exception as exc:
                    return f"unavailable ({exc})"
    try:
        return sum(1 for _ in path.open("rb")) - 1  # header-ish for csv; wrong for binary already handled
    except Exception:
        return "unavailable"


def csv_rows(path: Path) -> int:
    return max(0, sum(1 for _ in path.open()) - 1)


def task3_manifest() -> Path:
    print("\n=== TASK 3: dataset checksum manifest ===")
    entries = [
        # Tracked processed / labels
        (DATA / "processed" / "ckd_cohort_final.csv", "git-tracked", "Day XGB + meal meal-builder cohort join"),
        (DATA / "processed" / "ckd_patients.csv", "git-tracked", "Processed CKD patient table (EDA / pipeline)"),
        (DATA / "processed" / "ckd_patients_clean.csv", "git-tracked", "Cleaned CKD patient table"),
        (DATA / "processed" / "food_nutrients_clean.csv", "git-tracked", "Legacy USDA nutrients (not live GuidaPlate DB)"),
        (STATS_DIR / "05_risk_labels_v3.csv", "git-tracked", "Day XGB v3 labels + clinical_score"),
        (STATS_DIR / "05_risk_labels_v3_meal.csv", "git-tracked", "Meal XGB v3 labels (train/eval table)"),
        # NHANES totals / demo / lab — CSV tracked
        (DATA / "raw" / "nhanes" / "DR1TOT_J.csv", "git-tracked", "NHANES day-1 totals (CSV)"),
        (DATA / "raw" / "nhanes" / "DR2TOT_J.csv", "git-tracked", "NHANES day-2 totals (CSV)"),
        (DATA / "raw" / "nhanes" / "DEMO_J.csv", "git-tracked", "NHANES demographics (CSV)"),
        (DATA / "raw" / "nhanes" / "BIOPRO_J.csv", "git-tracked", "NHANES biochemistry (CSV)"),
        (DATA / "raw" / "nhanes" / "BMX_J.csv", "git-tracked", "NHANES body measures (CSV)"),
        # Local-only XPTs
        (DATA / "raw" / "nhanes" / "DR1TOT_J.xpt", "local-only (gitignored)", "NHANES day-1 totals (XPT)"),
        (DATA / "raw" / "nhanes" / "DR2TOT_J.xpt", "local-only (gitignored)", "NHANES day-2 totals (XPT)"),
        (DATA / "raw" / "nhanes" / "DEMO_J.xpt", "local-only (gitignored)", "NHANES demographics (XPT)"),
        (DATA / "raw" / "nhanes" / "BIOPRO_J.xpt", "local-only (gitignored)", "NHANES biochemistry (XPT)"),
        (DATA / "raw" / "nhanes" / "BMX_J.xpt", "local-only (gitignored)", "NHANES body measures (XPT)"),
        (DATA / "raw" / "nhanes" / "DR1IFF_J.xpt", "local-only (gitignored)", "Meal XGB / IFF day-1 foods (XPT)"),
        (DATA / "raw" / "nhanes" / "DR2IFF_J.xpt", "local-only (gitignored)", "Meal XGB / IFF day-2 foods (XPT)"),
        (DATA / "raw" / "nhanes" / "DR1IFF_J.csv", "local-only (gitignored)", "IFF day-1 CSV export (local)"),
        (DATA / "raw" / "nhanes" / "DR2IFF_J.csv", "local-only (gitignored)", "IFF day-2 CSV export (local)"),
        # USDA (+ food DB used at app runtime — related training context)
        (DATA / "raw" / "usda" / "food.csv", "git-tracked", "USDA FoodData Central foods"),
        (DATA / "raw" / "usda" / "food_nutrient.csv", "git-tracked", "USDA food-nutrient links"),
        (DATA / "raw" / "usda" / "nutrient.csv", "git-tracked", "USDA nutrient dictionary"),
        (ROOT / "backend" / "data" / "food_database.csv", "git-tracked", "Live Rwandan food DB (app; not NHANES train)"),
    ]

    rows = []
    for path, tracking, role in entries:
        rel = str(path.relative_to(ROOT)) if path.exists() else str(path)
        if not path.exists():
            rows.append(
                {
                    "path": rel,
                    "sha256": "MISSING",
                    "bytes": None,
                    "approx_rows": None,
                    "git_status": tracking,
                    "role": role,
                }
            )
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        if path.suffix.lower() == ".csv":
            nrows = csv_rows(path)
        elif path.suffix.lower() == ".xpt":
            # Skip full XPT row parse for huge IFF in manifest pass — note size only;
            # hash is the traceability signal. Optional light note.
            nrows = "not_enumerated (large XPT; hash is authoritative)"
            if path.stat().st_size < 20_000_000:
                nrows = row_count(path)
        else:
            nrows = None
        rows.append(
            {
                "path": rel,
                "sha256": digest,
                "bytes": size,
                "approx_rows": nrows,
                "git_status": tracking,
                "role": role,
            }
        )
        print(f"  {digest[:12]}…  {rel}")

    md_path = DATA / "DATASET_MANIFEST.md"
    assert_no_models_write(md_path)
    lines = [
        "# GuidaPlate dataset checksum manifest",
        "",
        "Traceability record for training/evaluation inputs. SHA256 hashes identify",
        "exact file bytes even when large NHANES XPT/IFF files are **gitignored**.",
        "",
        f"Generated by `scripts/generate_xgboost_training_evidence.py`.",
        "",
        "| Path | SHA256 | Bytes | Rows | Git status | Role |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['path']}` | `{r['sha256']}` | {r['bytes']} | {r['approx_rows']} | {r['git_status']} | {r['role']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **git-tracked**: present in the repository; hash should match a clean checkout.",
            "- **local-only (gitignored)**: must be re-downloaded from CDC NHANES 2017–2018 (J cycle)",
            "  to reproduce meal training; this manifest is the version pin for those bytes.",
            "- Day XGB training merges `data/processed/ckd_cohort_final.csv` with",
            "  `outputs/stats/05_risk_labels_v3.csv`.",
            "- Meal XGB training originally builds from DR1IFF/DR2IFF XPTs + cohort; the",
            "  persisted table `outputs/stats/05_risk_labels_v3_meal.csv` is the labeled",
            "  training export used for reproducible search/eval in this evidence script.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")

    json_path = DATA / "DATASET_MANIFEST.json"
    assert_no_models_write(json_path)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    return md_path


def main() -> None:
    # Preflight: confirm diagnostic destinations are not production
    for forbidden in (DAY_MODEL, MEAL_MODEL):
        print(f"[safety] will never write: {forbidden}")

    day_mtime0 = file_mtime(DAY_MODEL)
    meal_mtime0 = file_mtime(MEAL_MODEL)
    assert_production_untouched("before")

    bundle = task1_search_logs()
    # Re-check after search (search must not touch models/)
    assert_production_untouched("after Task 1 search")

    task2_loss_curves(bundle)
    assert_production_untouched("after Task 2 diagnostic refits")

    task3_manifest()
    digests = assert_production_untouched("after Task 3 manifest")

    day_mtime1 = file_mtime(DAY_MODEL)
    meal_mtime1 = file_mtime(MEAL_MODEL)
    if day_mtime1 != day_mtime0 or meal_mtime1 != meal_mtime0:
        raise RuntimeError(
            f"Production mtime changed!\n"
            f" day  {day_mtime0} -> {day_mtime1}\n"
            f" meal {meal_mtime0} -> {meal_mtime1}"
        )

    print("\n========== FINAL VERIFICATION ==========")
    print(f"day  SHA256: {digests[DAY_MODEL.name]}  (expected {DAY_HASH})")
    print(f"meal SHA256: {digests[MEAL_MODEL.name]}  (expected {MEAL_HASH})")
    print(f"day  mtime unchanged: {day_mtime0}")
    print(f"meal mtime unchanged: {meal_mtime0}")
    print("Production models UNTOUCHED.")


if __name__ == "__main__":
    # Fail fast if someone remaps cwd oddly
    os.chdir(ROOT)
    main()
