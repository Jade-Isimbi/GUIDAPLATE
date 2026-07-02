"""
generate_testing_evidence.py
Reads existing saved stats files and
renders them as labeled table images
for the testing documentation folder.

Does NOT retrain or re-run any models.
Reads only from outputs/stats/ and
outputs/figures/ which already exist.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "outputs" / "stats"
FIGS = ROOT / "outputs" / "figures"
DOCS = ROOT / "docs" / "testing"

plt.rcParams["font.family"] = "sans-serif"


def save_df_as_image(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    figsize=(10, None),
):
    """Render a DataFrame as a clean table image."""
    n_rows = len(df) + 1
    height = max(2, n_rows * 0.4)
    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for i in range(len(df.columns)):
        tbl[0, i].set_facecolor("#0D9488")
        tbl[0, i].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_mod2_vs_mod3_image(data: dict, title: str, out_path: Path):
    """Render MOD=2 vs MOD=3 model rows and metadata footer separately."""
    model_rows = []
    meta: dict[str, object] = {}
    for key, vals in data.items():
        if isinstance(vals, dict):
            model_rows.append({"Model": key, **vals})
        else:
            meta[key] = vals

    df = pd.DataFrame(model_rows)
    order = [name for name in ("MOD=3", "MOD=2") if name in df["Model"].values]
    extra = [m for m in df["Model"] if m not in order]
    df = df.set_index("Model").loc[order + extra].reset_index()

    display_cols = [
        "Model",
        "accuracy",
        "f1_macro",
        "auc",
        "low_recall",
        "mod_recall",
        "high_recall",
        "cv_f1",
        "edge_pass",
        "deployed",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    for col in df.columns:
        if col == "Model":
            continue
        if col == "deployed":
            df[col] = df[col].map({True: "Yes", False: "No", "True": "Yes", "False": "No"})
        elif col == "edge_pass":
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")

    df.columns = [
        c.replace("_", " ").title().replace("Mod ", "MOD ").replace("Cv ", "CV ")
        for c in df.columns
    ]

    n_rows = len(df) + 1
    height = max(3.0, n_rows * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(14, height))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="upper center",
        bbox=[0, 0.22, 1, 0.72],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    for i in range(len(df.columns)):
        tbl[0, i].set_facecolor("#0D9488")
        tbl[0, i].set_text_props(color="white", fontweight="bold")

    footer_lines = [
        f"McNemar p = {meta.get('mcnemar_p', '—')}",
        f"Deployed model: {meta.get('deploy', '—')}",
        str(meta.get("reason", "")),
    ]
    fig.text(
        0.5,
        0.08,
        "\n".join(footer_lines),
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0FDFA", edgecolor="#0D9488"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_dict_as_text_image(data: dict, title: str, out_path: Path):
    """Render a flat dict as a key:value table image."""
    rows = [[str(k), str(v)] for k, v in data.items()]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    save_df_as_image(df, title, out_path)


print("=" * 60)
print("GENERATING TESTING EVIDENCE IMAGES")
print("=" * 60)

# ─────────────────────────────────────
# 1 — McNemar tests
# ─────────────────────────────────────
print("\n[1] McNemar tests")
target = DOCS / "01_mcnemar_tests"
f = STATS / "14_mcnemar_results.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "McNemar Significance Tests — Model Comparisons",
        target / "mcnemar_results.png",
        figsize=(11, None),
    )
else:
    print(f"  ⚠ {f} not found — skipping")

f2 = STATS / "20_mod2_vs_mod3_comparison.json"
if f2.exists():
    with open(f2) as fh:
        data = json.load(fh)
    save_mod2_vs_mod3_image(
        data,
        "Tier 3 — MOD=2 vs MOD=3 Comparison",
        target / "mod2_vs_mod3.png",
    )
else:
    print(f"  ⚠ {f2} not found — skipping")

# ─────────────────────────────────────
# 2 — Cross-validation
# ─────────────────────────────────────
print("\n[2] Cross-validation")
target = DOCS / "02_cross_validation"
f = STATS / "15_lstm_cv_results.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "5-Fold Cross-Validation Results",
        target / "cv_results.png",
    )
else:
    print(f"  ⚠ {f} not found — skipping")

f2 = STATS / "19_winner_full_metrics.json"
if f2.exists():
    with open(f2) as fh:
        data = json.load(fh)
    cv_subset = {k: v for k, v in data.items() if "CV" in k or "cv" in k}
    if cv_subset:
        save_dict_as_text_image(
            cv_subset,
            "Tier 3 RF — Cross-Validation Summary",
            target / "tier3_cv_summary.png",
        )
else:
    print(f"  ⚠ {f2} not found — skipping")

# ─────────────────────────────────────
# 3 — Overfitting analysis
# ─────────────────────────────────────
print("\n[3] Overfitting analysis")
target = DOCS / "03_overfitting_analysis"
f = STATS / "13_overfitting_analysis.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "Overfitting Analysis — Train vs Test Gap",
        target / "overfitting_analysis.png",
    )
else:
    print(f"  ⚠ {f} not found — skipping")

# ─────────────────────────────────────
# 4 — Confusion matrices
# ─────────────────────────────────────
print("\n[4] Confusion matrices")
target = DOCS / "04_confusion_matrices"
patterns = [
    "*confusion*.png",
]
copied = 0
for pattern in patterns:
    for fig_file in FIGS.glob(pattern):
        dest = target / fig_file.name
        shutil.copy(fig_file, dest)
        print(f"  Copied: {fig_file.name}")
        copied += 1
if copied == 0:
    print(f"  ⚠ No confusion matrix figures found in {FIGS}")

# ─────────────────────────────────────
# 5 — Per-stage breakdown
# ─────────────────────────────────────
print("\n[5] Per-stage breakdown")
target = DOCS / "05_per_stage_breakdown"
f = STATS / "16_per_stage_breakdown.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "Per-CKD-Stage Performance Breakdown",
        target / "per_stage_breakdown.png",
    )
else:
    print(f"  ⚠ {f} not found — skipping")

f2 = STATS / "19_winner_full_metrics.csv"
if f2.exists():
    df2 = pd.read_csv(f2)
    save_df_as_image(
        df2,
        "Tier 3 RF — Per-Class Metrics (Precision/Recall/F1/Specificity)",
        target / "tier3_per_class.png",
        figsize=(13, None),
    )
else:
    print(f"  ⚠ {f2} not found — skipping")

# ─────────────────────────────────────
# 6 — Edge case testing
# ─────────────────────────────────────
print("\n[6] Edge case testing")
target = DOCS / "06_edge_case_testing"
print("  ⚠ Edge case results were printed in notebook, not saved as CSV.")
print("  → Manually screenshot the printed edge case tables from")
print("    notebooks/11_weekly_tier3.ipynb Section 13 and Section 15 outputs,")
print("    then save them into:")
print(f"    {target}/")

# ─────────────────────────────────────
# 7 — Model comparison
# ─────────────────────────────────────
print("\n[7] Model comparison")
target = DOCS / "07_model_comparison"
f = STATS / "12_model_comparison.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "Full Model Comparison — All Tiers",
        target / "model_comparison.png",
        figsize=(13, None),
    )
else:
    print(f"  ⚠ {f} not found — skipping")

# ─────────────────────────────────────
# 8 — Hyperparameter sweep
# ─────────────────────────────────────
print("\n[8] Hyperparameter sweep")
target = DOCS / "08_hyperparameter_sweep"
f = STATS / "18_weekly_improvement_sweep.csv"
if f.exists():
    df = pd.read_csv(f)
    save_df_as_image(
        df,
        "Tier 3 — Full Improvement Sweep (All Options & Combinations)",
        target / "improvement_sweep.png",
        figsize=(13, None),
    )
else:
    print(f"  ⚠ {f} not found — skipping")

f2 = STATS / "21_section15_all_results.csv"
if f2.exists():
    df2 = pd.read_csv(f2)
    save_df_as_image(
        df2,
        "Tier 3 — Threshold/Calibration/Ensemble Comparison (Section 15)",
        target / "section15_results.png",
        figsize=(13, None),
    )
else:
    print(f"  ⚠ {f2} not found — skipping")

# ─────────────────────────────────────
# 9 — Calibration
# ─────────────────────────────────────
print("\n[9] Calibration")
target = DOCS / "09_calibration"
f = FIGS / "34_calibration_curves.png"
if f.exists():
    shutil.copy(f, target / "calibration_curves.png")
    print(f"  Copied: {f.name}")
else:
    print(f"  ⚠ {f} not found — skipping")

# ─────────────────────────────────────
# 10 — Integration verification
# ─────────────────────────────────────
print("\n[10] Integration verification")
target = DOCS / "10_integration_verification"
print("  ⚠ This is terminal output from verify_tier3.py — not a saved file.")
print("  → Manually screenshot your terminal showing the 9/9 PASSED summary,")
print(f"    then save it into:")
print(f"    {target}/")

print("\n" + "=" * 60)
print("DONE — review docs/testing/ folder")
print("=" * 60)
print("\nFolders needing MANUAL screenshots:")
print("  06_edge_case_testing/  — notebook printed tables")
print("  10_integration_verification/ — terminal output")
print("\nAll other folders should now contain auto-generated images.")
