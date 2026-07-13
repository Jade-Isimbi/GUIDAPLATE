#!/usr/bin/env python3
"""Abandoned Tier 3 weekly RF — offline artifact checks only.

NOT a live integration test. The weekly Random Forest was trained and
evaluated, then archived; it is not loaded by the production API
(weekly_trend.py uses nutrient aggregates + LSTM only).

Artifacts (archived):
  models/archive/weekly_rf.pkl
  models/archive/weekly_rf_config.json

This script only verifies those files exist, load with joblib, and can
run a smoke predict. It does NOT expect (or prove) live API wiring.

Run from repo root:
  python3 scripts/archive/verify_tier3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

# scripts/archive/verify_tier3.py → repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RF_PATH = ROOT / "models" / "archive" / "weekly_rf.pkl"
CFG_PATH = ROOT / "models" / "archive" / "weekly_rf_config.json"
LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{suffix}")
    return ok


def main() -> int:
    print("GuidaPlate Tier 3 — offline archived-artifact check")
    print("(Abandoned approach — not a live API integration test)")
    print("=" * 50)
    print(f"Repo root: {ROOT}")
    print(f"RF path:   {RF_PATH}")
    print(f"Config:    {CFG_PATH}")
    print("=" * 50)

    results: list[bool] = []

    results.append(check("archive/weekly_rf.pkl exists", RF_PATH.is_file()))
    results.append(check("archive/weekly_rf_config.json exists", CFG_PATH.is_file()))

    config: dict | None = None
    if CFG_PATH.is_file():
        try:
            with open(CFG_PATH) as f:
                config = json.load(f)
            ok = isinstance(config, dict) and config.get("model") == "weekly_rf.pkl"
            results.append(check("Config JSON valid", ok))
        except json.JSONDecodeError as e:
            results.append(check("Config JSON valid", False, str(e)))
    else:
        results.append(check("Config JSON valid", False))

    rf = None
    if RF_PATH.is_file():
        try:
            rf = joblib.load(RF_PATH)
            results.append(check("Model loads with joblib", True))
        except Exception as e:
            results.append(check("Model loads with joblib", False, str(e)))
    else:
        results.append(check("Model loads with joblib", False))

    n_features = config.get("n_features", 21) if config else 21
    results.append(
        check("Config declares 21 input features", n_features == 21, f"n_features={n_features}")
    )

    if rf is not None:
        X = np.array([[0.2, 0.3, 0.5]] * 7, dtype=float).flatten().reshape(1, -1)
        try:
            pred = int(rf.predict(X)[0])
            proba = rf.predict_proba(X)[0]
            ok = (
                pred in LABELS
                and len(proba) == 3
                and abs(float(proba.sum()) - 1.0) < 1e-6
            )
            results.append(
                check(
                    "Offline smoke inference (predict + proba)",
                    ok,
                    f"{LABELS[pred]}, sum={float(proba.sum()):.4f}",
                )
            )
        except Exception as e:
            results.append(check("Offline smoke inference (predict + proba)", False, str(e)))
    else:
        results.append(check("Offline smoke inference (predict + proba)", False))

    if config:
        cw = config.get("class_weight", {})
        ok = cw.get("MODERATE") == 3
        winner = config.get("winner", "")
        results.append(
            check(
                "Archived winner metadata is RF + CW MOD=3",
                ok,
                winner or "missing winner",
            )
        )
    else:
        results.append(check("Archived winner metadata is RF + CW MOD=3", False))

    # Explicitly document that live wiring is NOT expected.
    try:
        from backend.api import weekly_trend as wt

        has_predict = hasattr(wt, "_predict_weekly_tier3")
        results.append(
            check(
                "Live API has no Tier-3 RF predict hook (expected)",
                not has_predict,
                (
                    "_predict_weekly_tier3 absent — weekly_trend is LSTM/aggregates only"
                    if not has_predict
                    else "UNEXPECTED: _predict_weekly_tier3 still present"
                ),
            )
        )
    except Exception as e:
        results.append(
            check("Live API has no Tier-3 RF predict hook (expected)", False, str(e))
        )

    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"Result: {passed}/{total} offline artifact checks passed")
    print("Note: this does not certify production Tier-3 deployment.")

    if passed == total:
        print("All offline archived-artifact checks PASSED")
        return 0

    print("Some offline checks FAILED — review output above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
