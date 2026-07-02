#!/usr/bin/env python3
"""End-to-end wiring verification for Tier 3 weekly RF (9 checks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RF_PATH = ROOT / "models" / "weekly_rf.pkl"
CFG_PATH = ROOT / "models" / "weekly_rf_config.json"
WEEKLY_NEUTRAL = [1 / 3, 1 / 3, 1 / 3]
LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{suffix}")
    return ok


def main() -> int:
    print("GuidaPlate Tier 3 integration verification")
    print("=" * 50)

    results: list[bool] = []

    results.append(check("weekly_rf.pkl exists", RF_PATH.is_file()))
    results.append(check("weekly_rf_config.json exists", CFG_PATH.is_file()))

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
                    "Model inference (predict + proba)",
                    ok,
                    f"{LABELS[pred]}, sum={float(proba.sum()):.4f}",
                )
            )
        except Exception as e:
            results.append(check("Model inference (predict + proba)", False, str(e)))
    else:
        results.append(check("Model inference (predict + proba)", False))

    if config:
        cw = config.get("class_weight", {})
        ok = cw.get("MODERATE") == 3
        winner = config.get("winner", "")
        results.append(
            check("Production winner is RF + CW MOD=3", ok, winner or "missing winner")
        )
    else:
        results.append(check("Production winner is RF + CW MOD=3", False))

    try:
        from backend.api.weekly_trend import _predict_weekly_tier3

        high_week = [[0.05, 0.15, 0.80]] * 7
        out = _predict_weekly_tier3(high_week)
        ok = out["method"] == "random_forest" and out["risk_label"] in LABELS.values()
        results.append(
            check(
                "Backend _predict_weekly_tier3 wired",
                ok,
                f"{out['risk_label']} via {out['method']}",
            )
        )
    except Exception as e:
        results.append(check("Backend _predict_weekly_tier3 wired", False, str(e)))

    try:
        from backend.api.weekly_trend import _predict_weekly_tier3

        padded = _predict_weekly_tier3([WEEKLY_NEUTRAL.copy()])
        ok = padded["risk_label"] in LABELS.values() and padded["days_analyzed"] == 0
        results.append(
            check(
                "Neutral padding + days_analyzed",
                ok,
                f"label={padded['risk_label']}, days={padded['days_analyzed']}",
            )
        )
    except Exception as e:
        results.append(check("Neutral padding + days_analyzed", False, str(e)))

    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("All checks PASSED — Tier 3 integration OK")
        return 0

    print("Some checks FAILED — review output above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
