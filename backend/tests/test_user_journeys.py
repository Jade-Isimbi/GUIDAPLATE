"""
Full-chain integration tests that mirror real Meal Check user journeys.

Unlike single-endpoint unit/API tests, these sequences exercise:
  register → food-log → predict/risk → daily-budget / history / substitutes

Re-check (chain 3) mirrors the frontend skipSave path: predict/risk is
called again WITHOUT a second POST /patient/food-log, proving the check
endpoint does not persist logs and count stays stable.
"""

from __future__ import annotations

import pytest

# Explicit nutrients so budget/history expectations are exact (bypass CSV scale).
CABBAGE_LUNCH_LOG = {
    "food_name": "cabbage",
    "category": "Vegetable",
    "stage_safe_range": "1-5",
    "portion_grams": 100.0,
    "meal_occasion": "Lunch",
    "potassium_mg": 200.0,
    "phosphorus_mg": 50.0,
    "protein_g": 2.0,
    "sodium_mg": 30.0,
}

WEIGHT_KG = 70.0

# High meal totals for Breakfast G3a (meal K cap = 750) → HIGH + exceeded.
HIGH_RISK_MEAL_PAYLOAD = {
    "potassium": 5000.0,
    "phosphorus": 2000.0,
    "protein_per_kg": 2.0,
    "sodium": 5000.0,
    "ckd_stage": "G3a",
    "occasion": "Breakfast",
    "food_name": "beef meat",
}

SAME_INPUT_FOR_SCALE_SWITCH = {
    "potassium": 375.0,
    "phosphorus": 100.0,
    "protein_per_kg": 0.09,
    "sodium": 287.5,
    "ckd_stage": "G3a",
    "occasion": "Breakfast",
}


def _post_food_log(client, headers, payload: dict) -> dict:
    r = client.post("/api/patient/food-log", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "saved"
    assert body["log_id"]
    return body


def _history(client, headers) -> list[dict]:
    r = client.get("/api/patient/food-log/history", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def test_chain_log_check_budget_reflects_food(client, auth_headers):
    """1. Log food → check meal → daily budget reflects logged nutrients."""
    _post_food_log(client, auth_headers, CABBAGE_LUNCH_LOG)

    risk = client.post(
        "/api/predict/risk",
        json={
            "potassium": CABBAGE_LUNCH_LOG["potassium_mg"],
            "phosphorus": CABBAGE_LUNCH_LOG["phosphorus_mg"],
            "protein_per_kg": CABBAGE_LUNCH_LOG["protein_g"] / WEIGHT_KG,
            "sodium": CABBAGE_LUNCH_LOG["sodium_mg"],
            "ckd_stage": "G3a",
            "occasion": "Lunch",
            "food_name": CABBAGE_LUNCH_LOG["food_name"],
        },
        headers=auth_headers,
    )
    assert risk.status_code == 200, risk.text
    assert risk.json()["scoring_scale"] == "meal"
    assert risk.json()["occasion"] == "Lunch"

    budget = client.get("/api/patient/daily-budget", headers=auth_headers)
    assert budget.status_code == 200, budget.text
    nutrients = budget.json()["nutrients"]

    assert nutrients["potassium"]["consumed"] == pytest.approx(200.0)
    assert nutrients["phosphorus"]["consumed"] == pytest.approx(50.0)
    assert nutrients["sodium"]["consumed"] == pytest.approx(30.0)
    # protein_per_kg = 2.0 / 70.0 ≈ 0.0286 → rounded to 4 dp by _nutrient_budget
    assert nutrients["protein_per_kg"]["consumed"] == pytest.approx(
        round(2.0 / WEIGHT_KG, 4)
    )


def test_chain_logged_food_appears_in_history(client, auth_headers):
    """2. After logging, history returns the entry with correct values."""
    _post_food_log(client, auth_headers, CABBAGE_LUNCH_LOG)

    logs = _history(client, auth_headers)
    assert len(logs) == 1
    entry = logs[0]
    assert entry["food_name"] == "cabbage"
    assert entry["meal_occasion"] == "Lunch"
    assert entry["portion_grams"] == pytest.approx(100.0)
    assert entry["potassium_mg"] == pytest.approx(200.0)
    assert entry["phosphorus_mg"] == pytest.approx(50.0)
    assert entry["protein_g"] == pytest.approx(2.0)
    assert entry["sodium_mg"] == pytest.approx(30.0)
    assert entry["category"] == "Vegetable"


def test_chain_recheck_does_not_duplicate_logs(client, auth_headers):
    """
    3. Re-check (skipSave) does not create extra food logs.

    Frontend Re-check calls predict/risk again without POST /patient/food-log.
    Backend proof: one persist + two risk checks → history count stays 1.
    """
    _post_food_log(client, auth_headers, CABBAGE_LUNCH_LOG)
    assert len(_history(client, auth_headers)) == 1

    risk_payload = {
        "potassium": CABBAGE_LUNCH_LOG["potassium_mg"],
        "phosphorus": CABBAGE_LUNCH_LOG["phosphorus_mg"],
        "protein_per_kg": CABBAGE_LUNCH_LOG["protein_g"] / WEIGHT_KG,
        "sodium": CABBAGE_LUNCH_LOG["sodium_mg"],
        "ckd_stage": "G3a",
        "occasion": "Lunch",
        "food_name": "cabbage",
    }
    r1 = client.post("/api/predict/risk", json=risk_payload, headers=auth_headers)
    r2 = client.post("/api/predict/risk", json=risk_payload, headers=auth_headers)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    assert len(_history(client, auth_headers)) == 1


def test_chain_high_risk_substitutes_full_path(client, auth_headers):
    """
    4. Log high-risk beef meal → check → exceeded nutrients + safe substitutes.
    """
    beef_log = {
        "food_name": "beef meat",
        "category": "Meat",
        "stage_safe_range": "1-4",
        "portion_grams": 100.0,
        "meal_occasion": "Breakfast",
        # Persist the meal totals used for risk (explicit, like the UI does)
        "potassium_mg": HIGH_RISK_MEAL_PAYLOAD["potassium"],
        "phosphorus_mg": HIGH_RISK_MEAL_PAYLOAD["phosphorus"],
        "protein_g": HIGH_RISK_MEAL_PAYLOAD["protein_per_kg"] * WEIGHT_KG,
        "sodium_mg": HIGH_RISK_MEAL_PAYLOAD["sodium"],
    }
    _post_food_log(client, auth_headers, beef_log)

    risk = client.post(
        "/api/predict/risk",
        json=HIGH_RISK_MEAL_PAYLOAD,
        headers=auth_headers,
    )
    assert risk.status_code == 200, risk.text
    body = risk.json()
    assert body["risk_label"] == "HIGH"
    assert body["exceeded_nutrients"]
    assert "potassium" in body["exceeded_nutrients"]

    names = [s["english"].lower() for s in body["substitutes"]]
    assert names == ["eggs", "chicken meat", "pork"]
    categories = {s["category"] for s in body["substitutes"]}
    assert categories <= {"Meat", "Fish", "Egg"}
    assert not categories & {"Dairy", "Fruit", "Vegetable"}

    # History still has the single logged beef meal
    logs = _history(client, auth_headers)
    assert len(logs) == 1
    assert logs[0]["food_name"] == "beef meat"


def test_chain_model_failure_uses_meal_rule_fallback(client, auth_headers, monkeypatch):
    """
    5. Model failure remains meal-scale and returns the transparent rule fallback.
    """
    monkeypatch.setattr(
        "backend.api.risk_prediction.get_meal_predictor",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated outage")),
    )
    response = client.post(
        "/api/predict/risk",
        json=SAME_INPUT_FOR_SCALE_SWITCH,
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scoring_scale"] == "meal"
    assert body["prediction_source"] == "rule_fallback"
    assert body["meal_feature_set"] == "rule_fallback"
    assert body["meal_limits"]["potassium"] == pytest.approx(750.0)
    assert body["occasion"] == "Breakfast"
