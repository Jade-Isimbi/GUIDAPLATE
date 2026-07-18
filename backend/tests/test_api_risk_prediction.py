"""
Integration tests for /api/predict/risk and /api/predict/thresholds/{stage}.
"""

from __future__ import annotations

import pytest


def test_thresholds_g2(client):
    r = client.get("/api/predict/thresholds/G2")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "G2"
    assert body["thresholds"]["potassium"] == 3500.0
    assert body["thresholds"]["phosphorus"] == 1000.0
    assert body["thresholds"]["protein_per_kg"] == 0.8
    assert body["thresholds"]["sodium"] == 2300.0


def test_thresholds_g3a(client):
    r = client.get("/api/predict/thresholds/G3a")
    assert r.status_code == 200
    t = r.json()["thresholds"]
    assert t == {
        "potassium": 3000.0,
        "phosphorus": 800.0,
        "protein_per_kg": 0.6,
        "sodium": 2300.0,
    }


def test_thresholds_g3b(client):
    r = client.get("/api/predict/thresholds/G3b")
    assert r.status_code == 200
    t = r.json()["thresholds"]
    assert t["potassium"] == 3000.0
    assert t["phosphorus"] == 800.0
    assert t["protein_per_kg"] == 0.6


def test_thresholds_g4(client):
    r = client.get("/api/predict/thresholds/G4")
    assert r.status_code == 200
    t = r.json()["thresholds"]
    assert t["potassium"] == 2500.0
    assert t["phosphorus"] == 700.0
    assert t["protein_per_kg"] == 0.55
    assert t["sodium"] == 2300.0


def test_thresholds_invalid_stage_404(client):
    r = client.get("/api/predict/thresholds/G9")
    assert r.status_code == 404


def test_predict_risk_meal_scale_low(client):
    # Half of G3a Breakfast caps → clinical score 0.5 → expect LOW
    payload = {
        "potassium": 375.0,
        "phosphorus": 100.0,
        "protein_per_kg": 0.09,
        "sodium": 287.5,
        "ckd_stage": "G3a",
        "occasion": "Breakfast",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scoring_scale"] == "meal"
    assert body["occasion"] == "Breakfast"
    assert body["meal_limits"]["potassium"] == pytest.approx(750.0)
    assert body["meal_limits"]["phosphorus"] == pytest.approx(200.0)
    assert body["meal_limits"]["protein_per_kg"] == pytest.approx(0.18)
    assert body["meal_limits"]["sodium"] == pytest.approx(575.0)
    assert body["risk_label"] == "LOW"
    assert body["exceeded_nutrients"] == []
    assert set(body["probabilities"]) >= {"LOW", "MODERATE", "HIGH"}
    assert "confidence" in body
    assert body["ckd_stage"] == "G3a"


def test_predict_risk_meal_scale_high(client):
    # Far above meal caps → HIGH
    payload = {
        "potassium": 5000.0,
        "phosphorus": 2000.0,
        "protein_per_kg": 2.0,
        "sodium": 5000.0,
        "ckd_stage": "G3a",
        "occasion": "Breakfast",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scoring_scale"] == "meal"
    assert body["meal_limits"] is not None
    assert body["risk_label"] == "HIGH"
    assert "potassium" in body["exceeded_nutrients"]
    assert len(body["exceeded_nutrients"]) == 4


def test_predict_risk_rule_fallback_low(client, monkeypatch):
    def _fail():
        raise RuntimeError("simulated model outage")

    monkeypatch.setattr("backend.api.risk_prediction.get_meal_predictor", _fail)
    payload = {
        "potassium": 375.0,
        "phosphorus": 100.0,
        "protein_per_kg": 0.09,
        "sodium": 287.5,
        "ckd_stage": "G3a",
        "occasion": "Breakfast",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scoring_scale"] == "meal"
    assert body["meal_limits"]["potassium"] == pytest.approx(750.0)
    assert body["prediction_source"] == "rule_fallback"
    assert body["meal_feature_set"] == "rule_fallback"
    assert body["confidence"] == 0.0
    assert body["risk_label"] == "LOW"
    assert body["exceeded_nutrients"] == []
    assert body["shap_contributions"] is None


def test_predict_risk_rule_fallback_high(client, monkeypatch):
    class BrokenPredictor:
        score_mode = "meal_noscore"

        def predict(self, *args, **kwargs):
            raise RuntimeError("simulated inference outage")

    monkeypatch.setattr(
        "backend.api.risk_prediction.get_meal_predictor",
        lambda: BrokenPredictor(),
    )
    payload = {
        "potassium": 1000.0,
        "phosphorus": 300.0,
        "protein_per_kg": 0.30,
        "sodium": 900.0,
        "ckd_stage": "G3a",
        "occasion": "Breakfast",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scoring_scale"] == "meal"
    assert body["prediction_source"] == "rule_fallback"
    assert len(body["exceeded_nutrients"]) >= 2
    assert body["risk_label"] == "HIGH"


def test_predict_risk_invalid_ckd_stage_400(client):
    payload = {
        "potassium": 100.0,
        "phosphorus": 100.0,
        "protein_per_kg": 0.1,
        "sodium": 100.0,
        "ckd_stage": "G9",
        "occasion": "Breakfast",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 400
    assert "ckd_stage" in r.json()["detail"].lower() or "G2" in r.json()["detail"]


def test_predict_risk_invalid_occasion_422(client):
    # occasion is a Pydantic Literal → request validation 422, not handler 400
    payload = {
        "potassium": 100.0,
        "phosphorus": 100.0,
        "protein_per_kg": 0.1,
        "sodium": 100.0,
        "ckd_stage": "G3a",
        "occasion": "Brunch",
    }
    r = client.post("/api/predict/risk", json=payload)
    assert r.status_code == 422
