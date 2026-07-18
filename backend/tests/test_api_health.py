"""
Health-check must smoke-test the meal XGBoost model selected for live serving.
The day and legacy meal models are offline research artifacts and are never health-checked.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def health_client(db_session):
    """Minimal app exposing main.api_health without full lifespan/RAG warmup."""
    from backend.database.db import get_db
    from backend.main import api_health

    app = FastAPI(title="GuidaPlate Health Test")
    app.add_api_route("/api/health", api_health, methods=["GET"])

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as client:
        yield client


def test_health_exercises_noscore_meal_model(health_client, monkeypatch):
    meal_calls = {"n": 0}
    day_calls = {"n": 0}

    from backend.models import xgboost_model as xgb_mod

    real_meal = xgb_mod.get_meal_predictor

    def _meal():
        meal_calls["n"] += 1
        return real_meal()

    monkeypatch.setattr(xgb_mod, "get_meal_predictor", _meal)

    response = health_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    xgb = body["models"]["xgboost_v3"]
    assert xgb["status"] == "loaded"
    assert xgb["scoring_scale"] == "meal"
    assert xgb["score_mode"] == "meal_noscore"
    assert xgb.get("uses_clinical_score_feature") is False
    assert meal_calls["n"] >= 1
    assert day_calls["n"] == 0


def test_health_is_degraded_when_meal_model_fails(health_client, monkeypatch):
    def _fail():
        raise RuntimeError("simulated meal model outage")

    monkeypatch.setattr(
        "backend.main.smoke_predict_live_risk_predictor",
        _fail,
    )
    response = health_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models"]["xgboost_v3"]["status"] == "failed"
