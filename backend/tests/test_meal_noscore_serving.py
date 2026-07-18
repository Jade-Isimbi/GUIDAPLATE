"""Live meal serving uses only the no-clinical_score model (occasion + meal caps)."""

from __future__ import annotations


def test_meal_predictor_is_noscore():
    from backend.models.xgboost_model import get_meal_predictor

    predictor = get_meal_predictor()
    assert predictor.score_mode == "meal_noscore"
    result = predictor.predict(
        potassium=400.0,
        phosphorus=200.0,
        protein_per_kg=0.15,
        sodium=400.0,
        ckd_stage="G3b",
        occasion="Lunch",
    )
    assert "clinical_score" not in result["features_used"]
    assert "occasion_encoded" in result["features_used"]
    assert "meal_cap_potassium" in result["features_used"]
    assert result["risk_label"] in {"LOW", "MODERATE", "HIGH"}


def test_predict_risk_reports_noscore_feature_set(client):
    response = client.post(
        "/api/predict/risk",
        json={
            "potassium": 400.0,
            "phosphorus": 200.0,
            "protein_per_kg": 0.15,
            "sodium": 400.0,
            "ckd_stage": "G3b",
            "occasion": "Lunch",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scoring_scale"] == "meal"
    assert body["meal_feature_set"] == "noscore_occasion_caps"
    assert body["prediction_source"] == "xgboost"
    assert "clinical_score" not in body["features_used"]
