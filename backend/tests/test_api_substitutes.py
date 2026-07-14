"""
Integration tests for POST /api/recommendations/substitutes.

Verified example: beef meat + potassium exceeded + G3a →
Rwandan Meat/Fish/Egg under K threshold (3000*0.15=450), stage-safe,
lowest-K first (excluding queried beef): eggs, chicken meat, pork.
"""

from __future__ import annotations

from backend.models.recommender import get_recommender


def test_substitutes_beef_meat_g3a_potassium(client):
    r = client.post(
        "/api/recommendations/substitutes",
        json={
            "food_name": "beef meat",
            "ckd_stage": "G3a",
            "exceeded_nutrients": ["potassium"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["food_queried"] == "beef meat"
    assert body["ckd_stage"] == "G3a"
    assert body["count"] == 3
    names = [s["english"].lower() for s in body["substitutes"]]
    assert names == ["eggs", "chicken meat", "pork"]
    categories = {s["category"] for s in body["substitutes"]}
    assert categories <= {"Meat", "Fish", "Egg"}
    assert "Dairy" not in categories
    assert "Fruit" not in categories
    assert "Vegetable" not in categories


def test_substitutes_direct_recommender_matches_api(db_session):
    # Same filter logic via FoodRecommender (no HTTP)
    results = get_recommender().get_substitutes(
        food_name="beef meat",
        ckd_stage="G3a",
        risk_label="HIGH",
        exceeded_nutrients=["potassium"],
        db=db_session,
    )
    assert [s["english"].lower() for s in results] == [
        "eggs",
        "chicken meat",
        "pork",
    ]


def test_substitutes_unknown_food_404(client):
    r = client.post(
        "/api/recommendations/substitutes",
        json={
            "food_name": "not-a-real-food-xyz",
            "ckd_stage": "G3a",
            "exceeded_nutrients": ["potassium"],
        },
    )
    assert r.status_code == 404


def test_substitutes_invalid_stage_400(client):
    r = client.post(
        "/api/recommendations/substitutes",
        json={
            "food_name": "beef meat",
            "ckd_stage": "G9",
            "exceeded_nutrients": ["potassium"],
        },
    )
    assert r.status_code == 400
