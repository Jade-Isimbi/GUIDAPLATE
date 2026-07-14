"""
Unit tests for compute_exceeded_nutrients / compute_exceeded_nutrients_meal.

Classification (NEAR_LIMIT_RATIO = 0.8):
  ratio >= 1.0 → exceeded
  0.8 <= ratio < 1.0 → near_limit
  ratio < 0.8 → neither
List keys use "protein" (not protein_per_kg).
"""

from __future__ import annotations

from backend.api.risk_prediction import compute_exceeded_nutrients
from backend.models.xgboost_model import compute_exceeded_nutrients_meal

# G3a daily K limit = 3000 → 80% = 2400


def test_day_exactly_80_percent_is_near_limit_not_exceeded():
    exceeded, near = compute_exceeded_nutrients(
        potassium=2400.0,
        phosphorus=100.0,
        protein_per_kg=0.1,
        sodium=100.0,
        ckd_stage="G3a",
    )
    assert exceeded == []
    assert near == ["potassium"]


def test_day_exactly_100_percent_is_exceeded():
    exceeded, near = compute_exceeded_nutrients(
        potassium=3000.0,
        phosphorus=100.0,
        protein_per_kg=0.1,
        sodium=100.0,
        ckd_stage="G3a",
    )
    assert exceeded == ["potassium"]
    assert near == []


def test_day_just_under_80_percent_is_clear():
    exceeded, near = compute_exceeded_nutrients(
        potassium=2399.0,
        phosphorus=100.0,
        protein_per_kg=0.1,
        sodium=100.0,
        ckd_stage="G3a",
    )
    assert exceeded == []
    assert near == []


def test_day_just_over_100_percent_is_exceeded():
    exceeded, near = compute_exceeded_nutrients(
        potassium=3001.0,
        phosphorus=100.0,
        protein_per_kg=0.1,
        sodium=100.0,
        ckd_stage="G3a",
    )
    assert exceeded == ["potassium"]
    assert "potassium" not in near


def test_meal_breakfast_g3a_boundaries():
    # Breakfast K cap = 750; 80% = 600; 100% = 750
    exceeded, near = compute_exceeded_nutrients_meal(
        potassium=600.0,
        phosphorus=10.0,
        protein_per_kg=0.01,
        sodium=10.0,
        ckd_stage="G3a",
        occasion="Breakfast",
    )
    assert exceeded == []
    assert near == ["potassium"]

    exceeded2, near2 = compute_exceeded_nutrients_meal(
        potassium=750.0,
        phosphorus=10.0,
        protein_per_kg=0.01,
        sodium=10.0,
        ckd_stage="G3a",
        occasion="Breakfast",
    )
    assert exceeded2 == ["potassium"]
    assert near2 == []


def test_protein_key_name_in_exceeded_list():
    # G3a protein_per_kg limit 0.6 → value 0.6 exceeds as "protein"
    exceeded, _ = compute_exceeded_nutrients(
        potassium=100.0,
        phosphorus=100.0,
        protein_per_kg=0.6,
        sodium=100.0,
        ckd_stage="G3a",
    )
    assert exceeded == ["protein"]
