"""
Unit tests for meal_limits_for_occasion.

meal_limits = KDOQI_DAILY_LIMITS[stage] × OCCASION_RULES[occasion]["nutrient_caps"]
Breakfast caps: (0.25, 0.25, 0.30, 0.25)  # K, P, protein, Na
Lunch/Dinner:   (0.40, 0.40, 0.40, 0.40)
Snack:          (0.15, 0.15, 0.10, 0.15)
"""

from __future__ import annotations

import pytest

from backend.models.xgboost_model import meal_limits_for_occasion


def test_g3a_breakfast_limits():
    # 3000*0.25, 800*0.25, 0.6*0.30, 2300*0.25
    limits = meal_limits_for_occasion("G3a", "Breakfast")
    assert limits["potassium"] == pytest.approx(750.0)
    assert limits["phosphorus"] == pytest.approx(200.0)
    assert limits["protein_per_kg"] == pytest.approx(0.18)
    assert limits["sodium"] == pytest.approx(575.0)


def test_g2_lunch_limits():
    # 3500*0.4, 1000*0.4, 0.8*0.4, 2300*0.4
    limits = meal_limits_for_occasion("G2", "Lunch")
    assert limits["potassium"] == pytest.approx(1400.0)
    assert limits["phosphorus"] == pytest.approx(400.0)
    assert limits["protein_per_kg"] == pytest.approx(0.32)
    assert limits["sodium"] == pytest.approx(920.0)


def test_g3b_dinner_matches_lunch_fractions():
    # Dinner shares 0.40 caps with Lunch; G3b daily = G3a daily
    lunch = meal_limits_for_occasion("G3b", "Lunch")
    dinner = meal_limits_for_occasion("G3b", "Dinner")
    assert dinner == lunch
    assert dinner["potassium"] == pytest.approx(1200.0)  # 3000*0.4


def test_g4_snack_limits():
    # 2500*0.15, 700*0.15, 0.55*0.10, 2300*0.15
    limits = meal_limits_for_occasion("G4", "Snack")
    assert limits["potassium"] == pytest.approx(375.0)
    assert limits["phosphorus"] == pytest.approx(105.0)
    assert limits["protein_per_kg"] == pytest.approx(0.055)
    assert limits["sodium"] == pytest.approx(345.0)


def test_unknown_stage_or_occasion_raises():
    with pytest.raises(KeyError):
        meal_limits_for_occasion("G9", "Breakfast")
    with pytest.raises(KeyError):
        meal_limits_for_occasion("G3a", "Brunch")
