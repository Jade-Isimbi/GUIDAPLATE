"""
Unit tests for recommender safety rules:
  - _parse_stage_safe
  - SUBSTITUTE_CATEGORY_GROUPS
  - _is_rwandan_candidate (CSV flag + oats bypass)
"""

from __future__ import annotations

from backend.database.db import Food
from backend.models.recommender import (
    SUBSTITUTE_CATEGORY_GROUPS,
    FoodRecommender,
    _is_rwandan_candidate,
)
import backend.models.recommender as recommender_mod


def test_parse_stage_safe_range_includes_patient():
    # STAGE_NUMERIC: G3a → 2; range "1-4" includes 2
    assert FoodRecommender._parse_stage_safe("1-4", 2) is True


def test_parse_stage_safe_range_excludes_patient_above():
    # G4 → 4; range "1-3" excludes 4
    assert FoodRecommender._parse_stage_safe("1-3", 4) is False


def test_parse_stage_safe_single_stage():
    assert FoodRecommender._parse_stage_safe("1", 1) is True
    assert FoodRecommender._parse_stage_safe("1", 2) is False


def test_parse_stage_safe_empty_or_none():
    assert FoodRecommender._parse_stage_safe(None, 2) is False
    assert FoodRecommender._parse_stage_safe("", 2) is False
    assert FoodRecommender._parse_stage_safe("   ", 2) is False


def test_parse_stage_safe_boundary_low_and_high():
    assert FoodRecommender._parse_stage_safe("2-3", 2) is True
    assert FoodRecommender._parse_stage_safe("2-3", 3) is True
    assert FoodRecommender._parse_stage_safe("2-3", 1) is False
    assert FoodRecommender._parse_stage_safe("2-3", 4) is False


def test_meat_category_group_allows_meat_fish_egg_only():
    meat_group = SUBSTITUTE_CATEGORY_GROUPS["Meat"]
    assert meat_group == frozenset({"Meat", "Fish", "Egg"})
    assert "Dairy" not in meat_group
    assert "Fruit" not in meat_group
    assert "Vegetable" not in meat_group
    assert SUBSTITUTE_CATEGORY_GROUPS["Dairy"] == frozenset({"Dairy"})
    assert SUBSTITUTE_CATEGORY_GROUPS["Fruit"] == frozenset({"Fruit"})


def test_is_rwandan_candidate_true_for_flagged_food_id(monkeypatch):
    monkeypatch.setattr(
        recommender_mod, "_rwandan_food_ids", lambda: {"10", "13"}
    )
    food = Food(
        food_id="10",
        english="beef meat",
        category="Meat",
        potassium_mg=230.0,
        phosphorus_mg=194.0,
        protein_g=28.7,
        sodium_mg=67.0,
    )
    assert _is_rwandan_candidate(food) is True


def test_is_rwandan_candidate_false_for_non_rwandan_non_oats(monkeypatch):
    monkeypatch.setattr(recommender_mod, "_rwandan_food_ids", lambda: {"10"})
    food = Food(
        food_id="77",
        english="Beef, chuck, roast",
        category="Meat",
        potassium_mg=281.0,
        phosphorus_mg=151.0,
        protein_g=18.4,
        sodium_mg=48.4,
    )
    assert _is_rwandan_candidate(food) is False


def test_is_rwandan_candidate_oats_bypass_even_without_flag(monkeypatch):
    monkeypatch.setattr(recommender_mod, "_rwandan_food_ids", lambda: set())
    oats = Food(
        food_id="999",
        english="Oats, whole grain, rolled",
        category="Grain",
        potassium_mg=100.0,
        phosphorus_mg=100.0,
        protein_g=5.0,
        sodium_mg=10.0,
    )
    assert _is_rwandan_candidate(oats) is True


def test_is_rwandan_candidate_goat_does_not_match_oats_bypass(monkeypatch):
    monkeypatch.setattr(recommender_mod, "_rwandan_food_ids", lambda: set())
    goat = Food(
        food_id="15",
        english="goat meat",
        category="Meat",
        potassium_mg=335.0,
        phosphorus_mg=211.0,
        protein_g=28.4,
        sodium_mg=73.0,
    )
    assert _is_rwandan_candidate(goat) is False
