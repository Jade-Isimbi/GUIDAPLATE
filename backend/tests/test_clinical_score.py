"""
Unit tests for day- and meal-scale clinical scores.

Formula (xgboost_model.compute_clinical_score*):
  for each nutrient: ratio = value / limit
  if ratio > 1: score += weight * (1 + (ratio - 1) * 2)
  else:         score += weight * ratio
Weights: K=0.35, P=0.30, protein=0.25, Na=0.10
"""

from __future__ import annotations

import pytest

from backend.models.xgboost_model import (
    compute_clinical_score,
    compute_clinical_score_meal,
)

# Hand-computed against CLINICAL_SEVERITY_WEIGHTS + KDOQI_DAILY_LIMITS["G3a"]
# G3a daily: K=3000, P=800, protein_per_kg=0.6, Na=2300


def test_day_score_all_nutrients_at_50_percent():
    # ratios all 0.5 → 0.35*0.5 + 0.30*0.5 + 0.25*0.5 + 0.10*0.5 = 0.5
    score = compute_clinical_score(
        potassium=1500.0,
        phosphorus=400.0,
        protein_per_kg=0.3,
        sodium=1150.0,
        ckd_stage="G3a",
    )
    assert score == pytest.approx(0.5)


def test_day_score_exactly_at_limit_uses_linear_branch():
    # ratio == 1.0 is NOT > 1.0 → linear: sum(weights) = 1.0
    score = compute_clinical_score(
        potassium=3000.0,
        phosphorus=800.0,
        protein_per_kg=0.6,
        sodium=2300.0,
        ckd_stage="G3a",
    )
    assert score == pytest.approx(1.0)


def test_day_score_potassium_overshoot_1_5x():
    # K ratio=1.5 → 0.35 * (1 + 0.5*2) = 0.35 * 2 = 0.7
    # others at 0.5 → 0.325; total = 1.025
    score = compute_clinical_score(
        potassium=4500.0,
        phosphorus=400.0,
        protein_per_kg=0.3,
        sodium=1150.0,
        ckd_stage="G3a",
    )
    assert score == pytest.approx(1.025)


def test_meal_score_breakfast_g3a_half_caps():
    # Breakfast caps G3a: (750, 200, 0.18, 575); half → same 0.5 as day case
    score = compute_clinical_score_meal(
        potassium=375.0,
        phosphorus=100.0,
        protein_per_kg=0.09,
        sodium=287.5,
        ckd_stage="G3a",
        occasion="Breakfast",
    )
    assert score == pytest.approx(0.5)


def test_meal_score_breakfast_g3a_potassium_1_5x_cap():
    # K=1125 / 750 = 1.5 → same overshoot math → 1.025
    score = compute_clinical_score_meal(
        potassium=1125.0,
        phosphorus=100.0,
        protein_per_kg=0.09,
        sodium=287.5,
        ckd_stage="G3a",
        occasion="Breakfast",
    )
    assert score == pytest.approx(1.025)
