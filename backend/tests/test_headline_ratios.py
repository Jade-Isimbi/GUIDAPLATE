"""
Unit tests for nutrient_limit_ratios and _headline_by_ratio.

Headline among exceeded/near-limit nutrients is highest value/limit
(ratio), not CLINICAL_SEVERITY_WEIGHTS.
"""

from __future__ import annotations

import pytest

from backend.api.risk_prediction import _headline_by_ratio, nutrient_limit_ratios
from backend.clinical_constants import KDOQI_DAILY_LIMITS


def test_nutrient_limit_ratios_g3a():
    limits = KDOQI_DAILY_LIMITS["G3a"]
    ratios = nutrient_limit_ratios(
        potassium=3600.0,  # 3600/3000 = 1.2
        phosphorus=1200.0,  # 1200/800 = 1.5
        protein_per_kg=0.3,  # 0.3/0.6 = 0.5
        sodium=2530.0,  # 2530/2300 = 1.1
        limits=limits,
    )
    assert ratios["potassium"] == pytest.approx(1.2)
    assert ratios["phosphorus"] == pytest.approx(1.5)
    assert ratios["protein"] == pytest.approx(0.5)
    assert ratios["sodium"] == pytest.approx(1.1)


def test_headline_highest_ratio_wins_over_severity_weight():
    # Potassium has higher severity weight (0.35) than phosphorus (0.30),
    # but phosphorus has the higher ratio (1.5 > 1.2) → phosphorus wins.
    ratios = {
        "potassium": 1.2,
        "phosphorus": 1.5,
        "sodium": 1.1,
    }
    exceeded = ["potassium", "phosphorus", "sodium"]
    assert _headline_by_ratio(exceeded, ratios) == "phosphorus"


def test_headline_single_candidate():
    assert _headline_by_ratio(["sodium"], {"sodium": 1.05, "potassium": 2.0}) == "sodium"
