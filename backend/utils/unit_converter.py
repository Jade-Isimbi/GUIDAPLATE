"""
Unit conversion helpers for patient-facing portion input.
"""

CATEGORY_UNITS = {
    "Fruit": {
        "unit": "piece",
        "grams_per_unit": 150,
        "examples": "apple, orange, mango",
    },
    "Fruit_small": {
        "unit": "piece",
        "grams_per_unit": 80,
        "examples": "passion fruit, plum",
    },
    "Starch/Grain": {
        "unit": "cup",
        "grams_per_unit": 185,
        "examples": "rice, ugali, maize meal",
    },
    "Bread": {
        "unit": "slice",
        "grams_per_unit": 30,
        "examples": "white bread, brown bread",
    },
    "Root Vegetable": {
        "unit": "piece",
        "grams_per_unit": 200,
        "examples": "potato, sweet potato, cassava",
    },
    "Vegetable": {
        "unit": "cup",
        "grams_per_unit": 90,
        "examples": "cabbage, spinach, carrot",
    },
    "Meat": {
        "unit": "piece",
        "grams_per_unit": 85,
        "examples": "chicken, beef, goat",
    },
    "Fish": {
        "unit": "piece",
        "grams_per_unit": 85,
        "examples": "tilapia, sardine",
    },
    "Egg": {
        "unit": "egg",
        "grams_per_unit": 50,
        "examples": "whole egg",
    },
    "Legume": {
        "unit": "cup",
        "grams_per_unit": 180,
        "examples": "beans, lentils, chickpeas",
    },
    "Dairy": {
        "unit": "cup",
        "grams_per_unit": 240,
        "examples": "milk, ikivuguto, yogurt",
    },
    "Beverage": {
        "unit": "cup",
        "grams_per_unit": 240,
        "examples": "tea, juice",
    },
    "Fat/Oil": {
        "unit": "tablespoon",
        "grams_per_unit": 14,
        "examples": "cooking oil, butter",
    },
    "Nut": {
        "unit": "tablespoon",
        "grams_per_unit": 30,
        "examples": "peanuts, groundnuts",
    },
    "Other": {
        "unit": "serving",
        "grams_per_unit": 100,
        "examples": "general serving",
    },
}

CATEGORY_ALIASES = {
    "Starch": "Starch/Grain",
    "Grain": "Starch/Grain",
}

FOOD_UNIT_OVERRIDES = {
    "banana": {"unit": "banana", "grams": 120},
    "egg": {"unit": "egg", "grams": 50},
    "bread": {"unit": "slice", "grams": 30},
    "avocado": {"unit": "piece", "grams": 200},
    "pineapple": {"unit": "slice", "grams": 80},
    "watermelon": {"unit": "slice", "grams": 200},
    "sugarcane": {"unit": "piece", "grams": 100},
    "milk": {"unit": "glass", "grams": 240},
    "tea": {"unit": "cup", "grams": 240},
    "ikivuguto": {"unit": "cup", "grams": 240},
    "sweet potatoes": {"unit": "piece", "grams": 200},
    "sweet potato": {"unit": "piece", "grams": 200},
    "cassava": {"unit": "piece", "grams": 200},
    "irish potatoes": {"unit": "piece", "grams": 200},
    "yams": {"unit": "cup", "grams": 150},
    "plantains": {"unit": "piece", "grams": 179},
    "rice": {"unit": "cup", "grams": 185},
    "ugali": {"unit": "serving", "grams": 200},
    "beans": {"unit": "cup", "grams": 180},
    "chicken": {"unit": "piece", "grams": 85},
    "tilapia": {"unit": "piece", "grams": 85},
    "oats": {"unit": "cup", "grams": 80},
    "peanut butter": {"unit": "tablespoon", "grams": 30},
}


def get_unit_info(food_name: str, category: str) -> dict:
    """Return unit name and grams per unit for a given food."""
    food_lower = food_name.lower()
    for key, override in FOOD_UNIT_OVERRIDES.items():
        if key in food_lower:
            return {
                "unit": override["unit"],
                "grams_per_unit": override["grams"],
            }

    category_key = CATEGORY_ALIASES.get(category, category)
    cat_info = CATEGORY_UNITS.get(category_key, CATEGORY_UNITS["Other"])
    return {
        "unit": cat_info["unit"],
        "grams_per_unit": cat_info["grams_per_unit"],
    }


def units_to_grams(food_name: str, category: str, quantity: float) -> float:
    """Convert quantity in units to grams."""
    unit_info = get_unit_info(food_name, category)
    return quantity * unit_info["grams_per_unit"]


def format_unit_display(quantity: float, unit: str, grams: float) -> str:
    """Build a patient-friendly display string like '2 bananas (240g)'."""
    plural = quantity > 1 and not unit.endswith("s")
    unit_label = f"{unit}{'s' if plural else ''}"
    return f"{quantity:g} {unit_label} ({round(grams)}g)"
