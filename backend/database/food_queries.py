"""
food_queries.py
GuidaPlate — SQLAlchemy food lookup helpers
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.database.db import Food


def find_food_by_name(db: Session, food_name: str) -> Food | None:
    """Exact English match first, then case-insensitive contains."""
    name = food_name.strip()
    if not name:
        return None

    food = db.query(Food).filter(Food.english.ilike(name)).first()
    if food is not None:
        return food

    return db.query(Food).filter(Food.english.ilike(f"%{name}%")).first()


def get_food_nutrients(
    food_name: str,
    portion_grams: float,
    db: Session,
) -> dict | None:
    """
    Look up food by name and compute nutrients for given portion.
    Uses exact match first then case-insensitive contains.
    """
    food = find_food_by_name(db, food_name)
    if food is None:
        return None

    ratio = portion_grams / 100.0

    return {
        "food_name": food.english,
        "food_id": food.food_id,
        "category": food.category,
        "portion_grams": portion_grams,
        "potassium_mg": round(food.potassium_mg * ratio, 1),
        "phosphorus_mg": round(food.phosphorus_mg * ratio, 1),
        "protein_g": round(food.protein_g * ratio, 2),
        "sodium_mg": round(food.sodium_mg * ratio, 1),
    }
