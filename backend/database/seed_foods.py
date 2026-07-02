"""
seed_foods.py
GuidaPlate — Load food_database.csv into the foods SQLAlchemy table
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from backend.config import FOOD_DATABASE_CSV
from backend.database.db import Food, engine


def seed_foods(db: Session) -> int:
    """
    Load food_database.csv into foods table.
    Only seeds if table is empty.
    Returns number of foods in the table.
    """
    existing = db.query(Food).count()
    if existing > 0:
        return existing

    df = pd.read_csv(FOOD_DATABASE_CSV)

    foods: list[Food] = []
    for _, row in df.iterrows():
        stage_safe = row.get("ckd_stage_safe", row.get("stage_safe_range"))
        foods.append(
            Food(
                food_id=str(row.get("food_id", row.get("id", ""))),
                english=str(row["english"]),
                french=str(row["french"]) if pd.notna(row.get("french")) else None,
                kinyarwanda=(
                    str(row["kinyarwanda"])
                    if pd.notna(row.get("kinyarwanda"))
                    else None
                ),
                category=str(row.get("category", "Other")),
                potassium_mg=float(row.get("potassium_mg", 0) or 0),
                phosphorus_mg=float(row.get("phosphorus_mg", 0) or 0),
                protein_g=float(row.get("protein_g", 0) or 0),
                sodium_mg=float(row.get("sodium_mg", 0) or 0),
                energy_kcal=(
                    float(row["energy_kcal"])
                    if pd.notna(row.get("energy_kcal"))
                    else None
                ),
                stage_safe_range=str(stage_safe) if pd.notna(stage_safe) else None,
                preparation_method=(
                    str(row["preparation_method"])
                    if pd.notna(row.get("preparation_method"))
                    else None
                ),
            )
        )

    db.bulk_save_objects(foods)
    db.commit()
    return len(foods)


def backfill_preparation_method(db: Session) -> int:
    """Ensure preparation_method column exists and is populated from CSV."""
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("foods")]
    if "preparation_method" not in cols:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE foods ADD COLUMN preparation_method VARCHAR"))
            conn.commit()

    df = pd.read_csv(FOOD_DATABASE_CSV)
    lookup = {
        str(row["food_id"]): (
            str(row["preparation_method"])
            if pd.notna(row.get("preparation_method"))
            else ""
        )
        for _, row in df.iterrows()
    }

    updated = 0
    for food in db.query(Food).all():
        prep = lookup.get(str(food.food_id), "") or None
        if food.preparation_method != prep:
            food.preparation_method = prep
            updated += 1
    if updated:
        db.commit()
    return updated
