#!/usr/bin/env python3
"""
ONE-OFF MANUAL SCRIPT — NOT for Playwright, CI, or automatic runs.

Purpose:
  Insert backdated food_log rows for the E2E Shared User only, so the
  Diet Pattern / weekly-trend page shows a multi-day chart for screenshots.

Scope (hard):
  - Only email: e2e.shared@example.com
  - Only food_logs for that user's user_id / patient_id
  - Does NOT touch other users, model files, or model hashes

Usage (local SQLite default: guidaplate.db at repo root):
  python3 scripts/seed_demo_weekly_pattern.py \\
      --confirm-email e2e.shared@example.com

Optional DB override (same as the app):
  DATABASE_PATH=/path/to/guidaplate.db python3 scripts/seed_demo_weekly_pattern.py \\
      --confirm-email e2e.shared@example.com

This script is intentionally inert unless --confirm-email matches exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "backend" / "data" / "food_database.csv"

# Strict allowlist — script refuses any other account.
ALLOWED_EMAIL = "e2e.shared@example.com"

# Believable escalating week (UTC): low → higher nutrient burden toward today.
# Portions in grams; nutrients scaled from food_database.csv (per 100g).
# Each tuple: (days_ago, occasion, english_name, portion_g)
PLAN: list[tuple[int, str, str, float]] = [
    # Day −6 — light vegetable / starch day
    (6, "Breakfast", "sorghum porridge", 220),
    (6, "Lunch", "cabbage", 150),
    (6, "Lunch", "rice", 160),
    # Day −5 — still modest
    (5, "Breakfast", "maize porridge", 250),
    (5, "Lunch", "ugali", 180),
    (5, "Lunch", "green beans", 120),
    (5, "Dinner", "pumpkin", 150),
    # Day −4 — legumes + plantain rise
    (4, "Breakfast", "sweet potatoes", 180),
    (4, "Lunch", "beans", 160),
    (4, "Lunch", "plantains", 200),
    (4, "Snack", "pineapples", 120),
    # Day −3 — animal protein enters
    (3, "Breakfast", "eggs", 100),
    (3, "Breakfast", "milk", 200),
    (3, "Lunch", "chicken meat", 140),
    (3, "Lunch", "rice", 150),
    (3, "Lunch", "spinach", 100),
    # Day −2 — fish + high-K potato
    (2, "Breakfast", "banana", 180),
    (2, "Lunch", "fish", 170),
    (2, "Lunch", "Irish potatoes", 200),
    (2, "Dinner", "tomatoes", 120),
    # Day −1 — richer meats / avocado / groundnuts
    (1, "Breakfast", "avocados", 120),
    (1, "Lunch", "goat meat", 150),
    (1, "Lunch", "beans", 140),
    (1, "Snack", "groundnuts", 35),
    # Day 0 (today UTC) — peak burden for a clear chart cliff
    (0, "Breakfast", "banana", 200),
    (0, "Lunch", "pork", 160),
    (0, "Lunch", "Irish potatoes", 220),
    (0, "Lunch", "spinach", 140),
    (0, "Dinner", "cassava leaves", 100),
]


def _load_rwandan_foods() -> dict[str, dict[str, str]]:
    """Index is_rwandan=1 rows by lowercased english name → CSV row."""
    if not CSV_PATH.is_file():
        raise SystemExit(f"Missing food database CSV: {CSV_PATH}")

    by_name: dict[str, dict[str, str]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("is_rwandan") not in {"1", "true", "True"}:
                continue
            key = (row.get("english") or "").strip().lower()
            if key:
                by_name[key] = row
    return by_name


def _nutrients_from_csv(row: dict[str, str], portion_g: float) -> dict[str, float]:
    """Scale per-100g CSV columns to the given portion — no fabricated values."""
    ratio = portion_g / 100.0
    return {
        "potassium_mg": round(float(row["potassium_mg"]) * ratio, 1),
        "phosphorus_mg": round(float(row["phosphorus_mg"]) * ratio, 1),
        "protein_g": round(float(row["protein_g"]) * ratio, 2),
        "sodium_mg": round(float(row["sodium_mg"]) * ratio, 1),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "ONE-OFF: backdate food_logs for e2e.shared@example.com only "
            "(demo / screenshot). Not CI."
        )
    )
    p.add_argument(
        "--confirm-email",
        required=True,
        help=f"Must be exactly {ALLOWED_EMAIL!r} (safety gate).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve user + print plan without writing.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.confirm_email != ALLOWED_EMAIL:
        print(
            f"Refusing: --confirm-email must be {ALLOWED_EMAIL!r}, "
            f"got {args.confirm_email!r}",
            file=sys.stderr,
        )
        return 2

    # Import DB after ROOT is on path so we use the same guidaplate.db as the app.
    sys.path.insert(0, str(ROOT))
    from backend.database.db import FoodLog, SessionLocal, User  # noqa: WPS433

    foods = _load_rwandan_foods()
    missing = sorted(
        {
            name.lower()
            for _, _, name, _ in PLAN
            if name.lower() not in foods
        }
    )
    if missing:
        print(f"Foods not in food_database.csv (is_rwandan=1): {missing}", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        # --- Confirmation: resolve the one allowed account ----------------
        user = (
            db.query(User)
            .filter(User.email == ALLOWED_EMAIL)  # WHERE email = 'e2e.shared@example.com'
            .one_or_none()
        )
        if user is None:
            print(f"No user found for {ALLOWED_EMAIL}", file=sys.stderr)
            return 1

        user_id = user.user_id
        print("=== TARGET ACCOUNT (read-only confirmation) ===")
        print(f"  email    : {user.email}")
        print(f"  user_id  : {user_id}")
        print(f"  name     : {user.name}")

        today = datetime.utcnow().date()
        window_start = datetime.combine(today - timedelta(days=6), time.min)
        window_end = datetime.combine(today, time.max)

        # --- Clear ONLY this user's logs in the past 7 UTC days ------------
        # WHERE patient_id = :user_id AND logged_at within [today-6, today]
        clear_q = (
            db.query(FoodLog)
            .filter(
                FoodLog.patient_id == user_id,  # STRICT: this account only
                FoodLog.logged_at >= window_start,
                FoodLog.logged_at <= window_end,
            )
        )
        to_delete = clear_q.count()
        print(f"\n=== CLEAR last-7-days food_logs for this user_id only ===")
        print(f"  WHERE patient_id = {user_id!r}")
        print(f"    AND logged_at >= {window_start.isoformat()}")
        print(f"    AND logged_at <= {window_end.isoformat()}")
        print(f"  rows matched: {to_delete}")

        if args.dry_run:
            print("\n[dry-run] skipping delete + insert")
            for days_ago, occasion, name, portion in PLAN:
                d = today - timedelta(days=days_ago)
                n = _nutrients_from_csv(foods[name.lower()], portion)
                print(
                    f"  would insert {d.isoformat()} {occasion:10} "
                    f"{name:18} {portion:5.0f}g  K={n['potassium_mg']}"
                )
            return 0

        deleted = clear_q.delete(synchronize_session=False)
        db.commit()
        print(f"  deleted: {deleted}")

        # --- Insert backdated week (explicit logged_at) -------------------
        inserted = 0
        for days_ago, occasion, name, portion in PLAN:
            row = foods[name.lower()]
            nutrients = _nutrients_from_csv(row, portion)
            day = today - timedelta(days=days_ago)
            # Mid-day stamp so .date() is unambiguous in UTC
            logged_at = datetime.combine(day, time(12, 0, 0))

            db.add(
                FoodLog(
                    log_id=str(uuid.uuid4()),
                    patient_id=user_id,  # STRICT: this account only
                    food_name=row["english"],
                    category=row["category"] or None,
                    stage_safe_range=row.get("ckd_stage_safe") or None,
                    portion_grams=portion,
                    meal_occasion=occasion,
                    potassium_mg=nutrients["potassium_mg"],
                    phosphorus_mg=nutrients["phosphorus_mg"],
                    protein_g=nutrients["protein_g"],
                    sodium_mg=nutrients["sodium_mg"],
                    logged_at=logged_at,
                )
            )
            inserted += 1

        db.commit()
        print(f"\n=== INSERTED {inserted} food_log rows (backdated logged_at) ===")

        # --- Verify: group by UTC date for this user_id only --------------
        print("\n=== VERIFY: food_logs by date (this user_id only) ===")
        verify = (
            db.query(FoodLog)
            .filter(
                FoodLog.patient_id == user_id,  # STRICT: this account only
                FoodLog.logged_at >= window_start,
                FoodLog.logged_at <= window_end,
            )
            .order_by(FoodLog.logged_at.asc())
            .all()
        )
        by_day: dict[date, list[FoodLog]] = {}
        for log in verify:
            by_day.setdefault(log.logged_at.date(), []).append(log)

        for day in sorted(by_day):
            logs = by_day[day]
            k = sum(float(l.potassium_mg or 0) for l in logs)
            p = sum(float(l.phosphorus_mg or 0) for l in logs)
            print(f"\n  {day.isoformat()}  ({len(logs)} items, K={k:.0f}mg P={p:.0f}mg)")
            for l in logs:
                print(
                    f"    {l.meal_occasion:10} {l.food_name:18} "
                    f"{l.portion_grams:5.0f}g  "
                    f"K={l.potassium_mg} Ph={l.phosphorus_mg} "
                    f"Pro={l.protein_g} Na={l.sodium_mg}"
                )

        print(
            "\nDone. No other users, model files, or model hashes were modified."
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
