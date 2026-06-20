#!/usr/bin/env python3
"""One-time backfill: create patients rows for legacy users missing them."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session

from backend.database.db import Patient, SessionLocal, User


def age_from_dob(dob: str | None) -> int | None:
    if not dob or not str(dob).strip():
        return None
    try:
        born = date.fromisoformat(str(dob).strip())
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def backfill(db: Session) -> list[str]:
    """Create Patient rows for users with no matching patients.patient_id."""
    users = db.query(User).all()
    created_for: list[str] = []

    for user in users:
        existing = db.query(Patient).filter(Patient.patient_id == user.user_id).first()
        if existing:
            continue

        patient = Patient(
            patient_id=user.user_id,
            ckd_stage=user.ckd_stage,
            body_weight_kg=user.weight_kg,
            age=age_from_dob(user.dob),
            sex=user.sex,
        )
        db.add(patient)
        created_for.append(user.email)

    if created_for:
        db.commit()

    return created_for


def main() -> None:
    db = SessionLocal()
    try:
        before_missing = (
            db.query(User)
            .outerjoin(Patient, User.user_id == Patient.patient_id)
            .filter(Patient.patient_id.is_(None))
            .count()
        )
        print(f"Users without patients (before): {before_missing}")

        created = backfill(db)

        after_missing = (
            db.query(User)
            .outerjoin(Patient, User.user_id == Patient.patient_id)
            .filter(Patient.patient_id.is_(None))
            .count()
        )
        user_count = db.query(User).count()
        patient_count = db.query(Patient).count()

        print(f"Patients rows created: {len(created)}")
        for email in created:
            print(f"  - {email}")
        print(f"Users without patients (after): {after_missing}")
        print(f"Final counts — users: {user_count}, patients: {patient_count}")

        if after_missing != 0:
            raise SystemExit("Backfill incomplete: some users still lack patients rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
