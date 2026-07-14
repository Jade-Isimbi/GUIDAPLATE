"""
Pytest fixtures for GuidaPlate backend tests.

Isolates DATABASE_PATH to a temp SQLite file so tests never touch
guidaplate.db. Provides a seeded Session and a lightweight TestClient
(risk + recommendations only — no RAG/LSTM lifespan).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Must run before importing backend.database.db (engine binds at import).
_TEST_DIR = Path(tempfile.mkdtemp(prefix="guidaplate_pytest_"))
os.environ["DATABASE_PATH"] = str(_TEST_DIR / "test.db")


@pytest.fixture()
def db_session():
    """Fresh schema + full food_database.csv seed for each test."""
    from backend.database.db import Base, SessionLocal, engine
    from backend.database.seed_foods import seed_foods
    import backend.models.recommender as recommender_mod

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    seed_foods(db)

    # Reset singletons / caches that can hold stale state across tests.
    recommender_mod._RWANDAN_FOOD_IDS = None
    recommender_mod._recommender = None

    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_session):
    """FastAPI TestClient with get_db overridden to the test session."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.recommendations import router as recommendations_router
    from backend.api.risk_prediction import router as risk_prediction_router
    from backend.database.db import get_db

    app = FastAPI(title="GuidaPlate Test API")
    app.include_router(risk_prediction_router, prefix="/api")
    app.include_router(recommendations_router, prefix="/api")

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db

    with TestClient(app) as test_client:
        yield test_client
