"""
GuidaPlate FastAPI entry point.

Serves the GuidaPlate REST API for food lookup, dietary risk prediction,
pattern analysis, and KDOQI-grounded food recommendations for CKD patients
in Rwanda.
"""

from dotenv import load_dotenv

load_dotenv()

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.api.chat_sessions import router as chat_sessions_router
from backend.api.daily_budget import router as daily_budget_router
from backend.api.next_meal import router as next_meal_router
from backend.api.patient_data import router as patient_data_router
from backend.api.auth import router as auth_router
from backend.api.auth_reset import router as auth_reset_router
from backend.api.food_lookup import router as food_lookup_router
from backend.api.pattern_analysis import router as pattern_analysis_router
from backend.api.meal_planner import router as meal_planner_router
from backend.api.recommendations import router as recommendations_router
from backend.api.risk_prediction import router as risk_prediction_router
from backend.api.weekly_suggestions import router as weekly_suggestions_router
from backend.api.weekly_trend import router as weekly_trend_router
from backend.database.db import Base, Food, SessionLocal, engine, get_db
from backend.database.seed_foods import backfill_preparation_method, seed_foods
from backend.models.lstm_model import warmup_lstm
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import get_predictor
from backend.rag.retriever import get_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        n = seed_foods(db)
        backfill_preparation_method(db)
        print(f"Foods table: {n} foods ready")
    finally:
        db.close()

    get_recommender()
    get_predictor()
    warmup_lstm()
    print("Initializing RAG retriever...")
    get_retriever()
    print("RAG retriever ready")
    yield


app = FastAPI(
    title="GuidaPlate API",
    version="1.0.0",
    description="AI-powered dietary decision-support for CKD patients in Rwanda",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,"
    "http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(daily_budget_router, prefix="/api")
app.include_router(next_meal_router, prefix="/api")
app.include_router(patient_data_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(auth_reset_router, prefix="/api")
app.include_router(food_lookup_router, prefix="/api")
app.include_router(risk_prediction_router, prefix="/api")
app.include_router(recommendations_router, prefix="/api")
app.include_router(pattern_analysis_router, prefix="/api")
app.include_router(weekly_trend_router, prefix="/api")
app.include_router(weekly_suggestions_router, prefix="/api")
app.include_router(meal_planner_router, prefix="/api")
app.include_router(chat_sessions_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "system": "GuidaPlate", "version": "1.0.0"}


@app.get("/api/health")
def api_health(db: Session = Depends(get_db)) -> dict:
    model_status = {}
    overall = "healthy"

    # Check XGBoost v3
    try:
        predictor = get_predictor()
        result = predictor.predict(
            potassium=2800,
            phosphorus=650,
            protein_per_kg=0.55,
            sodium=1800,
            ckd_stage="G3b",
        )
        model_status["xgboost_v3"] = {
            "status": "loaded",
            "test_prediction": result["risk_label"],
        }
    except Exception as e:
        model_status["xgboost_v3"] = {
            "status": "failed",
            "error": str(e),
        }
        overall = "degraded"

    # Check LSTM v3
    try:
        from backend.models.lstm_model import get_analyzer

        analyzer = get_analyzer()
        r = analyzer.analyze([
            [2800, 650, 0.55, 1800, 0],
            [3000, 700, 0.60, 2000, 1],
        ])
        model_status["lstm_v3"] = {
            "status": "loaded",
            "test_trend": r["trend"],
            "trend_method": r.get("trend_method"),
        }
    except Exception as e:
        model_status["lstm_v3"] = {
            "status": "failed",
            "error": str(e),
        }
        overall = "degraded"

    # Check Weekly RF
    try:
        import joblib
        import numpy as np

        rf = joblib.load("models/weekly_rf.pkl")
        X = np.array([[0.2, 0.3, 0.5]] * 7).flatten().reshape(1, -1)
        pred = rf.predict(X)[0]
        labels = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        model_status["weekly_rf"] = {
            "status": "loaded",
            "test_prediction": labels[int(pred)],
        }
    except Exception as e:
        model_status["weekly_rf"] = {
            "status": "failed",
            "error": str(e),
        }
        overall = "degraded"

    # Check database
    try:
        from sqlalchemy import text

        db.execute(text("SELECT COUNT(*) FROM foods"))
        model_status["database"] = {"status": "connected"}
    except Exception as e:
        model_status["database"] = {
            "status": "failed",
            "error": str(e),
        }
        overall = "degraded"

    # Check RAG
    try:
        ret = get_retriever()
        _ = ret
        model_status["rag"] = {
            "status": "loaded",
            "chunks": 5874,
        }
    except Exception as e:
        model_status["rag"] = {
            "status": "failed",
            "error": str(e),
        }
        overall = "degraded"

    return {
        "status": overall,
        "version": "1.0.0",
        "models": model_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
