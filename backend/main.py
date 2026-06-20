"""
GuidaPlate FastAPI entry point.

Serves the GuidaPlate REST API for food lookup, dietary risk prediction,
pattern analysis, and KDOQI-grounded food recommendations for CKD patients
in Rwanda.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.patient_data import router as patient_data_router
from backend.api.auth import router as auth_router
from backend.api.food_lookup import router as food_lookup_router
from backend.api.pattern_analysis import router as pattern_analysis_router
from backend.api.recommendations import router as recommendations_router
from backend.api.risk_prediction import router as risk_prediction_router
from backend.models.lstm_model import warmup_lstm
from backend.models.recommender import get_recommender
from backend.models.xgboost_model import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_recommender()
    get_predictor()
    warmup_lstm()
    yield


app = FastAPI(
    title="GuidaPlate API",
    version="1.0.0",
    description="AI-powered dietary decision-support for CKD patients in Rwanda",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patient_data_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(food_lookup_router, prefix="/api")
app.include_router(risk_prediction_router, prefix="/api")
app.include_router(recommendations_router, prefix="/api")
app.include_router(pattern_analysis_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "system": "GuidaPlate", "version": "1.0.0"}


@app.get("/api/health")
def api_health() -> dict:
    food_count = len(get_recommender().foods)
    return {
        "status": "healthy",
        "models": {"xgboost": "loaded", "lstm": "loaded"},
        "food_database": f"{food_count} foods",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
