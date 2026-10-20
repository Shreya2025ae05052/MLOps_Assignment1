"""Step 6 - Model serving API (FastAPI).

Endpoints:
  GET  /health   -> {"status": "ok"}
  POST /predict  -> {"prediction": 0 or 1, "confidence": 0.0 - 1.0}
  GET  /metrics  -> Prometheus metrics (request count, latency)

Usage: uvicorn api.main:app --port 8000
"""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Log format: time, level, message
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
model = None


def get_model():
    """Loads the model the first time it is needed."""
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return model


app = FastAPI(title="Heart Disease Prediction API")

# Adds the /metrics endpoint for Prometheus
Instrumentator().instrument(app).expose(app)


class PatientData(BaseModel):
    """The 13 input features. Requests with missing fields are rejected (422)."""
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request: method, path, status code and time taken."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s status=%d time_ms=%.1f",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientData):
    data = pd.DataFrame([patient.model_dump()])
    current_model = get_model()

    prediction = int(current_model.predict(data)[0])
    confidence = float(current_model.predict_proba(data)[0].max())

    logger.info(
        "prediction input=%s prediction=%d confidence=%.4f",
        patient.model_dump(), prediction, confidence,
    )
    return {"prediction": prediction, "confidence": round(confidence, 4)}
