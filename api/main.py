from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import joblib
import time
import os
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="MLOps Cybersecurity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl") if os.path.exists("models/scaler.pkl") else None

REQUEST_COUNT   = Counter("api_requests_total", "Nombre total de requetes", ["endpoint"])
PREDICTION_TIME = Histogram("prediction_duration_seconds", "Duree des predictions")
ATTACK_COUNT    = Counter("attacks_detected_total", "Attaques detectees")

class NetworkTraffic(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    processing_time_ms: float
    model: str

if os.path.exists("api/static"):
    app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def home():
    return {"message": "MLOps Cybersecurity API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model":  type(model).__name__,
        "scaler": "loaded" if scaler else "not found",
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: NetworkTraffic):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start = time.time()

    if len(data.features) != 41:
        raise HTTPException(
            status_code=422,
            detail=f"41 features attendus, {len(data.features)} recus"
        )

    X = np.array(data.features).reshape(1, -1)

    if scaler is not None:
        X = scaler.transform(X)

    with PREDICTION_TIME.time():
        prediction = int(model.predict(X)[0])
        proba = (
            model.predict_proba(X)[0]
            if hasattr(model, "predict_proba")
            else [1 - prediction, float(prediction)]
        )

    if prediction == 1:
        ATTACK_COUNT.inc()

    elapsed = (time.time() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        label="ATTAQUE" if prediction == 1 else "NORMAL",
        confidence=round(float(max(proba)), 4),
        processing_time_ms=round(elapsed, 2),
        model=type(model).__name__,
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
