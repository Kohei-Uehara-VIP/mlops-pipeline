# src/api/main.py
# FastAPI application for serving wine quality predictions

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import structlog
import logging

# ── 1. Configure structlog ───────────────────────────────────────────────────
logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()

# ── 2. Load model ────────────────────────────────────────────────────────────
def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
logger.info("model_loaded", status="success")

# ── 3. Define input schema with Pydantic ────────────────────────────────────
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# ── 4. Create FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="Wine Quality Prediction API")

@app.get("/health")
def health():
    logger.info("health_check", status="ok")
    return {"status": "ok"}

@app.post("/predict")
def predict(features: WineFeatures):
    data = pd.DataFrame([features.model_dump()])
    data.columns = [col.replace("_", " ") for col in data.columns]
    prediction = model.predict(data)
    result = int(prediction[0])
    logger.info("prediction_made", prediction=result)
    return {"prediction": result}
