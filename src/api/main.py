# src/api/main.py
# FastAPI application for serving wine quality predictions

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

# ── 1. Load model from MLflow Model Registry ────────────────────────────────
def load_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model("models:/WineQualityModel/1")
    return model

model = load_model()

# ── 2. Define input schema with Pydantic ────────────────────────────────────
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

# ── 3. Create FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="Wine Quality Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: WineFeatures):
    data = pd.DataFrame([features.model_dump()])
    # Rename columns: underscores → spaces (to match training data)
    data.columns = [col.replace("_", " ") for col in data.columns]
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}