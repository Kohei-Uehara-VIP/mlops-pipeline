# src/train.py
# Load data, train a Random Forest model, and log everything with MLflow

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os
import argparse

# ── 0. Parse command-line arguments ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=10)
args = parser.parse_args()

# ── 1. Load processed data ──────────────────────────────────────────────────
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# ── 2. Define hyperparameters ────────────────────────────────────────────────
params = {
    "n_estimators": args.n_estimators,   # number of trees in the forest
    "max_depth": args.max_depth,        # maximum depth of each tree
    "random_state": 42     # fix randomness for reproducibility
}

# ── 3. Start MLflow run and log everything ───────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-quality")

with mlflow.start_run() as run:

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Create and log confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log parameters and metrics to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log the trained model as an artifact
    model_info = mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        registered_model_name="WineQualityModel"  # ← Model Registryに登録
    )

    # Save model locally as well
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Parameters:", params)
    print("Metrics:", metrics)
    print("MLflow run complete. Model saved to models/model.pkl")
    print("Model registered as 'WineQualityModel' in MLflow Model Registry")
