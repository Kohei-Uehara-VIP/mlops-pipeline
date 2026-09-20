# src/train.py
# Load data, train a Random Forest model, and log everything with MLflow

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

# Default hyperparameters (used when the script is imported, not run)
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 10


def parse_args():
    """Read hyperparameters from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS
    )
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH)
    return parser.parse_args()


def load_processed_data():
    """Load the processed train/test data."""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_model(
    n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH
):
    """Train a Random Forest model and log the run with MLflow.

    Returns:
        dict: accuracy and f1_score of the trained model
    """
    X_train, X_test, y_train, y_test = load_processed_data()

    params = {
        "n_estimators": n_estimators,   # number of trees in the forest
        "max_depth": max_depth,         # maximum depth of each tree
        "random_state": 42,             # fix randomness for reproducibility
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-quality")

    with mlflow.start_run():

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
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            registered_model_name="WineQualityModel",
        )

        # Save model locally as well
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        print("Parameters:", params)
        print("Metrics:", metrics)
        print("MLflow run complete. Model saved to models/model.pkl")
        print(
            "Model registered as 'WineQualityModel' "
            "in MLflow Model Registry"
        )

    return metrics


def main():
    args = parse_args()
    train_model(n_estimators=args.n_estimators, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
