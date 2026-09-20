"""Orchestrate the training pipeline with Prefect (Week 9, Day 3).

Wraps the existing preprocessing and training steps as Prefect tasks
so that every run is tracked in the local Prefect dashboard.

Usage (from the repository root):
    python -m src.training_flow
"""

from prefect import flow, task

from src.preprocessing import preprocess_data
from src.train import train_model

RAW_DATA_PATH = "data/raw/winequality-red.csv"
PROCESSED_DIR = "data/processed"


@task(name="Preprocess data")
def preprocess_task():
    """Split the raw data and scale the features."""
    preprocess_data(input_path=RAW_DATA_PATH, output_dir=PROCESSED_DIR)
    return PROCESSED_DIR


@task(name="Train model")
def train_task():
    """Train the Random Forest model and log the run with MLflow."""
    metrics = train_model()
    return metrics


@flow(name="Wine quality training")
def training_flow():
    """Run preprocessing and training in order."""
    preprocess_task()
    metrics = train_task()

    print(f"Flow finished. Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    training_flow()
