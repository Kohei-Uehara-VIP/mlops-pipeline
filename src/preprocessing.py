# src/preprocessing.py
# Build a reproducible preprocessing pipeline using scikit-learn

import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
from sklearn.pipeline import Pipeline  # Chain multiple steps
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.model_selection import train_test_split  # Split data
import os  # File operations


def build_pipeline() -> Pipeline:
    """
    Build a preprocessing pipeline.

    Returns:
        Scikit-learn Pipeline object
    """
    pipeline = Pipeline([
        # Step 1: Scale all features to mean=0, std=1
        ("scaler", StandardScaler()),
    ])
    return pipeline


def preprocess_data(input_path: str, output_dir: str) -> None:
    """
    Load raw data, preprocess it, and save to processed directory.

    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save processed data
    """
    # Load raw data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Separate features and target
    X = df.drop("quality", axis=1)  # Features (all columns except quality)
    y = df["quality"]               # Target (quality score)

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Build and fit pipeline on training data only
    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train_processed, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_processed, columns=X.columns)

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)

    X_train_df.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test_df.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/winequality-red.csv",
        output_dir="data/processed"
    )