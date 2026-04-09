# src/data_ingestion.py
# Script to download and save raw Wine Quality dataset

import pandas as pd  # Data manipulation library
import os  # File/directory operations


def download_data(url: str, save_path: str) -> pd.DataFrame:
    """
    Download dataset from URL and save to local path.

    Args:
        url: URL to download data from
        save_path: Local path to save the raw data

    Returns:
        DataFrame containing the downloaded data
    """
    print(f"Downloading data from {url}...")

    # Download CSV directly from URL using pandas
    df = pd.read_csv(url, sep=";")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save raw data to local file
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    # Wine Quality Dataset URL (UCI Repository)
    URL = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/wine-quality/winequality-red.csv"
    )

    # Save path for raw data
    SAVE_PATH = "data/raw/winequality-red.csv"

    # Run ingestion
    df = download_data(URL, SAVE_PATH)
    print(df.head())