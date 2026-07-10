# tests/test_pipeline.py
# First quality-gate tests: data validation schema and preprocessing pipeline.
# These run without the trained model artifact, so they pass in any fresh clone.

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.data_validation import validate_data
from src.preprocessing import build_pipeline, preprocess_data


def make_valid_df(n_rows: int = 20) -> pd.DataFrame:
    """Build a small dataframe that satisfies the Pandera schema."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "fixed acidity": rng.uniform(4.0, 16.0, n_rows),
        "volatile acidity": rng.uniform(0.1, 1.5, n_rows),
        "citric acid": rng.uniform(0.0, 1.0, n_rows),
        "residual sugar": rng.uniform(0.5, 15.0, n_rows),
        "chlorides": rng.uniform(0.01, 0.6, n_rows),
        "free sulfur dioxide": rng.uniform(1.0, 70.0, n_rows),
        "total sulfur dioxide": rng.uniform(6.0, 280.0, n_rows),
        "density": rng.uniform(0.99, 1.004, n_rows),
        "pH": rng.uniform(2.8, 4.0, n_rows),
        "sulphates": rng.uniform(0.3, 2.0, n_rows),
        "alcohol": rng.uniform(8.5, 14.5, n_rows),
        "quality": rng.integers(3, 9, n_rows),
    })


def test_validate_data_accepts_valid_rows():
    df = make_valid_df()
    validated = validate_data(df)
    assert len(validated) == len(df)


def test_validate_data_rejects_out_of_range_ph():
    df = make_valid_df()
    df.loc[0, "pH"] = 9.9  # schema allows only 2.0-5.0
    with pytest.raises(SchemaError):
        validate_data(df)


def test_build_pipeline_standardizes_features():
    df = make_valid_df().drop("quality", axis=1)
    pipeline = build_pipeline()
    result = pipeline.fit_transform(df)
    assert result.shape == df.shape
    # StandardScaler contract: mean ~ 0, std ~ 1 for every feature
    assert np.allclose(result.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(result.std(axis=0), 1.0, atol=1e-6)


def test_preprocess_data_writes_four_splits(tmp_path):
    raw = tmp_path / "raw.csv"
    out_dir = tmp_path / "processed"
    make_valid_df(25).to_csv(raw, index=False)

    preprocess_data(input_path=str(raw), output_dir=str(out_dir))

    for name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert (out_dir / name).exists(), f"{name} was not written"
    x_train = pd.read_csv(out_dir / "X_train.csv")
    x_test = pd.read_csv(out_dir / "X_test.csv")
    assert len(x_train) == 20  # 80% of 25
    assert len(x_test) == 5    # 20% of 25
    assert "quality" not in x_train.columns
