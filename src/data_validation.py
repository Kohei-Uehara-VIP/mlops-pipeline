# src/data_validation.py
# Validate the raw Wine Quality dataset using Pandera

import pandas as pd  # Data manipulation library
import pandera.pandas as pa  # Data validation library
from pandera.pandas import Column, DataFrameSchema, Check # Validation components


def get_schema() -> DataFrameSchema:
    """
    Define the expected schema for Wine Quality dataset.
    
    Returns:
        DataFrameSchema object with validation rules
    """
    schema = DataFrameSchema({
        # Each column: name, type, and validation rules
        "fixed acidity": Column(float, Check.greater_than(0)),
        "volatile acidity": Column(float, Check.greater_than(0)),
        "citric acid": Column(float, Check.greater_than_or_equal_to(0)),
        "residual sugar": Column(float, Check.greater_than(0)),
        "chlorides": Column(float, Check.greater_than(0)),
        "free sulfur dioxide": Column(float, Check.greater_than(0)),
        "total sulfur dioxide": Column(float, Check.greater_than(0)),
        "density": Column(float, Check.greater_than(0)),
        "pH": Column(float, Check.in_range(2.0, 5.0)),
        "sulphates": Column(float, Check.greater_than(0)),
        "alcohol": Column(float, Check.in_range(8.0, 15.0)),
        "quality": Column(int, Check.in_range(3, 9)),
    })
    return schema


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataframe against schema.

    Args:
        df: Raw dataframe to validate

    Returns:
        Validated dataframe
    """
    schema = get_schema()

    print("Validating data schema...")
    validated_df = schema.validate(df)
    print("✅ Data validation passed!")

    return validated_df


if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/raw/winequality-red.csv")

    # Run validation
    validated_df = validate_data(df)
    print(f"Validated data shape: {validated_df.shape}")