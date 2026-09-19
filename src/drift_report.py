"""Generate a data drift report with Evidently (Week 9, Day 1).

Compares the training data (reference) with simulated production
data (current) and saves the result as an interactive HTML report.
"""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# Input files (paths are relative to the repository root)
REFERENCE_PATH = Path("data/processed/X_train.csv")
CURRENT_PATH = Path("data/processed/X_test.csv")

# Output file
REPORT_PATH = Path("reports/drift_report.html")

# Columns to shift on purpose, and how much to shift them.
# The data is standardized, so +1.0 means "+1 standard deviation".
DRIFT_COLUMNS = ["alcohol", "sulphates"]
DRIFT_SHIFT = 1.0


def load_data():
    """Load reference (training) and current (test) data."""
    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)
    return reference, current


def simulate_drift(current):
    """Return a copy of the current data with artificial drift added."""
    drifted = current.copy()
    for column in DRIFT_COLUMNS:
        drifted[column] = drifted[column] + DRIFT_SHIFT
    return drifted


def build_report(reference, current):
    """Compare the two datasets with Evidently's DataDriftPreset."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report


def main():
    reference, current = load_data()
    current = simulate_drift(current)

    report = build_report(reference, current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    print(f"Drift report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
