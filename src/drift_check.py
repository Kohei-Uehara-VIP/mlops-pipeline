"""Run an Evidently Test Suite as an automated data quality gate.

Week 9, Day 2. Checks simulated production data (current) against the
training data (reference), saves the results as JSON, and raises an
error when the share of drifted columns reaches the threshold. This
mimics how an automated pipeline would stop on drifted data.

Week 9, Day 4 adds is_drift_detected(), which returns True or False
instead of raising, so that a Prefect flow can branch on the result.

Usage (from the repository root):
    python -m src.drift_check              # with simulated drift -> fails
    python -m src.drift_check --no-drift   # without drift -> passes
"""

import argparse
import json
from pathlib import Path

from evidently.test_preset import DataQualityTestPreset
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

from src.drift_report import load_data, simulate_drift

# Output file for the structured JSON results
RESULTS_PATH = Path("reports/drift_check.json")

# The test fails when the share of drifted columns is 0.3 or more.
# With 11 columns: up to 3 drifted columns pass, 4 or more fail.
DRIFT_SHARE_THRESHOLD = 0.3

# Name that Evidently gives to TestShareOfDriftedColumns in the JSON
DRIFT_TEST_NAME = "Share of Drifted Columns"


class DataDriftError(Exception):
    """Raised when data drift exceeds the allowed threshold."""


def run_test_suite(reference, current):
    """Run stability, quality and drift tests and return the suite."""
    suite = TestSuite(
        tests=[
            DataStabilityTestPreset(),
            DataQualityTestPreset(),
            # Added so that Task 4 can check the data drift score
            TestShareOfDriftedColumns(lt=DRIFT_SHARE_THRESHOLD),
        ]
    )
    suite.run(reference_data=reference, current_data=current)
    return suite


def save_results(results_json):
    """Save the JSON results to a file."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(results_json)
    print(f"Test results saved to {RESULTS_PATH}")


def get_drift_score(results_json):
    """Parse the JSON results and return the share of drifted columns."""
    results = json.loads(results_json)

    # Overall summary of all tests (for information only)
    summary = results["summary"]
    print(
        f"Tests: {summary['total_tests']} total, "
        f"{summary['success_tests']} passed, "
        f"{summary['failed_tests']} failed"
    )

    # Find the drift test among all test results
    drift_test = None
    for test in results["tests"]:
        if test["name"] == DRIFT_TEST_NAME:
            drift_test = test
            break
    if drift_test is None:
        raise KeyError(f"'{DRIFT_TEST_NAME}' was not found in the results")

    # Calculate the drift score: share of columns with drift detected
    features = drift_test["parameters"]["features"]
    drifted = [name for name, info in features.items() if info["detected"]]
    drift_score = len(drifted) / len(features)

    print(f"Drifted columns: {drifted}")
    print(
        f"Drift score: {drift_score:.3f} "
        f"(threshold: {DRIFT_SHARE_THRESHOLD})"
    )
    return drift_score


def is_drift_detected(results_json):
    """Return True when the drift score reaches the threshold.

    Used by the Prefect flow to decide whether to retrain.
    """
    return get_drift_score(results_json) >= DRIFT_SHARE_THRESHOLD


def check_drift(results_json):
    """Raise DataDriftError if the drift score is too high."""
    drift_score = get_drift_score(results_json)

    if drift_score >= DRIFT_SHARE_THRESHOLD:
        raise DataDriftError(
            f"Data drift detected: drift score {drift_score:.3f} "
            f">= threshold {DRIFT_SHARE_THRESHOLD}"
        )
    print("No significant data drift. Pipeline can continue.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="use the current data as-is, without simulated drift",
    )
    args = parser.parse_args()

    reference, current = load_data()
    if not args.no_drift:
        current = simulate_drift(current)

    suite = run_test_suite(reference, current)
    results_json = suite.json()

    save_results(results_json)
    check_drift(results_json)


if __name__ == "__main__":
    main()
