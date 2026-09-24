"""Automated retraining pipeline (Week 9, Day 4).

A conditional Prefect flow: it ingests simulated production data,
checks it for data drift with Evidently, and retrains the model only
when drift is detected.

Usage (from the repository root):
    python -m src.retraining_flow              # drifted data -> retrains
    python -m src.retraining_flow --no-drift   # clean data -> no retraining
"""

import argparse

from prefect import flow, get_run_logger, task

from src.drift_check import is_drift_detected, run_test_suite, save_results
from src.drift_report import load_data, simulate_drift
from src.train import train_model


@task(name="Ingest production data")
def ingest_task(add_drift=True):
    """Load the reference data and the simulated production data."""
    logger = get_run_logger()

    reference, current = load_data()
    if add_drift:
        current = simulate_drift(current)
        logger.info("Ingested production data WITH simulated drift")
    else:
        logger.info("Ingested production data without drift")

    logger.info(
        "Reference rows: %d, current rows: %d", len(reference), len(current)
    )
    return reference, current


@task(name="Check data drift")
def check_drift_task(reference, current):
    """Run the Evidently Test Suite and return True if drift is found."""
    logger = get_run_logger()

    suite = run_test_suite(reference, current)
    results_json = suite.json()
    save_results(results_json)

    drift_detected = is_drift_detected(results_json)
    logger.info("Drift detected: %s", drift_detected)
    return drift_detected


@task(name="Retrain model")
def retrain_task():
    """Retrain the model and return its metrics."""
    logger = get_run_logger()

    logger.info("Drift detected - starting model retraining")
    metrics = train_model()
    logger.info("Retraining finished with metrics: %s", metrics)
    return metrics


@flow(name="Drift-triggered retraining")
def retraining_flow(add_drift=True):
    """Retrain the model only when data drift is detected."""
    logger = get_run_logger()

    reference, current = ingest_task(add_drift=add_drift)
    drift_detected = check_drift_task(reference, current)

    if not drift_detected:
        logger.info("No significant drift - skipping retraining")
        return None

    return retrain_task()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="use the current data as-is, without simulated drift",
    )
    args = parser.parse_args()

    retraining_flow(add_drift=not args.no_drift)


if __name__ == "__main__":
    main()
