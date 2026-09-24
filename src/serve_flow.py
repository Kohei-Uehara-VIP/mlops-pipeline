"""Schedule the retraining flow with Prefect (Week 9, Day 5).

Creates a local deployment of the drift-triggered retraining flow and
runs it on a fixed interval. Keep this process running: it waits for
each scheduled run and executes it.

Usage (from the repository root, with the Prefect server running):
    python -m src.serve_flow
"""

from datetime import timedelta

from src.retraining_flow import retraining_flow

# How often the pipeline runs. Ten minutes is short on purpose, so the
# scheduling can be demonstrated quickly.
INTERVAL_MINUTES = 10

# Always feed drifted data, so every scheduled run triggers retraining.
# Set to False to watch the flow skip retraining instead.
ADD_DRIFT = True


def main():
    retraining_flow.serve(
        name="drift-retraining-every-10-min",
        interval=timedelta(minutes=INTERVAL_MINUTES),
        parameters={"add_drift": ADD_DRIFT},
        description=(
            "Checks simulated production data for drift and retrains "
            "the wine quality model when drift is detected."
        ),
        tags=["week9", "monitoring"],
    )


if __name__ == "__main__":
    main()
