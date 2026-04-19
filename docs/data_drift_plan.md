# Data Drift Monitoring Plan

## What is Data Drift?
Data drift occurs when the statistical properties of model input data change
over time, causing model performance to degrade.

## Tool: Evidently AI
Evidently AI is an open-source library for monitoring ML models in production.
It generates reports to detect data drift, target drift, and model performance.

## Implementation Plan

### Step 1: Collect Production Data
- Log incoming prediction requests to a database or file
- Store input features and predictions with timestamps

### Step 2: Generate Drift Reports
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=production_data)
report.save_html("drift_report.html")
```

### Step 3: Schedule Regular Checks
- Run drift detection daily or weekly
- Alert when drift score exceeds threshold (e.g., p-value < 0.05)

### Step 4: Trigger Retraining
- If drift is detected, trigger dvc repro via GitHub Actions
- Register new model version in MLflow Model Registry