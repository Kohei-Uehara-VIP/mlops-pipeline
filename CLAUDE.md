# CLAUDE.md — laws for this repository

Repo: end-to-end MLOps pipeline for red wine quality prediction.
Stack: Python 3.10 / scikit-learn (Random Forest) / Pandera / DVC / MLflow /
FastAPI / Docker / Google Cloud Run. Conda env: `mlops-pipeline`.

## NEVER (laws; exceptions require asking the owner first)
- Never edit, weaken, or delete a test to make it pass. That is always a fail.
- Never report work as "done" from your own assessment.
  Done = `./scripts/verify.sh` exits 0. Nothing else counts.
- Never exceed 200 changed lines in one commit.
- Never commit data files, model artifacts (*.pkl, *.joblib), `.env`,
  credentials, or notebook checkpoints. Data goes through DVC.
- Never invent a metric, endpoint, config value, or API key.
  If it is not in this repo, stop and ask.
- Never add a new dependency without asking first.
- Never claim results that do not come from an actual run. The model in this
  repo is a Random Forest — reported algorithms and metrics must match the
  trained artifact exactly.

## DONE
- Every task states a machine-checkable done condition before work starts.
- `./scripts/verify.sh` has the final vote. Confidence is not evidence.
- On unexpected deviations: take the conservative option, note it in the
  commit message, continue.

## WORDS
- "done"  = verify.sh green; nothing else
- "small" = under 50 changed lines
- "quick" = under 10 minutes of the owner's time

## CONTEXT (facts, so you never guess them)
- Deploy flow: `docker build` creates the image locally -> `docker push`
  sends the image (not a running container) to Artifact Registry ->
  Cloud Run starts an independent container from that image.
- API: `src/api/main.py` exposes `/health` and `/predict`; it loads
  `models/model.pkl` at import time, so importing it requires the artifact.
- Pipeline stages are defined in `dvc.yaml`; experiments tracked in MLflow.
