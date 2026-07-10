#!/usr/bin/env bash
# Quality gate for mlops-pipeline.
# Law: if any check fails here, the work is NOT done. No exceptions.
# Run from anywhere: ./scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/3 lint (same rule as CI) ==="
flake8 src/ tests/ --max-line-length=88

echo "=== 2/3 tests ==="
pytest -q

echo "=== 3/3 no artifacts or secrets tracked by git ==="
BAD=$(git ls-files | grep -E '\.(pkl|joblib|h5)$|(^|/)\.env$|ipynb_checkpoints' || true)
if [ -n "$BAD" ]; then
  echo "FAIL: these files must never be committed:"
  echo "$BAD"
  exit 1
fi

echo ""
echo "ALL GREEN - verify.sh passed (this is the only definition of done)"
