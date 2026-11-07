#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106
mkdir cell-pretraining/execution-lock
echo "Waiting for complete training input upload"
until test -s cell-pretraining/data/manifest.json && test -s cell-pretraining/data/patient_split.json && test -s cell-pretraining/data/excluded.json; do
  sleep 5
done
echo "Input files present; training entry will verify full tar SHA before starting"
bash cell-pretraining/run.sh
echo "FULL PIPELINE COMPLETED"
