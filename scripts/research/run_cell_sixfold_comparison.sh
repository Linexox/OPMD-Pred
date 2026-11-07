#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106
export PYTHONPATH="$PWD/scripts"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache"
export OMP_NUM_THREADS=2
/home/zjkj/24334067/OPMD-Pred/.venv/bin/python -u scripts/run_cell_sixfold_ensemble.py \
  --data data/attempt1 --protocol cell-pretraining/sixfold-app-protocol.json --config config.json \
  --methyl-checkpoint outputs/attempt2/methylation_encoder.pt \
  --features-root cell-pretraining/outputs/sysu-attempt1 \
  --output cell-pretraining/outputs/sixfold-attempt1 --device cuda
