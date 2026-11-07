#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106
export PYTHONPATH="$PWD/scripts:$PWD/../OPMD-Pred-SYSU-Binary-20261029/vendor"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache"
export OMP_NUM_THREADS=2
python_bin=/home/zjkj/24334067/OPMD-Pred/.venv/bin/python
"$python_bin" scripts/prepare_cell_random_protocols.py --bundle data/attempt1/bundle.json \
  --output cell-pretraining/random-protocols --seeds 43 44 45
"$python_bin" -u scripts/run_cell_multiseed.py --data data/attempt1 --config config.json \
  --protocols cell-pretraining/random-protocols --seeds 43 44 45 \
  --features cell-pretraining/outputs/sysu-attempt1/oral_pretrained/cell_features.npz \
  --methyl-checkpoint outputs/attempt2/methylation_encoder.pt \
  --output cell-pretraining/outputs/multiseed-attempt1 --device cuda
