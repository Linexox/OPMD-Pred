#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106
export PYTHONPATH="$PWD/scripts:/home/zjkj/24334067/OPMD-Pred-SYSU-Binary-20261029/vendor"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export TORCH_HOME="$PWD/cache/torch"
export OMP_NUM_THREADS=2
mkdir -p outputs
/home/zjkj/24334067/OPMD-Pred/.venv/bin/python -u scripts/run_methylation_cell_transfer.py \
  --data data/attempt1 --config config.json --output "outputs/${1:-attempt2}" --device cuda
