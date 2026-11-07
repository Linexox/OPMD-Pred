#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106
export PYTHONPATH="$PWD/scripts"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/cache"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export TORCH_HOME="$PWD/cache/torch"
export OMP_NUM_THREADS=2
python_bin=/home/zjkj/24334067/OPMD-Pred/.venv/bin/python
"$python_bin" -u scripts/train_oral_cytology_encoder.py --data cell-pretraining/data \
  --config cell-pretraining/config.json --cache cache --output cell-pretraining/outputs/pretrain-attempt1 --device cuda
"$python_bin" -u scripts/run_sysu_cell_encoder_transfer.py --data data/attempt1 --config config.json \
  --cell-checkpoint cell-pretraining/outputs/pretrain-attempt1/best.pt \
  --methyl-checkpoint outputs/attempt2/methylation_encoder.pt --cache cache \
  --output cell-pretraining/outputs/sysu-attempt1 --device cuda
"$python_bin" scripts/summarize_cell_encoder_transfer.py --pretrain cell-pretraining/outputs/pretrain-attempt1 \
  --sysu cell-pretraining/outputs/sysu-attempt1 --data-summary cell-pretraining/data/summary.json \
  --report cell-pretraining/final-report.md
