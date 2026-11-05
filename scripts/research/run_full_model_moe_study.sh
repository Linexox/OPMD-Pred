#!/usr/bin/env bash
set -euo pipefail
cd /home/zjkj/24334067/OPMD-Pred-SYSU-Binary-20261029
test "$(pwd)" = /home/zjkj/24334067/OPMD-Pred-SYSU-Binary-20261029
mkdir -p outputs/experiments/sysu/full-model-moe-20261104/attempt1 cache/tmp cache/matplotlib
export PYTHONPATH="$PWD/vendor:$PWD/src:$PWD/scripts/research"
export TMPDIR="$PWD/cache/tmp"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
export HF_HOME="$PWD/cache/huggingface"
export TORCH_HOME="$PWD/cache/torch"
export TABPFN_MODEL_CACHE_DIR="$PWD/cache/tabpfn"
export HF_HUB_OFFLINE=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=2
python_bin=/home/zjkj/24334067/OPMD-Pred/.venv/bin/python
"$python_bin" scripts/research/check_full_model_moe.py --output outputs/checks/full-model-moe-20261104/unit-checks.json
"$python_bin" -u scripts/research/run_sysu_full_model_moe.py --scope both --device cuda
"$python_bin" -u scripts/research/evaluate_full_model_moe_inference.py --device cuda
"$python_bin" scripts/research/verify_full_model_moe.py --fusion-device cuda --output outputs/experiments/sysu/full-model-moe-20261104/attempt1/inference-verification.json
