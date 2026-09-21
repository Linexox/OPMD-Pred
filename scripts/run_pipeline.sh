# 远程一键跑整个流水线（阶段A → B → C → D）
# 用法：在服务器上 bash scripts/run_pipeline.sh [TAG前缀]
# 日志落在 logs/ 下，全部 nohup 后台执行，断 SSH 不影响

set -u
PREFIX=${1:-exp}
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs artifacts reports

echo "== 阶段A: 图像嵌入缓存 =="
HF_ENDPOINT=https://hf-mirror.com PYTHONUNBUFFERED=1 uv run python \
  scripts/extract_features.py --device cuda 2>&1 | tee "logs/A_extract.log"
tail -3 "logs/A_extract.log"

echo "== 阶段C: MSE 主线（含阶段B 对比预训练） =="
PYTHONUNBUFFERED=1 uv run python scripts/train_cv.py --tag "${PREFIX}_mse" --head mse --device cuda \
  2>&1 | tee "logs/C_mse.log"

echo "== 消融: CORAL 头 =="
PYTHONUNBUFFERED=1 uv run python scripts/train_cv.py --tag "${PREFIX}_coral" --head coral --device cuda \
  2>&1 | tee "logs/C_coral.log"

echo "== 消融: 去掉对比预训练 =="
PYTHONUNBUFFERED=1 uv run python scripts/train_cv.py --tag "${PREFIX}_nopre" --head mse --skip-pretrain --device cuda \
  2>&1 | tee "logs/C_nopre.log"

echo "== 阶段D: 可解释性（用主线模型） =="
PYTHONUNBUFFERED=1 uv run python scripts/explain.py --tag "${PREFIX}_mse" --device cuda \
  2>&1 | tee "logs/D_explain.log"

echo "== 完成，查看汇总 =="
cat reports/RESULTS.md
