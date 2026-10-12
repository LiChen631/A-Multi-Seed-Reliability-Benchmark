#!/usr/bin/env bash
# 02_eval_metrics_split.sh —— 分离统计真实碰撞 / 安全距离违反 / 惩罚触发 + 个体到达 / 无碰撞成功
# 对应审稿意见 #2（几何与碰撞/安全违反分开统计）与 #4（个体到达 / 全队到达 / 无碰撞成功）
# 成本：纯评估。4 算法 × 3 seeds × 50 episodes ≈ 1–2 小时（CPU）
set -e
cd "$(dirname "$0")/.."
export UAV_RESULTS_DIR="${UAV_RESULTS_DIR:-$PWD/../results}"
cd uav_marl
python eval_metrics_split.py \
  --algos maddpg matd3 muav_mtd3 muav_std3 \
  --seeds 42 123 456 \
  --n_episodes 50 \
  --threshold 120 \
  --results-dir "$UAV_RESULTS_DIR"
cd ..
echo "结果：$UAV_RESULTS_DIR/metrics_split.json"
