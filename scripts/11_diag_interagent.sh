#!/usr/bin/env bash
# 11_diag_interagent.sh —— 【方案 A｜零重训】检验"编队/机间避碰"这两项奖励到底有没有被触发
#
# 回答审稿意见 #3 的前提：奖励要求保持编队并避免机间碰撞，但观测里没有队友信息。
# 成本：纯评估，4 算法 × 2 种子 × 50 episodes ≈ 10–20 分钟（CPU 即可）
set -e
cd "$(dirname "$0")/.."
export UAV_RESULTS_DIR="${UAV_RESULTS_DIR:-$PWD/../results}"
mkdir -p "$UAV_RESULTS_DIR"
cd uav_marl
python eval_interagent_diag.py \
  --algos maddpg matd3 muav_mtd3 muav_std3 \
  --seeds 42 123 \
  --n_episodes 50 \
  --results-dir "$UAV_RESULTS_DIR" | tee ../log_interagent_diag.txt
cd ..
echo
echo "结果：$UAV_RESULTS_DIR/interagent_diag.json （把它发回即可）"
echo "判读要点："
echo "  * frac_steps_with_interagent_pen  ≈ 0  → 机间处罚几乎未触发 ⇒ '避免机间碰撞'在本任务中几乎不是活跃约束"
echo "  * form_err_mean / ideal(37.71)    远大于 1 ⇒ 编队并未被维持（该奖励项未产生编队行为）"
echo "  * centroid_d_first10 -> last10     明显下降 ⇒ 整体在向目标推进；几乎不变 ⇒ 未推进"
