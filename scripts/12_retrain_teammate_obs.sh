#!/usr/bin/env bash
# 12_retrain_teammate_obs.sh —— 【方案 B｜需重训】队友观测对照实验（仅在方案 A 显示"机间项频繁触发"时才做）
#
# 设计（预先注册，结果无论正负都如实报告）：
#   对照条件：观测 10 维（无队友信息，= 论文主实验）  vs  14 维（加入 2 架队友的相对位置）
#   算法：maddpg / matd3 / muav_mtd3 / muav_std3
#   种子：42 123（非退化种子；如资源允许再加 456）
#   预算：1500 episodes/seed，超参与论文完全一致（不调参）
#   评估：与论文同一套脚本（分离指标 + 饱和诊断 + 机间诊断）
#   假设 H1：加入队友观测后，至少一个种子出现非零"全队到达"（team arrival > 0）
#   零假设 H0：与 10 维条件无可区分差异 ⇒ 退化并非"缺队友信息"所致
#
# 成本：8 个 run × 约 1.7–3 h；16 核可并行 8 个 ⇒ 约 2–4 h 墙钟（≈¥5–10）
set -e
cd "$(dirname "$0")/.."
export UAV_TEAMMATE_OBS=1
export UAV_RESULTS_DIR="${UAV_RESULTS_DIR_TOBS:-$PWD/../results_tobs}"
mkdir -p "$UAV_RESULTS_DIR/checkpoints"
SEEDS="${SEEDS:-42 123}"
NPROC="${NPROC:-4}"        # 同时并行的 run 数（按核数调整，16 核建议 8）
echo "teammate-obs 对照实验：results_dir=$UAV_RESULTS_DIR seeds=$SEEDS 并行=$NPROC"

cd uav_marl
n=0
for S in $SEEDS; do
  for A in maddpg matd3 muav_mtd3 muav_std3; do
    while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 10; done
    mkdir -p "$UAV_RESULTS_DIR/checkpoints/${A}_seed${S}"
    nohup python multi_seed_train.py --algo "$A" --seeds "$S" --n_eps 1500 \
        > "../log_tobs_${A}_seed${S}.txt" 2>&1 &
    echo "  launched $A seed=$S"
    n=$((n+1))
  done
done
wait
echo "训练完成（$n 个 run）。下一步跑评估："

cd ..
# ① 分离指标（个体/全队到达、碰撞/安全违反）
python uav_marl/eval_metrics_split.py --algos maddpg matd3 muav_mtd3 muav_std3 \
      --seeds $SEEDS --n_episodes 50 --threshold 120 --results-dir "$UAV_RESULTS_DIR"
# ② 机间/编队诊断
python uav_marl/eval_interagent_diag.py --algos maddpg matd3 muav_mtd3 muav_std3 \
      --seeds $SEEDS --n_episodes 50 --results-dir "$UAV_RESULTS_DIR"
echo
echo "把 $UAV_RESULTS_DIR 下的 metrics_split.json 与 interagent_diag.json 发回；"
echo "对照基准（10 维条件）= results/metrics_split.json（team_arr 全为 0、indiv_arr 0.31–0.57、MUAV-MTD3 为 0）。"
