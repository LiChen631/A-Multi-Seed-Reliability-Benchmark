#!/usr/bin/env bash
# 10_multiseed_sweeps.sh —— 同组多 seed、统一预算的扫描与消融（对应审稿意见 #11）
#
# 做法：每个 seed 用独立的 results 根目录（避免原脚本"按配置命名"互相覆盖），
#       跑完后合并成多 seed 汇总表。
# 成本：11 个配置 × 3 seeds = 33 次 1500-episode 训练。
#       单进程约 5 h/次 → 串行约 165 h；**用 8 进程并行约 20–25 h**（见 20_parallel_retrain.sh 的思路）。
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/sweep_multiseed"
mkdir -p "$OUT"
EPISODES=${EPISODES:-1500}
SEEDS=${SEEDS:-"42 123 456"}

run_one() {  # $1=seed  $2=script  $3..=args
  local seed=$1; shift
  local script=$1; shift
  export UAV_FORCE_SEED="$seed"
  export UAV_RESULTS_DIR="$ROOT/results_seed$seed"
  mkdir -p "$UAV_RESULTS_DIR"
  ( cd uav_marl && python "$script" "$@" ) | tee "$OUT/log_${script%.py}_seed${seed}.txt"
  cp -v "$UAV_RESULTS_DIR/${script%.py}_results.json" "$OUT/${script%.py}_seed${seed}.json" 2>/dev/null || true
}

for S in $SEEDS; do
  echo "===== seed $S ====="
  run_one $S beta_sweep_v3.py          --beta 0.1 0.5 1.0 5.0 10.0 --episodes "$EPISODES"
  run_one $S quick_k_sweep_v3.py       --k 5 10 20                  --episodes "$EPISODES"
  run_one $S retrain_mtd3_ablation_v2.py --beta 0.1 0.5 0.9          --episodes "$EPISODES"
done

python "$ROOT/merge/merge_sweeps.py" --dir "$OUT" --out "$OUT/sweep_multiseed_summary.json"
echo "汇总：$OUT/sweep_multiseed_summary.json"
