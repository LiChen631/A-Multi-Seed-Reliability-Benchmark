#!/usr/bin/env bash
# 01_fix_seed456_robustness.sh —— 【最要紧】重跑鲁棒性评估，修掉 Table 6 的数据串档
#
# 问题：robustness_multi_seed.json 中 muav_std3.seed456 与 muav_mtd3.seed456 在 12 个条件里
#       9 个逐位相同、其余差 <=0.268（checkpoint 不同）→ 属串档，Table 6 的 MUAV-STD3 列不可用。
# 做法：只删除 muav_std3 的旧键，再重跑；脚本会**自动跳过其他算法已完成的条目**。
# 成本：有旧 JSON 时 ~10–20 分钟；无旧 JSON 时全表重算 ~1–1.5 小时。
set -e
cd "$(dirname "$0")/.."
RD="${UAV_RESULTS_DIR:-$PWD/../results}"
export UAV_RESULTS_DIR="$RD"
echo "RESULTS_DIR = $RD"
mkdir -p "$RD/checkpoints"

# ---- 1) 定位已有的 robustness_multi_seed.json（根目录，或审稿包里的 鲁棒性与成功率/ 子目录）----
if [ ! -f "$RD/robustness_multi_seed.json" ] && [ -f "$RD/鲁棒性与成功率/robustness_multi_seed.json" ]; then
  cp -v "$RD/鲁棒性与成功率/robustness_multi_seed.json" "$RD/robustness_multi_seed.json"
fi

if [ -f "$RD/robustness_multi_seed.json" ]; then
  echo "[1/3] 备份并只删除 MUAV-STD3 的旧结果（其他算法保持不动，脚本会自动跳过）"
  cp -v "$RD/robustness_multi_seed.json" "$RD/robustness_multi_seed.before_rerun.json"
  python - <<'PY'
import json, os
p = os.path.join(os.environ['UAV_RESULTS_DIR'], 'robustness_multi_seed.json')
d = json.load(open(p, encoding='utf-8'))
for s in ['seed42', 'seed123', 'seed456']:
    d.get('muav_std3', {}).pop(s, None)
d.get('muav_std3', {}).pop('aggregate', None)
json.dump(d, open(p, 'w', encoding='utf-8'), indent=2)
print('  已移除 muav_std3 的旧键 → 将被重新计算（其他算法会显示跳过）')
PY
else
  echo "[提示] 未找到已有 robustness_multi_seed.json：将重算全部 4 个算法 × 3 seeds（约 1–1.5 小时）"
fi

# ---- 2) 重跑（缺什么算什么）----
echo "[2/3] 开始评估（Ctrl+C 安全，可重复执行续跑）"
( cd uav_marl && python eval_robustness_multi.py --seeds 42 123 456 --n_episodes 30 --results-dir "$RD" )

# ---- 3) 校验 ----
echo "[3/3] 校验 seed456 是否还存在串档"
python verify/verify_outputs.py --results-dir "$RD" --check seed456_duplication || {
  echo "[FAIL] 仍有近乎相同的条件；请把上面的日志发我排查。"; exit 1; }
echo "完成：请把 $RD/robustness_multi_seed.json 与本次输出发回。"
