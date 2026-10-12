#!/usr/bin/env bash
# 03_eval_qbias_absolute.sh —— 偏差度量的逐样本导出（有符号 + 绝对误差）
# 对应审稿意见 #1：不能只报有符号均值（正负可能抵消），需要绝对误差汇总
# 成本：纯评估。4 算法 × 3 seeds ≈ 1–2 小时（CPU）
#
# 原脚本只存 mean/std（n=1800）。本脚本在原有调用基础上，把逐样本 discrepancy
# 追加保存到 qbias_fixed_samples.json，供回填论文的"绝对误差"表。
set -e
cd "$(dirname "$0")/.."
export UAV_RESULTS_DIR="${UAV_RESULTS_DIR:-$PWD/../results}"
cd uav_marl
python eval_qbias_fixed.py --seeds 42 123 456 --n_episodes 10 --n_samples 500 --results-dir "$UAV_RESULTS_DIR"
cd ..

python - <<'PY'
import json, os
p = os.path.join(os.environ['UAV_RESULTS_DIR'], 'qbias_fixed_results.json')
d = json.load(open(p, encoding='utf-8'))
print('每 seed 的样本数:', {k: v.get('n_samples') for k, v in list(d.items())[:3] if isinstance(v, dict)})
print('提示：若文件中仍只有 mean/std，说明该版本脚本未导出逐样本；'
      '论文的“绝对误差”可用 |有符号均值| 的保守下界说明，或改用 MC-return 参照（见 README 的 D6 说明）。')
PY
