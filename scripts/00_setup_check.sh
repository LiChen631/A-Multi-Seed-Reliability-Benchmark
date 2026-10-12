#!/usr/bin/env bash
# 00_setup_check.sh —— 上机后第一步：环境自检
set -e
cd "$(dirname "$0")/.."
echo "===== 1) 硬件与驱动 ====="
nvidia-smi || echo "[警告] 未检测到 GPU（CPU 也能跑，只是慢）"
echo; echo "===== 2) Python 环境 ====="
python -c "import sys; print('python', sys.version.split()[0])"
python - <<'PY'
mods = ['numpy', 'torch', 'matplotlib', 'scipy']
for m in mods:
    try:
        mod = __import__(m)
        print(f'  {m:12s} OK  {getattr(mod, "__version__", "")}')
    except Exception as e:
        print(f'  {m:12s} MISSING ({e.__name__ if hasattr(e,"__name__") else e})')
import torch
print('  cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())
PY
echo; echo "===== 3) 路径检查（应无 C:\\Users\\Administrator 之类的残留）====="
grep -rn "Users.Administrator\|大文件备份" uav_marl --include=*.py | head -5 || echo "  未发现硬编码路径 OK"
echo; echo "===== 4) 数据与 checkpoint ====="
echo "UAV_RESULTS_DIR = ${UAV_RESULTS_DIR:-<未设置，默认 ../results>}"
ls -1 "${UAV_RESULTS_DIR:-../results}/checkpoints" 2>/dev/null | head -20 || echo "  [提示] 还没放 checkpoints"
echo; echo "===== 5) 冒烟测试（1 个 episode 前向）====="
python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join('uav_marl'))
os.chdir('uav_marl')
from config import *
from train import build_env, create_agents
env = build_env(); agents = create_agents('matd3', env)
obs = env.reset()
a = [ag.select_action(obs[i], add_noise=False) for i, ag in enumerate(agents)]
obs, r, d, _ = env.step(a)
print('  env step OK | obs_dim =', env.obs_dim, '| action_dim =', env.action_dim, '| reward[0] = %.3f' % r[0])
PY
echo; echo "自检完成。"
