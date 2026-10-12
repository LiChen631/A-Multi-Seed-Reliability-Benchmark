#!/usr/bin/env bash
# 04_timing.sh —— 补测训练步时（论文 §4.4 的 8.2/9.1/12.4/15.7 ms 目前无出处）
# 成本：几分钟
set -e
cd "$(dirname "$0")/.."
cd uav_marl
python - <<'PY'
import os, sys, time
sys.path.insert(0, '.')
import numpy as np, torch
from config import *
from train import build_env, create_agents

for algo in ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']:
    env = build_env(); agents = create_agents(algo, env)
    obs = env.reset()
    # 预热
    for _ in range(5):
        acts = [a.select_action(obs[i], add_noise=True) for i, a in enumerate(agents)]
        obs, r, d, _ = env.step(acts)
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        acts = [a.select_action(obs[i], add_noise=True) for i, a in enumerate(agents)]
        obs, r, d, _ = env.step(acts)
        # 触发一次学习更新（与训练循环一致的调用方式，视 train.py 的接口而定）
        for a in agents:
            if hasattr(a, 'update'):
                try:
                    a.update()
                except TypeError:
                    pass
    dt = (time.perf_counter() - t0) / N
    print('%-10s %.1f ms / env-step (3 agents)' % (algo, dt * 1000))
print('device:', DEVICE)
PY
