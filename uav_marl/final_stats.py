# -*- coding: utf-8 -*-
"""final_stats.py — 用 5 个完整 1500-ep 种子计算正式统计并生成论文表格数据"""
import os, sys, json
import numpy as np
from scipy import stats as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from train import build_env, create_agents
import torch

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

# 正式 5 种子：main(=42) + 123, 456, 789, 1024
SEED_DIRS = {
    'maddpg':   ['maddpg', 'maddpg_seed123', 'maddpg_seed456', 'maddpg_seed789', 'maddpg_seed1024'],
    'matd3':    ['matd3', 'matd3_seed123', 'matd3_seed456', 'matd3_seed789', 'matd3_seed1024'],
    'muav_mtd3':['muav_mtd3', 'muav_mtd3_seed123', 'muav_mtd3_seed456', 'muav_mtd3_seed789', 'muav_mtd3_seed1024'],
    'muav_std3':['muav_std3', 'muav_std3_seed123', 'muav_std3_seed456', 'muav_std3_seed789', 'muav_std3_seed1024'],
}
ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}


def evaluate(agents, env, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            actions = [a.select_action(obs_list[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, r, dones, _ = env.step(actions)
            ep_r += sum(r)
            if dones[0]:
                break
        rewards.append(ep_r)
    return rewards


def load_agents(algo, ckpt_dir):
    env = build_env()
    agents = create_agents(algo, env)
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ckpt_dir, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
        if 'target_actor' in st:
            ag.target_actor.load_state_dict(st['target_actor'])
    return agents, env


def main():
    print("=" * 70)
    print("FINAL 5-SEED STATISTICS (1500-ep complete runs only)")
    print("=" * 70)

    results = {}
    for algo in SEED_DIRS:
        means = []
        details = {}
        for sd in SEED_DIRS[algo]:
            try:
                agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
                r = evaluate(agents, env, n_episodes=20)
                m = float(np.mean(r))
                means.append(m)
                details[sd] = {'mean': m, 'std': float(np.std(r, ddof=1)), 'n': len(r)}
            except Exception as e:
                print(f"  ERROR {algo} {sd}: {e}")
        results[algo] = {'seed_means': means, 'details': details}

    # ── Aggregate stats ──
    print("\n--- AGGREGATE (per-seed means, n=5 seeds) ---")
    table = {}
    for algo in SEED_DIRS:
        means = np.array(results[algo]['seed_means'])
        mean, std = np.mean(means), np.std(means, ddof=1)
        sem = std / np.sqrt(len(means))
        ci = (mean - 1.96*sem, mean + 1.96*sem)
        med = np.median(means)
        mn, mx = np.min(means), np.max(means)
        table[algo] = {'mean': float(mean), 'std': float(std), 'sem': float(sem),
                       'ci95_low': float(ci[0]), 'ci95_high': float(ci[1]),
                       'median': float(med), 'min': float(mn), 'max': float(mx),
                       'seed_values': [round(v, 1) for v in results[algo]['seed_means']]}
        print(f"{ALGO_LABELS[algo]:<12} {mean:8.1f} +/- {std:7.1f}  "
              f"[{ci[0]:8.1f}, {ci[1]:8.1f}]  median={med:7.1f}  "
              f"min={mn:8.1f} max={mx:8.1f}")
        print(f"             seeds: {[round(v,1) for v in results[algo]['seed_means']]}")

    # ── Pairwise tests ──
    print("\n--- PAIRWISE COMPARISONS (n=5 seeds) ---")
    pairs = [('matd3', 'muav_std3'), ('matd3', 'muav_mtd3'), ('matd3', 'maddpg'),
             ('muav_std3', 'muav_mtd3'), ('muav_std3', 'maddpg'), ('muav_mtd3', 'maddpg')]
    pairwise = {}
    for a, b in pairs:
        x = np.array(results[a]['seed_means'])
        y = np.array(results[b]['seed_means'])
        t, p = sp.ttest_ind(x, y)
        u, pu = sp.mannwhitneyu(x, y, alternative='two-sided')
        # Cohen's d
        nx, ny = len(x), len(y)
        pool = np.sqrt(((nx-1)*np.var(x,ddof=1) + (ny-1)*np.var(y,ddof=1)) / (nx+ny-2))
        d = (np.mean(x) - np.mean(y)) / pool if pool > 0 else 0
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        pairwise[f'{a}_vs_{b}'] = {
            't_test': {'statistic': float(t), 'p_value': float(p)},
            'mann_whitney': {'statistic': float(u), 'p_value': float(pu)},
            'cohens_d': float(d),
        }
        print(f"{ALGO_LABELS[a]:<12} vs {ALGO_LABELS[b]:<12}  t={t:7.2f} p={p:.4f} {sig}  d={d:6.2f}")

    # ── Save ──
    output = {'aggregate': table, 'pairwise': pairwise,
              'methodology': {'n_seeds': 5, 'episodes': 1500, 'eval_episodes': 20}}
    out_path = os.path.join(RESULTS_DIR, 'final_5seed_stats.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── LaTeX table for paper ──
    print("\n--- LaTeX TABLE 2 (5-seed, mean +/- std) ---")
    order = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
    for algo in order:
        s = table[algo]
        print(f"{ALGO_LABELS[algo]} & ${s['mean']:.1f} \\pm {s['std']:.1f}$ & "
              f"$[{s['ci95_low']:.1f}, {s['ci95_high']:.1f}]$ \\\\")


if __name__ == '__main__':
    main()
