# -*- coding: utf-8 -*-
"""success_rate_v2.py — 相对阈值成功率 + episode级 Levene 检验"""
import os, sys, json
import numpy as np
from scipy import stats as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from train import build_env, create_agents

CKPT_DIR = os.path.join(os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')), 'checkpoints')
RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}
SEED_DIRS = {
    'maddpg':   ['maddpg', 'maddpg_seed123', 'maddpg_seed456', 'maddpg_seed789', 'maddpg_seed1024'],
    'matd3':    ['matd3', 'matd3_seed123', 'matd3_seed456', 'matd3_seed789', 'matd3_seed1024'],
    'muav_mtd3':['muav_mtd3', 'muav_mtd3_seed123', 'muav_mtd3_seed456', 'muav_mtd3_seed789', 'muav_mtd3_seed1024'],
    'muav_std3':['muav_std3', 'muav_std3_seed123', 'muav_std3_seed456', 'muav_std3_seed789', 'muav_std3_seed1024'],
}
REL_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def load_agents(algo, ckpt_dir):
    env = build_env()
    agents = create_agents(algo, env)
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ckpt_dir, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents, env


def eval_episode(agents, env):
    """单集评估，返回 (ep_reward, final_goal_dist, init_goal_dist)。"""
    obs_list = env.reset()
    for a in agents:
        a.reset_noise()
    ep_r = 0.0
    final_dist = None
    for step in range(MAX_EPISODE_STEPS):
        actions = [a.select_action(obs_list[i], add_noise=False) for i, a in enumerate(agents)]
        obs_list, r, dones, info = env.step(actions)
        ep_r += sum(r)
        if dones[0] or step == MAX_EPISODE_STEPS - 1:
            states = env.states
            final_dist = float(np.mean([np.linalg.norm(states[i, 0:2] - env.goal_positions[i])
                                        for i in range(env.n_agents)]))
            break
    init_dist = float(np.mean([np.linalg.norm(env.start_positions[i] - env.goal_positions[i])
                               for i in range(env.n_agents)]))
    return ep_r, final_dist, init_dist


def main():
    print("=" * 70)
    print("SUCCESS RATE (relative thresholds) + EPISODE-LEVEL LEVENE")
    print("=" * 70)

    out = {}

    # ── 1. seed=42, 200 eps: 相对阈值成功率 ──
    print("\n--- SEED=42 (200 episodes) ---")
    per_algo_episodes = {}
    for algo in SEED_DIRS:
        agents, env = load_agents(algo, os.path.join(CKPT_DIR, algo))
        rewards, finals = [], []
        for _ in range(200):
            r, fd, _ = eval_episode(agents, env)
            rewards.append(r)
            finals.append(fd)
        per_algo_episodes[algo] = {'rewards': rewards, 'finals': finals}
        print(f"  {ALGO_LABELS[algo]:<12} reward={np.mean(rewards):8.1f}±{np.std(rewards,ddof=1):6.1f}")

    print("\n  Success rate by relative threshold (final_dist < init_dist x t):")
    t2_success = {}
    for t in REL_THRESHOLDS:
        row = []
        for algo in SEED_DIRS:
            finals = per_algo_episodes[algo]['finals']
            sr = np.mean([1.0 if f < 400.0 * t else 0.0 for f in finals])
            row.append(round(float(sr), 3))
        print(f"    t={t}: " + "  ".join(f"{ALGO_LABELS[a][:8]}:{v*100:.0f}%" for a, v in zip(SEED_DIRS, row)))
    out['table2_success_relative'] = {
        algo: {f't{t}': round(float(np.mean([1.0 if f < 400.0*t else 0.0 for f in per_algo_episodes[algo]['finals']])), 3)
               for t in REL_THRESHOLDS} for algo in SEED_DIRS}

    # ── 2. 5-seed 成功率（每 seed 40 eps）t=0.3 ──
    print("\n--- 5-SEED SUCCESS RATE (40 eps/seed, t=0.3) ---")
    five = {}
    for algo in SEED_DIRS:
        seed_srs = []
        seed_rewards = []
        for sd in SEED_DIRS[algo]:
            agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
            srs, revs = [], []
            for _ in range(40):
                r, fd, _ = eval_episode(agents, env)
                revs.append(r)
                srs.append(1.0 if fd < 400.0 * 0.3 else 0.0)
            seed_srs.append(np.mean(srs))
            seed_rewards.append(np.mean(revs))
        five[algo] = {
            'success_mean': float(np.mean(seed_srs)), 'success_std': float(np.std(seed_srs, ddof=1)),
            'reward_mean': float(np.mean(seed_rewards)), 'reward_std': float(np.std(seed_rewards, ddof=1)),
            'seed_success': [round(s, 3) for s in seed_srs],
        }
        print(f"  {ALGO_LABELS[algo]:<12} success={np.mean(seed_srs)*100:5.1f}%±{np.std(seed_srs,ddof=1)*100:4.1f} "
              f"seeds={[f'{s*100:.0f}%' for s in seed_srs]}")
    out['five_seed_success_t03'] = five

    # ── 3. Episode-level Levene (5 seeds x 40 eps = 200 values/group) ──
    print("\n--- EPISODE-LEVEL LEVENE (200 values/group) ---")
    group_vals = {}
    for algo in SEED_DIRS:
        vals = []
        for sd in SEED_DIRS[algo]:
            agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
            for _ in range(40):
                r, _, _ = eval_episode(agents, env)
                vals.append(r)
        group_vals[algo] = np.array(vals)
    levene = {}
    for algo in group_vals:
        if algo == 'matd3':
            continue
        stat, p = sp.levene(group_vals['matd3'], group_vals[algo])
        levene[f'matd3_vs_{algo}'] = {'statistic': float(stat), 'p_value': float(p)}
        # also var ratio
        vr = np.var(group_vals['matd3'], ddof=1) / np.var(group_vals[algo], ddof=1)
        print(f"  MATD3 vs {ALGO_LABELS[algo]:<12} Levene F={stat:6.2f} p={p:.4f} var_ratio={vr:.1f}")
    out['levene_episode_level'] = levene

    path = os.path.join(RESULTS_DIR, 'success_rate_v2.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
