# -*- coding: utf-8 -*-
"""success_rate_and_levene.py — 任务成功率 + Levene 检验 + 大样本评估 (200 eps)"""
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
GOAL_THRESHOLD = 50.0  # 到达目标阈值（米）


def load_agents(algo, ckpt_dir):
    env = build_env()
    agents = create_agents(algo, env)
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ckpt_dir, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents, env


def evaluate_with_metrics(agents, env, n_episodes=200):
    """大样本评估：reward + 任务成功率（最终距目标 < 阈值）+ goal dist"""
    rewards, successes, goal_dists = [], [], []
    for _ in range(n_episodes):
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
                # 最终各 agent 到目标距离
                states = env.states
                final_dist = np.mean([np.linalg.norm(states[i, 0:2] - env.goal_positions[i])
                                      for i in range(env.n_agents)])
                break
        rewards.append(ep_r)
        goal_dists.append(final_dist if final_dist is not None else 999)
        successes.append(1.0 if final_dist is not None and final_dist < GOAL_THRESHOLD else 0.0)
    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards, ddof=1)),
        'ci95_low': float(np.mean(rewards) - 1.96 * np.std(rewards, ddof=1) / np.sqrt(n_episodes)),
        'ci95_high': float(np.mean(rewards) + 1.96 * np.std(rewards, ddof=1) / np.sqrt(n_episodes)),
        'success_rate': float(np.mean(successes)),
        'goal_dist': float(np.mean(goal_dists)),
        'n_episodes': n_episodes,
    }


def main():
    print("=" * 70)
    print("LARGE-SAMPLE EVALUATION (200 eps) + SUCCESS RATE + LEVENE")
    print("=" * 70)

    out = {}
    # 1) seed=42 大样本评估（Table 2 升级）
    print("\n--- SEED=42, 200-episode evaluation ---")
    t2 = {}
    for algo in SEED_DIRS:
        agents, env = load_agents(algo, os.path.join(CKPT_DIR, algo))
        r = evaluate_with_metrics(agents, env, n_episodes=200)
        t2[algo] = r
        print(f"  {ALGO_LABELS[algo]:<12} reward={r['reward_mean']:8.1f}±{r['reward_std']:6.1f} "
              f"CI=[{r['ci95_low']:.1f},{r['ci95_high']:.1f}] "
              f"success={r['success_rate']*100:5.1f}% goal={r['goal_dist']:7.1f}")
    out['table2_seed42_200ep'] = t2

    # 2) 5-seed 任务成功率（每 seed 50 eps 平均 → 每算法 250 eps）
    print("\n--- 5-SEED SUCCESS RATE (50 eps/seed) ---")
    five_seed = {}
    for algo in SEED_DIRS:
        seed_success = []
        seed_rewards = []
        for sd in SEED_DIRS[algo]:
            agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
            r = evaluate_with_metrics(agents, env, n_episodes=50)
            seed_success.append(r['success_rate'])
            seed_rewards.append(r['reward_mean'])
        five_seed[algo] = {
            'success_mean': float(np.mean(seed_success)),
            'success_std': float(np.std(seed_success, ddof=1)),
            'reward_mean': float(np.mean(seed_rewards)),
            'reward_std': float(np.std(seed_rewards, ddof=1)),
            'seed_success': [round(s, 3) for s in seed_success],
        }
        print(f"  {ALGO_LABELS[algo]:<12} success={np.mean(seed_success)*100:5.1f}%±{np.std(seed_success,ddof=1)*100:4.1f} "
              f"seeds={[f'{s*100:.0f}%' for s in seed_success]}")
    out['five_seed_success'] = five_seed

    # 3) Levene 检验（5-seed 均值的方差齐性 vs MATD3）
    print("\n--- LEVENE'S TEST (variance homogeneity vs MATD3) ---")
    matd3_means = [np.mean([evaluate_reward(algo, sd) for sd in SEED_DIRS['matd3']])]  # placeholder
    # 用 five_seed 数据做 Levene：需要每个 seed 的 episode 级 reward，这里用 5 个 seed 均值近似
    # 真正的 Levene 用 5-seed 均值的方差比较（n=5 过小，需说明）
    data = {}
    for algo in SEED_DIRS:
        vals = []
        for sd in SEED_DIRS[algo]:
            agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
            r = evaluate_with_metrics(agents, env, n_episodes=20)
            vals.append(r['reward_mean'])
        data[algo] = np.array(vals)
    levene = {}
    for algo in data:
        if algo == 'matd3':
            continue
        stat, p = sp.levene(data['matd3'], data[algo])
        levene[f'matd3_vs_{algo}'] = {'statistic': float(stat), 'p_value': float(p)}
        print(f"  MATD3 vs {ALGO_LABELS[algo]:<12} Levene F={stat:6.2f} p={p:.4f}")
    out['levene'] = levene

    path = os.path.join(RESULTS_DIR, 'large_sample_eval.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


def evaluate_reward(algo, seed_dir):
    """单次评估辅助函数"""
    agents, env = load_agents(algo, os.path.join(CKPT_DIR, seed_dir))
    r = evaluate_with_metrics(agents, env, n_episodes=20)
    return r['reward_mean']


if __name__ == '__main__':
    main()
