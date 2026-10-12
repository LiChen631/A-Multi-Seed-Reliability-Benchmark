# -*- coding: utf-8 -*-
"""full_eval.py — 基于最新模型完整评估：reward, goal dist, formation err, collisions, 5-seed"""
import os, sys, json
import numpy as np
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
# 期望队形间距（用于 formation error）
EXPECTED_SPACING = 15.0  # meters, 从环境配置


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


def evaluate_full(agents, env, n_episodes=20):
    """评估并收集 reward, goal distance, formation error, collisions."""
    rewards, goal_dists, form_errs, colls = [], [], [], []
    spacing = getattr(env, 'formation_spacing', 15.0)
    for _ in range(n_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        ep_goal, ep_form, ep_coll = [], [], 0
        for _ in range(MAX_EPISODE_STEPS):
            actions = [a.select_action(obs_list[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, r, dones, info = env.step(actions)
            ep_r += sum(r)
            ep_goal.extend(info['dist_to_goal'])
            # Formation error: mean pairwise deviation from expected spacing
            states = env.states  # shape (N, state_dim), first 2 cols = x,y
            pairs = 0
            dev_sum = 0.0
            for i in range(env.n_agents):
                for j in range(i + 1, env.n_agents):
                    d = np.linalg.norm(states[i, 0:2] - states[j, 0:2])
                    dev_sum += abs(d - spacing)
                    pairs += 1
            if pairs > 0:
                ep_form.append(dev_sum / pairs)
            ep_coll += info.get('collisions', 0)
            if dones[0]:
                break
        rewards.append(ep_r)
        goal_dists.append(np.mean(ep_goal) if ep_goal else 0)
        form_errs.append(np.mean(ep_form) if ep_form else 0)
        colls.append(ep_coll)
    return {
        'reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards, ddof=1)),
        'goal_dist': float(np.mean(goal_dists)),
        'form_err': float(np.mean(form_errs)),
        'collisions': float(np.mean(colls)),
    }


def main():
    print("=" * 70)
    print("FULL 5-SEED EVALUATION (all metrics)")
    print("=" * 70)

    out = {}
    for algo in SEED_DIRS:
        seed_results = []
        for sd in SEED_DIRS[algo]:
            try:
                agents, env = load_agents(algo, os.path.join(CKPT_DIR, sd))
                r = evaluate_full(agents, env, n_episodes=20)
                seed_results.append({'seed_dir': sd, **r})
                print(f"  {algo:<12} {sd:<20} reward={r['reward']:8.1f} "
                      f"goal={r['goal_dist']:7.1f} form={r['form_err']:7.1f} coll={r['collisions']:5.2f}")
            except Exception as e:
                print(f"  ERROR {algo} {sd}: {e}")
        out[algo] = seed_results

    # ── Aggregate per algorithm (across 5 seeds) ──
    print("\n--- 5-SEED AGGREGATE ---")
    agg = {}
    for algo in SEED_DIRS:
        if not out[algo]:
            continue
        rew = np.array([s['reward'] for s in out[algo]])
        goal = np.array([s['goal_dist'] for s in out[algo]])
        form = np.array([s['form_err'] for s in out[algo]])
        coll = np.array([s['collisions'] for s in out[algo]])
        agg[algo] = {
            'reward_mean': float(rew.mean()), 'reward_std': float(rew.std(ddof=1)),
            'goal_mean': float(goal.mean()), 'form_mean': float(form.mean()),
            'coll_mean': float(coll.mean()),
            'seed_details': out[algo],
        }
        print(f"{ALGO_LABELS[algo]:<12} reward={rew.mean():8.1f}±{rew.std(ddof=1):6.1f} "
              f"goal={goal.mean():7.1f} form={form.mean():7.1f} coll={coll.mean():5.2f}")

    # ── Table 2 (single seed=42) ──
    print("\n--- TABLE 2 (seed=42 single run, 20 episodes) ---")
    t2 = {}
    for algo in SEED_DIRS:
        if not out[algo]:
            continue
        s42 = out[algo][0]  # first entry = main (seed 42)
        t2[algo] = {'reward': s42['reward'], 'goal_dist': s42['goal_dist'],
                    'form_err': s42['form_err'], 'collisions': s42['collisions']}
        print(f"{ALGO_LABELS[algo]:<12} reward={s42['reward']:8.1f} "
              f"goal={s42['goal_dist']:7.1f} form={s42['form_err']:7.1f} coll={s42['collisions']:5.2f}")

    save = {'table2_seed42': t2, 'aggregate_5seed': agg}
    path = os.path.join(RESULTS_DIR, 'full_eval_all_metrics.json')
    with open(path, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
