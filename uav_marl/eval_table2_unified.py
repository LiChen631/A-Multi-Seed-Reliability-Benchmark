# -*- coding: utf-8 -*-
"""eval_table2_unified.py — Table 2 统一单次 200-episode 评估（生成 table2_unified_200ep.json）

动机（对照审稿意见 M2）：Table 2 的 reward/CI 与 goal/form/collisions 必须来自同一次评估运行，
消除此前"奖励列与指标列来自两次运行"的口径混用。

口径：
- seed=42 单 checkpoint，200 evaluation episodes（add_noise=False）
- reward：episode 累计团队奖励；95% CI = mean ± 1.96 * std / sqrt(200)
- goal_dist：episode 内各 agent 到目标的距离均值（§5.4 "Average minimum distance"）
- form_err：三机两两间距相对期望间距的偏差（spacing=15 m，与 full_eval.py 一致）
- collisions：episode 内障碍碰撞事件计数（env.info['collisions']）

覆盖 4 个 MARL 算法 + 独立 TD3 基线（td3/ 目录为 seed=42）。

用法：python eval_table2_unified.py [--n_episodes 200] [--results-dir PATH]
"""
import os, sys, json, argparse
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import *
from train import build_env, create_agents
from agents.td3 import TD3Agent

ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
SPACING = 15.0  # formation error 期望间距（与 full_eval.py 一致）


def default_results_dir():
    return os.path.join(os.path.dirname(CODE_DIR), 'results')


def load_marl_agents(algo, ckpt_dir):
    env = build_env()
    agents = create_agents(algo, env)
    ck = os.path.join(ckpt_dir, algo)
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ck, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents, env


def load_td3_agents(ckpt_dir):
    env = build_env()
    agents = [TD3Agent(agent_id=i, obs_dim=env.obs_dim, action_dim=env.action_dim,
                       hidden_dim=HIDDEN_DIM, device=DEVICE) for i in range(env.n_agents)]
    ck = os.path.join(ckpt_dir, 'td3')
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ck, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents, env


def evaluate_full(agents, env, n_episodes=200):
    rewards, goal_dists, form_errs, colls = [], [], [], []
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
            states = env.states
            pairs, dev_sum = 0, 0.0
            for i in range(env.n_agents):
                for j in range(i + 1, env.n_agents):
                    d = np.linalg.norm(states[i, 0:2] - states[j, 0:2])
                    dev_sum += abs(d - SPACING)
                    pairs += 1
            if pairs > 0:
                ep_form.append(dev_sum / pairs)
            ep_coll += info.get('collisions', 0)
            if dones[0]:
                break
        rewards.append(ep_r)
        goal_dists.append(np.mean(ep_goal) if ep_goal else 0.0)
        form_errs.append(np.mean(ep_form) if ep_form else 0.0)
        colls.append(ep_coll)

    mean = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=1))
    sem = std / np.sqrt(len(rewards))
    return {
        'reward_mean': mean,
        'reward_std': std,
        'ci95_low': mean - 1.96 * sem,
        'ci95_high': mean + 1.96 * sem,
        'goal_dist': float(np.mean(goal_dists)),
        'form_err': float(np.mean(form_errs)),
        'collisions': float(np.mean(colls)),
        'n_episodes': len(rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=200)
    parser.add_argument('--results-dir', type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir or default_results_dir()
    CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
    OUT_PATH = os.path.join(RESULTS_DIR, 'table2_unified_200ep.json')

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print(f"TABLE 2 UNIFIED EVALUATION (seed=42, {args.n_episodes} episodes)")
    print(f"results_dir={RESULTS_DIR}")
    print("=" * 70)

    out = {}
    for algo in ALGOS:
        try:
            agents, env = load_marl_agents(algo, CKPT_DIR)
            r = evaluate_full(agents, env, args.n_episodes)
            out[algo] = r
            print(f"  {algo:<12} reward={r['reward_mean']:8.1f}±{r['reward_std']:6.1f} "
                  f"goal={r['goal_dist']:7.1f} form={r['form_err']:7.1f} coll={r['collisions']:5.2f}")
        except Exception as e:
            print(f"  ERROR {algo}: {e}")

    try:
        agents, env = load_td3_agents(CKPT_DIR)
        r = evaluate_full(agents, env, args.n_episodes)
        out['td3'] = r
        print(f"  {'td3':<12} reward={r['reward_mean']:8.1f}±{r['reward_std']:6.1f} "
              f"goal={r['goal_dist']:7.1f} form={r['form_err']:7.1f} coll={r['collisions']:5.2f}")
    except Exception as e:
        print(f"  ERROR td3: {e}")

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
