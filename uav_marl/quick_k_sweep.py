# -*- coding: utf-8 -*-
"""quick_k_sweep.py — K sweep at β=0.5（600-ep 重训）
β=0.5 是 600-ep 预算下唯一能收敛的温度，此扫描才能体现 K 的真实影响。"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import *
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents, evaluate

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
K_VALUES = [5, 10, 20]
EPISODES = 600
BETA = 0.5


def train_k(k, episodes=EPISODES, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    old_beta = config.BETA_SOFT_MUAV_STD3
    old_k = config.K_SAMPLES_MUAV_STD3
    config.BETA_SOFT_MUAV_STD3 = BETA
    config.K_SAMPLES_MUAV_STD3 = k
    try:
        env = build_env()
        agents = create_agents('muav_std3', env)
        buffer = ReplayBuffer(REPLAY_SIZE)
        total_steps = 0
        ep_rewards = []
        for episode in range(1, episodes + 1):
            obs_list = env.reset()
            global_state = env._get_global_state()
            for a in agents:
                a.reset_noise()
            ep_r = 0.0
            for step in range(MAX_EPISODE_STEPS):
                actions = [agents[i].select_action(obs_list[i], add_noise=True) for i in range(N_AGENTS)]
                next_obs, rewards, dones, info = env.step(actions)
                next_global_state = env._get_global_state()
                d = np.array([1.0 if dones[0] else 0.0] * N_AGENTS, dtype=np.float32)
                buffer.push(obs_list, actions, rewards, next_obs, global_state, next_global_state, d)
                ep_r += sum(rewards)
                obs_list = next_obs
                global_state = next_global_state
                total_steps += 1
                if total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                    update_agents(agents, buffer, 'muav_std3')
                if dones[0]:
                    break
            ep_rewards.append(ep_r)
            if episode % 100 == 0:
                print(f"    [β={BETA},K={k}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}", flush=True)
        eval_r = evaluate(agents, env, num_episodes=20)
        return float(eval_r), float(np.mean(ep_rewards[-100:]))
    finally:
        config.BETA_SOFT_MUAV_STD3 = old_beta
        config.K_SAMPLES_MUAV_STD3 = old_k


def main():
    print("=" * 70)
    print(f"K SWEEP AT β=0.5 (RETRAINED, {EPISODES} episodes each)")
    print("=" * 70)
    results = {}
    for k in K_VALUES:
        print(f"\n  Training K={k} (β={BETA})...")
        eval_r, final100 = train_k(k)
        results[str(k)] = {'eval_reward': eval_r, 'final100': final100, 'beta': BETA}
        print(f"    -> eval={eval_r:.1f} final100={final100:.1f}")
    print("\n--- K SWEEP SUMMARY (β=0.5) ---")
    for k, r in results.items():
        print(f"  K={k}: eval={r['eval_reward']:.1f} final100={r['final100']:.1f}")
    path = os.path.join(RESULTS_DIR, 'k_sweep_beta05.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
