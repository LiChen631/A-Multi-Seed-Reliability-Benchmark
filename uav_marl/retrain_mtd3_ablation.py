# -*- coding: utf-8 -*-
"""retrain_mtd3_ablation.py — MUAV-MTD3 β ablation 真实验（重训 β∈{0.1,0.5,0.9}）
评审要求：ablation 必须重新训练，不能只改评估参数。"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import *
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents, evaluate

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ABLATION_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'ablation_retrain')
os.makedirs(ABLATION_DIR, exist_ok=True)

BETA_VALUES = [0.1, 0.5, 0.9]
EPISODES = 600


def train_beta(beta, episodes=EPISODES, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    old_beta = config.BETA_MUAV_MTD3
    config.BETA_MUAV_MTD3 = beta
    try:
        env = build_env()
        agents = create_agents('muav_mtd3', env)
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
                    update_agents(agents, buffer, 'muav_mtd3')
                if dones[0]:
                    break
            ep_rewards.append(ep_r)
            if episode % 100 == 0:
                print(f"    [β={beta}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}", flush=True)
        eval_r = evaluate(agents, env, num_episodes=20)
        return eval_r, float(np.mean(ep_rewards[-100:]))
    finally:
        config.BETA_MUAV_MTD3 = old_beta


def main():
    print("=" * 70)
    print(f"MUAV-MTD3 BETA ABLATION (RETRAINED, {EPISODES} episodes each)")
    print("=" * 70)

    results = {}
    for beta in BETA_VALUES:
        print(f"\n  Training MUAV-MTD3 with β={beta}...")
        eval_r, final100 = train_beta(beta)
        results[str(beta)] = {'eval_reward': eval_r, 'final100': final100}
        print(f"    -> eval={eval_r:.1f} final100={final100:.1f}")

    print("\n--- ABLATION SUMMARY ---")
    for beta, r in results.items():
        print(f"  β={beta}: eval={r['eval_reward']:.1f} final100={r['final100']:.1f}")

    path = os.path.join(RESULTS_DIR, 'ablation_retrain_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
