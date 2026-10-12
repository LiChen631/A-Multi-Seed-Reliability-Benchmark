# -*- coding: utf-8 -*-
"""retrain_hyperparam_sweep.py — 重训版超参扫描（β_soft 与 K），500-800 eps
评审要求：超参扫描必须是"真实验"（重新训练），不能只用训练好的策略改参数评估。"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents, evaluate

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
SWEEP_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'hyperparam_retrain')
os.makedirs(SWEEP_DIR, exist_ok=True)

# β_soft sweep 与 K sweep（重训配置）
BETA_SWEEP = [0.1, 0.5, 1.0, 5.0, 10.0]
K_SWEEP = [5, 10, 20]
EPISODES = 600  # 预算受限的敏感性验证


def train_with_params(algo, beta_soft, k_samples, episodes=EPISODES, seed=42):
    """以指定超参重训 MUAV-STD3 类算法（500-800 eps）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 临时覆盖 config
    old_beta = config.BETA_SOFT_MUAV_STD3
    old_k = config.K_SAMPLES_MUAV_STD3
    config.BETA_SOFT_MUAV_STD3 = beta_soft
    config.K_SAMPLES_MUAV_STD3 = k_samples

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
                avg = np.mean(ep_rewards[-100:])
                print(f"    [β={beta_soft},K={k_samples}] ep {episode}/{episodes} avg100={avg:.1f}", flush=True)

        # 评估
        eval_r = evaluate(agents, env, num_episodes=20)
        return eval_r, np.mean(ep_rewards[-100:])
    finally:
        config.BETA_SOFT_MUAV_STD3 = old_beta
        config.K_SAMPLES_MUAV_STD3 = old_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', choices=['beta', 'k', 'all'], default='all')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    args = parser.parse_args()

    print("=" * 70)
    print(f"RETRAINED HYPERPARAMETER SWEEP ({args.episodes} episodes, budget-constrained)")
    print("=" * 70)

    results = {}

    if args.which in ('beta', 'all'):
        print("\n--- BETA_SOFT SWEEP (K=5 fixed) ---")
        results['beta_soft'] = {}
        for beta in BETA_SWEEP:
            print(f"  Training β_soft={beta}...")
            eval_r, final100 = train_with_params('muav_std3', beta, 5, args.episodes)
            results['beta_soft'][str(beta)] = {'eval_reward': eval_r, 'final100': final100}
            print(f"    -> eval={eval_r:.1f} final100={final100:.1f}")

    if args.which in ('k', 'all'):
        print("\n--- K SWEEP (β_soft=5.0 fixed, config default) ---")
        results['k_samples'] = {}
        for k in K_SWEEP:
            print(f"  Training K={k}...")
            eval_r, final100 = train_with_params('muav_std3', 5.0, k, args.episodes)
            results['k_samples'][str(k)] = {'eval_reward': eval_r, 'final100': final100}
            print(f"    -> eval={eval_r:.1f} final100={final100:.1f}")

    import json
    path = os.path.join(RESULTS_DIR, 'hyperparam_retrain_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")
    print(f"Total time: {(time.time() - __import__('time').time()) if False else ''}")


if __name__ == '__main__':
    main()
