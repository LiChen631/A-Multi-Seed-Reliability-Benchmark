# -*- coding: utf-8 -*-
"""retrain_std3_beta05.py — MUAV-STD3 seed=42 最优配置(β=0.5) 1500-ep 重训（控制变量实验）
评审 §2.1-4：用论文声称的最优配置重训 seed=42，若接近/超过 MATD3 主叙事成立。"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import *
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents, evaluate, save_full_checkpoint

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
OUT_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'muav_std3_beta05')
os.makedirs(OUT_DIR, exist_ok=True)

EPISODES = 1500
BETA = 0.5
K = 5
SEED = 42


def main():
    print("=" * 70)
    print(f"MUAV-STD3 SEED=42 RETRAIN (β={BETA}, K={K}, {EPISODES} eps) — CONTROL EXPERIMENT")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    old_beta = config.BETA_SOFT_MUAV_STD3
    old_k = config.K_SAMPLES_MUAV_STD3
    config.BETA_SOFT_MUAV_STD3 = BETA
    config.K_SAMPLES_MUAV_STD3 = K
    try:
        env = build_env()
        agents = create_agents('muav_std3', env)
        buffer = ReplayBuffer(REPLAY_SIZE)
        total_steps = 0
        ep_rewards = []
        eval_rewards = []
        for episode in range(1, EPISODES + 1):
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
                eval_r = evaluate(agents, env, num_episodes=10)
                eval_rewards.append((episode, eval_r))
                print(f"  [β={BETA}] ep {episode}/{EPISODES} avg100={np.mean(ep_rewards[-100:]):.1f} "
                      f"eval={eval_r:.1f}", flush=True)
            if episode % 500 == 0:
                save_full_checkpoint(agents, buffer, 'muav_std3_beta05', episode, ep_rewards, eval_rewards)

        # 最终评估 200 eps
        print("\n  Final evaluation (200 episodes)...")
        rewards = []
        for _ in range(200):
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
        final = {
            'beta': BETA, 'k': K, 'seed': SEED, 'episodes': EPISODES,
            'final200_mean': float(np.mean(rewards)),
            'final200_std': float(np.std(rewards, ddof=1)),
            'final200_ci95': [float(np.mean(rewards) - 1.96*np.std(rewards,ddof=1)/np.sqrt(200)),
                              float(np.mean(rewards) + 1.96*np.std(rewards,ddof=1)/np.sqrt(200))],
        }
        print(f"  FINAL 200-ep: {final['final200_mean']:.1f} ± {final['final200_std']:.1f} "
              f"CI={final['final200_ci95']}")
        print(f"  对比: MATD3 seed=42 = -468.9 ± 89.8 | MUAV-STD3 β=5.0 = -552.4 ± 36.2")

        save_full_checkpoint(agents, buffer, 'muav_std3_beta05', EPISODES, ep_rewards, eval_rewards)
        path = os.path.join(RESULTS_DIR, 'std3_beta05_retrain.json')
        with open(path, 'w') as f:
            json.dump(final, f, indent=2)
        print(f"Saved: {path}")
    finally:
        config.BETA_SOFT_MUAV_STD3 = old_beta
        config.K_SAMPLES_MUAV_STD3 = old_k


if __name__ == '__main__':
    main()
