# -*- coding: utf-8 -*-
"""quick_k_sweep_v2.py — K sweep 修复版
修复点：
1. 每个 K 配置使用不同 seed（避免共享 seed 导致相同训练轨迹）
2. 使用生产配置 β=5.0（β=0.5 在 600ep 下发散，会掩盖 K 的影响）
3. 加发散检测：评估 std 异常大或奖励 < -1500 判定为 diverged
4. 验证 K 确实传入采样器（打印 self.k_samples）"""
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
SEEDS = {5: 42, 10: 123, 20: 456}   # 每 K 不同 seed
EPISODES = 1000
BETA = 5.0                          # 生产配置
DIVERGE_THRESHOLD = -1500.0


def check_diverged(rewards, final100):
    """发散检测：最终奖励过低或评估完全确定（std≈0）视为发散"""
    if final100 < DIVERGE_THRESHOLD:
        return True
    if len(rewards) >= 20 and np.std(rewards) < 1e-6:
        return True
    return False


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
        # 验证 K 传入
        assert all(a.k_samples == k for a in agents), f"K not propagated: {[a.k_samples for a in agents]}"
        print(f"    [VERIFY] k_samples={agents[0].k_samples} (K={k})", flush=True)
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
                print(f"    [β={BETA},K={k},s{seed}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}", flush=True)
        # 评估 20 集
        eval_rewards = []
        for _ in range(20):
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
            eval_rewards.append(ep_r)
        eval_r = float(np.mean(eval_rewards))
        eval_std = float(np.std(eval_rewards, ddof=1))
        final100 = float(np.mean(ep_rewards[-100:]))
        diverged = check_diverged(eval_rewards, final100)
        return {'eval_reward': eval_r, 'eval_std': eval_std, 'final100': final100,
                'diverged': diverged, 'beta': BETA, 'k': k, 'seed': seed}
    finally:
        config.BETA_SOFT_MUAV_STD3 = old_beta
        config.K_SAMPLES_MUAV_STD3 = old_k


def main():
    print("=" * 70)
    print(f"K SWEEP v2 (β={BETA}, {EPISODES} eps, per-K seeds, divergence detection)")
    print("=" * 70)
    results = {}
    for k in K_VALUES:
        seed = SEEDS[k]
        print(f"\n  Training K={k} (β={BETA}, seed={seed})...")
        r = train_k(k, EPISODES, seed)
        results[str(k)] = r
        status = "DIVERGED" if r['diverged'] else "OK"
        print(f"    -> eval={r['eval_reward']:.1f}±{r['eval_std']:.1f} final100={r['final100']:.1f} [{status}]")
    print("\n--- K SWEEP SUMMARY (β=5.0, 1000-ep) ---")
    for k, r in results.items():
        print(f"  K={k}: eval={r['eval_reward']:.1f}±{r['eval_std']:.1f} "
              f"{'DIVERGED' if r['diverged'] else 'OK'}")
    path = os.path.join(RESULTS_DIR, 'k_sweep_v2_beta5.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
