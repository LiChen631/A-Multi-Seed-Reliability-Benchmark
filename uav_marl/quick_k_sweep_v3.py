# -*- coding: utf-8 -*-
"""quick_k_sweep_v3.py — K sweep 可续传版
功能：
1. 每个 K 配置保存到独立 checkpoint 目录（agents + train_state）
2. 每 100 集自动保存，Ctrl+C 安全
3. 重新运行自动检测 checkpoint 并从断点续传（支持不同 episode 预算）
4. 发散检测：最终奖励过低标 DIVERGED
用法：
    python quick_k_sweep_v3.py                    # 训练/续传 K=5,10,20
    python quick_k_sweep_v3.py --k 5              # 只跑 K=5
    python quick_k_sweep_v3.py --episodes 1500    # 提高预算（续传）
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import *
from utils.replay_buffer import ReplayBuffer
import train
from train import build_env, create_agents, update_agents

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
SWEEP_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'k_sweep_v3')
DEFAULT_K = [5, 10, 20]
SEEDS = {5: 42, 10: 123, 20: 456}

# --- 重跑包补丁：允许用环境变量强制所有配置使用同一组种子（多 seed 复现）---
_force = os.environ.get('UAV_FORCE_SEED')
if _force:
    SEEDS = {k: int(_force) for k in SEEDS}
BETA = 5.0
DIVERGE_THRESHOLD = -1500.0
SAVE_EVERY = 100


def ckpt_path(k):
    return os.path.join(SWEEP_DIR, f'K{k}')


def load_checkpoint(k, agents, buffer, n_agents):
    """从 checkpoint 恢复。返回 (start_episode, ep_rewards) 或 (1, [])。"""
    d = ckpt_path(k)
    ts_path = os.path.join(d, 'train_state.pt')
    if not os.path.exists(ts_path):
        return 1, []
    try:
        ts = torch.load(ts_path, map_location='cpu', weights_only=False)
    except Exception:
        return 1, []
    if ts.get('completed', False):
        return None, None  # 已完成
    start = ts.get('episode', 0) + 1
    ep_rewards = ts.get('episode_rewards', [])
    total_steps = ts.get('total_steps', 0)
    # 恢复 agent 网络 + optimizer
    for i in range(n_agents):
        st = torch.load(os.path.join(d, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        ag = agents[i]
        for key in ['actor', 'critic', 'target_actor', 'target_critic', 'critic2', 'target_critic2']:
            if hasattr(ag, key) and key in st:
                getattr(ag, key).load_state_dict(st[key])
        for key in ['actor_optimizer', 'critic_optimizer', 'critic_optimizer2']:
            if hasattr(ag, key) and key in st:
                try:
                    getattr(ag, key).load_state_dict(st[key])
                except Exception:
                    pass
    # 恢复 buffer
    if 'buffer_data' in ts and ts['buffer_data']:
        for entry in ts['buffer_data']:
            buffer.buffer.append(entry)
    print(f"    [RESUME K={k}] episode {start} | buffer {len(buffer)} | steps {total_steps}", flush=True)
    return start, ep_rewards


def save_checkpoint(k, agents, buffer, episode, ep_rewards, total_steps, completed=False):
    d = ckpt_path(k)
    os.makedirs(d, exist_ok=True)
    for i, ag in enumerate(agents):
        st = {}
        for key in ['actor', 'critic', 'target_actor', 'target_critic', 'critic2', 'target_critic2']:
            if hasattr(ag, key):
                st[key] = getattr(ag, key).state_dict()
        for key in ['actor_optimizer', 'critic_optimizer', 'critic_optimizer2']:
            if hasattr(ag, key):
                st[key] = getattr(ag, key).state_dict()
        torch.save(st, os.path.join(d, f'agent_{i}.pt'))
    ts = {
        'episode': episode,
        'episode_rewards': ep_rewards,
        'total_steps': total_steps,
        'buffer_data': list(buffer.buffer)[-50000:],
        'completed': completed,
        'beta': BETA, 'k': k,
    }
    torch.save(ts, os.path.join(d, 'train_state.pt'))


def train_k(k, episodes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    old_beta = config.BETA_SOFT_MUAV_STD3
    old_k = config.K_SAMPLES_MUAV_STD3
    config.BETA_SOFT_MUAV_STD3 = BETA
    config.K_SAMPLES_MUAV_STD3 = k
    train.BETA_SOFT_MUAV_STD3 = BETA
    train.K_SAMPLES_MUAV_STD3 = k
    try:
        env = build_env()
        agents = create_agents('muav_std3', env)
        assert all(a.k_samples == k for a in agents), "K not propagated"
        buffer = ReplayBuffer(REPLAY_SIZE)
        total_steps = 0

        start_ep, ep_rewards = load_checkpoint(k, agents, buffer, env.n_agents)
        if start_ep is None:
            print(f"    [SKIP K={k}] already completed", flush=True)
            return None
        if start_ep > 1 and ep_rewards:
            total_steps = start_ep * MAX_EPISODE_STEPS // 4

        print(f"    [TRAIN K={k} beta={BETA} seed={seed}] from ep {start_ep} to {episodes}", flush=True)
        interrupted = False
        episode = start_ep
        try:
            for episode in range(start_ep, episodes + 1):
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
                    print(f"    [K={k}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}",
                          flush=True)
                if episode % SAVE_EVERY == 0:
                    save_checkpoint(k, agents, buffer, episode, ep_rewards, total_steps)
        except KeyboardInterrupt:
            print(f"\n    [Ctrl+C K={k}] saving at ep {episode-1}...", flush=True)
            if episode > start_ep:
                save_checkpoint(k, agents, buffer, episode - 1, ep_rewards, total_steps)
                print(f"    [SAVED] resume with same command", flush=True)
            interrupted = True

        if interrupted:
            return None

        # 完成：保存 + 评估
        save_checkpoint(k, agents, buffer, episodes, ep_rewards, total_steps, completed=True)
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
        diverged = final100 < DIVERGE_THRESHOLD
        result = {'eval_reward': eval_r, 'eval_std': eval_std, 'final100': final100,
                  'diverged': diverged, 'beta': BETA, 'k': k, 'seed': seed}
        print(f"    [DONE K={k}] eval={eval_r:.1f}+/-{eval_std:.1f} "
              f"{'DIVERGED' if diverged else 'OK'}", flush=True)
        return result
    finally:
        config.BETA_SOFT_MUAV_STD3 = old_beta
        config.K_SAMPLES_MUAV_STD3 = old_k
        train.BETA_SOFT_MUAV_STD3 = old_beta
        train.K_SAMPLES_MUAV_STD3 = old_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, nargs='+', default=DEFAULT_K)
    parser.add_argument('--episodes', type=int, default=1000)
    args = parser.parse_args()

    print("=" * 70)
    print(f"K SWEEP v3 (resumable) | beta={BETA} | {args.episodes} eps | Ctrl+C safe")
    print("=" * 70)

    results = {}
    for k in args.k:
        print(f"\n  === K={k} (seed={SEEDS[k]}) ===")
        r = train_k(k, args.episodes, SEEDS[k])
        if r is not None:
            results[str(k)] = r

    if results:
        print("\n--- K SWEEP SUMMARY ---")
        for k, r in results.items():
            print(f"  K={k}: eval={r['eval_reward']:.1f}+/-{r['eval_std']:.1f} "
                  f"{'DIVERGED' if r['diverged'] else 'OK'}")
        path = os.path.join(RESULTS_DIR, 'k_sweep_v3_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
