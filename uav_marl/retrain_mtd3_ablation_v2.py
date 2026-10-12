# -*- coding: utf-8 -*-
"""retrain_mtd3_ablation_v2.py — MUAV-MTD3 β ablation 修复版
修复：用 train.BETA_MUAV_MTD3 传参（config 修改不生效的 bug）
可续传：每 β 独立 checkpoint，Ctrl+C 安全，支持断点续传
用法：
    python retrain_mtd3_ablation_v2.py                # β=0.1,0.5,0.9
    python retrain_mtd3_ablation_v2.py --beta 0.3     # 指定
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import train
from config import *
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, update_agents

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ABL_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'mtd3_ablation_v2')
DEFAULT_BETAS = [0.1, 0.5, 0.9]
SEEDS = {0.1: 42, 0.5: 123, 0.9: 456}

# --- 重跑包补丁：允许用环境变量强制所有配置使用同一组种子（多 seed 复现）---
_force = os.environ.get('UAV_FORCE_SEED')
if _force:
    SEEDS = {k: int(_force) for k in SEEDS}
EPISODES = 800
DIVERGE_THRESHOLD = -1500.0
SAVE_EVERY = 100


def ckpt_path(beta):
    return os.path.join(ABL_DIR, f'beta{beta}')


def load_checkpoint(beta, agents, buffer, n_agents):
    d = ckpt_path(beta)
    ts_path = os.path.join(d, 'train_state.pt')
    if not os.path.exists(ts_path):
        return 1, []
    try:
        ts = torch.load(ts_path, map_location='cpu', weights_only=False)
    except Exception:
        return 1, []
    if ts.get('completed', False):
        return None, None
    start = ts.get('episode', 0) + 1
    ep_rewards = ts.get('episode_rewards', [])
    total_steps = ts.get('total_steps', 0)
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
    if 'buffer_data' in ts and ts['buffer_data']:
        for entry in ts['buffer_data']:
            buffer.buffer.append(entry)
    print(f"    [RESUME beta={beta}] episode {start} | buffer {len(buffer)}", flush=True)
    return start, ep_rewards


def save_checkpoint(beta, agents, buffer, episode, ep_rewards, total_steps, completed=False):
    d = ckpt_path(beta)
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
    ts = {'episode': episode, 'episode_rewards': ep_rewards, 'total_steps': total_steps,
          'buffer_data': list(buffer.buffer)[-50000:], 'completed': completed, 'beta': beta}
    torch.save(ts, os.path.join(d, 'train_state.pt'))


def train_beta(beta, episodes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    old = config.BETA_MUAV_MTD3
    config.BETA_MUAV_MTD3 = beta
    train.BETA_MUAV_MTD3 = beta   # 关键修复：改 train 模块副本
    try:
        env = build_env()
        agents = create_agents('muav_mtd3', env)
        assert all(a.beta_mutual == beta for a in agents), f"beta not propagated: {agents[0].beta_mutual}"
        print(f"    [VERIFY] beta_mutual={agents[0].beta_mutual} (β={beta})", flush=True)
        buffer = ReplayBuffer(REPLAY_SIZE)
        total_steps = 0
        start_ep, ep_rewards = load_checkpoint(beta, agents, buffer, env.n_agents)
        if start_ep is None:
            print(f"    [SKIP beta={beta}] already completed", flush=True)
            return None
        if start_ep > 1 and ep_rewards:
            total_steps = start_ep * MAX_EPISODE_STEPS // 4
        print(f"    [TRAIN beta={beta} seed={seed}] from ep {start_ep} to {episodes}", flush=True)
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
                        update_agents(agents, buffer, 'muav_mtd3')
                    if dones[0]:
                        break
                ep_rewards.append(ep_r)
                if episode % 100 == 0:
                    print(f"    [beta={beta}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}",
                          flush=True)
                if episode % SAVE_EVERY == 0:
                    save_checkpoint(beta, agents, buffer, episode, ep_rewards, total_steps)
        except KeyboardInterrupt:
            print(f"\n    [Ctrl+C beta={beta}] saving at ep {episode-1}...", flush=True)
            if episode > start_ep:
                save_checkpoint(beta, agents, buffer, episode - 1, ep_rewards, total_steps)
            interrupted = True
        if interrupted:
            return None
        save_checkpoint(beta, agents, buffer, episodes, ep_rewards, total_steps, completed=True)
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
                  'diverged': diverged, 'beta': beta, 'seed': seed}
        print(f"    [DONE beta={beta}] eval={eval_r:.1f}+/-{eval_std:.1f} "
              f"{'DIVERGED' if diverged else 'OK'}", flush=True)
        return result
    finally:
        config.BETA_MUAV_MTD3 = old
        train.BETA_MUAV_MTD3 = old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, nargs='+', default=DEFAULT_BETAS)
    parser.add_argument('--episodes', type=int, default=EPISODES)
    args = parser.parse_args()
    print("=" * 70)
    print(f"MUAV-MTD3 BETA ABLATION v2 (resumable) | {args.episodes} eps | Ctrl+C safe")
    print("=" * 70)
    results = {}
    for beta in args.beta:
        print(f"\n  === beta={beta} (seed={SEEDS[beta]}) ===")
        r = train_beta(beta, args.episodes, SEEDS[beta])
        if r is not None:
            results[str(beta)] = r
    if results:
        print("\n--- ABLATION SUMMARY ---")
        for beta, r in results.items():
            print(f"  beta={beta}: eval={r['eval_reward']:.1f}+/-{r['eval_std']:.1f} "
                  f"{'DIVERGED' if r['diverged'] else 'OK'}")
        path = os.path.join(RESULTS_DIR, 'mtd3_ablation_v2_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
