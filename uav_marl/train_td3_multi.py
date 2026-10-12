# -*- coding: utf-8 -*-
"""train_td3_multi.py — TD3 基线多种子可续传版
用法：
    python train_td3_multi.py --seeds 123 456 789 1024
    python train_td3_multi.py --seed 123            # 单种子
    python train_td3_multi.py --episodes 1500
每种子独立 checkpoint，Ctrl+C 安全，断点续传。
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils.replay_buffer import ReplayBuffer
from agents.td3 import TD3Agent
from train import build_env

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
DEFAULT_SEEDS = [123, 456, 789, 1024]   # 42 已有
SAVE_EVERY = 100


def ckpt_dir(seed):
    return os.path.join(RESULTS_DIR, 'checkpoints', f'td3_seed{seed}')


def load_ckpt(seed, agents, buffers):
    d = ckpt_dir(seed)
    ts_path = os.path.join(d, 'train_state.pt')
    if not os.path.exists(ts_path):
        return 1, []
    ts = torch.load(ts_path, map_location='cpu', weights_only=False)
    if ts.get('completed', False):
        return None, None
    start = ts.get('episode', 0) + 1
    ep_rewards = ts.get('episode_rewards', [])
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(d, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        for key in ['actor', 'critic', 'critic2', 'target_actor', 'target_critic', 'target_critic2']:
            if hasattr(ag, key) and key in st:
                getattr(ag, key).load_state_dict(st[key])
        for key in ['actor_optimizer', 'critic_optimizer', 'critic_optimizer2']:
            if hasattr(ag, key) and key in st:
                try:
                    getattr(ag, key).load_state_dict(st[key])
                except Exception:
                    pass
    print(f"    [RESUME TD3 seed={seed}] episode {start}", flush=True)
    return start, ep_rewards


def save_ckpt(seed, agents, episode, ep_rewards, total_steps, completed=False):
    d = ckpt_dir(seed)
    os.makedirs(d, exist_ok=True)
    for i, ag in enumerate(agents):
        st = {}
        for key in ['actor', 'critic', 'critic2', 'target_actor', 'target_critic', 'target_critic2']:
            if hasattr(ag, key):
                st[key] = getattr(ag, key).state_dict()
        for key in ['actor_optimizer', 'critic_optimizer', 'critic_optimizer2']:
            if hasattr(ag, key):
                st[key] = getattr(ag, key).state_dict()
        torch.save(st, os.path.join(d, f'agent_{i}.pt'))
    torch.save({'episode': episode, 'episode_rewards': ep_rewards,
                'total_steps': total_steps, 'completed': completed},
               os.path.join(d, 'train_state.pt'))


def train_seed(seed, episodes):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"  === TD3 seed={seed} ===")
    env = build_env()
    agents = [TD3Agent(agent_id=i, obs_dim=env.obs_dim, action_dim=env.action_dim,
                       hidden_dim=HIDDEN_DIM, device=DEVICE) for i in range(env.n_agents)]
    buffers = [ReplayBuffer(REPLAY_SIZE) for _ in range(env.n_agents)]
    total_steps = 0

    start_ep, ep_rewards = load_ckpt(seed, agents, buffers)
    if start_ep is None:
        print(f"    [SKIP TD3 seed={seed}] already completed", flush=True)
        return None
    if start_ep > 1 and ep_rewards:
        total_steps = start_ep * MAX_EPISODE_STEPS // 4
    print(f"    [TRAIN TD3 seed={seed}] from ep {start_ep} to {episodes}", flush=True)
    interrupted = False
    episode = start_ep
    try:
        for episode in range(start_ep, episodes + 1):
            obs_list = env.reset()
            for a in agents:
                a.reset_noise()
            ep_r = np.zeros(env.n_agents)
            for step in range(MAX_EPISODE_STEPS):
                actions = [a.select_action(obs_list[i], add_noise=True) for i, a in enumerate(agents)]
                next_obs, rewards, dones, info = env.step(actions)
                d = np.array([1.0 if dones[0] else 0.0] * env.n_agents, dtype=np.float32)
                for i in range(env.n_agents):
                    buffers[i].push([obs_list[i]], [actions[i]], [rewards[i]],
                                    [next_obs[i]], None, None, [d[i]])
                ep_r += np.array(rewards)
                obs_list = next_obs
                total_steps += 1
                if total_steps >= WARMUP_STEPS:
                    for i in range(env.n_agents):
                        if len(buffers[i]) >= BATCH_SIZE:
                            batch = buffers[i].sample(BATCH_SIZE)
                            obs_b, act_b, rew_b, n_obs_b, _, _, dones_b = batch
                            y = agents[i].compute_target(n_obs_b[0], rew_b[0], dones_b[0])
                            agents[i].update_critic(obs_b[0], act_b[0], y)
                            agents[i].update_actor(obs_b[0])
                            agents[i].soft_update()
                if dones[0]:
                    break
            ep_rewards.append(ep_r.sum())
            if episode % 100 == 0:
                print(f"    [TD3 s{seed}] ep {episode}/{episodes} avg100={np.mean(ep_rewards[-100:]):.1f}",
                      flush=True)
            if episode % SAVE_EVERY == 0:
                save_ckpt(seed, agents, episode, ep_rewards, total_steps)
    except KeyboardInterrupt:
        print(f"\n    [Ctrl+C TD3 seed={seed}] saving at ep {episode-1}...", flush=True)
        if episode > start_ep:
            save_ckpt(seed, agents, episode - 1, ep_rewards, total_steps)
        interrupted = True
    if interrupted:
        return None

    save_ckpt(seed, agents, episodes, ep_rewards, total_steps, completed=True)
    # 评估 50 eps
    eval_rewards = []
    for _ in range(50):
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
    result = {'seed': seed, 'eval_mean': float(np.mean(eval_rewards)),
              'eval_std': float(np.std(eval_rewards, ddof=1)),
              'final100': float(np.mean(ep_rewards[-100:]))}
    print(f"    [DONE TD3 seed={seed}] eval={result['eval_mean']:.1f}+/-{result['eval_std']:.1f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--episodes', type=int, default=1500)
    args = parser.parse_args()
    print("=" * 70)
    print(f"TD3 MULTI-SEED (resumable) | {args.episodes} eps | Ctrl+C safe")
    print("=" * 70)
    results = []
    for seed in args.seeds:
        r = train_seed(seed, args.episodes)
        if r is not None:
            results.append(r)
    if results:
        print("\n--- TD3 MULTI-SEED SUMMARY ---")
        for r in results:
            print(f"  seed={r['seed']}: eval={r['eval_mean']:.1f}+/-{r['eval_std']:.1f}")
        path = os.path.join(RESULTS_DIR, 'td3_multi_seed.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {path}")


if __name__ == '__main__':
    main()
