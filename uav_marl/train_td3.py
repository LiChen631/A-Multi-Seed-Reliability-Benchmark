# -*- coding: utf-8 -*-
"""train_td3.py — TD3 基线训练（每 UAV 独立单智能体 TD3，1500 eps）
评审 §2.2-2：补 TD3 单智能体基线。"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from agents.td3 import TD3Agent
from train import build_env, evaluate

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'td3')
os.makedirs(CKPT_DIR, exist_ok=True)

EPISODES = 1500
SEED = 42


def main():
    print("=" * 70)
    print(f"TD3 BASELINE (independent per-UAV, seed={SEED}, {EPISODES} eps)")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = build_env()
    agents = [TD3Agent(agent_id=i, obs_dim=env.obs_dim, action_dim=env.action_dim,
                       hidden_dim=HIDDEN_DIM, device=DEVICE) for i in range(env.n_agents)]
    buffers = [ReplayBuffer(REPLAY_SIZE) for _ in range(env.n_agents)]
    total_steps = 0
    ep_rewards = []
    eval_rewards = []

    for episode in range(1, EPISODES + 1):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = np.zeros(env.n_agents)
        transitions = [[] for _ in range(env.n_agents)]

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

            # Per-agent independent update
            if total_steps >= WARMUP_STEPS:
                for i in range(env.n_agents):
                    if len(buffers[i]) >= BATCH_SIZE:
                        batch = buffers[i].sample(BATCH_SIZE)
                        obs_b, act_b, rew_b, n_obs_b, _, _, dones_b = batch
                        # ReplayBuffer returns dict-of-arrays keyed by agent id (here always 0)
                        y = agents[i].compute_target(n_obs_b[0], rew_b[0], dones_b[0])
                        agents[i].update_critic(obs_b[0], act_b[0], y)
                        agents[i].update_actor(obs_b[0])
                        agents[i].soft_update()
            if dones[0]:
                break

        ep_rewards.append(ep_r.sum())
        if episode % 100 == 0:
            eval_r = evaluate(agents, env, num_episodes=10)
            eval_rewards.append((episode, eval_r))
            print(f"  [TD3] ep {episode}/{EPISODES} avg100={np.mean(ep_rewards[-100:]):.1f} "
                  f"eval={eval_r:.1f}", flush=True)
        if episode % 500 == 0:
            for i, ag in enumerate(agents):
                torch.save({'actor': ag.actor.state_dict(), 'critic': ag.critic.state_dict(),
                            'critic2': ag.critic2.state_dict(),
                            'target_actor': ag.target_actor.state_dict(),
                            'target_critic': ag.target_critic.state_dict(),
                            'target_critic2': ag.target_critic2.state_dict()},
                           os.path.join(CKPT_DIR, f'agent_{i}.pt'))
            torch.save({'episode': episode, 'episode_rewards': ep_rewards,
                        'eval_rewards': eval_rewards}, os.path.join(CKPT_DIR, 'train_state.pt'))

    # Final save
    for i, ag in enumerate(agents):
        torch.save({'actor': ag.actor.state_dict(), 'critic': ag.critic.state_dict(),
                    'critic2': ag.critic2.state_dict(),
                    'target_actor': ag.target_actor.state_dict(),
                    'target_critic': ag.target_critic.state_dict(),
                    'target_critic2': ag.target_critic2.state_dict()},
                   os.path.join(CKPT_DIR, f'agent_{i}.pt'))
    torch.save({'episode': EPISODES, 'episode_rewards': ep_rewards,
                'eval_rewards': eval_rewards, 'completed': True},
               os.path.join(CKPT_DIR, 'train_state.pt'))

    # Final 200-ep evaluation
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
        'algo': 'td3', 'seed': SEED, 'episodes': EPISODES,
        'final200_mean': float(np.mean(rewards)),
        'final200_std': float(np.std(rewards, ddof=1)),
        'final200_ci95': [float(np.mean(rewards) - 1.96*np.std(rewards,ddof=1)/np.sqrt(200)),
                          float(np.mean(rewards) + 1.96*np.std(rewards,ddof=1)/np.sqrt(200))],
    }
    print(f"  FINAL 200-ep: {final['final200_mean']:.1f} ± {final['final200_std']:.1f} "
          f"CI={final['final200_ci95']}")
    path = os.path.join(RESULTS_DIR, 'td3_results.json')
    with open(path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
