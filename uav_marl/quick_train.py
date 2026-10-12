"""Fast training script that flushes output immediately."""
import sys, os, json
sys.stdout.reconfigure(line_buffering=True)  # Force line-buffered output

import numpy as np
import torch

from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, save_full_checkpoint, evaluate, plot_training_curve

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
torch.manual_seed(42)
np.random.seed(42)

def quick_train(algo_name, n_episodes=500):
    print(f"\n{'='*50}", flush=True)
    print(f"Training: {algo_name.upper()} | Episodes: {n_episodes}", flush=True)
    print(f"{'='*50}", flush=True)

    env = build_env()
    agents = create_agents(algo_name, env)
    buffer = ReplayBuffer(REPLAY_SIZE)

    ep_rewards, eval_rewards = [], []
    total_steps = 0

    for episode in range(1, n_episodes + 1):
        obs_list = env.reset()
        global_state = env._get_global_state()
        for a in agents:
            a.reset_noise()

        ep_r = np.zeros(N_AGENTS)
        for step in range(MAX_EPISODE_STEPS):
            actions = [agents[i].select_action(obs_list[i], add_noise=True) for i in range(N_AGENTS)]
            next_obs, rewards, dones, info = env.step(actions)
            next_global_state = env._get_global_state()

            d = np.array([1.0 if dones[0] else 0.0] * N_AGENTS, dtype=np.float32)
            buffer.push(obs_list, actions, rewards, next_obs, global_state, next_global_state, d)
            ep_r += np.array(rewards)
            obs_list = next_obs
            global_state = next_global_state
            total_steps += 1

            if total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                # Inline update for speed (avoid function call overhead)
                obs_b, act_b, rew_b, nobs_b, gs_b, ngs_b, done_b = buffer.sample(BATCH_SIZE)
                joint_actions = np.concatenate([act_b[i] for i in range(N_AGENTS)], axis=1)
                rewards_tensor = {i: torch.FloatTensor(rew_b[i]).to(DEVICE) for i in range(N_AGENTS)}
                dones_tensor = {i: torch.FloatTensor(done_b[i]).to(DEVICE) for i in range(N_AGENTS)}
                target_actors = [a.target_actor for a in agents]

                for i, ag in enumerate(agents):
                    if algo_name == 'muav_mtd3':
                        peer_target_critics = [(agents[j].target_critic, agents[j].target_critic2)
                                               for j in range(N_AGENTS) if j != i]
                        yt = ag.compute_target(ngs_b, nobs_b, target_actors, rewards_tensor, dones_tensor,
                                               peer_critics=peer_target_critics)
                    else:
                        yt = ag.compute_target(ngs_b, nobs_b, target_actors, rewards_tensor, dones_tensor)
                    ag.update_critic(gs_b, joint_actions, yt)

                actor_actors = [a.actor for a in agents]
                for ag in agents:
                    ag.update_actor(obs_b, actor_actors, gs_b)

                for ag in agents:
                    ag.soft_update()

            if dones[0]:
                break

        total_r = ep_r.sum()
        ep_rewards.append(total_r)

        if episode % 100 == 0 or episode == 1:
            eval_r = evaluate(agents, env, num_episodes=5)
            eval_rewards.append((episode, eval_r))
            avg_r = np.mean(ep_rewards[-min(100, len(ep_rewards)):])
            print(f"  Ep {episode:4d} | Avg: {avg_r:8.2f} | Eval: {eval_r:8.2f} | Steps: {total_steps}", flush=True)

        if episode % 500 == 0 or episode == n_episodes:
            save_full_checkpoint(agents, buffer, algo_name, episode, ep_rewards, eval_rewards)
            print(f"  [Checkpoint saved]", flush=True)

    # Save results
    plot_training_curve(ep_rewards, eval_rewards, algo_name)
    results = {'algo': algo_name, 'episode_rewards': ep_rewards,
               'eval_rewards': [(int(e), float(r)) for e, r in eval_rewards]}
    with open(os.path.join(RESULTS_DIR, f'{algo_name}_results.json'), 'w') as f:
        json.dump(results, f)
    print(f"  [{algo_name.upper()} complete!]", flush=True)
    return ep_rewards, eval_rewards

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='matd3')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()

    quick_train(args.algo, args.episodes)
