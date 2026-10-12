"""
Train multi-agent DRL algorithms on UAV formation environment.
Usage:
    python train.py --algo matd3          # Train MATD3 baseline
    python train.py --algo matd3 --resume # Resume from last checkpoint
    python train.py --algo all            # Train all sequentially
    python train.py --algo all --resume   # Resume unfinished algorithms
"""
import os, sys, argparse, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer

torch.manual_seed(42)
np.random.seed(42)

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))


def build_env():
    # 补丁（对照组）：UAV_TEAMMATE_OBS=1 时把队友相对位置加入观测（obs 10 -> 14 维）
    use_tobs = os.environ.get('UAV_TEAMMATE_OBS', '0') == '1'
    return UAVFormationEnv(n_agents=N_AGENTS, n_obstacles=N_OBSTACLES, map_size=MAP_SIZE,
        obstacle_radius=OBSTACLE_RADIUS, d_safe=D_SAFE, d_comm=D_COMM,
        v_max=V_MAX, a_max=A_MAX, dt=DT, max_steps=MAX_EPISODE_STEPS,
        use_teammate_obs=use_tobs)


def create_agents(algo_name, env):
    global_state_dim = env.global_state_dim
    base_kwargs = {
        'obs_dim': env.obs_dim, 'global_state_dim': global_state_dim,
        'action_dim': env.action_dim, 'n_agents': env.n_agents,
        'actor_lr': ACTOR_LR, 'critic_lr': CRITIC_LR,
        'gamma': GAMMA, 'tau': TAU, 'hidden_dim': HIDDEN_DIM, 'device': DEVICE,
    }
    agents = []
    for i in range(env.n_agents):
        if algo_name == 'maddpg':
            from agents.maddpg import MADDPGAgent
            agent = MADDPGAgent(agent_id=i, **base_kwargs)
        elif algo_name == 'matd3':
            from agents.matd3 import MATD3Agent
            agent = MATD3Agent(agent_id=i, target_noise=TARGET_NOISE,
                               noise_clip=NOISE_CLIP, policy_delay=POLICY_DELAY, **base_kwargs)
        elif algo_name == 'muav_mtd3':
            from agents.muav_mtd3 import MUAVMTD3Agent
            agent = MUAVMTD3Agent(agent_id=i, target_noise=TARGET_NOISE,
                                  noise_clip=NOISE_CLIP, policy_delay=POLICY_DELAY,
                                  beta_mutual=BETA_MUAV_MTD3, **base_kwargs)
        elif algo_name == 'muav_std3':
            from agents.muav_std3 import MUAVSTD3Agent
            agent = MUAVSTD3Agent(agent_id=i, target_noise=TARGET_NOISE,
                                  noise_clip=NOISE_CLIP, policy_delay=POLICY_DELAY,
                                  beta_soft=BETA_SOFT_MUAV_STD3,
                                  k_samples=K_SAMPLES_MUAV_STD3, **base_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        agents.append(agent)
    return agents


def save_full_checkpoint(agents, buffer, algo_name, episode, ep_rewards, eval_rewards):
    """Save everything needed to resume training."""
    ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints', algo_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save agent models + optimizers
    for i, ag in enumerate(agents):
        state = {
            'actor': ag.actor.state_dict(),
            'critic': ag.critic.state_dict(),
            'actor_optimizer': ag.actor_optimizer.state_dict(),
            'critic_optimizer': ag.critic_optimizer.state_dict(),
            'target_actor': ag.target_actor.state_dict(),
            'target_critic': ag.target_critic.state_dict(),
        }
        if hasattr(ag, 'critic2'):
            state['critic2'] = ag.critic2.state_dict()
            state['target_critic2'] = ag.target_critic2.state_dict()
            state['critic_optimizer2'] = ag.critic_optimizer2.state_dict()
        if hasattr(ag, 'update_count'):
            state['update_count'] = ag.update_count
        torch.save(state, os.path.join(ckpt_dir, f'agent_{i}.pt'))

    # Save training state
    train_state = {
        'episode': episode,
        'episode_rewards': ep_rewards,
        'eval_rewards': [(int(e), float(r)) for e, r in eval_rewards],
        'buffer_data': list(buffer.buffer)[-50000:],  # Keep last 50k for resume
    }
    torch.save(train_state, os.path.join(ckpt_dir, 'train_state.pt'))
    return ckpt_dir


def load_full_checkpoint(agents, algo_name):
    """Load checkpoint and return (start_episode, ep_rewards, eval_rewards, buffer)."""
    ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints', algo_name)
    train_state_path = os.path.join(ckpt_dir, 'train_state.pt')
    if not os.path.exists(train_state_path):
        return 0, [], [], None

    train_state = torch.load(train_state_path, map_location=DEVICE, weights_only=False)
    start_ep = train_state['episode'] + 1
    ep_rewards = train_state['episode_rewards']
    eval_rewards = train_state['eval_rewards']

    # Load agent states
    for i, ag in enumerate(agents):
        state = torch.load(os.path.join(ckpt_dir, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        ag.actor.load_state_dict(state['actor'])
        ag.critic.load_state_dict(state['critic'])
        ag.actor_optimizer.load_state_dict(state['actor_optimizer'])
        ag.critic_optimizer.load_state_dict(state['critic_optimizer'])
        ag.target_actor.load_state_dict(state['target_actor'])
        ag.target_critic.load_state_dict(state['target_critic'])
        if hasattr(ag, 'critic2') and 'critic2' in state:
            ag.critic2.load_state_dict(state['critic2'])
            ag.target_critic2.load_state_dict(state['target_critic2'])
            ag.critic_optimizer2.load_state_dict(state['critic_optimizer2'])
        if hasattr(ag, 'update_count') and 'update_count' in state:
            ag.update_count = state['update_count']

    # Reconstruct buffer
    buffer = ReplayBuffer(REPLAY_SIZE)
    if 'buffer_data' in train_state and train_state['buffer_data']:
        for entry in train_state['buffer_data']:
            buffer.buffer.append(entry)

    return start_ep, ep_rewards, eval_rewards, buffer


def check_if_done(algo_name):
    """Check if this algorithm has completed training."""
    ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints', algo_name)
    train_state_path = os.path.join(ckpt_dir, 'train_state.pt')
    if not os.path.exists(train_state_path):
        return False
    train_state = torch.load(train_state_path, map_location='cpu', weights_only=False)
    return train_state.get('completed', False)


def update_agents(agents, buffer, algo_name):
    obs_b, act_b, rew_b, nobs_b, gs_b, ngs_b, done_b = buffer.sample(BATCH_SIZE)
    joint_actions = np.concatenate([act_b[i] for i in range(len(agents))], axis=1)
    rewards_tensor = {i: torch.FloatTensor(rew_b[i]).to(DEVICE) for i in range(len(agents))}
    dones_tensor = {i: torch.FloatTensor(done_b[i]).to(DEVICE) for i in range(len(agents))}
    target_actors = [a.target_actor for a in agents]

    for i, ag in enumerate(agents):
        if algo_name == 'muav_mtd3':
            peer_target_critics = [(agents[j].target_critic, agents[j].target_critic2)
                                   for j in range(len(agents)) if j != i]
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


def evaluate(agents, env, num_episodes=5):
    total_rewards = []
    for _ in range(num_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            actions = [a.select_action(obs_list[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, rewards, dones, _ = env.step(actions)
            ep_r += sum(rewards)
            if dones[0]:
                break
        total_rewards.append(ep_r)
    return np.mean(total_rewards)


def plot_training_curve(ep_rewards, eval_rewards, algo_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = max(1, len(ep_rewards) // 100)
    if len(ep_rewards) >= window:
        smoothed = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(len(smoothed)) + window, smoothed, alpha=0.8, label='Train (smoothed)')
    if eval_rewards:
        evals_x, evals_y = zip(*eval_rewards)
        ax.plot(evals_x, evals_y, 'ro-', markersize=4, label='Eval')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'{algo_name.upper()} Training Curve')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, f'{algo_name}_training.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def train(algo_name, n_episodes=MAX_EPISODES, resume=False):
    print(f"\n{'='*60}")
    print(f"Training: {algo_name.upper()} | Device: {DEVICE}")
    print(f"Episodes: {n_episodes} | Hidden: {HIDDEN_DIM} | Batch: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    env = build_env()
    agents = create_agents(algo_name, env)

    if resume:
        start_ep, ep_rewards, eval_rewards, saved_buffer = load_full_checkpoint(agents, algo_name)
        if start_ep > 0:
            buffer = saved_buffer if saved_buffer else ReplayBuffer(REPLAY_SIZE)
            total_steps = max(WARMUP_STEPS, len(buffer))  # Skip warmup on resume
            print(f"[Resume] Starting from episode {start_ep} | Buffer: {len(buffer)}")
        else:
            buffer = ReplayBuffer(REPLAY_SIZE)
            start_ep = 1
            ep_rewards, eval_rewards = [], []
            total_steps = 0
            print("[Resume] No checkpoint found, starting from scratch")
    else:
        start_ep = 1
        ep_rewards, eval_rewards = [], []
        buffer = ReplayBuffer(REPLAY_SIZE)
        total_steps = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        for episode in range(start_ep, n_episodes + 1):
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
                    update_agents(agents, buffer, algo_name)

                if dones[0]:
                    break

            total_r = ep_r.sum()
            ep_rewards.append(total_r)

            if episode % EVAL_INTERVAL == 0 or episode == start_ep:
                eval_r = evaluate(agents, env, num_episodes=5)
                eval_rewards.append((episode, eval_r))
                avg_r = np.mean(ep_rewards[-EVAL_INTERVAL:])
                print(f"Ep {episode:5d} | Avg100: {avg_r:8.2f} | Eval: {eval_r:8.2f} | Buffer: {len(buffer)}")

            if episode % 500 == 0 or episode == n_episodes:
                save_full_checkpoint(agents, buffer, algo_name, episode, ep_rewards, eval_rewards)
                print(f"  [Checkpoint] Ep {episode} saved")

    except KeyboardInterrupt:
        print(f"\n[STOPPED] Saving checkpoint at episode {episode-1}...")
        save_full_checkpoint(agents, buffer, algo_name, episode-1, ep_rewards, eval_rewards)
        print(f"  To resume: python train.py --algo {algo_name} --resume")
        return ep_rewards, eval_rewards

    # Mark as completed
    ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints', algo_name)
    ts = torch.load(os.path.join(ckpt_dir, 'train_state.pt'), map_location='cpu', weights_only=False)
    ts['completed'] = True
    torch.save(ts, os.path.join(ckpt_dir, 'train_state.pt'))

    # Save results
    plot_training_curve(ep_rewards, eval_rewards, algo_name)
    results = {'algo': algo_name, 'episode_rewards': ep_rewards,
               'eval_rewards': [(int(e), float(r)) for e, r in eval_rewards]}
    with open(os.path.join(RESULTS_DIR, f'{algo_name}_results.json'), 'w') as f:
        json.dump(results, f)
    print(f"\n{algo_name.upper()} training complete!")
    return ep_rewards, eval_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='matd3',
                       choices=['maddpg', 'matd3', 'muav_mtd3', 'muav_std3', 'all'])
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    ALL_ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']

    if args.algo == 'all':
        for algo in ALL_ALGOS:
            # Skip completed algorithms
            if check_if_done(algo):
                print(f"[Skip] {algo} already completed")
                continue
            try:
                train(algo, args.episodes, resume=args.resume)
                # Only resume the first incomplete one, then run normally
                args.resume = False
            except KeyboardInterrupt:
                print(f"\n[PAUSED] {algo} checkpoint saved. Run again with --resume to continue.")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] {algo}: {e}")
                import traceback; traceback.print_exc()
        print("\n=== ALL ALGORITHMS COMPLETE ===")
    else:
        train(args.algo, args.episodes, resume=args.resume)
