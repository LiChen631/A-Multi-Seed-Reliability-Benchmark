"""
Multi-seed training launcher for all 4 algorithms.
Trains each algorithm with 5 independent seeds for 1500 episodes.
Run overnight on CPU.

Usage:
    python multi_seed_train.py                    # Train all incomplete seeds
    python multi_seed_train.py --algo matd3       # Train only MATD3
    python multi_seed_train.py --seeds 42 123 456 # Custom seeds
    python multi_seed_train.py --n_eps 800        # Fewer episodes
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, save_full_checkpoint, update_agents, evaluate, plot_training_curve

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))

ALL_ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def get_seed_dir(algo, seed):
    """Get checkpoint directory for a specific seed."""
    if seed == 42:
        return os.path.join(RESULTS_DIR, 'checkpoints', algo)
    return os.path.join(RESULTS_DIR, 'checkpoints', f'{algo}_seed{seed}')


def train_one_seed(algo, seed, n_episodes=1500, resume=True):
    """Train one algorithm with one seed. Ctrl+C safe — auto-saves and resume works."""
    print(f"\n{'='*60}")
    print(f"Training: {algo.upper()} | Seed: {seed} | Episodes: {n_episodes}")
    print(f"Device: {DEVICE}  |  Ctrl+C to stop safely")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = build_env()
    agents = create_agents(algo, env)

    seed_str = get_seed_dir(algo, seed)
    start_ep = 1
    ep_rewards, eval_rewards = [], []
    buffer = ReplayBuffer(REPLAY_SIZE)
    total_steps = 0

    # ── Resume from checkpoint ──
    if resume:
        train_state_path = os.path.join(seed_str, 'train_state.pt')
        if os.path.exists(train_state_path):
            ts = torch.load(train_state_path, map_location='cpu', weights_only=False)
            if ts.get('completed', False):
                print(f"  [SKIP] Already completed.")
                return
            start_ep = ts.get('episode', 0) + 1
            # Compat: both 'episode_rewards' (new) and 'rewards' (old seed format)
            ep_rewards = ts.get('episode_rewards', ts.get('rewards', []))
            eval_rewards = ts.get('eval_rewards', [])
            total_steps = ts.get('total_steps', max(WARMUP_STEPS, (start_ep-1) * MAX_EPISODE_STEPS // 4))

            # Restore buffer if saved
            if 'buffer_data' in ts and ts['buffer_data']:
                for entry in ts['buffer_data']:
                    buffer.buffer.append(entry)

            print(f"  [RESUME] Episode {start_ep} | Buffer: {len(buffer)} | Steps: {total_steps}")

            # Load agent networks + optimizers
            for i, ag in enumerate(agents):
                st = torch.load(os.path.join(seed_str, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
                for key in ['actor', 'critic', 'target_actor', 'target_critic']:
                    if key in st:
                        getattr(ag, key).load_state_dict(st[key])
                # Optimizers (optional — can resume without them from older checkpoints)
                for key in ['actor_optimizer', 'critic_optimizer']:
                    if key in st:
                        try:
                            getattr(ag, key).load_state_dict(st[key])
                        except Exception:
                            pass  # older checkpoints may not have saved optimizer state
                # Twin critic (MATD3 family only)
                for key in ['critic2', 'target_critic2', 'critic_optimizer2']:
                    if hasattr(ag, key) and key in st:
                        try:
                            getattr(ag, key).load_state_dict(st[key])
                        except Exception:
                            pass
                if hasattr(ag, 'update_count') and 'update_count' in st:
                    ag.update_count = st['update_count']

    # ── Training loop ──
    interrupted = False
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
                    update_agents(agents, buffer, algo)
                if dones[0]:
                    break
            total_r = ep_r.sum()
            ep_rewards.append(total_r)
            if episode % 100 == 0 or episode == 1:
                eval_r = evaluate(agents, env, num_episodes=5)
                eval_rewards.append((episode, eval_r))
                avg = np.mean(ep_rewards[-min(100, len(ep_rewards)):])
                print(f"  [{algo} s{seed}] Ep {episode:5d} | Avg100: {avg:8.2f} | Eval: {eval_r:8.2f}", flush=True)

            # Save every 100 episodes (so you lose ≤99 episodes on Ctrl+C)
            if episode % 100 == 0:
                os.makedirs(seed_str, exist_ok=True)
                _save_checkpoint(seed_str, agents, buffer, episode, ep_rewards, eval_rewards, total_steps)

    except KeyboardInterrupt:
        print(f"\n  [Ctrl+C] Interrupted at episode {episode-1}. Saving checkpoint...", flush=True)
        if episode > start_ep:
            os.makedirs(seed_str, exist_ok=True)
            _save_checkpoint(seed_str, agents, buffer, episode-1, ep_rewards, eval_rewards, total_steps)
            print(f"  [SAVED] To resume: python multi_seed_train.py --algo {algo} --seeds {seed}", flush=True)
        interrupted = True

    if interrupted:
        return  # Don't mark completed

    # ── Final save + mark completed ──
    os.makedirs(seed_str, exist_ok=True)
    _save_checkpoint(seed_str, agents, buffer, n_episodes, ep_rewards, eval_rewards, total_steps)
    ts_path = os.path.join(seed_str, 'train_state.pt')
    ts = torch.load(ts_path, map_location='cpu', weights_only=False)
    ts['completed'] = True
    torch.save(ts, ts_path)
    # Also save training curve
    try:
        plot_training_curve(ep_rewards, eval_rewards, os.path.basename(seed_str))
    except Exception:
        pass
    print(f"  [{algo} s{seed}] COMPLETE! ({n_episodes} episodes)", flush=True)


def _save_checkpoint(ckpt_dir, agents, buffer, episode, ep_rewards, eval_rewards, total_steps):
    """Save everything needed to resume: networks, optimizers, buffer, rewards history."""
    for i, ag in enumerate(agents):
        state = {key: getattr(ag, key).state_dict()
                 for key in ['actor', 'critic', 'target_actor', 'target_critic']
                 if hasattr(ag, key)}
        for key in ['actor_optimizer', 'critic_optimizer']:
            if hasattr(ag, key):
                state[key] = getattr(ag, key).state_dict()
        if hasattr(ag, 'critic2'):
            state['critic2'] = ag.critic2.state_dict()
            state['target_critic2'] = ag.target_critic2.state_dict()
            state['critic_optimizer2'] = ag.critic_optimizer2.state_dict()
        if hasattr(ag, 'update_count'):
            state['update_count'] = ag.update_count
        torch.save(state, os.path.join(ckpt_dir, f'agent_{i}.pt'))

    train_state = {
        'episode': episode,
        'episode_rewards': ep_rewards,
        'eval_rewards': [(int(e), float(r)) for e, r in eval_rewards],
        'total_steps': total_steps,
        'buffer_data': list(buffer.buffer)[-50000:],  # Last 50k transitions
    }
    torch.save(train_state, os.path.join(ckpt_dir, 'train_state.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--n_eps', type=int, default=1500)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    algos = ALL_ALGOS if args.algo == 'all' else [args.algo]

    total_runs = len(algos) * len(args.seeds)
    print(f"Total training runs: {total_runs}")
    print(f"Seeds: {args.seeds}")
    print(f"Algorithms: {algos}")
    print(f"Episodes per run: {args.n_eps}")
    print(f"Estimated time (CPU): {total_runs * args.n_eps * 5 // 3600} hours")

    start_time = time.time()
    run_count = 0

    for algo in algos:
        for seed in args.seeds:
            run_count += 1
            print(f"\n{'#'*60}")
            print(f"# RUN {run_count}/{total_runs}: {algo} seed={seed}")
            print(f"{'#'*60}")
            try:
                train_one_seed(algo, seed, args.n_eps, resume=not args.no_resume)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL DONE! Total time: {elapsed/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
