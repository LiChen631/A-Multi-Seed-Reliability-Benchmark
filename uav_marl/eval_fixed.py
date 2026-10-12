"""
Fixed multi-seed evaluation that handles different checkpoint formats.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from train import build_env, create_agents

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

ALL_ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}

def load_agents_manual(algo_name, ckpt_dir):
    """Manually load agents from checkpoint. Only needs actor for eval."""
    env = build_env()
    agents = create_agents(algo_name, env)
    
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ckpt_dir, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        # Always load actor + target_actor (needed for evaluation)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
        if 'target_actor' in st:
            ag.target_actor.load_state_dict(st['target_actor'])
        # Critics are optional for evaluation
        if 'critic' in st:
            ag.critic.load_state_dict(st['critic'])
        if 'target_critic' in st:
            ag.target_critic.load_state_dict(st['target_critic'])
        if hasattr(ag, 'critic2') and 'critic2' in st:
            ag.critic2.load_state_dict(st['critic2'])
        if hasattr(ag, 'target_critic2') and 'target_critic2' in st:
            ag.target_critic2.load_state_dict(st['target_critic2'])
        if hasattr(ag, 'actor_optimizer') and 'actor_optimizer' in st:
            ag.actor_optimizer.load_state_dict(st['actor_optimizer'])
        if hasattr(ag, 'critic_optimizer') and 'critic_optimizer' in st:
            ag.critic_optimizer.load_state_dict(st['critic_optimizer'])
    return agents, env

def find_seeds(algo):
    """Find all seeds with valid checkpoints."""
    seeds = {}
    for dname in os.listdir(CKPT_DIR):
        # Match: algo (main), algo_seedN, algo_seedN_v2, etc.
        if dname == algo or (dname.startswith(algo + '_') and 'bak' not in dname):
            dpath = os.path.join(CKPT_DIR, dname)
            a0 = os.path.join(dpath, 'agent_0.pt')
            ts = os.path.join(dpath, 'train_state.pt')
            if os.path.exists(a0) and os.path.exists(ts):
                tstate = torch.load(ts, map_location='cpu', weights_only=False)
                # Skip if no agent files have target networks
                try:
                    a0s = torch.load(a0, map_location='cpu', weights_only=False)
                    has_target = 'target_actor' in a0s and 'target_critic' in a0s
                    if not has_target:
                        continue
                except:
                    continue
                ep = tstate.get('episode', tstate.get('episode', 0))
                seeds[dname] = ep
    return seeds

def evaluate_agents(agents, env, n_episodes=15):
    rewards = []
    for _ in range(n_episodes):
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
    return rewards

def main():
    print("=" * 70)
    print("MULTI-SEED EVALUATION (FIXED)")
    print("=" * 70)

    all_data = {}

    for algo in ALL_ALGOS:
        seeds = find_seeds(algo)
        print(f"\n{algo}: {len(seeds)} valid checkpoints")
        algo_evals = []
        
        for seed_name, n_eps in sorted(seeds.items()):
            ckpt_path = os.path.join(CKPT_DIR, seed_name)
            print(f"  {seed_name} (trained {n_eps} eps)...", end=' ', flush=True)
            try:
                agents, env = load_agents_manual(algo, ckpt_path)
                rewards = evaluate_agents(agents, env, n_episodes=15)
                mean_r = np.mean(rewards)
                print(f"eval_mean={mean_r:.1f}")
                algo_evals.append({
                    'seed': seed_name,
                    'episodes_trained': n_eps,
                    'eval_rewards': rewards,
                })
            except Exception as e:
                print(f"ERROR: {e}")
        
        if algo_evals:
            all_data[algo] = algo_evals

    # Aggregate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    output = {}
    for algo in ALL_ALGOS:
        if algo not in all_data:
            continue
        evals = all_data[algo]
        seed_means = [np.mean(sd['eval_rewards']) for sd in evals]
        
        mean = np.mean(seed_means)
        std = np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0
        sem = std / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0
        ci = (mean - 1.96*sem, mean + 1.96*sem) if len(seed_means) > 1 else (mean, mean)
        
        output[algo] = {
            'algo': ALGO_LABELS[algo],
            'n_seeds': len(seed_means),
            'mean': float(mean),
            'std': float(std),
            'ci95': [float(ci[0]), float(ci[1])],
            'seed_details': {sd['seed']: {
                'trained_episodes': sd['episodes_trained'],
                'eval_mean': float(np.mean(sd['eval_rewards'])),
                'eval_std': float(np.std(sd['eval_rewards'], ddof=1)),
            } for sd in evals},
        }
        
        label = ALGO_LABELS[algo]
        print(f"{label}: {mean:.1f} ± {std:.1f} (n={len(seed_means)} seeds), "
              f"95% CI=[{ci[0]:.1f}, {ci[1]:.1f}]")
        for sd in evals:
            print(f"  {sd['seed']} (@{sd['episodes_trained']}ep): {np.mean(sd['eval_rewards']):.1f}")
    
    out_path = os.path.join(RESULTS_DIR, 'multi_seed_fixed.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return output

if __name__ == '__main__':
    main()
