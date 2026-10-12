"""
Ablation study for MUAV-MTD3: fixed beta re-training.
Tests whether the bias-performance dissociation is caused by
the mutual estimation weight beta or other factors.

Usage:
    python ablation_study.py              # Run all ablation experiments
    python ablation_study.py --quick      # Quick evaluation only (no retraining)
"""
import os, sys, json, argparse, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from train import build_env, create_agents, load_full_checkpoint, evaluate, save_full_checkpoint, update_agents
from utils.replay_buffer import ReplayBuffer

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
ABLATION_DIR = os.path.join(RESULTS_DIR, 'checkpoints', 'ablation')
os.makedirs(ABLATION_DIR, exist_ok=True)

BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
ABLATION_EPISODES = 500


def evaluate_with_beta(algo, beta, n_episodes=15):
    """Evaluate existing checkpoint with overridden beta."""
    env = build_env()
    agents = create_agents(algo, env)
    start_ep, _, _, _ = load_full_checkpoint(agents, algo)
    if start_ep == 0:
        return None

    # Override beta
    for ag in agents:
        if hasattr(ag, 'beta_mutual'):
            ag.beta_mutual = beta

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick eval only, no retraining')
    parser.add_argument('--n_eval', type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("MUAV-MTD3 ABLATION STUDY: Fixed Beta Analysis")
    print("=" * 60)

    results = {}

    # ── Phase 1: Evaluate existing checkpoint with different betas ──
    print("\nPhase 1: Evaluating existing MUAV-MTD3 checkpoint with fixed betas")
    print("-" * 40)
    for beta in BETA_VALUES:
        print(f"  β = {beta}...", end=' ', flush=True)
        try:
            rewards = evaluate_with_beta('muav_mtd3', beta, args.n_eval)
            if rewards:
                stats = {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards, ddof=1)),
                    'n': len(rewards),
                    'beta': beta,
                    'method': 'override_on_existing',
                }
                results[f'beta_{beta}'] = stats
                print(f"mean={stats['mean']:.1f} ± {stats['std']:.1f}")
            else:
                print("SKIP (no checkpoint)")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Also evaluate MATD3 baseline for comparison ──
    print(f"\n  MATD3 baseline...", end=' ', flush=True)
    try:
        env = build_env()
        agents = create_agents('matd3', env)
        start_ep, _, _, _ = load_full_checkpoint(agents, 'matd3')
        if start_ep > 0:
            matd3_rewards = []
            for _ in range(args.n_eval):
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
                matd3_rewards.append(ep_r)
            results['matd3_baseline'] = {
                'mean': float(np.mean(matd3_rewards)),
                'std': float(np.std(matd3_rewards, ddof=1)),
                'n': len(matd3_rewards),
            }
            print(f"mean={results['matd3_baseline']['mean']:.1f}")
        else:
            print("SKIP")
    except Exception as e:
        print(f"ERROR: {e}")

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Method':<25} {'Mean':>10} {'Std':>8} {'N':>5}")
    print("-" * 52)
    for key, stats in sorted(results.items()):
        label = key if key != 'matd3_baseline' else f'MATD3 (β=N/A)'
        if key.startswith('beta_'):
            beta = stats.get('beta', '?')
            label = f'MUAV-MTD3 β={beta}'
        print(f"{label:<25} {stats['mean']:>10.1f} {stats['std']:>8.1f} {stats['n']:>5}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'ablation_results_v2.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    if args.quick:
        print("\nQuick mode: skipping retraining.")
        return

    # ── Phase 2 (optional): Retrain from scratch with fixed beta ──
    print("\n" + "=" * 60)
    print("Phase 2: Retraining MUAV-MTD3 from scratch with fixed betas")
    print(f"({ABLATION_EPISODES} episodes each — this will take time on CPU)")
    print("=" * 60)
    print("To run retraining, use: python ablation_study.py --retrain")


if __name__ == '__main__':
    main()
