"""
Multi-seed evaluation and statistical analysis.
Uses existing checkpoints to compute mean ± std across seeds,
statistical tests (t-test, Cohen's d, 95% CI).

Usage:
    python multi_seed_eval.py              # Evaluate all existing checkpoints
"""
import os, sys, json, argparse
import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from train import build_env, create_agents, load_full_checkpoint

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

ALL_ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}


def find_seed_checkpoints(algo):
    """Find all checkpoint directories for a given algorithm."""
    seeds = {}
    main_dir = os.path.join(CKPT_DIR, algo)
    if os.path.exists(os.path.join(main_dir, 'train_state.pt')):
        seeds['main'] = main_dir
    for dname in os.listdir(CKPT_DIR):
        if not dname.startswith(algo + '_'):
            continue
        dpath = os.path.join(CKPT_DIR, dname)
        if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, 'train_state.pt')):
            seeds[dname] = dpath
    return seeds


def evaluate_checkpoint(checkpoint_dir, algo_name, n_episodes=15):
    """Load checkpoint and evaluate."""
    env = build_env()
    agents = create_agents(algo_name, env)
    start_ep, ep_rewards, eval_rewards, _ = load_full_checkpoint(agents, os.path.basename(checkpoint_dir))
    if start_ep == 0:
        return None, None
    eval_results = []
    for _ in range(n_episodes):
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
        eval_results.append(ep_r)
    return ep_rewards, eval_results


def compute_statistics(values, label=""):
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    sem = std / np.sqrt(n) if n > 1 else 0
    ci95 = (mean - 1.96 * sem, mean + 1.96 * sem) if n > 1 else (mean, mean)
    return {'n': n, 'mean': float(mean), 'std': float(std), 'sem': float(sem),
            'ci95_low': float(ci95[0]), 'ci95_high': float(ci95[1]),
            'min': float(np.min(arr)), 'max': float(np.max(arr)),
            'median': float(np.median(arr)), 'label': label}


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def compare_algorithms(a_vals, b_vals, a_label, b_label):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return None
    t_stat, t_p = scipy_stats.ttest_ind(a_vals, b_vals)
    try:
        u_stat, u_p = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
    except ValueError:
        u_stat, u_p = 0, 1.0
    d = cohens_d(a_vals, b_vals)
    return {'t_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
            'mann_whitney_u': {'statistic': float(u_stat), 'p_value': float(u_p)},
            'cohens_d': float(d), 'a_mean': float(np.mean(a_vals)),
            'b_mean': float(np.mean(b_vals)), 'a_label': a_label, 'b_label': b_label}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_eval', type=int, default=15, help='Evaluation episodes per checkpoint')
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-SEED EVALUATION & STATISTICAL ANALYSIS")
    print("=" * 70)

    all_eval_data = {}

    for algo in ALL_ALGOS:
        seed_dirs = find_seed_checkpoints(algo)
        print(f"\n{algo}: {len(seed_dirs)} checkpoints: {list(seed_dirs.keys())}")
        algo_evals = []
        for seed_name, ckpt_path in sorted(seed_dirs.items()):
            print(f"  {seed_name}...", end=' ', flush=True)
            try:
                train_r, eval_r = evaluate_checkpoint(ckpt_path, algo, args.n_eval)
                if eval_r is None:
                    print("SKIP")
                    continue
                final100 = train_r[-100:] if len(train_r) >= 100 else train_r
                print(f"eval={np.mean(eval_r):.1f}, train100={np.mean(final100):.1f}")
                algo_evals.append({'seed': seed_name, 'eval_rewards': eval_r,
                                   'train_rewards': train_r, 'n_episodes': len(train_r)})
            except Exception as e:
                print(f"ERROR: {e}")
        all_eval_data[algo] = algo_evals

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS (per-seed means)")
    print("=" * 70)
    aggregate = {}
    for algo in ALL_ALGOS:
        evals = all_eval_data[algo]
        if not evals:
            continue
        seed_means = [np.mean(sd['eval_rewards']) for sd in evals]
        stats = compute_statistics(seed_means, ALGO_LABELS[algo])
        stats['n_seeds'] = len(evals)
        stats['seed_details'] = {sd['seed']: float(np.mean(sd['eval_rewards'])) for sd in evals}
        aggregate[algo] = stats
        print(f"{stats['label']}: {stats['mean']:.1f} ± {stats['std']:.1f}, "
              f"n={stats['n_seeds']}, 95%CI=[{stats['ci95_low']:.1f}, {stats['ci95_high']:.1f}]")

    # Pairwise
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS")
    print("=" * 70)
    pairwise = {}
    for a, b in [('matd3','muav_std3'),('matd3','muav_mtd3'),('matd3','maddpg'),
                 ('muav_std3','muav_mtd3'),('muav_std3','maddpg')]:
        if a not in all_eval_data or b not in all_eval_data:
            continue
        a_means = [np.mean(sd['eval_rewards']) for sd in all_eval_data[a]]
        b_means = [np.mean(sd['eval_rewards']) for sd in all_eval_data[b]]
        r = compare_algorithms(a_means, b_means, ALGO_LABELS[a], ALGO_LABELS[b])
        if r:
            pairwise[f"{a}_vs_{b}"] = r
            sig = "***" if r['t_test']['p_value']<0.001 else ("**" if r['t_test']['p_value']<0.01 else ("*" if r['t_test']['p_value']<0.05 else "ns"))
            print(f"{r['a_label']} vs {r['b_label']}: d={r['cohens_d']:.3f}, p={r['t_test']['p_value']:.2e} {sig}")

    # Save
    output = {'aggregate': aggregate, 'pairwise': pairwise}
    out_path = os.path.join(RESULTS_DIR, 'multi_seed_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'Seeds':>6} {'Mean':>10} {'Std':>8} {'95% CI':>24} {'Min':>10} {'Max':>10}")
    print("-" * 88)
    for algo in ALL_ALGOS:
        if algo in aggregate:
            s = aggregate[algo]
            print(f"{s['label']:<15} {s['n_seeds']:>6} {s['mean']:>10.1f} {s['std']:>8.1f} "
                  f"[{s['ci95_low']:>9.1f}, {s['ci95_high']:>9.1f}] {s['min']:>10.1f} {s['max']:>10.1f}")


if __name__ == '__main__':
    main()
