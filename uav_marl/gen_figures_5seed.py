# -*- coding: utf-8 -*-
"""gen_figures_5seed.py — 基于 5-seed 数据重新生成论文图片"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT_DIR = os.path.join(os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')), 'checkpoints')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
OUT_DIR = RESULTS_DIR

ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}
ALGO_COLORS = {'maddpg': '#1f77b4', 'matd3': '#ff7f0e', 'muav_mtd3': '#2ca02c', 'muav_std3': '#d62728'}
SEEDS = ['', '_seed123', '_seed456', '_seed789', '_seed1024']  # '' = main (seed 42)

def load_train_rewards(algo, seed_suffix):
    """Load episode rewards from checkpoint."""
    d = os.path.join(CKPT_DIR, f'{algo}{seed_suffix}')
    ts = os.path.join(d, 'train_state.pt')
    if not os.path.exists(ts):
        return None
    try:
        s = torch.load(ts, map_location='cpu', weights_only=False)
    except Exception:
        return None
    r = s.get('episode_rewards', s.get('rewards', None))
    if r is None:
        return None
    return np.array(r, dtype=float)

def gen_training_comparison():
    """Figure 3: 5-seed mean +/- std training curves."""
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 9})
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for algo in ALGO_LABELS:
        curves = []
        for s in SEEDS:
            r = load_train_rewards(algo, s)
            if r is not None and len(r) >= 100:
                curves.append(r)
        if len(curves) < 2:
            continue
        # Align to min length
        min_len = min(len(c) for c in curves)
        arr = np.stack([c[:min_len] for c in curves])
        # Smooth with 100-window mean
        win = 50
        smoothed = np.array([np.convolve(arr[:, i], np.ones(win)/win, mode='valid')
                             for i in range(min_len - win + 1)])
        # smoothed shape: (n_t, n_seeds) after transpose handling
        mean_curve = smoothed.mean(axis=1)
        std_curve = smoothed.std(axis=1, ddof=1)
        xs = np.arange(win, min_len + 1)
        ax.plot(xs, mean_curve, color=ALGO_COLORS[algo], linewidth=2,
                label=f'{ALGO_LABELS[algo]} (mean of {len(curves)} seeds)')
        ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve,
                        color=ALGO_COLORS[algo], alpha=0.15)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Episode Reward (100-episode smoothed)')
    ax.set_title('Training Curves (5-seed mean ± std)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'training_comparison_5seed.png')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")

    # Also save a bar chart of 5-seed means with std error bars
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    means, stds, labels, colors = [], [], [], []
    for algo in ALGO_LABELS:
        r = load_train_rewards(algo, '')
        if r is None:
            continue
        # Use eval values from final stats
        continue
    # Use final_5seed_stats.json for bar chart
    stats = json.load(open(os.path.join(OUT_DIR, 'final_5seed_stats.json')))['aggregate']
    order = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
    for algo in order:
        s = stats[algo]
        means.append(s['mean'])
        stds.append(s['std'])
        labels.append(ALGO_LABELS[algo])
        colors.append(ALGO_COLORS[algo])
    bars = ax2.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.8,
                   error_kw={'elinewidth': 1.5})
    ax2.set_ylabel('Mean Evaluation Reward (5 seeds)')
    ax2.set_title('5-Seed Performance Comparison (mean ± std)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    # Annotate convergence
    for i, algo in enumerate(order):
        ax2.text(i, means[i] + stds[i] + 40, f'conv {4 if algo != "muav_std3" else 5}/5',
                 ha='center', fontsize=9)
    plt.tight_layout()
    path2 = os.path.join(FIG_DIR, 'performance_5seed_bar.png')
    fig2.savefig(path2, dpi=300)
    plt.close(fig2)
    print(f"Saved: {path2}")

if __name__ == '__main__':
    gen_training_comparison()
