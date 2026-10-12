"""
Comprehensive experiment suite for multi-UAV cooperative path planning paper.
Usage:
    python experiments.py --task qbias        # Q-value bias evaluation
    python experiments.py --task hyperparam   # Hyperparameter sensitivity sweeps
    python experiments.py --task robustness   # Robustness tests
    python experiments.py --task trajectory   # Trajectory visualization
    python experiments.py --task all          # Run all experiments
"""
import os, sys, argparse, json, copy, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import defaultdict

from config import *
from env.uav_env import UAVFormationEnv
from utils.replay_buffer import ReplayBuffer
from train import build_env, create_agents, load_full_checkpoint

RESULTS_DIR = os.environ.get('UAV_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

ALL_ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}
ALGO_COLORS = {'maddpg': '#e41a1c', 'matd3': '#377eb8', 'muav_mtd3': '#4daf4a', 'muav_std3': '#ff7f00'}


def load_agents(algo_name):
    """Load trained agents from checkpoint."""
    env = build_env()
    agents = create_agents(algo_name, env)
    start_ep, _, _, _ = load_full_checkpoint(agents, algo_name)
    if start_ep == 0:
        raise FileNotFoundError(f"No checkpoint found for {algo_name}")
    return agents, env


def mc_q_value(agents, env, state, action, n_rollouts=5, rollout_steps=100, gamma=0.99):
    """Estimate true Q(s,a) via Monte Carlo rollouts."""
    returns = []
    for _ in range(n_rollouts):
        # Restore environment state (approximate)
        env.reset()
        # Use current state's global state directly
        total_return = 0.0
        # Roll forward using current policy (no exploration noise)
        obs_list = env._get_observations()
        for step in range(rollout_steps):
            acts = [agents[i].select_action(obs_list[i], add_noise=False) for i in range(env.n_agents)]
            obs_list, rewards, dones, _ = env.step(acts)
            total_return += sum(rewards) * (gamma ** step)
            if dones[0]:
                break
        returns.append(total_return)
    return np.mean(returns)


def run_qbias_experiment():
    """Experiment 1: Q-value estimation bias verification.

    Computes TD-error bias = Q_est(s,a) - y_target(s,a) for each algorithm
    using the trained critics and target networks. Positive bias = overestimation,
    negative bias = underestimation.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Q-Value Estimation Bias")
    print("=" * 60)

    env = build_env()
    n_eval_episodes = 5  # Episodes to collect transitions
    n_samples = 500      # Total bias samples to collect
    all_results = {}

    for algo in ALL_ALGOS:
        print(f"\n--- {ALGO_LABELS[algo]} ---")
        try:
            agents, _ = load_agents(algo)
        except FileNotFoundError:
            print(f"  SKIP: no checkpoint")
            continue

        biases = []
        samples_collected = 0

        for ep in range(n_eval_episodes):
            obs_list = env.reset()
            global_state = env._get_global_state()
            for a in agents:
                a.reset_noise()

            for step in range(MAX_EPISODE_STEPS):
                if samples_collected >= n_samples // n_eval_episodes:
                    break

                # Select actions
                actions = [agents[i].select_action(obs_list[i], add_noise=False)
                          for i in range(N_AGENTS)]
                joint_actions = np.concatenate([a.reshape(1, -1) for a in actions], axis=1)

                # Step environment
                next_obs, rewards, dones, info = env.step(actions)
                next_global_state = env._get_global_state()

                # Compute Q_est = critic(s, a) for all agents
                gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(DEVICE)
                ja_t = torch.FloatTensor(joint_actions).to(DEVICE)
                ngs_t = torch.FloatTensor(next_global_state).unsqueeze(0).to(DEVICE)
                done_val = 1.0 if dones[0] else 0.0

                for i, ag in enumerate(agents):
                    # Q estimated by the critic
                    q_est = ag.critic(gs_t, ja_t).item()

                    # Q target: r + γ * target_critic(next_s, next_a) * (1-done)
                    if algo == 'maddpg':
                        next_acts = []
                        for j in range(N_AGENTS):
                            nj = torch.FloatTensor(next_obs[j]).unsqueeze(0).to(DEVICE)
                            next_acts.append(ag.target_actor(nj))
                        next_ja = torch.cat(next_acts, dim=1)
                        q_target = ag.target_critic(ngs_t, next_ja).item()
                    else:
                        # MATD3 family: use min(Q1_target, Q2_target)
                        next_acts = []
                        for j in range(N_AGENTS):
                            nj = torch.FloatTensor(next_obs[j]).unsqueeze(0).to(DEVICE)
                            a_j = ag.target_actor(nj)
                            noise = torch.randn_like(a_j) * ag.target_noise
                            noise = torch.clamp(noise, -ag.noise_clip, ag.noise_clip)
                            a_j = torch.clamp(a_j + noise, -1.0, 1.0)
                            next_acts.append(a_j)
                        next_ja = torch.cat(next_acts, dim=1)
                        q1 = ag.target_critic(ngs_t, next_ja).item()
                        q2 = ag.target_critic2(ngs_t, next_ja).item()
                        q_target = min(q1, q2)

                    y_target = sum(rewards) + GAMMA * q_target * (1 - done_val)
                    bias = q_est - y_target
                    biases.append(bias)

                samples_collected += 1
                obs_list = next_obs
                global_state = next_global_state

                if dones[0]:
                    break

        if biases:
            biases = np.array(biases)
            all_results[algo] = {
                'mean_bias': float(np.mean(biases)),
                'std_bias': float(np.std(biases)),
                'biases': biases.tolist(),
            }
            bias_str = 'Overestimation (+)' if np.mean(biases) > 0 else 'Underestimation (-)'
            print(f"  Samples: {len(biases)}, Mean bias: {np.mean(biases):.4f} +/- {np.std(biases):.4f}")
            print(f"  Direction: {bias_str}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bias distribution
    ax = axes[0]
    for algo, res in all_results.items():
        ax.hist(res['biases'], bins=40, alpha=0.5, label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo], density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Q-Value Estimation Bias (Q_est - y_target)')
    ax.set_ylabel('Density')
    ax.set_title('Q-Value Bias Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mean bias bar chart
    ax = axes[1]
    algo_names = [ALGO_LABELS[a] for a in all_results]
    mean_biases = [all_results[a]['mean_bias'] for a in all_results]
    std_biases = [all_results[a]['std_bias'] for a in all_results]
    colors_list = [ALGO_COLORS[a] for a in all_results]
    bars = ax.bar(algo_names, mean_biases, yerr=std_biases, color=colors_list,
                  edgecolor='black', capsize=5)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Mean Q-Value Estimation Bias')
    ax.set_title('Mean Bias Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mean_biases):
        offset = 0.015 if val >= 0 else -0.06
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q_bias_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: figures/q_bias_distribution.png")

    with open(os.path.join(RESULTS_DIR, 'qbias_results.json'), 'w') as f:
        json.dump({k: {'mean_bias': v['mean_bias'], 'std_bias': v['std_bias']}
                   for k, v in all_results.items()}, f, indent=2)


def run_hyperparam_sweep():
    """Experiment 3: Hyperparameter sensitivity analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Hyperparameter Sensitivity")
    print("=" * 60)

    env = build_env()

    # ── MUAV-MTD3: β sweep ──
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"\n--- MUAV-MTD3 β sweep: {beta_values} ---")
    beta_results = []
    for beta in beta_values:
        print(f"  β = {beta}...", end=' ')
        try:
            agents, _ = load_agents('muav_mtd3')
            # Override beta_mutual
            for ag in agents:
                ag.beta_mutual = beta
            rewards = evaluate_agents(agents, env, num_episodes=20)
            beta_results.append({'beta': beta, 'reward': float(rewards)})
            print(f"reward = {rewards:.2f}")
        except FileNotFoundError:
            print("SKIP (no checkpoint)")
            break

    # ── MUAV-STD3: β_soft sweep ──
    beta_soft_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    print(f"\n--- MUAV-STD3 β_soft sweep: {beta_soft_values} ---")
    betasoft_results = []
    for bs_val in beta_soft_values:
        print(f"  β_soft = {bs_val}...", end=' ')
        try:
            agents, _ = load_agents('muav_std3')
            for ag in agents:
                ag.beta_soft = bs_val
            rewards = evaluate_agents(agents, env, num_episodes=20)
            betasoft_results.append({'beta_soft': bs_val, 'reward': float(rewards)})
            print(f"reward = {rewards:.2f}")
        except FileNotFoundError:
            print("SKIP (no checkpoint)")
            break

    # ── MUAV-STD3: K sweep ──
    k_values = [5, 10, 20]
    print(f"\n--- MUAV-STD3 K sweep: {k_values} ---")
    k_results = []
    for k in k_values:
        print(f"  K = {k}...", end=' ')
        try:
            agents, _ = load_agents('muav_std3')
            for ag in agents:
                ag.k_samples = k
            rewards = evaluate_agents(agents, env, num_episodes=20)
            k_results.append({'K': k, 'reward': float(rewards)})
            print(f"reward = {rewards:.2f}")
        except FileNotFoundError:
            print("SKIP (no checkpoint)")
            break

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # β plot
    ax = axes[0]
    if beta_results:
        xs = [r['beta'] for r in beta_results]
        ys = [r['reward'] for r in beta_results]
        ax.plot(xs, ys, 'o-', color='#4daf4a', markersize=8, linewidth=2)
        ax.set_xlabel(r'$\beta$ (Mutual Estimation Weight)')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title(r'MUAV-MTD3: $\beta$ Sensitivity')
        ax.grid(True, alpha=0.3)
        # Mark best
        best = max(beta_results, key=lambda r: r['reward'])
        ax.annotate(f'Best: β={best["beta"]}', xy=(best['beta'], best['reward']),
                    xytext=(best['beta'] + 0.15, best['reward'] + 5),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    # β_soft plot
    ax = axes[1]
    if betasoft_results:
        xs = [r['beta_soft'] for r in betasoft_results]
        ys = [r['reward'] for r in betasoft_results]
        ax.semilogx(xs, ys, 'o-', color='#ff7f00', markersize=8, linewidth=2)
        ax.set_xlabel(r'$\beta_{soft}$ (Softmax Temperature)')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title(r'MUAV-STD3: $\beta_{soft}$ Sensitivity')
        ax.grid(True, alpha=0.3)
        best = max(betasoft_results, key=lambda r: r['reward'])
        ax.annotate(f'Best: β_soft={best["beta_soft"]}', xy=(best['beta_soft'], best['reward']),
                    xytext=(best['beta_soft'] * 2, best['reward'] + 5),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    # K plot
    ax = axes[2]
    if k_results:
        xs = [r['K'] for r in k_results]
        ys = [r['reward'] for r in k_results]
        ax.plot(xs, ys, 's-', color='#ff7f00', markersize=8, linewidth=2)
        ax.set_xlabel('K (Action Samples)')
        ax.set_ylabel('Average Episode Reward')
        ax.set_title(r'MUAV-STD3: $K$ Sensitivity')
        ax.grid(True, alpha=0.3)
        best = max(k_results, key=lambda r: r['reward'])
        ax.annotate(f'Best: K={best["K"]}', xy=(best['K'], best['reward']),
                    xytext=(best['K'] + 3, best['reward'] + 5),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hyperparam_sensitivity.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {os.path.join(FIG_DIR, 'hyperparam_sensitivity.png')}")

    # Save data
    sweep_data = {
        'muav_mtd3_beta': beta_results,
        'muav_std3_beta_soft': betasoft_results,
        'muav_std3_K': k_results,
    }
    with open(os.path.join(RESULTS_DIR, 'hyperparam_results.json'), 'w') as f:
        json.dump(sweep_data, f, indent=2)


def run_robustness_experiment():
    """Experiment 4: Robustness under non-ideal conditions."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Robustness Analysis")
    print("=" * 60)

    env = build_env()
    num_episodes = 20

    results = defaultdict(dict)

    # ── Communication delay ──
    delay_values = [0, 50, 100, 200]  # ms
    print(f"\n--- Communication Delay: {delay_values} ms ---")
    for algo in ALL_ALGOS:
        print(f"  {ALGO_LABELS[algo]}:", end=' ')
        try:
            agents, _ = load_agents(algo)
        except FileNotFoundError:
            print("SKIP")
            continue
        algo_delays = []
        for delay_ms in delay_values:
            rewards = evaluate_with_delay(agents, env, delay_ms, num_episodes)
            algo_delays.append({'delay_ms': delay_ms, 'reward': float(rewards)})
            print(f"{delay_ms}ms={rewards:.1f}", end=' ')
        results['delay'][algo] = algo_delays
        print()

    # ── Packet loss ──
    loss_values = [0.0, 0.1, 0.2, 0.3]
    print(f"\n--- Packet Loss: {loss_values} ---")
    for algo in ALL_ALGOS:
        print(f"  {ALGO_LABELS[algo]}:", end=' ')
        try:
            agents, _ = load_agents(algo)
        except FileNotFoundError:
            print("SKIP")
            continue
        algo_losses = []
        for p_loss in loss_values:
            rewards = evaluate_with_packet_loss(agents, env, p_loss, num_episodes)
            algo_losses.append({'loss_rate': p_loss, 'reward': float(rewards)})
            print(f"{p_loss}={rewards:.1f}", end=' ')
        results['packet_loss'][algo] = algo_losses
        print()

    # ── Sensor noise ──
    noise_values = [0.0, 0.01, 0.05, 0.1]
    print(f"\n--- Sensor Noise σ: {noise_values} ---")
    for algo in ALL_ALGOS:
        print(f"  {ALGO_LABELS[algo]}:", end=' ')
        try:
            agents, _ = load_agents(algo)
        except FileNotFoundError:
            print("SKIP")
            continue
        algo_noises = []
        for sigma in noise_values:
            rewards = evaluate_with_noise(agents, env, sigma, num_episodes)
            algo_noises.append({'noise_sigma': sigma, 'reward': float(rewards)})
            print(f"σ={sigma}={rewards:.1f}", end=' ')
        results['sensor_noise'][algo] = algo_noises
        print()

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Delay plot
    ax = axes[0]
    for algo in ALL_ALGOS:
        if algo in results['delay']:
            xs = [d['delay_ms'] for d in results['delay'][algo]]
            ys = [d['reward'] for d in results['delay'][algo]]
            ax.plot(xs, ys, 'o-', label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                    markersize=7, linewidth=2)
    ax.set_xlabel('Communication Delay (ms)')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Robustness to Communication Delay')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Packet loss plot
    ax = axes[1]
    for algo in ALL_ALGOS:
        if algo in results['packet_loss']:
            xs = [d['loss_rate'] for d in results['packet_loss'][algo]]
            ys = [d['reward'] for d in results['packet_loss'][algo]]
            ax.plot(xs, ys, 's-', label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                    markersize=7, linewidth=2)
    ax.set_xlabel('Packet Loss Rate')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Robustness to Packet Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Sensor noise plot
    ax = axes[2]
    for algo in ALL_ALGOS:
        if algo in results['sensor_noise']:
            xs = [d['noise_sigma'] for d in results['sensor_noise'][algo]]
            ys = [d['reward'] for d in results['sensor_noise'][algo]]
            ax.plot(xs, ys, 'D-', label=ALGO_LABELS[algo], color=ALGO_COLORS[algo],
                    markersize=7, linewidth=2)
    ax.set_xlabel('Sensor Noise σ')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Robustness to Sensor Noise')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'robustness_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {os.path.join(FIG_DIR, 'robustness_analysis.png')}")

    # Save data
    with open(os.path.join(RESULTS_DIR, 'robustness_results.json'), 'w') as f:
        json.dump({k: dict(v) for k, v in results.items()}, f, indent=2)


def run_trajectory_experiment():
    """Experiment 5: Trajectory visualization."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Trajectory Visualization")
    print("=" * 60)

    env = build_env()
    n_agents = env.n_agents

    all_trajectories = {}

    for algo in ALL_ALGOS:
        print(f"\n--- {ALGO_LABELS[algo]} ---")
        try:
            agents, _ = load_agents(algo)
        except FileNotFoundError:
            print(f"  SKIP: no checkpoint")
            continue

        # Record one full episode
        obs_list = env.reset()
        obstacles = env.obstacles.copy()
        for a in agents:
            a.reset_noise()

        trajectories = [[] for _ in range(n_agents)]
        total_reward = 0.0

        for step in range(MAX_EPISODE_STEPS):
            actions = [agents[i].select_action(obs_list[i], add_noise=False) for i in range(n_agents)]
            for i in range(n_agents):
                trajectories[i].append(env.states[i, 0:2].copy())
            obs_list, rewards, dones, info = env.step(actions)
            total_reward += sum(rewards)
            if dones[0]:
                break

        # Convert to arrays
        for i in range(n_agents):
            trajectories[i] = np.array(trajectories[i])

        all_trajectories[algo] = {
            'trajectories': [t.tolist() for t in trajectories],
            'total_reward': float(total_reward),
            'obstacles': obstacles.tolist(),
        }
        print(f"  Total reward: {total_reward:.2f}, Steps: {len(trajectories[0])}")

    # ── Plot: 3 algorithms side by side ──
    plot_algos = [a for a in ALL_ALGOS if a in all_trajectories]
    n_plots = len(plot_algos)
    if n_plots == 0:
        print("  No trajectories to plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    start_positions = np.array([[-200.0, 0.0], [-200.0, 80.0], [-200.0, -80.0]])
    goal_positions = np.array([[200.0, 0.0], [200.0, 80.0], [200.0, -80.0]])

    for ax_idx, algo in enumerate(plot_algos):
        ax = axes[ax_idx]
        data = all_trajectories[algo]
        obstacles_arr = np.array(data['obstacles'])

        # Draw obstacles
        for ox, oy in obstacles_arr:
            circle = Circle((ox, oy), D_SAFE, fill=True, color='gray', alpha=0.4, edgecolor='black', linewidth=0.5)
            ax.add_patch(circle)

        # Draw trajectories
        agent_colors_traj = ['#e41a1c', '#377eb8', '#4daf4a']
        for i in range(n_agents):
            traj = np.array(data['trajectories'][i])
            ax.plot(traj[:, 0], traj[:, 1], '-', color=agent_colors_traj[i], linewidth=1.5, alpha=0.8,
                    label=f'UAV {i+1}')
            # Start marker
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=agent_colors_traj[i], markersize=8)
            # End marker
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=agent_colors_traj[i], markersize=8)

        # Draw start and goal areas
        for i in range(n_agents):
            ax.plot(start_positions[i, 0], start_positions[i, 1], 'o', color='green',
                    markersize=10, markeredgecolor='black', markeredgewidth=1)
            ax.plot(goal_positions[i, 0], goal_positions[i, 1], '*', color='gold',
                    markersize=14, markeredgecolor='black', markeredgewidth=1)

        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{ALGO_LABELS[algo]}\nReward: {data["total_reward"]:.1f}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'trajectory_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {os.path.join(FIG_DIR, 'trajectory_comparison.png')}")

    # ── Plot: Overlay comparison (all algorithms on same map) ──
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw obstacles once
    first_algo = plot_algos[0]
    obstacles_arr = np.array(all_trajectories[first_algo]['obstacles'])
    for ox, oy in obstacles_arr:
        circle = Circle((ox, oy), D_SAFE, fill=True, color='gray', alpha=0.3, edgecolor='black', linewidth=0.5)
        ax.add_patch(circle)

    # Draw UAV 1 trajectories from all algorithms
    for algo in plot_algos:
        traj = np.array(all_trajectories[algo]['trajectories'][0])
        ax.plot(traj[:, 0], traj[:, 1], '-', color=ALGO_COLORS[algo], linewidth=2,
                label=f'{ALGO_LABELS[algo]} (UAV 1)', alpha=0.85)

    for i in range(n_agents):
        ax.plot(start_positions[i, 0], start_positions[i, 1], 'o', color='green',
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(goal_positions[i, 0], goal_positions[i, 1], '*', color='gold',
                markersize=16, markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison: UAV 1 Paths Across Algorithms')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'trajectory_overlay.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(FIG_DIR, 'trajectory_overlay.png')}")

    # Save data
    with open(os.path.join(RESULTS_DIR, 'trajectory_results.json'), 'w') as f:
        json.dump(all_trajectories, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════

def evaluate_agents(agents, env, num_episodes=10):
    """Standard evaluation."""
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


def evaluate_with_delay(agents, env, delay_ms, num_episodes=10):
    """Evaluate with simulated communication delay during execution.

    Delay affects the freshness of observations. At delay_ms > 0, we simulate
    this by adding zero-mean Gaussian perturbation to the observation, with
    variance proportional to the delay (stale data = noisier effective observation).
    """
    total_rewards = []
    delay_factor = delay_ms / 1000.0  # Convert to seconds, scale noise
    for _ in range(num_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            # Delay perturbs observations: stale position/velocity data
            noisy_obs = []
            for i in range(env.n_agents):
                o = obs_list[i].copy()
                # Position noise proportional to delay * velocity
                v_scale = np.linalg.norm(o[2:4]) if np.linalg.norm(o[2:4]) > 0 else 1.0
                pos_noise = delay_factor * v_scale * np.random.randn(2) * 2.0
                o[0:2] += pos_noise
                o[2:4] += delay_factor * np.random.randn(2) * 1.0
                noisy_obs.append(o)
            actions = [a.select_action(noisy_obs[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, rewards, dones, _ = env.step(actions)
            ep_r += sum(rewards)
            if dones[0]:
                break
        total_rewards.append(ep_r)
    return np.mean(total_rewards)


def evaluate_with_packet_loss(agents, env, p_loss, num_episodes=10):
    """Evaluate with packet loss: randomly zero out observation components."""
    total_rewards = []
    for _ in range(num_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            # Apply packet loss to observations
            noisy_obs = []
            for i in range(env.n_agents):
                o = obs_list[i].copy()
                if np.random.random() < p_loss:
                    # Simulate losing obstacle information (most critical for path planning)
                    o[6:] = 0.0  # Zero out obstacle relative positions
                noisy_obs.append(o)
            actions = [a.select_action(noisy_obs[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, rewards, dones, _ = env.step(actions)
            ep_r += sum(rewards)
            if dones[0]:
                break
        total_rewards.append(ep_r)
    return np.mean(total_rewards)


def evaluate_with_noise(agents, env, noise_sigma, num_episodes=10):
    """Evaluate with sensor noise added to observations."""
    total_rewards = []
    for _ in range(num_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r = 0.0
        for _ in range(MAX_EPISODE_STEPS):
            noisy_obs = []
            for i in range(env.n_agents):
                o = obs_list[i].copy()
                # Add Gaussian noise to position and velocity components (indices 0-3)
                o[:4] += np.random.randn(4) * noise_sigma * np.array([5.0, 5.0, 1.0, 1.0])
                noisy_obs.append(o)
            actions = [a.select_action(noisy_obs[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, rewards, dones, _ = env.step(actions)
            ep_r += sum(rewards)
            if dones[0]:
                break
        total_rewards.append(ep_r)
    return np.mean(total_rewards)


# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                       choices=['qbias', 'hyperparam', 'robustness', 'trajectory', 'all'])
    args = parser.parse_args()

    tasks = ['qbias', 'hyperparam', 'robustness', 'trajectory'] if args.task == 'all' else [args.task]

    for task in tasks:
        if task == 'qbias':
            run_qbias_experiment()
        elif task == 'hyperparam':
            run_hyperparam_sweep()
        elif task == 'robustness':
            run_robustness_experiment()
        elif task == 'trajectory':
            run_trajectory_experiment()

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)
