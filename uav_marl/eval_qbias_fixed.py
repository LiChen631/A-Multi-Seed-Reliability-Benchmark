# -*- coding: utf-8 -*-
"""eval_qbias_fixed.py — F2 修复版：TD-target discrepancy 度量（可续传）

修复点（对照审稿意见 F2）：
1. y_target 使用单机奖励 r_i（不再用 sum(rewards)）
2. 每算法用其自身训练目标构造 y_target —— 直接复用各 agent 的 compute_target()，
   与训练时（train.py 的 update_agents）口径完全一致，消除手写目标公式的偏差
3. 移除 MC-bias（原实现存在状态恢复 / 量纲 / 配对缺陷，且并非本文结论所需）
4. 对每个 (algo, seed) 重置随机种子，保证评估可复现、扰动独立
5. 路径由脚本位置推导，可用 --results-dir 覆盖（不再硬编码绝对路径）

输出：TD-target discrepancy = Q_online(s,a) - y_target，其中
    y_target = r_i + γ·target_i(s', a')（target_i 为各算法自身的 bootstrap 目标）
该量是自参照代理度量（self-referential proxy），而非相对 ground-truth Q* 的统计偏差，
论文中以 "TD-target discrepancy" 命名（勿称 estimation bias）。

用法：python eval_qbias_fixed.py [--seeds 42 123 456] [--n_episodes 10] [--n_samples 500] [--results-dir PATH]
"""
import os, sys, json, argparse
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import *
from train import build_env, create_agents

ALGO_LABELS = {'maddpg': 'MADDPG', 'matd3': 'MATD3', 'muav_mtd3': 'MUAV-MTD3', 'muav_std3': 'MUAV-STD3'}
ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']


def default_results_dir():
    # 默认：脚本所在目录的上一级 results（即 <root>/results）
    return os.path.join(os.path.dirname(CODE_DIR), 'results')


def load_existing(out_path):
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_results(out, out_path):
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, out_path)


def load_agents(algo, seed, ckpt_dir):
    env = build_env()
    agents = create_agents(algo, env)
    ck = os.path.join(ckpt_dir, algo) if seed == 42 else os.path.join(ckpt_dir, f'{algo}_seed{seed}')
    for i, ag in enumerate(agents):
        st = torch.load(os.path.join(ck, f'agent_{i}.pt'), map_location=DEVICE, weights_only=False)
        for key in ['actor', 'critic', 'target_actor', 'target_critic', 'critic2', 'target_critic2']:
            if hasattr(ag, key) and key in st:
                getattr(ag, key).load_state_dict(st[key])
    return agents, env


def run_one(algo, seed, n_episodes, n_samples):
    """跑单个 (algo, seed)，返回结果 dict（TD-target discrepancy）。"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    agents, env = load_agents(algo, seed, CKPT_DIR)
    n_agents = env.n_agents
    disc = []      # TD-target discrepancy (Q_est - y_target)，每个 agent 每条样本
    q_ests = []
    y_targets = []
    samples, episodes = 0, 0

    while samples < n_samples and episodes < n_episodes:
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        gs = env._get_global_state()
        for _ in range(MAX_EPISODE_STEPS):
            actions = [a.select_action(obs_list[i], add_noise=False) for i in range(n_agents)]
            ja = np.concatenate(actions)                       # (action_dim * n_agents,)
            gs_t = torch.FloatTensor(gs).unsqueeze(0).to(DEVICE)       # (1, global_dim)
            ja_t = torch.FloatTensor(ja).unsqueeze(0).to(DEVICE)       # (1, action_dim * n_agents)

            # 执行动作，取 next obs / next global state / rewards / dones
            obs_list, rewards, dones, _ = env.step(actions)
            ngs = env._get_global_state()

            # 用各 agent 自己的 compute_target 构造 y_target（与训练口径一致）
            target_actors = [a.target_actor for a in agents]
            # compute_target 内部对 next_global_state / next_obs 做 torch.FloatTensor()，必须传 numpy/cpu
            ngs_np = ngs.reshape(1, -1)                                     # numpy (1, global_dim)
            nobs_np = [obs.reshape(1, -1) for obs in obs_list]              # list of numpy (1, obs_dim)
            done_val = 1.0 if dones[0] else 0.0
            rew_t = {j: torch.FloatTensor([rewards[j]]).to(DEVICE) for j in range(n_agents)}   # (1,)
            done_t = {j: torch.FloatTensor([done_val]).to(DEVICE) for j in range(n_agents)}    # (1,)

            for i, ag in enumerate(agents):
                if algo == 'muav_mtd3':
                    peer = [(agents[j].target_critic, agents[j].target_critic2)
                            for j in range(n_agents) if j != i]
                    yt = ag.compute_target(ngs_np, nobs_np, target_actors, rew_t, done_t, peer_critics=peer)
                else:
                    yt = ag.compute_target(ngs_np, nobs_np, target_actors, rew_t, done_t)
                q_est = ag.critic(gs_t, ja_t)                   # (1,1)
                disc.append(float((q_est - yt).item()))
                q_ests.append(float(q_est.item()))
                y_targets.append(float(yt.item()))

            samples += 1
            gs = ngs
            if dones[0]:
                break
        episodes += 1

    disc = np.array(disc)
    return {
        'n_samples': int(len(disc)),
        'td_target_discrepancy_mean': float(disc.mean()),
        'td_target_discrepancy_std': float(disc.std(ddof=1)) if len(disc) > 1 else 0.0,
        'q_est_mean': float(np.mean(q_ests)),
        'y_target_mean': float(np.mean(y_targets)),
        'method': 'per-agent reward r_i; per-algorithm own compute_target; TD-target discrepancy = Q_est - y_target (self-referential proxy)',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--results-dir', type=str, default=None,
                        help='results 根目录（默认脚本所在目录上一级的 results）')
    args = parser.parse_args()

    global RESULTS_DIR, OUT_PATH, CKPT_DIR
    RESULTS_DIR = args.results_dir or default_results_dir()
    OUT_PATH = os.path.join(RESULTS_DIR, 'qbias_fixed_results.json')
    CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

    print("=" * 70)
    print("F2-FIXED EVALUATION (resumable): TD-target discrepancy")
    print(f"seeds={args.seeds}, episodes={args.n_episodes}, samples={args.n_samples}")
    print(f"results_dir={RESULTS_DIR}")
    print("Ctrl+C 安全；重新运行自动跳过已完成项。")
    print("=" * 70)

    out = load_existing(OUT_PATH)
    if 'meta' not in out:
        out['meta'] = {'seeds': args.seeds, 'n_episodes': args.n_episodes, 'n_samples': args.n_samples,
                       'metric': 'TD-target discrepancy (Q_est - y_target), per-agent reward, per-algorithm own target'}

    interrupted = False
    try:
        for algo in ALGOS:
            if algo not in out:
                out[algo] = {}
            for seed in args.seeds:
                key = str(seed)
                if key in out[algo]:
                    print(f"[SKIP] {algo} seed={seed} (already done)", flush=True)
                    continue
                print(f"[RUN] {algo} seed={seed} ...", flush=True)
                try:
                    result = run_one(algo, seed, args.n_episodes, args.n_samples)
                    out[algo][key] = result
                    save_results(out, OUT_PATH)
                    print(f"  -> TD-disc={result['td_target_discrepancy_mean']:+.2f} "
                          f"[saved]", flush=True)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"  !! {algo} seed={seed} error: {e}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        save_results(out, OUT_PATH)
        print("\n[Ctrl+C] 已完成部分已保存，重新运行同一命令即可续跑剩余项。", flush=True)

    if not interrupted:
        print("\n--- SUMMARY (mean across seeds) ---")
        for algo in ALGOS:
            if algo not in out:
                continue
            vals = [out[algo][str(s)]['td_target_discrepancy_mean'] for s in args.seeds
                    if str(s) in out[algo]]
            if vals:
                print(f"  {algo:<12} TD-disc(mean over seeds)={np.mean(vals):+.2f}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
