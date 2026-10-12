# -*- coding: utf-8 -*-
"""eval_robustness_multi.py — F3 修复版：Observation Perturbation 鲁棒性评估（可续传）

修复点（对照审稿意见 F3）：
1. 语义改名 Observation Perturbation（delay/loss/noise 均为对 actor 局部观测的扰动，与实现一致）
2. 多种子 × 多 checkpoint（默认 seeds=[42,123,456]）
3. 每条件 30 episodes + 95% CI
4. 增加成功率随扰动变化指标
5. 对每个 (algo, seed) 重置随机种子，保证扰动噪声与 env.reset 可复现、跨种子独立
6. 路径由脚本位置推导，可用 --results-dir 覆盖（不再硬编码绝对路径）

可续传：每个 (algo, seed, condition) 完成即写入结果 json；重新运行自动跳过已完成项；Ctrl+C 安全。

用法：python eval_robustness_multi.py [--seeds 42 123 456] [--n_episodes 30] [--results-dir PATH]
"""
import os, sys, json, argparse
import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import *
from train import build_env, create_agents

ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
SUCCESS_THRESHOLD = 120.0

# 条件定义：均为对 actor 局部观测的扰动
CONDITIONS = []
for d in [0, 50, 100, 200]:
    CONDITIONS.append(('delay', d))
for p in [0.0, 0.1, 0.2, 0.3]:
    CONDITIONS.append(('loss', p))
for s in [0.0, 0.01, 0.05, 0.1]:
    CONDITIONS.append(('noise', s))


def default_results_dir():
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
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents, env


def eval_one(agents, env, kind, strength, n_episodes=30):
    rewards, successes = [], []
    for _ in range(n_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        ep_r, fd = 0.0, None
        for step in range(MAX_EPISODE_STEPS):
            noisy = []
            for i in range(env.n_agents):
                o = obs_list[i].copy()
                if kind == 'delay':
                    v_scale = np.linalg.norm(o[2:4]) if np.linalg.norm(o[2:4]) > 0 else 1.0
                    factor = strength / 1000.0
                    o[0:2] += factor * v_scale * np.random.randn(2) * 2.0
                    o[2:4] += factor * np.random.randn(2) * 1.0
                elif kind == 'loss':
                    if np.random.random() < strength:
                        o[6:] = 0.0
                elif kind == 'noise':
                    o += np.random.randn(*o.shape) * strength
                noisy.append(o)
            actions = [a.select_action(noisy[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, r, dones, _ = env.step(actions)
            ep_r += sum(r)
            if dones[0] or step == MAX_EPISODE_STEPS - 1:
                states = env.states
                fd = float(np.mean([np.linalg.norm(states[i, 0:2] - env.goal_positions[i])
                                    for i in range(env.n_agents)]))
            if dones[0]:
                break
        rewards.append(ep_r)
        successes.append(1.0 if fd is not None and fd < SUCCESS_THRESHOLD else 0.0)
    mean = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=1))
    ci = [mean - 1.96 * std / np.sqrt(n_episodes), mean + 1.96 * std / np.sqrt(n_episodes)]
    return {'mean': mean, 'std': std, 'ci95': ci, 'success_rate': float(np.mean(successes)),
            'n': n_episodes, 'condition': f'{kind}{strength}'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--n_episodes', type=int, default=30)
    parser.add_argument('--results-dir', type=str, default=None,
                        help='results 根目录（默认脚本所在目录上一级的 results）')
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir or default_results_dir()
    OUT_PATH = os.path.join(RESULTS_DIR, 'robustness_multi_seed.json')
    CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')

    print("=" * 70)
    print("F3-FIXED: OBSERVATION PERTURBATION ROBUSTNESS (multi-checkpoint, resumable)")
    print(f"seeds={args.seeds}, episodes/condition={args.n_episodes}")
    print(f"results_dir={RESULTS_DIR}")
    print("语义：delay/loss/noise 均为对 actor 局部观测的扰动（与实现一致）。")
    print("Ctrl+C 安全；重新运行自动跳过已完成项。")
    print("=" * 70)

    out = load_existing(OUT_PATH)
    if 'meta' not in out:
        out['meta'] = {'seeds': args.seeds, 'n_episodes': args.n_episodes,
                       'semantics': 'observation perturbation (delay/loss/noise on actor obs)'}

    interrupted = False
    try:
        for algo in ALGOS:
            if algo not in out:
                out[algo] = {}
            for seed in args.seeds:
                # 每个 (algo, seed) 独立重置随机种子，保证可复现、跨种子独立
                np.random.seed(seed)
                torch.manual_seed(seed)
                agents, env = load_agents(algo, seed, CKPT_DIR)
                skey = f'seed{seed}'
                if skey not in out[algo]:
                    out[algo][skey] = {}
                for kind, strength in CONDITIONS:
                    ckey = f'{kind}{strength}'
                    if ckey in out[algo][skey]:
                        continue
                    print(f"[RUN] {algo} seed={seed} {ckey} ...", flush=True)
                    try:
                        r = eval_one(agents, env, kind, strength, args.n_episodes)
                        out[algo][skey][ckey] = r
                        save_results(out, OUT_PATH)
                        print(f"  -> {r['mean']:8.1f} (success {r['success_rate']*100:4.1f}%) [saved]",
                              flush=True)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"  !! {algo} {ckey} error: {e}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        save_results(out, OUT_PATH)
        print("\n[Ctrl+C] 已完成部分已保存，重新运行同一命令即可续跑。", flush=True)

    if not interrupted:
        print("\n--- SUMMARY (aggregate across seeds) ---")
        for algo in ALGOS:
            if algo not in out:
                continue
            agg = {}
            for kind, strength in CONDITIONS:
                ckey = f'{kind}{strength}'
                means = [out[algo][f'seed{s}'][ckey]['mean'] for s in args.seeds
                         if f'seed{s}' in out[algo] and ckey in out[algo][f'seed{s}']]
                srs = [out[algo][f'seed{s}'][ckey]['success_rate'] for s in args.seeds
                       if f'seed{s}' in out[algo] and ckey in out[algo][f'seed{s}']]
                if means:
                    agg[ckey] = {'mean_across_seeds': float(np.mean(means)),
                                 'std_across_seeds': float(np.std(means, ddof=1)),
                                 'success_rate_mean': float(np.mean(srs))}
            out[algo]['aggregate'] = agg
            print(f"  {algo}:", {k: round(v['mean_across_seeds'], 1) for k, v in agg.items()})
        save_results(out, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
