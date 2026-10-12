# -*- coding: utf-8 -*-
"""eval_interagent_diag.py —— 零重训诊断（审稿意见 #3 的前提检验）

回答的问题：奖励里写了"保持编队 + 避免机间碰撞"，但观测里没有队友信息。
             那么在本任务的实际执行中，这两项**到底有没有被触发**？

统计（逐 step，跨 episode 聚合）：
  * 机间接近事件：agent 对的中心距 d_ij < 20 m（奖励 inter-agent 项的触发条件）与 < 25 m
  * 最小机间距离（全局最小值）
  * 编队误差：三对机间距离的标准差（与论文 Formation Error 定义一致）
    参考值：起始/目标构型本身的 std = 37.71 m（等腰三角形 80/80/160），故"保持构型"对应 ≈37.7，而非 0
  * 编队质心到目标质心距离（整体是否在向目标推进）
  * 首/末 10% 步的编队误差与质心距离（趋势：是否在收拢/推进）

用法:
  python eval_interagent_diag.py --algos maddpg matd3 muav_mtd3 muav_std3 \
      --seeds 42 123 --n_episodes 50 --results-dir <UAV_RESULTS_DIR>
输出: <UAV_RESULTS_DIR>/interagent_diag.json
"""
import os, sys, json, argparse
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from config import *                       # noqa
from train import build_env, create_agents  # noqa
import torch

ALGOS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3']
D_PEN = 20.0
D_MARGIN = 25.0

# 起始/目标构型（等腰三角形）自身的编队误差参考值
_IDEAL = np.array([[200.0, 0.0], [200.0, 80.0], [200.0, -80.0]])
_pd = [np.linalg.norm(_IDEAL[0] - _IDEAL[1]), np.linalg.norm(_IDEAL[1] - _IDEAL[2]),
       np.linalg.norm(_IDEAL[0] - _IDEAL[2])]
IDEAL_FORM_ERR = float(np.std(_pd))
GOAL_CENTROID = _IDEAL.mean(axis=0)


def load_agents(algo, seed, ckpt_dir, env):
    agents = create_agents(algo, env)
    d = os.path.join(ckpt_dir, algo if seed == 42 else f'{algo}_seed{seed}')
    for i, ag in enumerate(agents):
        f = os.path.join(d, f'agent_{i}.pt')
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        st = torch.load(f, map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents


def pair_stats(states):
    n = len(states)
    ds = []
    for i in range(n):
        for j in range(i + 1, n):
            ds.append(float(np.linalg.norm(states[i, 0:2] - states[j, 0:2])))
    return ds


def run(algo, seed, ckpt_dir, n_episodes):
    env = build_env()
    agents = load_agents(algo, seed, ckpt_dir, env)
    eps = []
    for ep in range(n_episodes):
        obs = env.reset()
        for a in agents:
            a.reset_noise()
        rec = {'pair_min': [], 'form_err': [], 'centroid_d': [], 'n_pen': [], 'n_margin': []}
        for t in range(MAX_EPISODE_STEPS):
            ds = pair_stats(env.states)
            rec['pair_min'].append(min(ds))
            rec['form_err'].append(float(np.std(ds)))
            rec['centroid_d'].append(float(np.linalg.norm(env.states[:, 0:2].mean(axis=0) - GOAL_CENTROID)))
            rec['n_pen'].append(float(sum(1 for d in ds if d < D_PEN)))
            rec['n_margin'].append(float(sum(1 for d in ds if d < D_MARGIN)))
            acts = [agents[i].select_action(obs[i], add_noise=False) for i in range(env.n_agents)]
            obs, r, dones, _ = env.step(acts)
            if dones[0]:
                break
        fe = np.array(rec['form_err'])
        cd = np.array(rec['centroid_d'])
        k = max(1, len(fe) // 10)
        eps.append({
            'steps': len(fe),
            'pair_min': float(np.min(rec['pair_min'])),
            'form_err_mean': float(fe.mean()),
            'form_err_first10': float(fe[:k].mean()),
            'form_err_last10': float(fe[-k:].mean()),
            'centroid_d_first10': float(cd[:k].mean()),
            'centroid_d_last10': float(cd[-k:].mean()),
            'steps_with_interagent_pen': float(np.mean([1.0 if v > 0 else 0.0 for v in rec['n_pen']])),
            'steps_with_interagent_margin': float(np.mean([1.0 if v > 0 else 0.0 for v in rec['n_margin']])),
            'interagent_pen_events': float(np.sum(rec['n_pen'])),
            'interagent_margin_events': float(np.sum(rec['n_margin'])),
        })
    agg = {
        'algo': algo, 'seed': seed, 'n_episodes': n_episodes,
        'ideal_form_err_reference': IDEAL_FORM_ERR,
        'form_err_mean': float(np.mean([e['form_err_mean'] for e in eps])),
        'form_err_ratio_to_ideal': float(np.mean([e['form_err_mean'] for e in eps]) / IDEAL_FORM_ERR),
        'form_err_first10': float(np.mean([e['form_err_first10'] for e in eps])),
        'form_err_last10': float(np.mean([e['form_err_last10'] for e in eps])),
        'centroid_d_first10': float(np.mean([e['centroid_d_first10'] for e in eps])),
        'centroid_d_last10': float(np.mean([e['centroid_d_last10'] for e in eps])),
        'global_min_pair_distance': float(np.min([e['pair_min'] for e in eps])),
        'frac_steps_with_interagent_pen': float(np.mean([e['steps_with_interagent_pen'] for e in eps])),
        'frac_steps_with_interagent_margin': float(np.mean([e['steps_with_interagent_margin'] for e in eps])),
        'interagent_pen_events_per_episode': float(np.mean([e['interagent_pen_events'] for e in eps])),
        'interagent_margin_events_per_episode': float(np.mean([e['interagent_margin_events'] for e in eps])),
        'episodes': eps,
    }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algos', nargs='+', default=ALGOS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    ap.add_argument('--n_episodes', type=int, default=50)
    ap.add_argument('--results-dir', default=None)
    args = ap.parse_args()
    rd = args.results_dir or os.environ.get('UAV_RESULTS_DIR', os.path.join(CODE_DIR, '..', 'results'))
    ckpt = os.path.join(rd, 'checkpoints')
    out_path = os.path.join(rd, 'interagent_diag.json')
    out = {}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path, encoding='utf-8'))
        except Exception:
            out = {}
    out.setdefault('meta', {
        'purpose': 'check whether the reward terms for formation keeping and inter-agent collision avoidance '
                   'are actually active at execution time (reviewer item 3)',
        'd_pen': D_PEN, 'd_margin': D_MARGIN, 'ideal_form_err': IDEAL_FORM_ERR,
        'form_err_definition': 'standard deviation of the three pairwise inter-UAV distances',
        'n_episodes': args.n_episodes, 'seeds': args.seeds})
    print('results_dir =', rd)
    print('%-10s %6s %10s %9s %9s %9s %9s' % ('algo', 'seed', 'form_err', 'ratio', 'fe_first', 'fe_last', 'pen%'))
    for algo in args.algos:
        for seed in args.seeds:
            key = f'{algo}_seed{seed}'
            if key in out:
                print('  [SKIP]', key)
                continue
            np.random.seed(seed)
            torch.manual_seed(seed)
            a = run(algo, seed, ckpt, args.n_episodes)
            out[key] = a
            json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print('%-10s %6d %10.1f %9.1f %9.1f %9.1f %8.1f%%'
                  % (algo, seed, a['form_err_mean'], a['form_err_ratio_to_ideal'],
                     a['form_err_first10'], a['form_err_last10'],
                     100 * a['frac_steps_with_interagent_pen']))
    print('saved:', out_path)


if __name__ == '__main__':
    main()
