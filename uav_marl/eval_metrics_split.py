# -*- coding: utf-8 -*-
"""eval_metrics_split.py — 分离统计（对应审稿意见 #2 / #4，在 CPU 或 GPU 上均可跑，纯评估）。

与论文原评估的区别：
  * 真实碰撞      : agent–obstacle 中心距 < r(=25 m)
  * 安全距离违反  : r <= d < r + margin(=20 m) → 中心距 < 45 m
  * 惩罚触发事件  : d < d_pen(=20 m)（= 论文原 "Collisions" 列的口径）
  * 个体到达      : 每架 UAV 终点距目标 < 阈值（默认 120 m = 30% 相对阈值）
  * 全队到达      : 三架全部达标
  * 无碰撞成功    : 全队到达 且 全程无真实碰撞
  * 安全合规成功  : 全队到达 且 全程无安全距离违反
  * 另报最小距离、终点距离、违反步数等

用法:
  python eval_metrics_split.py --algos maddpg matd3 muav_mtd3 muav_std3 td3 \\
      --seeds 42 123 456 --n_episodes 50 --results-dir <UAV_RESULTS_DIR>
"""
import os, sys, json, argparse
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from config import *                      # noqa
from train import build_env, create_agents  # noqa
import torch

R_OBST = 25.0
MARGIN = 20.0
D_PEN = 20.0
ALGO_KEYS = ['maddpg', 'matd3', 'muav_mtd3', 'muav_std3', 'td3']


def load_agents(algo, seed, ckpt_dir, env):
    agents = create_agents(algo, env)
    if algo == 'td3':
        ck = os.path.join(ckpt_dir, 'td3') if seed == 42 else os.path.join(ckpt_dir, f'td3_seed{seed}')
    else:
        ck = os.path.join(ckpt_dir, algo) if seed == 42 else os.path.join(ckpt_dir, f'{algo}_seed{seed}')
    for i, ag in enumerate(agents):
        f = os.path.join(ck, f'agent_{i}.pt')
        if not os.path.exists(f):
            print('  !! missing', f)
            continue
        st = torch.load(f, map_location=DEVICE, weights_only=False)
        if 'actor' in st:
            ag.actor.load_state_dict(st['actor'])
    return agents


def episode_metrics(env):
    """跑完一个 episode 后统计：需要逐 step 记录的最坏情况"""
    n = env.n_agents
    true_coll = 0
    pen_events = 0
    safety_viol = 0
    min_d = np.inf
    for i in range(n):
        px, py = env.states[i, 0], env.states[i, 1]
        for ox, oy in env.obstacles:
            d = float(np.linalg.norm([px - ox, py - oy]))
            min_d = min(min_d, d)
            if d < R_OBST:
                true_coll += 1
            if d < D_PEN:
                pen_events += 1
            if R_OBST <= d < R_OBST + MARGIN:
                safety_viol += 1
    return true_coll, pen_events, safety_viol, min_d


def run(algo, seed, ckpt_dir, env, n_episodes, thresh):
    agents = load_agents(algo, seed, ckpt_dir, env)
    rows = []
    for ep in range(n_episodes):
        obs_list = env.reset()
        for a in agents:
            a.reset_noise()
        true_coll = pen_events = safety_viol = 0
        min_d = np.inf
        for step in range(MAX_EPISODE_STEPS):
            tc, pe, sv, md = episode_metrics(env)
            true_coll += tc
            pen_events += pe
            safety_viol += sv
            min_d = min(min_d, md)
            actions = [a.select_action(obs_list[i], add_noise=False) for i, a in enumerate(agents)]
            obs_list, r, dones, _ = env.step(actions)
            if dones[0]:
                break
        tc, pe, sv, md = episode_metrics(env)
        true_coll += tc
        pen_events += pe
        safety_viol += sv
        min_d = min(min_d, md)
        dists = [float(np.linalg.norm(env.states[i, 0:2] - env.goal_positions[i])) for i in range(env.n_agents)]
        arr = [d < thresh for d in dists]
        rows.append(dict(
            ep=ep,
            final_dists=dists,
            team_final_mean=float(np.mean(dists)),
            team_final_min=float(np.min(dists)),
            n_individual_arrived=int(sum(arr)),
            team_arrived=int(all(arr)),
            any_arrived=int(any(arr)),
            true_collisions=true_coll,
            safety_violations=safety_viol,
            penalty_events=pen_events,
            min_obstacle_dist=float(min_d),
            collision_free=int(true_coll == 0),
            safety_compliant=int(safety_viol == 0),
        ))
    agg = {
        'algo': algo, 'seed': seed, 'n_episodes': n_episodes, 'threshold_m': thresh,
        'team_arrival_rate': float(np.mean([r['team_arrived'] for r in rows])),
        'individual_arrival_rate': float(np.mean([r['n_individual_arrived'] / 3.0 for r in rows])),
        'any_arrival_rate': float(np.mean([r['any_arrived'] for r in rows])),
        'collision_free_rate': float(np.mean([r['collision_free'] for r in rows])),
        'safety_compliant_rate': float(np.mean([r['safety_compliant'] for r in rows])),
        'collision_free_team_success': float(np.mean([r['team_arrived'] and r['collision_free'] for r in rows])),
        'safety_compliant_team_success': float(np.mean([r['team_arrived'] and r['safety_compliant'] for r in rows])),
        'mean_team_final_dist': float(np.mean([r['team_final_mean'] for r in rows])),
        'mean_true_collisions': float(np.mean([r['true_collisions'] for r in rows])),
        'mean_safety_violations': float(np.mean([r['safety_violations'] for r in rows])),
        'mean_penalty_events': float(np.mean([r['penalty_events'] for r in rows])),
        'mean_min_obstacle_dist': float(np.mean([r['min_obstacle_dist'] for r in rows])),
    }
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algos', nargs='+', default=ALGO_KEYS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--n_episodes', type=int, default=50)
    ap.add_argument('--threshold', type=float, default=120.0)
    ap.add_argument('--results-dir', type=str, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir or os.environ.get('UAV_RESULTS_DIR', os.path.join(CODE_DIR, '..', 'results'))
    ckpt_dir = os.path.join(results_dir, 'checkpoints')
    out_path = os.path.join(results_dir, 'metrics_split.json')
    out = {}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path, encoding='utf-8'))
        except Exception:
            out = {}
    out.setdefault('meta', {'r_obstacle': R_OBST, 'margin': MARGIN, 'd_pen': D_PEN,
                            'semantics': 'separated collision / safety-violation / penalty-trigger statistics',
                            'seeds': args.seeds, 'n_episodes': args.n_episodes, 'threshold_m': args.threshold})
    print('results_dir =', results_dir)
    for algo in args.algos:
        for seed in args.seeds:
            key = f'{algo}_seed{seed}'
            if key in out:
                print(' [SKIP]', key)
                continue
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = build_env()
            agg, rows = run(algo, seed, ckpt_dir, env, args.n_episodes, args.threshold)
            out[key] = {'aggregate': agg, 'episodes': rows}
            json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print(' [DONE] %-22s team_arr=%.3f indiv_arr=%.3f true_coll=%.2f safety_viol=%.1f pen=%.1f'
                  % (key, agg['team_arrival_rate'], agg['individual_arrival_rate'],
                     agg['mean_true_collisions'], agg['mean_safety_violations'], agg['mean_penalty_events']))
    print('saved:', out_path)


if __name__ == '__main__':
    main()
