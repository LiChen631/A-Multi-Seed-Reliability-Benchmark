# -*- coding: utf-8 -*-
"""把 10_multiseed_sweeps.sh 产出的逐 seed 扫描结果合并成多 seed 汇总（均值±标准差）。

输入目录里形如 beta_sweep_v3_seed42.json / beta_sweep_v3_seed123.json ...
输出：{实验: {配置: {seeds:[...], eval_mean, eval_std, n_seeds, diverged_ratio}}}
"""
import os, json, argparse, glob, re
import numpy as np


def load_dir(d):
    res = {}
    for p in sorted(glob.glob(os.path.join(d, '*_seed*.json'))):
        m = re.match(r'(.+)_seed(\d+)\.json$', os.path.basename(p))
        if not m:
            continue
        name, seed = m.group(1), int(m.group(2))
        try:
            data = json.load(open(p, encoding='utf-8'))
        except Exception as e:
            print('  skip', p, e)
            continue
        for cfg, v in data.items():
            if not isinstance(v, dict) or 'eval_reward' not in v:
                continue
            res.setdefault(name, {}).setdefault(str(cfg), []).append(
                {'seed': seed, 'eval_reward': v['eval_reward'],
                 'eval_std': v.get('eval_std'), 'diverged': bool(v.get('diverged'))})
    out = {}
    for name, cfgs in res.items():
        out[name] = {}
        for cfg, rows in cfgs.items():
            vals = [r['eval_reward'] for r in rows]
            out[name][cfg] = {
                'seeds': [r['seed'] for r in rows],
                'n_seeds': len(vals),
                'eval_mean': float(np.mean(vals)),
                'eval_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'diverged_ratio': float(np.mean([r['diverged'] for r in rows])),
                'per_seed': rows,
            }
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    r = load_dir(a.dir)
    json.dump(r, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for name, cfgs in r.items():
        print('==', name)
        for cfg, v in sorted(cfgs.items(), key=lambda kv: kv[0]):
            print('   %-6s n=%d  mean=%8.1f ± %6.1f  diverged=%.0f%%' %
                  (cfg, v['n_seeds'], v['eval_mean'], v['eval_std'], 100 * v['diverged_ratio']))
    print('saved:', a.out)
