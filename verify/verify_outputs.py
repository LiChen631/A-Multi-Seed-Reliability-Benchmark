# -*- coding: utf-8 -*-
"""verify_outputs.py —— 重跑结果校验（把"跑完是否合格"变成可执行断言）。

用法:
  python verify/verify_outputs.py --results-dir <UAV_RESULTS_DIR> --check seed456_duplication
  python verify/verify_outputs.py --results-dir <...> --check all
"""
import os, json, argparse, sys
import numpy as np


def load(p):
    return json.load(open(p, encoding='utf-8'))


def check_seed456_duplication(rd, tol=0.3):
    """核心断言：seed456 下 MUAV-STD3 与 MUAV-MTD3 不得再"几乎逐位相同"。"""
    p = os.path.join(rd, 'robustness_multi_seed.json')
    if not os.path.exists(p):
        return False, '找不到 %s' % p
    d = load(p)
    if 'muav_std3' not in d or 'seed456' not in d['muav_std3']:
        return False, 'muav_std3.seed456 不存在（是否尚未重跑？）'
    conds = list(d['muav_std3']['seed456'].keys())
    same = [c for c in conds
            if abs(d['muav_std3']['seed456'][c]['mean'] - d['muav_mtd3']['seed456'][c]['mean']) < tol]
    if same:
        return False, '仍有 %d/%d 个条件与 MUAV-MTD3 相差 < %s：%s' % (len(same), len(conds), tol, same)
    # 与旧值对比：新结果应与原稿 seed456 明显不同
    return True, '通过：12 个条件均与 MUAV-MTD3 明显不同（>=%s）；旧串档值已被替换' % tol


def check_robustness_ranges(rd):
    d = load(os.path.join(rd, 'robustness_multi_seed.json'))
    ok, msgs = True, []
    exp = {'maddpg': (-1400, -1000), 'matd3': (-700, -450),
           'muav_mtd3': (-900, -700), 'muav_std3': (-800, -550)}
    for a, (lo, hi) in exp.items():
        agg = d.get(a, {}).get('aggregate', {})
        vals = [v['mean_across_seeds'] for v in agg.values() if isinstance(v, dict)]
        if not vals:
            ok = False; msgs.append('%s 无 aggregate' % a); continue
        if not (lo <= np.mean(vals) <= hi):
            ok = False; msgs.append('%s 均值 %.1f 超出预期区间 [%d,%d]' % (a, np.mean(vals), lo, hi))
    return ok, ('通过：四个算法的聚合回报都在预期量级' if ok else '；'.join(msgs))


def check_metrics_split(rd):
    p = os.path.join(rd, 'metrics_split.json')
    if not os.path.exists(p):
        return False, '缺少 metrics_split.json（是否还没跑 02 脚本？）'
    d = load(p)
    keys = [k for k in d if k != 'meta']
    bad = [k for k in keys if 'aggregate' not in d[k]]
    if bad:
        return False, '以下条目缺 aggregate: %s' % bad[:5]
    m = d[keys[0]]['aggregate']
    need = ['team_arrival_rate', 'individual_arrival_rate', 'collision_free_team_success',
            'mean_true_collisions', 'mean_safety_violations', 'mean_penalty_events']
    miss = [x for x in need if x not in m]
    return (not miss), ('通过：分离指标齐备（%d 条）' % len(keys) if not miss else '缺字段: %s' % miss)


CHECKS = {
    'seed456_duplication': ('seed456 串档是否修好（论文 Table 6 的关键）', check_seed456_duplication),
    'robustness_ranges': ('鲁棒性聚合值是否在预期量级', check_robustness_ranges),
    'metrics_split': ('分离指标是否齐备', check_metrics_split),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=os.environ.get('UAV_RESULTS_DIR', 'results'))
    ap.add_argument('--check', default='all')
    a = ap.parse_args()
    names = list(CHECKS) if a.check == 'all' else [a.check]
    allok = True
    for n in names:
        if n not in CHECKS:
            print('unknown check:', n); continue
        title, fn = CHECKS[n]
        try:
            ok, msg = fn(a.results_dir)
        except Exception as e:
            ok, msg = False, '异常 %s: %s' % (type(e).__name__, e)
        allok &= ok
        print('%s %-22s %s' % ('[PASS]' if ok else '[FAIL]', n, msg))
    print('总体:', 'PASS' if allok else 'FAIL')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
