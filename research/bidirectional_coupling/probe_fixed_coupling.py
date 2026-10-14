"""Exploratory SDP probe; numerical gaps are not exact certificates.

All four free Bob reflections, all eight Alice outcome blocks, and every
native word of length <= k are included. No same-party commutation is added.
"""
from pathlib import Path
from time import perf_counter
import argparse
import json
import math

import cvxpy as cp
import numpy as np


def reduced(word):
    stack = []
    for x in word:
        if stack and stack[-1] == x:
            stack.pop()
        else:
            stack.append(x)
    return tuple(stack)


def word_ball(level):
    out = [()]
    front = [()]
    for _ in range(level):
        front = [w+(y,) for w in front for y in range(4)
                 if not w or w[-1] != y]
        out.extend(front)
    return out


def problem(level):
    words = word_ball(level)
    n = len(words)
    keys = {}
    positions = []
    for left in words:
        for right in words:
            w = reduced(tuple(reversed(left))+right)
            key = min(w, tuple(reversed(w)))
            keys.setdefault(key, len(keys))
            positions.append(keys[key])
    y = cp.Variable((8, len(keys)))
    mats = [cp.reshape(y[a, positions], (n, n), order='C') for a in range(8)]
    constraints = [m >> 0 for m in mats]
    common = y[0]+y[1]
    constraints += [y[2*x]+y[2*x+1] == common for x in range(1, 4)]
    constraints += [common[keys[()]] == 1]
    alpha = cp.Parameter(nonneg=True)
    eta = cp.Parameter(nonneg=True)
    score = alpha*(y[0, keys[()]]-y[1, keys[()]])
    score += alpha*common[keys[(2,)]]
    for block in (0, 2):
        for x in range(2):
            for b in range(2):
                sign = -1 if x == b == 1 else 1
                score += sign*(y[2*(block+x), keys[(block+b,)]]
                               -y[2*(block+x)+1, keys[(block+b,)]])
    score += eta*(y[0, keys[(2,)]]-y[1, keys[(2,)]])
    return cp.Problem(cp.Maximize(score), constraints), alpha, eta, y, mats, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--alpha', type=float, nargs='+', default=[1.5, 1.9, 1.99])
    parser.add_argument('--eta', type=float, nargs='+', default=[0.01, 0.1, 1.0])
    parser.add_argument('--solver', default='CLARABEL')
    parser.add_argument('--output', default='fixed_coupling_probe.json')
    args = parser.parse_args()
    task, a, e, y, mats, keys = problem(args.level)
    records = []
    print(json.dumps({'level': args.level, 'Bob_words': mats[0].shape[0],
                      'moment_keys': len(keys), 'solver': args.solver}), flush=True)
    for alpha in args.alpha:
        for eta in args.eta:
            a.value, e.value = alpha, eta
            start = perf_counter()
            opts = {'max_iter': 150, 'tol_gap_abs': 1e-9,
                    'tol_gap_rel': 1e-9, 'tol_feas': 1e-9}
            if args.solver == 'SCS':
                opts = {'max_iters': 200000, 'eps': 1e-7}
            value = task.solve(solver=args.solver, warm_start=True, **opts)
            target = 2*math.sqrt(8+2*alpha*alpha)+eta
            minimum = min(float(np.linalg.eigvalsh(m.value)[0]) for m in mats)
            common = y.value[0]+y.value[1]
            sums = max(float(np.max(np.abs(y.value[2*x]+y.value[2*x+1]-common)))
                       for x in range(1, 4))
            record = {'alpha': alpha, 'eta': eta, 'level': args.level,
                      'status': task.status, 'value': value, 'quantum': target,
                      'gap_numerical': value-target, 'min_Gram_eigenvalue': minimum,
                      'max_common_sum_residual': sums,
                      'seconds': perf_counter()-start,
                      'interpretation': 'numerical evidence only; not an exact bound'}
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(__file__).with_name(args.output).write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
