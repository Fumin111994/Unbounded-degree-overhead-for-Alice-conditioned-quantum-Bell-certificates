"""Exploratory mixed-direction SDP; floating-point results are not proofs.

The intersection uses common Bell moments for full Alice- and
Bob-conditioned assemblages. Each orientation retains all its own higher
moments. Its support function is dual to the sum of the two Bell-gap cones.
"""
from pathlib import Path
import json

import cvxpy as cp
import numpy as np

from probe_minimal_symmetric import ball, red


def conditioned_data(k):
    words = ball(k)
    keys = {}
    pos = []
    for u in words:
        for v in words:
            w = red(tuple(reversed(u))+v)
            w = min(w, tuple(reversed(w)))
            keys.setdefault(w, len(keys))
            pos.append(keys[w])
    y = cp.Variable((4, len(keys)))
    grams = [cp.reshape(y[i, pos], (len(words), len(words)), order='C')
             for i in range(4)]
    common = y[0]+y[1]
    constraints = [*(g >> 0 for g in grams), common == y[2]+y[3], common[keys[()]] == 1]
    am = cp.hstack([y[2*x, keys[()]]-y[2*x+1, keys[()]] for x in range(2)])
    bm = cp.hstack([common[keys[(b,)]] for b in range(2)])
    corr = cp.vstack([cp.hstack([y[2*x, keys[(b,)]]-y[2*x+1, keys[(b,)]]
                                 for b in range(2)]) for x in range(2)])
    return y, grams, constraints, am, bm, corr, keys


def mixed_problem(k):
    left = conditioned_data(k)
    right = conditioned_data(k)
    y, grams, constraints, a, b, c, keys = left
    z, h, other, rb, ra, rc, rkeys = right
    theta = cp.Parameter(8)
    moments = cp.hstack([a, b, cp.reshape(c, (4,), order='C')])
    problem = cp.Problem(cp.Maximize(theta @ moments),
                         constraints+other+[a == ra, b == rb, c == rc.T])
    return problem, theta, moments, y, z, grams+h, keys, rkeys


def standard_problem():
    words = [(a,b) for a in ball(2) for b in ball(2) if len(a)+len(b)<=2]
    keys, pos = {}, []
    for a,b in words:
        for c,d in words:
            w = (red(tuple(reversed(a))+c), red(tuple(reversed(b))+d))
            w = min(w, (tuple(reversed(w[0])), tuple(reversed(w[1]))))
            keys.setdefault(w, len(keys))
            pos.append(keys[w])
    y = cp.Variable(len(keys))
    gram = cp.reshape(y[pos], (len(words),len(words)), order='C')
    theta = cp.Parameter(8)
    entries = [((x,),()) for x in range(2)]+[((),(x,)) for x in range(2)]
    entries += [((x,),(b,)) for x in range(2) for b in range(2)]
    moments = y[[keys[w] for w in entries]]
    problem = cp.Problem(cp.Maximize(theta @ moments), [gram >> 0, y[keys[((),())]] == 1])
    return problem, theta


def main():
    mixed = {k: mixed_problem(k) for k in (1,2,3)}
    std, st = standard_problem()
    candidates = {
        'archived_Gstar': [256/965,266/719,62/183,0,1,355/647,-767/846,857/980],
        'symmetric_double_tilt_095': [.95,0,.95,0,1,1,1,-1],
        'no_Alice_marginal': [0,0,1,-3,7,14,11,-5],
        'tilted_19': [1.9,0,0,0,1,1,1,-1],
    }
    rng = np.random.default_rng(413)
    for i in range(20):
        candidates[f'random_{i}'] = rng.normal(size=8).tolist()
    records = []
    opts = dict(solver='CLARABEL', tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                tol_feas=1e-9, max_iter=150)
    for name, coefficients in candidates.items():
        st.value = coefficients
        standard = std.solve(**opts)
        row = dict(name=name, coefficients=coefficients, standard2=standard,
                   standard_status=std.status, interpretation='numerical screening only')
        for k, (problem, theta, moments, y, z, grams, keys, rkeys) in mixed.items():
            if k>1 and name.startswith('random_'):
                continue
            theta.value = coefficients
            val = problem.solve(**opts)
            row[f'mixed{k}'] = dict(value=val, minus_standard2=val-standard,
                status=problem.status,
                min_Gram_eigenvalue=min(float(np.linalg.eigvalsh(g.value)[0]) for g in grams))
            if name=='archived_Gstar' and k==1:
                np.savez(Path(__file__).with_name('mixed_Gstar_candidate.npz'),
                         left=y.value,right=z.value, moments=moments.value)
        records.append(row)
        print(json.dumps(row), flush=True)
    Path(__file__).with_name('mixed_direction_probe.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
