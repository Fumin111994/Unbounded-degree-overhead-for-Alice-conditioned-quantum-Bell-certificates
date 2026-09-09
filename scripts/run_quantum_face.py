"""Generate exact optimal-bound certificates using quantum-strategy kernels.

Numerical optimization selects free coordinates; exact algebraic elimination
enforces the identity. Only independently verified certificates are written.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from npa2 import exact
from quantum_face import ARTIFACTS, strategy_data, verify_certificate


def generate(alpha, level, max_denominator=10000):
    alpha = sp.Rational(alpha)
    functional, q, maps, _, _ = strategy_data(alpha, level)
    kernels = [sp.Matrix.hstack(*V.nullspace()).applyfunc(sp.simplify) for V in maps]
    n, hdim = maps[0].cols, maps[0].cols-2
    hpos = [(i,j) for i in range(hdim) for j in range(i,hdim)]
    spos = [(i,j) for i in range(n) for j in range(i,n)]
    nh, ns = len(hpos), len(spos)
    names, full, b0 = exact.onesided_identity_system(functional, level, Fraction(0), False)
    _, _, b1 = exact.onesided_identity_system(functional, level, Fraction(1), False)
    _, pivlam = full[:,4*ns:-1].rref()
    selected = list(range(4*ns)) + [4*ns+j for j in pivlam] + [len(names)-1]
    multipliers = len(pivlam)+1
    nv = 4*nh+multipliers
    transform = sp.zeros(len(selected), nv)
    for block, K in enumerate(kernels):
        for h,(i,j) in enumerate(hpos):
            unit = sp.zeros(hdim)
            unit[i,j] = unit[j,i] = 1
            gram = (K*unit*K.T).applyfunc(sp.simplify)
            for index,(u,v) in enumerate(spos):
                transform[ns*block+index,nh*block+h] = gram[u,v]
    transform[4*ns:,4*nh:] = sp.eye(multipliers)
    A = (full[:,selected]*transform).applyfunc(sp.simplify)
    rhs = b0+q*(b1-b0)
    reduced, pivots = A.row_join(rhs).to_DM(extension=True).rref()
    reduced = reduced.to_Matrix().applyfunc(sp.simplify)
    if nv in pivots:
        raise RuntimeError('Optimal-face affine identity is inconsistent')
    free = [i for i in range(nv) if i not in pivots]
    base, null = sp.zeros(nv,1), sp.zeros(nv,len(free))
    for row,pivot in enumerate(pivots):
        base[pivot] = reduced[row,nv]
        for j,f in enumerate(free):
            null[pivot,j] = -reduced[row,f]
    for j,f in enumerate(free):
        null[f,j] = 1
    coordinates, margin = cp.Variable(len(free)), cp.Variable()
    numeric = np.array(base,dtype=float).ravel()+np.array(null,dtype=float)@coordinates
    constraints = [cp.norm(coordinates,2) <= 100]
    for block in range(4):
        H = cp.bmat([[numeric[block*nh+hpos.index((min(i,j),max(i,j)))]
                      for j in range(hdim)] for i in range(hdim)])
        constraints.append(H-margin*np.eye(hdim) >> 0)
    problem = cp.Problem(cp.Maximize(margin),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,
                  tol_feas=1e-10,max_iter=300)
    if coordinates.value is None:
        raise RuntimeError('No numerical candidate; this does not prove non-closure')
    rational = sp.Matrix([sp.Rational(float(v)).limit_denominator(max_denominator)
                          for v in coordinates.value])
    candidate = (base+null*rational).applyfunc(sp.simplify)
    grams = [sp.Matrix(hdim,hdim,lambda i,j:
                       candidate[block*nh+hpos.index((min(i,j),max(i,j)))])
             for block in range(4)]
    encode = lambda matrix: [[str(v) for v in row] for row in matrix.tolist()]
    payload = {
        'schema_version': 1, 'alpha': str(alpha), 'level': level, 'bound': str(q),
        'K': [encode(K) for K in kernels], 'H': [encode(H) for H in grams],
        'names': [names[j] for j in selected],
        'u': [str(v) for v in (transform*candidate).applyfunc(sp.simplify)],
        'affine_rank': len(pivots), 'free_dimension': len(free),
        'rho_numerical': float(margin.value),
    }
    report = verify_certificate(payload)
    if report['status'] != 'PASS':
        raise RuntimeError(f'Candidate failed exact verification: {report}')
    payload['status'] = 'PASS'
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', required=True)
    parser.add_argument('--level', type=int, choices=(2,3), required=True)
    parser.add_argument('--output', type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    certificate = generate(args.alpha,args.level)
    args.output.mkdir(parents=True,exist_ok=True)
    name = 'generic_face_'+certificate['alpha'].replace('/','_')+'_L'+str(args.level)+'.json'
    target = args.output/name
    target.write_text(json.dumps(certificate,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','file':str(target),'bound':certificate['bound']}))
