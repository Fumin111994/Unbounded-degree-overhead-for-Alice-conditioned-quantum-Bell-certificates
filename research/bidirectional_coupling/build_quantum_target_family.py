"""Discover a local-attained exact D2 certificate on the Gstar tilt path.

Numerical optimization is only the search step. The output is accepted
only after rational coefficient matching and exact positive definiteness.
"""
from pathlib import Path
import json
import itertools

import cvxpy as cp
import numpy as np
import sympy as sp

from probe_minimal_symmetric import ball,red

HERE=Path(__file__).resolve().parent
BASE=list(map(sp.Rational,['256/965','266/719','62/183','0','1','355/647','-767/846','857/980']))


def canonical(w):
    a,b=w
    return min(w,(a[::-1],b[::-1]))


def product(u,v):
    return canonical((red(u[0][::-1]+v[0]),red(u[1][::-1]+v[1])))


def system():
    unit=((),())
    words=[(a,b) for a in ball(2) for b in ball(2) if len(a)+len(b)<=2]
    nonunit=words[1:]
    pairs=list(itertools.combinations_with_replacement(range(len(nonunit)),2))
    columns=[]
    keys={}
    for i,j in pairs:
        p={}
        for u,v in [(nonunit[i],nonunit[j])]+([] if i==j else [(nonunit[j],nonunit[i])]):
            for w,c in [(product(u,v),1),(canonical((u[0][::-1],u[1][::-1])),-1),
                        (canonical(v),-1),(unit,1)]:
                p[w]=p.get(w,0)+c
        columns.append(p)
        for w in p:
            keys.setdefault(w,len(keys))
    A=sp.zeros(len(keys),len(pairs))
    for j,p in enumerate(columns):
        for w,c in p.items():A[keys[w],j]=c
    rows=list(A.T.rref()[1])
    return words,pairs,keys,A,rows


def main():
    words,pairs,keys,A,rows=system()
    n=len(words)-1
    reduced=A[rows,:]
    correction=reduced.T*(reduced*reduced.T).inv()
    atoms=[((x,),()) for x in range(2)]+[((),(y,)) for y in range(2)]
    atoms += [((x,),(y,)) for x in range(2) for y in range(2)]
    witness=json.loads((HERE/'mixed_Gstar_level2_exact.json').read_text())
    moments=list(map(sp.Rational,witness['shared_Bell_moments']))
    out=[]
    for t in map(sp.Rational,['101/100','203/200','51/50']):
        theta=[t*x if i<4 else x for i,x in enumerate(BASE)]
        target=sum(theta)
        rhs=sp.zeros(len(keys),1);rhs[keys[((),())]]=target
        for w,c in zip(atoms,theta):rhs[keys[w]]-=c
        matrix=cp.Variable((n,n),symmetric=True);margin=cp.Variable()
        vector=cp.hstack([matrix[i,j] for i,j in pairs])
        problem=cp.Problem(cp.Maximize(margin),
            [matrix-margin*np.eye(n)>>0,np.array(reduced,dtype=float)@vector==np.array(rhs[rows,:],dtype=float).ravel()])
        value=problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
        print(json.dumps(dict(t=str(t),minimum_face_eigenvalue_search=value,status=problem.status)),flush=True)
        if value<1e-8:continue
        rounded=sp.Matrix([sp.Rational(round(float(matrix.value[i,j])*10**9),10**9) for i,j in pairs])
        exact_vector=rounded-correction*(reduced*rounded-rhs[rows,:])
        assert A*exact_vector==rhs
        exact_matrix=sp.zeros(n)
        for (i,j),v in zip(pairs,exact_vector):exact_matrix[i,j]=exact_matrix[j,i]=v
        minors=[exact_matrix[:j,:j].det(method='domain-ge') for j in range(1,n+1)]
        assert all(x>0 for x in minors)
        score=sum(c*m for c,m in zip(theta,moments))
        record=dict(t=str(t),quantum_target=str(target),coefficients=list(map(str,theta)),
            basis=[[[*a],[*b]] for a,b in words[1:]],basis_convention='factor i is W_i-I',
            gram=[[str(v) for v in row] for row in exact_matrix.tolist()],
            exact_leading_minors=list(map(str,minors)),mixed_witness_score=str(score),
            mixed_witness_gap=str(score-target),status='EXACT_PASS',
            numerical_search_margin=float(value))
        out.append(record)
        print(json.dumps({k:record[k] for k in ['t','quantum_target','mixed_witness_gap','status']}),flush=True)
    assert out
    (HERE/'quantum_target_family_exact.json').write_text(json.dumps(dict(certificates=out),indent=2)+'\n')


if __name__=='__main__':main()
