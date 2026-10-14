"""Search and exactly rationalize a tight conditioned degree-three certificate."""
from pathlib import Path
import itertools
import json

import cvxpy as cp
import numpy as np
import sympy as sp

from build_quantum_target_family import BASE,canonical
from probe_minimal_symmetric import ball,red

HERE=Path(__file__).resolve().parent


def system():
    blocks=[];columns=[];keys={}
    for x in range(2):
        for outcome in range(2):
            factors=[{w:sp.S.One} for w in ball(3)]
            if outcome==0:
                factors=[{w:sp.S.One,():-sp.S.One} for w in ball(3)[1:]]
            pairs=list(itertools.combinations_with_replacement(range(len(factors)),2))
            blocks.append(dict(x=x,outcome=outcome,factors=factors,pairs=pairs))
            for i,j in pairs:
                poly={}
                for ii,jj in [(i,j)]+([] if i==j else [(j,i)]):
                    for u,cu in factors[ii].items():
                        for v,cv in factors[jj].items():
                            bw=red(u[::-1]+v)
                            for aw,weight in [((),sp.Rational(1,2)),((x,),sp.Rational((-1)**outcome,2))]:
                                w=canonical((aw,bw));poly[w]=poly.get(w,0)+weight*cu*cv
                columns.append(poly)
                for w in poly:keys.setdefault(w,len(keys))
    A=sp.zeros(len(keys),len(columns))
    for j,p in enumerate(columns):
        for w,c in p.items():A[keys[w],j]=c
    rows=list(A.T.rref()[1])
    return blocks,keys,A,rows


def main():
    blocks,keys,A,rows=system()
    reduced=A[rows,:]
    t=sp.Rational(101,100)
    theta=[t*c if i<4 else c for i,c in enumerate(BASE)]
    rhs=sp.zeros(len(keys),1);rhs[keys[((),())]]=sum(theta)
    atoms=[((x,),()) for x in range(2)]+[((),(y,)) for y in range(2)]+[((x,),(y,)) for x in range(2) for y in range(2)]
    for w,c in zip(atoms,theta):rhs[keys[w]]-=c
    gamma=cp.Variable()
    mats=[cp.Variable((len(b['factors']),len(b['factors'])),symmetric=True) for b in blocks]
    vector=cp.hstack([m[i,j] for m,b in zip(mats,blocks) for i,j in b['pairs']])
    constraints=[m-gamma*np.eye(m.shape[0])>>0 for m in mats]
    constraints += [np.array(reduced,dtype=float)@vector==np.array(rhs[rows,:],dtype=float).ravel()]
    problem=cp.Problem(cp.Maximize(gamma),constraints)
    value=problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
    print(json.dumps(dict(status=problem.status,search_margin=value)),flush=True)
    assert value>1e-8
    rounded=sp.Matrix([sp.Rational(round(float(m.value[i,j])*10**9),10**9) for m,b in zip(mats,blocks) for i,j in b['pairs']])
    correction=reduced.T*(reduced*reduced.T).inv()
    exact=rounded-correction*(reduced*rounded-rhs[rows,:])
    assert A*exact==rhs
    output=[];offset=0
    for block in blocks:
        size=len(block['factors']);matrix=sp.zeros(size)
        for i,j in block['pairs']:
            matrix[i,j]=matrix[j,i]=exact[offset];offset+=1
        minors=[matrix[:j,:j].det(method='domain-ge') for j in range(1,size+1)]
        assert all(m>0 for m in minors)
        output.append(dict(x=block['x'],outcome=block['outcome'],
                           words=[list(w) for w in (ball(3)[1:] if block['outcome']==0 else ball(3))],
                           subtract_identity=block['outcome']==0,
                           gram=[[str(v) for v in row] for row in matrix.tolist()]))
    payload=dict(t=str(t),level=3,quantum_target=str(sum(theta)),coefficients=list(map(str,theta)),
                 blocks=output,status='EXACT_PASS',search_margin=float(value))
    (HERE/'quantum_conditioned_upper_exact.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('EXACT_PASS: all four rational Gram matrices are positive definite and the complete identity holds.')


if __name__=='__main__':main()
