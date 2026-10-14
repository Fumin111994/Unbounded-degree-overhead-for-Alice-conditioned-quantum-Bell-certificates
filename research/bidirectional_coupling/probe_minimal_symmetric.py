"""Exploratory minimal-scenario search; no numerical equality proves exactness."""
from pathlib import Path
import itertools
import json
import math

import cvxpy as cp
import numpy as np
from scipy.optimize import differential_evolution


def red(w):
    out=[]
    for a in w:
        if out and out[-1]==a: out.pop()
        else: out.append(a)
    return tuple(out)


def ball(k):
    return [()]+[tuple((a+j)%2 for j in range(n)) for n in range(1,k+1) for a in (0,1)]


def std2():
    words=[(a,b) for a in ball(2) for b in ball(2) if len(a)+len(b)<=2]
    keys={}; pos=[]
    for a,b in words:
        for c,d in words:
            w=(red(tuple(reversed(a))+c),red(tuple(reversed(b))+d))
            w=min(w,(tuple(reversed(w[0])),tuple(reversed(w[1]))))
            keys.setdefault(w,len(keys)); pos.append(keys[w])
    y=cp.Variable(len(keys)); gram=cp.reshape(y[pos],(len(words),len(words)),order='C')
    alpha=cp.Parameter(nonneg=True); lam=cp.Parameter()
    score=alpha*(y[keys[((0,),())]]+y[keys[((),(0,))]])
    score+=lam*y[keys[((0,),(0,))]]+y[keys[((0,),(1,))]]+y[keys[((1,),(0,))]]-y[keys[((1,),(1,))]]
    p=cp.Problem(cp.Maximize(score),[gram>>0,y[keys[((),())]]==1])
    return p,alpha,lam


def conditioned(k):
    words=ball(k); keys={}; pos=[]
    for u in words:
        for v in words:
            w=red(tuple(reversed(u))+v); w=min(w,tuple(reversed(w)))
            keys.setdefault(w,len(keys)); pos.append(keys[w])
    y=cp.Variable((4,len(keys)))
    grams=[cp.reshape(y[i,pos],(len(words),len(words)),order='C') for i in range(4)]
    common=y[0]+y[1]
    alpha=cp.Parameter(nonneg=True); lam=cp.Parameter()
    score=alpha*(y[0,keys[()]]-y[1,keys[()]])+alpha*common[keys[(0,)]]
    coeff=[[lam,1],[1,-1]]
    for x in range(2):
        for b in range(2):score+=coeff[x][b]*(y[2*x,keys[(b,)]]-y[2*x+1,keys[(b,)]])
    p=cp.Problem(cp.Maximize(score),[*(g>>0 for g in grams),common==y[2]+y[3],common[keys[()]]==1])
    return p,alpha,lam


def qubit_search(alpha,lam):
    z=np.diag([1.,-1.]); x=np.array([[0.,1.],[1.,0.]]); eye=np.eye(2)
    const=alpha*(np.kron(z,eye)+np.kron(eye,z))+lam*np.kron(z,z)
    def objective(angles):
        u,v=angles
        a=np.cos(u)*z+np.sin(u)*x; b=np.cos(v)*z+np.sin(v)*x
        return -np.linalg.eigvalsh(const+np.kron(z,b)+np.kron(a,z)-np.kron(a,b))[-1]
    result=differential_evolution(objective,[(0,math.pi)]*2,seed=19,tol=1e-11,popsize=10,maxiter=200,polish=True)
    return -float(result.fun),result.x.tolist()


def main():
    models={'std2':std2(),'Acond1':conditioned(1),'Acond2':conditioned(2),'Acond4':conditioned(4)}
    records=[]
    for alpha,lam in itertools.product([0.8,0.95,0.99],[0.,1.,4.,16.]):
        q,angles=qubit_search(alpha,lam)
        row={'alpha':alpha,'lambda':lam,'qubit_search_lower_bound':q,'angles':angles,
             'scope':'numerical candidate screening only'}
        for name,(problem,a,l) in models.items():
            a.value=alpha;l.value=lam
            val=problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
            row[name]={'value':val,'minus_qubit_search':val-q,'status':problem.status}
        print(json.dumps(row),flush=True); records.append(row)
        Path(__file__).with_name('minimal_symmetric_probe.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__': main()
