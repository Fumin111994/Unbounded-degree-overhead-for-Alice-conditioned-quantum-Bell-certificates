"""Numerical path search only; no closure or quantum-optimum certificates."""
from pathlib import Path
import json
import math

import numpy as np
from scipy.optimize import differential_evolution

from probe_mixed_directions import mixed_problem, standard_problem


def qubit(theta):
    z=np.diag([1.,-1.]); x=np.array([[0.,1.],[1.,0.]]); eye=np.eye(2)
    def value(angles):
        a=[z,np.cos(angles[0])*z+np.sin(angles[0])*x]
        b=[z,np.cos(angles[1])*z+np.sin(angles[1])*x]
        h=sum(theta[i]*np.kron(a[i],eye)+theta[i+2]*np.kron(eye,b[i]) for i in range(2))
        h+=sum(theta[4+2*i+j]*np.kron(a[i],b[j]) for i in range(2) for j in range(2))
        return -np.linalg.eigvalsh(h)[-1]
    result=differential_evolution(value,[(0,math.pi)]*2,seed=72,tol=1e-11,popsize=12,maxiter=300,polish=True)
    searched=-float(result.fun)
    deterministic=local(theta)
    # A heuristic angle search can miss the endpoint optimum. Always retain
    # the explicit deterministic lower bound; [] denotes that branch.
    return (deterministic,[]) if deterministic>searched+1e-10 else (max(searched,deterministic),result.x.tolist())


def local(theta):
    import itertools
    return max(sum(theta[i]*v[i] for i in range(4))+
        sum(theta[4+2*i+j]*v[i]*v[2+j] for i in range(2) for j in range(2))
        for v in itertools.product((-1,1),repeat=4))


def main():
    base=np.array([256/965,266/719,62/183,0,1,355/647,-767/846,857/980])
    models={k:mixed_problem(k) for k in (2,3,4,6)}
    std,st=standard_problem()
    opts=dict(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
    records=[]
    for t in [0.,.5,1.,1.2,1.4,1.6,1.8,2.,2.2,2.4,2.6,2.8,3.,3.5,4.]:
        theta=base.copy();theta[:4]*=t
        q,angles=qubit(theta)
        st.value=theta
        sv=std.solve(**opts)
        row=dict(t=t,local_bound=local(theta),qubit_search_lower_bound=q,angles=angles,
                 standard2=sv,standard2_status=std.status,standard2_minus_qubit=sv-q,
                 interpretation='Numerical candidate screening; qubit search is not a certified quantum upper bound.')
        for k,(p,par,m,y,zz,grams,keys,rkeys) in models.items():
            par.value=theta
            v=p.solve(**opts)
            row[f'mixed{k}']=dict(value=v,minus_qubit=v-q,status=p.status,
                min_Gram_eigenvalue=min(float(np.linalg.eigvalsh(g.value)[0]) for g in grams))
        records.append(row)
        print(json.dumps(row),flush=True)
        Path(__file__).with_name('parameter_path_probe.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__': main()
