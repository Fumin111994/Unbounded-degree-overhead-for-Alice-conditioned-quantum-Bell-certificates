"""Numerical saturation survey, explicitly not an exact closure certificate.

The full reduced involution-word quotient is assembled with shared moment
variables. This is equivalent to numeric_problem in higher_level_witness,
but avoids thousands of redundant scalar equality constraints.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import warnings
import cvxpy as cp
import numpy as np
import scipy.sparse as sparse
from higher_level_witness import words,key,numeric_problem

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/prl/fejer/numerical_saturation.json'


def problem_at_level(k):
    w=words(k); n=len(w)
    keys=list(dict.fromkeys(key(u,v) for u in w for v in w))
    ids={v:i for i,v in enumerate(keys)}; m=len(keys)
    indices=[ids[key(u,v)] for u in w for v in w]
    mapping=sparse.csr_matrix((np.ones(n*n),(np.arange(n*n),indices)),shape=(n*n,m))
    common=cp.Variable(m); positive=[cp.Variable(m),cp.Variable(m)]
    moment_vectors=[positive[0],common-positive[0],positive[1],common-positive[1]]
    grams=[cp.reshape(mapping@v,(n,n),order='C') for v in moment_vectors]
    constraints=[G >> 0 for G in grams]+[common[ids[()]]==1]
    alpha=cp.Parameter(nonneg=True)
    a,b0,b1=ids[()],ids[(0,)],ids[(1,)]
    h=2*positive[0][a]-common[a]
    g=2*(positive[0][b0]+positive[0][b1]+positive[1][b0]-positive[1][b1]-common[b0])
    problem=cp.Problem(cp.Maximize(alpha*h+g),constraints)
    return problem,alpha,grams,constraints,mapping,(a,b0,b1),m


def solve(model,epsilon,k,tolerance=1e-10):
    p,alpha,grams,constraints,mapping,ids,m=model
    alpha.value=2-epsilon
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        value=p.solve(solver='CLARABEL',tol_gap_abs=tolerance,tol_gap_rel=tolerance,
                      tol_feas=tolerance,max_iter=500)
    q=float(np.sqrt(8+2*alpha.value**2))
    record={'epsilon':epsilon,'alpha':float(alpha.value),'level':k,'status':p.status,
            'value':float(value),'quantum_value':q,'gap':float(value-q),
            'solve_seconds':time.perf_counter()-started,
            'warnings':[str(w.message) for w in caught]}
    if any(G.value is None for G in grams):raise RuntimeError(record)
    duals=[c.dual_value for c in constraints[:4]]
    y=float(constraints[-1].dual_value)
    aa,b0,b1=ids
    cc=np.zeros(m);cc[aa]=-alpha.value;cc[b0]=-2
    c0=np.zeros(m);c0[aa]=2*alpha.value;c0[b0]=c0[b1]=2
    c1=np.zeros(m);c1[b0]=2;c1[b1]=-2
    adj=lambda Z:np.asarray(mapping.T@np.asarray(Z).ravel(order='C')).ravel()
    e=np.zeros(m);e[aa]=1
    residual=[-cc+y*e-adj(duals[1]+duals[3]),-c0+adj(duals[1]-duals[0]),
              -c1+adj(duals[3]-duals[2])]
    record.update(dual_value=y,primal_dual_gap=float(y-value),
                  minimum_primal_eigenvalue=min(float(np.linalg.eigvalsh(G.value).min()) for G in grams),
                  minimum_dual_eigenvalue=min(float(np.linalg.eigvalsh(Z).min()) for Z in duals),
                  normalization_residual=float(constraints[-1].violation()),
                  maximum_dual_stationarity_residual=max(float(np.max(np.abs(r))) for r in residual))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--probe',nargs=2,type=float,metavar=('EPSILON','LEVEL'))
    parser.add_argument('--audit',action='store_true',help='Cross-check sensitive points with the four-matrix model')
    args=parser.parse_args()
    if args.audit:
        payload=json.loads(args.output.read_text(encoding='utf-8'))
        checks=[]
        for a,k in [(1.97,12),(1.97,13),(1.98,14),(1.98,15)]:
            p,matrices,_=numeric_problem(a,k)
            value=p.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,
                          tol_feas=1e-9,max_iter=500)
            record={'alpha':a,'level':k,'gap':float(value-np.sqrt(8+2*a*a)),
                    'status':p.status,'maximum_constraint_violation':
                    max(float(np.max(np.abs(c.violation()))) for c in p.constraints)}
            checks.append(record)
            print(json.dumps(record),flush=True)
        payload['sensitive_four_matrix_cross_checks']=checks
        args.output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
        return
    if args.probe:
        e,k=args.probe;k=int(k)
        print(json.dumps(solve(problem_at_level(k),e,k),indent=2))
        return
    threshold=1e-6
    models={};cache={};records=[];summary=[]
    def sample(e,k):
        if (e,k) not in cache:
            if k not in models:models[k]=problem_at_level(k)
            record=solve(models[k],e,k)
            cache[e,k]=record;records.append(record)
            print(json.dumps(record),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({'complete':False,'records':records},indent=2)+'\n',encoding='utf-8')
        return cache[e,k]
    for e in [.5,.3,.15,.08,.04,.03,.02]:
        lo,hi=1,20
        if sample(e,lo)['gap']<=threshold or sample(e,hi)['gap']>threshold:
            raise RuntimeError('Survey bracket failed')
        while hi-lo>1:
            mid=(lo+hi)//2
            if sample(e,mid)['gap']>threshold:lo=mid
            else:hi=mid
        before,after=sample(e,lo),sample(e,hi)
        summary.append({'epsilon':e,'alpha':2-e,'numerical_saturation_level':hi,
                        'gap_at_previous_level':before['gap'],'gap_at_reported_level':after['gap'],
                        'degree_times_sqrt_epsilon':hi*np.sqrt(e)})
    # Independently assemble the original four-matrix model at a modest size.
    reference,_,_=numeric_problem(1.7,3)
    ref=reference.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=500)
    check=sample(.3,3)
    difference=abs(ref-check['value'])
    if difference>1e-7:raise RuntimeError('Independent model cross-check failed')
    fit=np.polyfit(np.log([r['epsilon'] for r in summary]),
                   np.log([r['numerical_saturation_level'] for r in summary]),1)
    payload={'complete':True,'kind':'numerical tolerance crossing, not exact finite closure',
             'threshold':threshold,'solver':'CLARABEL','solver_tolerances':1e-10,
             'maximum_search_level':20,'summary':summary,'records':records,
             'log_log_fit_slope':float(fit[0]),'log_log_fit_prefactor':float(np.exp(fit[1])),
             'independent_four_matrix_model_value_difference':float(difference),
             'environment':{'python':platform.python_version(),'cvxpy':cp.__version__,'numpy':np.__version__}}
    args.output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'fit':fit.tolist(),'cross_check':difference},indent=2))


if __name__=='__main__':main()
