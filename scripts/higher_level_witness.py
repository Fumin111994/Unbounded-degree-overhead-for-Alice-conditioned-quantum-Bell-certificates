"""Exact higher-level witnesses against a universal one-extra-level conversion.

Search uses an independently built involution-word SDP for conditioning.
Acceptance reconstructs and verifies the original projector-word hierarchy.
Stored rational data and exact PSD/sign checks are the proof, not solver values.
"""
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import sys

import cvxpy as cp
import numpy as np
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from npa2 import exact
from run_m6_closure import exact_tilted_functional
from standard_tilted_sos import verify_stored

ARTIFACTS=ROOT/'artifacts/prl/higher_level'


def reduced(word):
    result=[]
    for letter in word:
        if result and result[-1]==letter:result.pop()
        else:result.append(letter)
    return tuple(result)


def words(level):
    return [()]+[tuple((first+i)%2 for i in range(n))
                 for n in range(1,level+1) for first in (0,1)]


def key(left,right):
    w=reduced(tuple(reversed(left))+right)
    return min(w,tuple(reversed(w)))


def projector_transform(level):
    """P-word = T times U-word, with P_y=(I+U_y)/2."""
    uw=words(level)
    pw=exact_tilted_functional(Fraction(0)).scenario.algebra().words(level,party='B')
    T=sp.zeros(len(pw),len(uw))
    for i,word in enumerate(pw):
        expansion={():sp.Rational(1)}
        for _,y,_ in word:
            nxt={}
            for w,c in expansion.items():
                for v in [w,reduced(w+(y,))]:nxt[v]=nxt.get(v,0)+c/2
            expansion=nxt
        for w,c in expansion.items():T[i,uw.index(w)]=c
    return T


def numeric_problem(alpha,level):
    uw=words(level);n=len(uw)
    matrices=[cp.Variable((n,n),symmetric=True) for _ in range(4)]
    constraints=[]
    reps={}
    for i,left in enumerate(uw):
        for j in range(i,n):reps.setdefault(key(left,uw[j]),(i,j))
    for M in matrices:
        constraints.append(M >> 0)
        for i,left in enumerate(uw):
            for j in range(i,n):
                a,b=reps[key(left,uw[j])]
                if (i,j)!=(a,b):constraints.append(M[i,j]==M[a,b])
    constraints.extend([matrices[0]+matrices[1]==matrices[2]+matrices[3],
                        matrices[0][0,0]+matrices[1][0,0]==1])
    objective=float(alpha)*(matrices[0][0,0]-matrices[1][0,0])
    for x in (0,1):
        for y in (0,1):
            objective+=(-1 if x==y==1 else 1)*(matrices[2*x][0,1+y]-matrices[2*x+1][0,1+y])
    return cp.Problem(cp.Maximize(objective),constraints),matrices,reps


def payload_from_blocks(alpha,level,blocks):
    T=projector_transform(level);entries={}
    for b,G in enumerate(blocks):
        M=T*G*T.T
        for i in range(M.rows):
            for j in range(i,M.cols):entries[f'{b//2}:{b%2}:{i}:{j}']=str(M[i,j])
    return {'schema_version':1,'alpha':str(alpha),'level':level,
            'involution_grams':[[[str(v) for v in row] for row in G.tolist()] for G in blocks],
            'block_entries':entries,'gap_lower_bound':str(sp.Rational(1,1000 if level==3 else 10000))}


def analytic_blocks(level):
    """Closed-form truncated positive functionals on Z_2 * Z_2.

    The word ball is a path with n=2k+1 vertices. Subtracting r vv^T
    saturates the rank-one PSD bound for I + adjacency(path)/2.
    """
    if not isinstance(level,int) or level<1:
        raise ValueError('A positive integer level is required')
    uw=words(level);n=len(uw)
    C=[sp.Matrix(n,n,lambda i,j:int(key(uw[i],uw[j])==(y,))) for y in (0,1)]
    v=sp.Matrix([(-1)**len(w) for w in uw])
    r=sp.Rational(6,n*(n+1)*(n+2))
    M=sp.eye(n)+(C[0]+C[1])/2
    return [M-r*v*v.T,r*v*v.T,(sp.eye(n)+C[0])/2,(sp.eye(n)+C[1])/2]


def analytic_certificate(level):
    n=2*level+1;r=sp.Rational(6,n*(n+1)*(n+2))
    p=payload_from_blocks(2-8*r,level,analytic_blocks(level))
    # w^2-q^2=64 r^2(1-2r)^2 and w+q<8. This deliberately
    # conservative rational lower bound is rechecked by squaring.
    p['gap_lower_bound']=str(4*r**2*(1-2*r)**2)
    p['construction']='canonical-trace path family'
    p['r']=str(r)
    return p


def fejer_blocks(level):
    """Fejer-weighted functionals giving the stronger r=1/(8k^2+2).

    H_k=(sum_{j=0}^{k-1} U^j)^*(sum_{j=0}^{k-1} U^j)/k,
    U=B0 B1. H_k is central and positive, so tau(H_k .) is a trace.
    The paper proves positivity at every k; matrices here are assembled
    from the quotient functional, independently of the Toeplitz proof.
    """
    if not isinstance(level,int) or level<1:
        raise ValueError('A positive integer level is required')
    uw=words(level); n=len(uw)
    def trace(word):
        word=reduced(word)
        if len(word)%2:return sp.Rational(0)
        return max(sp.Rational(0),1-sp.Rational(len(word),2*level))
    def Ly(y,word):return (trace(word)+trace((y,)+word))/2
    blocks=[sp.Matrix(n,n,lambda i,j:Ly(y,tuple(reversed(uw[i]))+uw[j]))
            for y in (0,1)]
    v=sp.Matrix([(-1)**len(w) for w in uw])
    atom=v*v.T/(8*level*level+2)
    return [blocks[0]+blocks[1]-atom,atom,*blocks]


def fejer_certificate(level):
    r=sp.Rational(1,8*level*level+2)
    p=payload_from_blocks(2-8*r,level,fejer_blocks(level))
    p.update(construction='Fejer-weighted moving-average family',r=str(r),
             gap_lower_bound=str(4*r**2*(1-2*r)**2))
    return p


def verify_fejer_family():
    """Exact symbolic proof identities and independent full-PVM audits.

    Moving-average Gram positivity and the alternating inverse identity
    are proved for arbitrary k in the SM. Finite checks are not a proof
    by sampling. This routine never calls an optimization solver.
    """
    k,j=sp.symbols('k j',integer=True,positive=True)
    r,e=sp.symbols('r e',real=True)
    N=2*k
    # At the endpoint, pair j=2l-1,2l and include j=2k-1.
    paired=-(1-(2*j-1)/N)+(1-2*j/N)
    alternating_endpoint=sp.summation(paired,(j,1,k-1))-1/N
    residual=(4-(1-2*r)*e)**2-(8+2*(2-e)**2)
    identities=[alternating_endpoint+sp.Rational(1,2),
                N+1+2*N*alternating_endpoint-1,
                -sp.Rational(2)*2*N/N+4,
                2*(N+1)+(N-1)*2*N-(8*k*k+2),
                residual-e*(16*r-(1+4*r-4*r*r)*e),
                residual.subs(e,8*r)-64*r*r*(1-2*r)**2,
                (16*r/(1+4*r-4*r*r)).subs(r,sp.Rational(1,34))-sp.Rational(68,161)]
    failures=[f'Symbolic identity {i+1} failed'
              for i,value in enumerate(identities) if sp.simplify(value)!=0]
    records=[]
    for level in [1,2,3,4,5,8]:
        p=fejer_certificate(level)
        result=verify_certificate(p)
        if result['status']!='PASS':failures.append(result['failures'])
        rr=sp.Rational(p['r'])
        if result.get('g')!=str(2+4*rr) or result.get('h')!=str(1-2*rr):
            failures.append(f'Objective formula failed at level {level}')
        # Independent interval-indicator Gram and explicit inverse vector.
        path=[tuple((0+j)%2 for j in range(t)) for t in range(level,0,-1)]
        path += [()]+[tuple((1+j)%2 for j in range(t)) for t in range(1,level+1)]
        order=[words(level).index(w) for w in path]
        b=fejer_blocks(level)
        M=(b[0]+b[1]).extract(order,order); NN=2*level
        A=sp.Matrix(NN+1,2*NN,lambda i,j:int(bool(i<=j<i+NN)))
        v=sp.Matrix([(-1)**i for i in range(NN+1)])
        z=sp.Matrix([(-1)**i*(NN+1 if i in (0,NN) else 2*NN) for i in range(NN+1)])
        if M!=A*A.T/NN or M*z!=v or v.dot(z)!=1/rr:
            failures.append(f'Moving-average or inverse identity failed at level {level}')
        records.append(result)
    standard=verify_stored()
    if standard['status']!='PASS':failures.append('Full-parameter standard SOS failed')
    return {'status':'FAIL' if failures else 'PASS','failures':failures,
            'symbolic_identities_checked':len(identities),
            'universal_positivity_proof':'Interval-indicator Gram and alternating inverse identity',
            'finite_cross_checks_only':[1,2,3,4,5,8],
            'r_formula':'1/(8*k**2+2)','proved_growth_exponent':'1/2',
            'level_two_endpoint_interval':'(254/161,2)',
            'standard_symbolic_sos':standard['status'],'records':records}


def verify_analytic_family():
    """Symbolic identities for all k, plus original-PVM cross-checks.

    Universal positivity follows from the path-Laplacian argument in
    the paper; finitely many matrix checks are explicitly only audits.
    No numerical optimization is used to construct this family.
    """
    n,j=sp.symbols('n j',integer=True,positive=True)
    r,e=sp.symbols('r e',real=True)
    x=lambda t:t*(n+1-t)
    w=4-(1-2*r)*e
    residual=w**2-(8+2*(2-e)**2)
    identities=[x(j)-(x(j-1)+x(j+1))/2-1,
                sp.summation(x(j),(j,1,n))-n*(n+1)*(n+2)/6,
                residual-e*(16*r-(1+4*r-4*r**2)*e),
                residual.subs(e,8*r)-64*r**2*(1-2*r)**2]
    failures=[]
    if x(0)!=0 or x(n+1)!=0:failures.append('Dirichlet boundary identity failed')
    for i,value in enumerate(identities):
        if sp.simplify(value)!=0:failures.append(f'Symbolic identity {i+1} failed')
    records=[]
    for level in [1,2,3,4,5,8]:
        p=analytic_certificate(level)
        report=verify_certificate(p)
        if report['status']!='PASS':failures.append(report['failures'])
        rr=sp.Rational(p['r'])
        if report.get('g')!=str(2+4*rr) or report.get('h')!=str(1-2*rr):
            failures.append(f'Objective formula failed at level {level}')
        records.append(report)
    standard=verify_stored()
    if standard['status']!='PASS':failures.append('Full-parameter standard SOS failed')
    return {'status':'FAIL' if failures else 'PASS','failures':failures,
            'symbolic_identities_checked':len(identities),
            'universal_positivity_proof':'Dirichlet path Laplacian and rank-one Schur complement',
            'finite_cross_checks_only':[1,2,3,4,5,8],
            'standard_symbolic_sos':standard['status'],'records':records}


def verify_certificate(payload,require_separation=True):
    try:
        if payload['schema_version']!=1:raise ValueError('Unsupported schema')
        level=int(payload['level']);alpha=sp.Rational(payload['alpha']);n=2*level+1
        if not 1<=level<=12 or not 0<=alpha<2:raise ValueError('Invalid level or tilt')
        required={f'{x}:{a}:{i}:{j}' for x in (0,1) for a in (0,1)
                  for i in range(n) for j in range(i,n)}
        if set(payload['block_entries'])!=required:raise ValueError('Missing or extra moment entries')
        blocks=[sp.Matrix([[sp.Rational(v) for v in row] for row in G]) for G in payload['involution_grams']]
        if len(blocks)!=4:raise ValueError('Four involution blocks required')
        T=projector_transform(level)
        if T.shape!=(n,n) or T.det()==0:raise ValueError('Singular basis transformation')
        entries={};dets=[]
        for b,G in enumerate(blocks):
            if G.shape!=(n,n) or G!=G.T or not exact.is_psd_exact(G):
                raise ValueError('Involution Gram is not symmetric PSD')
            dets.append(str(G.det()))
            M=T*G*T.T
            entries[(b//2,b%2)]={}
            for i in range(n):
                for j in range(i,n):
                    value=sp.Rational(payload['block_entries'][f'{b//2}:{b%2}:{i}:{j}'])
                    if value!=M[i,j]:raise ValueError('Projector moment fails exact congruence')
                    entries[(b//2,b%2)][(i,j)]=Fraction(value)
        f=exact_tilted_functional(Fraction(alpha))
        report=exact.verify_onesided_witness(f,level,entries,enforce_probability_positivity=True)
        if report['status']!='PASS':raise ValueError('Original PVM verification failed: '+str(report['failures']))
        w=sp.Rational(report['objective_exact']);qsq=8+2*alpha**2
        h=blocks[0][0,0]-blocks[1][0,0]
        g=sum((-1 if x==y==1 else 1)*(blocks[2*x][0,1+y]-blocks[2*x+1][0,1+y])
              for x in (0,1) for y in (0,1))
        if w!=g+alpha*h:raise ValueError('Observable and projector objectives disagree')
        delta=sp.Rational(payload['gap_lower_bound'])
        if require_separation and not (delta>0 and w-delta>0 and (w-delta)**2>qsq):
            raise ValueError('Strict quantum separation is not proved')
        return {'status':'PASS','failures':[],'level':level,'alpha':str(alpha),
                'objective_exact':str(w),'g':str(g),'h':str(h),'quantum_bound_squared':str(qsq),
                'gap_lower_bound':str(delta),'gap_float_illustration':float(w-sp.sqrt(qsq)),
                'positive_definite_blocks':sum(bool(sp.Rational(d)>0) for d in dets),
                'original_pvm_moment_equalities_checked':True,
                'minimum_joint_probability':report['min_probability_exact']}
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        return {'status':'FAIL','failures':[str(exc)]}


def generate(alpha,level,denominator=10**9,epsilon=sp.Rational(1,100000)):
    problem,matrices,reps=numeric_problem(alpha,level)
    value=problem.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,
                        tol_feas=1e-11,max_iter=500)
    if any(M.value is None for M in matrices):raise RuntimeError('No numerical witness')
    raw=[M.value for M in matrices]
    def rounded(v):return sp.Rational(round(float(v)*denominator),denominator)
    vals=[{} for _ in range(4)]
    for k,(i,j) in reps.items():
        total=sp.Rational(1) if not k else rounded(raw[0][i,j]+raw[1][i,j])
        for x in (0,1):
            vals[2*x][k]=rounded(raw[2*x][i,j])
            vals[2*x+1][k]=total-vals[2*x][k]
    uw=words(level);n=len(uw)
    blocks=[(1-epsilon)*sp.Matrix(n,n,lambda i,j:vals[b][key(uw[i],uw[j])])+epsilon*sp.eye(n)/2
            for b in range(4)]
    payload=payload_from_blocks(alpha,level,blocks)
    with patch.object(cp.Problem,'solve',side_effect=RuntimeError('SDP disabled during acceptance')):
        check=verify_certificate(payload)
    if check['status']!='PASS':raise RuntimeError(str(check))
    payload['search']={'basis':'reduced involutions','solver_status':problem.status,
                       'value_float':float(value),'rounding_denominator':denominator,'trace_mixture':str(epsilon)}
    return payload,check


def verify_all():
    records=[]
    for level in [3,4]:
        p=json.loads((ARTIFACTS/f'alpha_9_5_L{level}.json').read_text(encoding='utf-8'))
        if p['alpha']!='9/5' or p['level']!=level:
            return {'status':'FAIL','failures':['Wrong mandatory witness parameter']}
        records.append(verify_certificate(p))
    standard=verify_stored()
    failures=[r.get('failures',[]) for r in records+[standard] if r['status']!='PASS']
    return {'status':'FAIL' if failures else 'PASS','failures':failures,
            'levels_checked':[3,4],'standard_symbolic_sos':standard['status'],'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate',action='store_true');p.add_argument('--alpha',default='9/5')
    p.add_argument('--analytic',action='store_true',help='Check the all-level analytic family without solving an SDP')
    p.add_argument('--fejer',action='store_true',help='Check the stronger square-root growth theorem without solving an SDP')
    p.add_argument('--level',type=int,default=3);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.analytic or args.fejer:
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('SDP disabled')):
            report=verify_fejer_family() if args.fejer else verify_analytic_family()
        if args.output and report['status']=='PASS':
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    elif args.generate:
        payload,report=generate(sp.Rational(args.alpha),args.level)
        target=args.output or ARTIFACTS/f'alpha_{args.alpha.replace("/","_")}_L{args.level}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    else:
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('SDP disabled')):report=verify_all()
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
