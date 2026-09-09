"""Generate a rational-function SOS certifying tilted-CHSH level-three closure.

Numerics choose quadratic free coordinates. The independent interval
verifier checks the original identity and whole-interval PSD before writing.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import cvxpy as cp
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npa2 import exact
from run_m6_closure import exact_tilted_functional
from quantum_interval import verify_certificate


def generate():
    u=sp.Symbol('u')
    c=(u*u-4*u+8)/(u*u-8)
    alpha=-2*(u*u-8*u+8)/(u*u-8)
    f=exact_tilted_functional(Fraction(0))
    words=f.scenario.algebra().words(3,party='B')
    # Similarity by diag(1,d), d^2=1-c^2, eliminates all square roots.
    projectors=[sp.Matrix([[1+c,1],[1-c*c,1-c]])/2,
                sp.Matrix([[1+c,-1],[-(1-c*c),1-c]])/2]
    vectors=[sp.Matrix([1,0]),sp.Matrix([0,1]),
             sp.Matrix([1,c-alpha/2]),sp.Matrix([1,-c+alpha/2])]
    kernels=[]
    for vector in vectors:
        columns=[]
        for word in words:
            matrix=sp.eye(2)
            for _,y,_ in word:matrix=(matrix*projectors[y]).applyfunc(sp.cancel)
            columns.append((matrix*vector).applyfunc(sp.cancel))
        V=sp.Matrix.hstack(*columns)
        K=sp.zeros(7,5)
        K[:2,:]=(-V[:,:2].inv()*V[:,2:]).applyfunc(sp.cancel)
        K[2:,:]=sp.eye(5)
        if (V*K).applyfunc(sp.cancel)!=sp.zeros(2,5):raise RuntimeError('Kernel identity failed')
        kernels.append(K)
    names,full,b0=exact.onesided_identity_system(f,3,Fraction(0),False)
    _,_,bq=exact.onesided_identity_system(f,3,Fraction(1),False)
    _,_,ba=exact.onesided_identity_system(exact_tilted_functional(Fraction(1)),3,Fraction(0),False)
    rhs=(b0+4*c*(bq-b0)+alpha*(ba-b0)).applyfunc(sp.cancel)
    _,pivlam=full[:,112:-1].rref()
    selected=list(range(112))+[112+j for j in pivlam]+[len(names)-1]
    nv=60+len(pivlam)+1
    transform=sp.zeros(len(selected),nv)
    hpos=[(i,j) for i in range(5) for j in range(i,5)]
    spos=[(i,j) for i in range(7) for j in range(i,7)]
    for block,K in enumerate(kernels):
        for h,(i,j) in enumerate(hpos):
            unit=sp.zeros(5);unit[i,j]=unit[j,i]=1
            gram=(K*unit*K.T).applyfunc(sp.cancel)
            for ind,(r,s) in enumerate(spos):transform[28*block+ind,15*block+h]=gram[r,s]
    transform[112:,60:]=sp.eye(len(pivlam)+1)
    A=(full[:,selected]*transform).applyfunc(sp.cancel)
    reduced,pivots=A.row_join(rhs).to_DM().rref()
    reduced=reduced.to_Matrix()
    if nv in pivots:raise RuntimeError('Parametric affine identity is inconsistent')
    free=[i for i in range(nv) if i not in pivots]
    base,null=sp.zeros(nv,1),sp.zeros(nv,len(free))
    for row,pivot in enumerate(pivots):
        base[pivot]=reduced[row,nv]
        for j,index in enumerate(free):null[pivot,j]=-reduced[row,index]
    for j,index in enumerate(free):null[index,j]=1
    lo,hi=sp.Rational(43,10),sp.Rational(447,100)
    center,radius=(lo+hi)/2,(hi-lo)/2
    coordinates=cp.Variable((len(free),3));margin=cp.Variable()
    constraints=[cp.norm(coordinates,'fro')<=100]
    for point in [lo+(hi-lo)*sp.Rational(j,8) for j in range(9)]:
        z=float((point-center)/radius)
        values=coordinates@np.array([1,z,z*z])
        x=np.array(base.subs(u,point),dtype=float).ravel()+np.array(null.subs(u,point),dtype=float)@values
        for block in range(4):
            H=cp.bmat([[x[15*block+hpos.index((min(i,j),max(i,j))) ] for j in range(5)] for i in range(5)])
            constraints.append(H-margin*np.eye(5)>>0)
    problem=cp.Problem(cp.Maximize(margin),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300)
    if coordinates.value is None:raise RuntimeError('No numerical candidate; not a nonexistence proof')
    rational=sp.Matrix([[sp.Rational(int(round(float(v)*10**8)),10**8) for v in row]
                        for row in coordinates.value])
    free_values=rational*sp.Matrix([1,(u-center)/radius,((u-center)/radius)**2])
    x=(base+null*free_values).applyfunc(sp.cancel)
    Hs=[sp.Matrix(5,5,lambda i,j:x[15*block+hpos.index((min(i,j),max(i,j)))]) for block in range(4)]
    original=(transform*x).applyfunc(sp.cancel)
    D=u*(u-4)*(u-2)*(u*u-8)**4*(u*u-4*u+8)**3
    encode=lambda matrix:[[str(sp.cancel(v)) for v in row] for row in matrix.tolist()]
    payload={'schema_version':1,'parameter':'u','parameter_interval':[str(lo),str(hi)],
             'alpha':str(sp.cancel(alpha)),'q':str(sp.cancel(4*c)),
             'positive_denominator':str(D),'level':3,
             'K':[encode(K) for K in kernels],
             'H_numerators':[encode(H.applyfunc(lambda z:sp.cancel(D*z))) for H in Hs],
             'names':[names[j] for j in selected],'u':[str(v) for v in original],
             'construction':{'free_coordinates':len(free),'polynomial_degree_in_normalized_parameter':2,
                             'center':str(center),'search_solver':'CLARABEL','search_status':problem.status,
                             'free_coordinate_grid':'1/100000000',
                             'search_margin_numerical':float(margin.value)},
             'sharp_interval':['13/10','3/2']}
    report=verify_certificate(payload)
    if report['status']!='PASS':raise RuntimeError(f'Exact interval verification failed: {report}')
    return payload,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/prl/interval')
    args=parser.parse_args()
    payload,report=generate()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'level3_interval.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
