"""Solver-free exact quantum-target interval separation from mixed degree two.

This verifier imports only Python standard-library modules and SymPy.
It reconstructs full noncommutative identities independently of the
search code and verifies all positive definiteness with exact minors.
"""
from pathlib import Path
import json

import sympy as sp

from verify_bidirectional_coupling import add,scale,multiply
from verify_mixed_separation import verify as verify_shared_witness

HERE=Path(__file__).resolve().parent


def verify():
    checks=[]
    def check(ok,name):
        if not ok:raise AssertionError(name)
        checks.append(name)
    unit={((),()):sp.S.One}
    alice=[{((x,),()):sp.S.One} for x in range(2)]
    bob=[{((),(y,)):sp.S.One} for y in range(2)]
    atoms=alice+bob+[multiply(a,b) for a in alice for b in bob]
    base=list(map(sp.Rational,['256/965','266/719','62/183','0','1','355/647','-767/846','857/980']))
    def dagger(poly):
        return {(a[::-1],b[::-1]):c for (a,b),c in poly.items()}
    def gram_sos(matrix,factors):
        result={}
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result=add(result,scale(multiply(dagger(factors[i]),factors[j]),matrix[i,j]))
        return result
    def positive(matrix,name):
        check(matrix==matrix.T,name+': Hermitian rational Gram')
        check(all(matrix[:j,:j].det(method='domain-ge')>0 for j in range(1,matrix.rows+1)),
              name+': positive definite by exact leading principal minors')
    def target(t):
        coefficients=[t*c if i<4 else c for i,c in enumerate(base)]
        return add(scale(unit,sum(coefficients)),*[scale(p,-c) for p,c in zip(atoms,coefficients)])
    lower=sp.Rational(101,100);upper=sp.Rational(51,50)
    certificates=json.loads((HERE/'quantum_target_family_exact.json').read_text())['certificates']
    for record in certificates:
        t=sp.Rational(record['t'])
        factors=[add({(tuple(w[0]),tuple(w[1])):sp.S.One},scale(unit,-1)) for w in record['basis']]
        matrix=sp.Matrix([[sp.Rational(c) for c in row] for row in record['gram']])
        check(max(len(a)+len(b) for f in factors for a,b in f)<=2,f't={t}: standard factors have degree at most two')
        positive(matrix,f't={t}: standard face')
        check(not add(gram_sos(matrix,factors),scale(target(t),-1)),f't={t}: complete standard noncommutative coefficient identity')
        check(sum(base[i]*(t if i<4 else 1) for i in range(8))==sp.Rational(record['quantum_target']),
              f't={t}: deterministic all-plus strategy attains exactly the upper bound')
    check(any(sp.Rational(p['t'])==lower for p in certificates),'left endpoint standard certificate is supplied')

    upper_payload=json.loads((HERE/'quantum_conditioned_upper_exact.json').read_text())
    check(sp.Rational(upper_payload['t'])==lower and upper_payload['level']==3,'conditioned upper certificate at the left endpoint')
    nice={}
    for block in upper_payload['blocks']:
        factors=[]
        for w in block['words']:
            f={((),tuple(w)):sp.S.One}
            if block['subtract_identity']:f=add(f,scale(unit,-1))
            factors.append(f)
        matrix=sp.Matrix([[sp.Rational(c) for c in row] for row in block['gram']])
        positive(matrix,f"conditioned block {block['x']},{block['outcome']}")
        check(all(not a and len(b)<=3 for f in factors for a,b in f),'conditioned SOS factors contain only Bob words through degree three')
        projector=scale(add(unit,scale(alice[block['x']],(-1)**block['outcome'])),sp.Rational(1,2))
        nice=add(nice,multiply(projector,gram_sos(matrix,factors)))
    check(not add(nice,scale(target(lower),-1)),'complete conditioned degree-three noncommutative identity')

    data=json.loads((HERE/'mixed_Gstar_level2_exact.json').read_text())
    witness_report=verify_shared_witness(data,archived=False)
    check(witness_report['status']=='PASS','shared full level-two assemblages pass exact verification')
    moments=list(map(sp.Rational,data['shared_Bell_moments']))
    t=sp.symbols('t',real=True)
    beta=sum(base[i]*(t if i<4 else 1) for i in range(8))
    score=sum(base[i]*moments[i]*(t if i<4 else 1) for i in range(8))
    gap=sp.expand(score-beta)
    check(sp.degree(gap,t)==1,'witness excess is affine in the parameter')
    check(sp.diff(gap,t)<0,'witness excess decreases along the entire parameter interval')
    endpoint=gap.subs(t,upper)
    check(endpoint>sp.Rational(59,100000),'uniform mixed degree-two excess exceeds 59/100000')
    # Exact affine transport supplies continuum certificates, not sampling.
    transport=add(*[scale(add(unit,scale(atoms[i],-1)),base[i]) for i in range(4)])
    check(not add(target(t),scale(target(lower),-1),scale(transport,-(t-lower))),
          'symbolic all-parameter transport identity')
    check(all(c>=0 for c in base[:4]),'transport is a nonnegative sum of elementary single-party gaps')
    check(all(-1<=m<=1 for m in moments),'all shared Bell moments lie in the physical coordinate range')
    return dict(status='PASS',checks=len(checks),passed_checks=checks,
        interval=[str(lower),str(upper)],quantum_bound=str(beta),
        deterministic_attainment=True,quantum_equals_local=True,
        exact_standard_degree=2,exact_mixed_degree=3,
        uniform_witness_gap=str(endpoint),uniform_witness_gap_display=float(endpoint),
        shared_witness_checks=witness_report['checks'],
        whole_ray_upper='D2 and O_A3 for every t>=101/100, by exact affine transport',
        limitations=['No degree-divergence theorem','No quantum violation on this interval'])


if __name__=='__main__':
    report=verify()
    (HERE/'quantum_target_family_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='passed_checks'},indent=2))
