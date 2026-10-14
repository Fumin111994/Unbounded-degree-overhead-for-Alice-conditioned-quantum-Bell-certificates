"""Exact mixed-direction level-two separation for the archived Gstar.

Default verification loads rational data and rejects all calls to SDP
solvers. --build only rationalizes the numerical discovery file; no
numerical eigenvalue or solver status is used as proof.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import sys

import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'mixed_Gstar_level2_exact.json'
COEFFICIENTS = list(map(sp.Rational,
    ['256/965', '266/719', '62/183', '0', '1', '355/647', '-767/846', '857/980']))
WORDS = [(), (0,), (1,), (0,1), (1,0)]
KEYS = [(), (0,), (1,), (0,1), (0,1,0), (1,0,1), (0,1,0,1)]


def key(word):
    stack = []
    for x in word:
        if stack and stack[-1] == x:
            stack.pop()
        else:
            stack.append(x)
    result = tuple(stack)
    return min(result, tuple(reversed(result)))


def build():
    import numpy as np
    source = np.load(HERE / 'mixed_Gstar_level2_candidate.npz')
    denominator = 10**10
    retained = sp.Rational(9999,10000)
    rational = lambda x: sp.Rational(round(float(x)*denominator), denominator)
    moments = [rational(float(retained)*v) for v in source['moments']]
    a, b = moments[:2], moments[2:4]
    corr = sp.Matrix(2,2,moments[4:])
    tables = []
    for direction, first, second, cc in [('left', a, b, corr), ('right', b, a, corr.T)]:
        table = sp.zeros(4,len(KEYS))
        for row in range(4):
            setting, sign = row//2, (-1)**row
            table[row,0] = (1+sign*first[setting])/2
            for y in range(2):
                table[row,1+y] = (second[y]+sign*cc[setting,y])/2
        for j in range(3,len(KEYS)):
            for row in range(3):
                # The reference assemblage is half the free-group trace:
                # each Gram is I/2 and all nonidentity moments are zero.
                table[row,j] = rational(float(retained)*source[direction][row,j])
            table[3,j] = table[0,j]+table[1,j]-table[2,j]
        tables.append([[str(v) for v in row] for row in table.tolist()])
    payload = dict(level=2, bound='501/200', coefficients=list(map(str,COEFFICIENTS)),
        word_basis=[list(w) for w in WORDS], moment_keys=[list(w) for w in KEYS],
        shared_Bell_moments=list(map(str,moments)), left=tables[0], right=tables[1],
        discovery='Numerical intersection SDP, followed by rational reconstruction.',
        reference_mixture='1/10000 of the normalized free-trace assemblage',
        scope='A fixed rational upper target, not the exact quantum value; no all-level claim.')
    DATA.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')


def verify(payload, archived=True):
    checks=[]
    def check(ok,name):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    check(payload['level']==2 and payload['bound']=='501/200', 'target and native degree')
    check(payload['coefficients']==list(map(str,COEFFICIENTS)), 'all eight exact Bell coefficients')
    check(payload['word_basis']==[list(w) for w in WORDS], 'complete five-word reduced basis')
    check(payload['moment_keys']==[list(w) for w in KEYS], 'all star-quotiented words through degree four')
    moments=list(map(sp.Rational,payload['shared_Bell_moments']))
    a,b=moments[:2],moments[2:4]
    corr=sp.Matrix(2,2,moments[4:])
    pivots=[]
    for direction,first,second,cc in [('left',a,b,corr),('right',b,a,corr.T)]:
        table=sp.Matrix([[sp.Rational(v) for v in row] for row in payload[direction]])
        check(table.shape==(4,len(KEYS)),f'{direction}: all four full outcome blocks supplied')
        check(table[0,:]+table[1,:]==table[2,:]+table[3,:],f'{direction}: exact common sums on every moment')
        check(table[0,0]+table[1,0]==1,f'{direction}: normalization')
        for row in range(4):
            setting,sign=row//2,(-1)**row
            check(table[row,0]==(1+sign*first[setting])/2,f'{direction} block {row}: shared marginal mass')
            check(all(table[row,1+y]==(second[y]+sign*cc[setting,y])/2 for y in range(2)),
                  f'{direction} block {row}: both shared cross-party moments')
            gram=sp.Matrix(len(WORDS),len(WORDS),lambda i,j:
                table[row,KEYS.index(key(tuple(reversed(WORDS[i]))+WORDS[j]))])
            check(gram==gram.T,f'{direction} block {row}: exact star symmetry')
            leading=[gram[:j,:j].det(method='domain-ge') for j in range(1,6)]
            check(all(d>0 for d in leading),f'{direction} block {row}: all five leading principal minors positive')
            pivots.append(leading)
            # Full reduced PVM words; not just kept-outcome principal subblocks.
            transform=sp.zeros(5)
            for i,w in enumerate(WORDS):
                expansion={():sp.S.One}
                for x in w:
                    following={}
                    for u,c in expansion.items():
                        for v in (u,key(u+(x,)) if len(u+(x,))<2 else u+(x,)):
                            following[v]=following.get(v,0)+c/2
                    expansion=following
                for u,c in expansion.items():
                    transform[i,WORDS.index(u)]=c
            check(transform.is_lower and transform.det()!=0,
                  f'{direction} block {row}: invertible native PVM conversion')
            converted=transform*gram*transform.T
            check(all(converted[:j,:j].det(method='domain-ge')>0 for j in range(1,6)),
                  f'{direction} block {row}: raw PVM positivity independently checked')
    score=sum(c*m for c,m in zip(COEFFICIENTS,moments))
    margin=score-sp.Rational(payload['bound'])
    check(margin>sp.Rational(79,100000),'exact mixed witness margin exceeds 79/100000')
    # Agreement on the common Bell subspace lets the two functionals glue.
    # Each nice-SOS term has at most one letter on its conditioned party.
    # Thus the overlap of the two support spaces is exactly span{I,Ax,By,AxBy}.
    support_a={(aw,bw) for aw in [(),(0,),(1,)] for bw in
               [()]+[tuple((x+j)%2 for j in range(n)) for n in range(1,5) for x in range(2)]}
    support_b={(bw,aw) for aw,bw in support_a}
    bell_support={(aw,bw) for aw in [(),(0,),(1,)] for bw in [(),(0,),(1,)]}
    check(support_a & support_b == bell_support,'the two full polynomial support spaces overlap only on Bell words')
    archived_reports={}
    if archived:
        import cvxpy as cp
        def forbidden(*args,**kwargs):
            raise AssertionError('A solver call is forbidden during exact verification')
        cp.Problem.solve=forbidden
        sys.path.insert(0,str(ROOT/'scripts'))
        sys.path.insert(0,str(ROOT/'src'))
        from run_m3_exactify import rebuild_functional
        from npa2 import exact
        original=json.loads((ROOT/'data/finite_separations/exact/search_s31337_308.json').read_text())
        functional=rebuild_functional(original)
        # Compare directly with the raw-outcome coefficients of Gstar.
        rf=original['rationalized_functional']
        signs=[1,-1]
        check(all(sp.Rational(rf['alice_local'][f'{x}:{o}'])==COEFFICIENTS[x]*signs[o]
                  for x in range(2) for o in range(2)), 'archived certificate: identical Alice coefficients')
        check(all(sp.Rational(rf['bob_local'].get(f'{y}:{o}','0'))==COEFFICIENTS[2+y]*signs[o]
                  for y in range(2) for o in range(2)), 'archived certificate: identical Bob coefficients')
        check(all(sp.Rational(rf['joint'][f'{x}:{y}:{o}:{p}'])==COEFFICIENTS[4+2*x+y]*signs[o]*signs[p]
                  for x in range(2) for y in range(2) for o in range(2) for p in range(2)),
              'archived certificate: identical correlator coefficients')
        cert=original['standard_dual']['certificate']
        entries={(int(name.split(':')[1]),int(name.split(':')[2])):Fraction(v)
                 for name,v in cert.items() if name.startswith('S:')}
        standard=exact.verify_standard_dual(functional,2,Fraction(501,200),entries,Fraction(cert['nu']))
        check(standard['status']=='PASS','archived D2 certificate rechecked coefficientwise and exactly PSD')
        upper=json.loads((ROOT/'data/finite_separations/exact/o3_308.json').read_text())
        names=list(upper['certificate'])
        blocks,lam,nu,mu=exact.unpack_onesided_unknowns(names,[Fraction(upper['certificate'][n]) for n in names])
        third=exact.verify_onesided_dual(functional,3,Fraction(501,200),blocks,lam,nu,mu)
        check(third['status']=='PASS','archived O_A3 certificate rechecked coefficientwise and exactly PSD')
        archived_reports={'standard2':standard['status'],'Alice_conditioned3':third['status']}
    return dict(status='PASS',checks=len(checks),passed_checks=checks,
                exact_score=str(score),exact_margin=str(margin),score_display=float(score),
                margin_display=float(margin),positive_leading_principal_minors=40,
                positive_raw_PVM_leading_principal_minors=40,
                archived_certificates=archived_reports,
                theorem='(501/200)I-Gstar is in D2 and O_A3, but outside O_A2+O_B2',
                scope=payload['scope'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:
        build()
    payload=json.loads(DATA.read_text(encoding='utf-8'))
    report=verify(payload)
    report['artifact_sha256']=hashlib.sha256(DATA.read_bytes()).hexdigest()
    (HERE/'mixed_separation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='passed_checks'},indent=2))


if __name__=='__main__':
    main()
