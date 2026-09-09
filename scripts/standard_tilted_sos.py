"""Exact BP15 Eq. (27), its attaining strategy, and both-outcome AQ restriction.

The symbolic check covers 0 <= alpha < 2, with no SDP calls.
"""
from pathlib import Path
import argparse
import json
import sys
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npa2 import exact
from npa2.bell import Scenario

ARTIFACT=ROOT/'artifacts/prl/standard_tilted/family.json'


def verify_family(second_square_sign=1):
    algebra=Scenario(2,2,2,2).algebra()
    alpha=sp.Symbol('alpha',nonnegative=True)
    q=sp.sqrt(8+2*alpha**2)
    I=exact.sym_identity()
    add,scale=exact.sym_add,exact.sym_scale
    mul=lambda x,y:exact.sym_multiply(algebra,x,y)
    A0,A1=[exact.sym_observable(algebra,'A',x) for x in (0,1)]
    B0,B1=[exact.sym_observable(algebra,'B',y) for y in (0,1)]
    F=add(scale(A0,alpha),mul(A0,add(B0,B1)),mul(A1,add(B0,scale(B1,-1))))
    sprime=add(mul(A0,add(B0,scale(B1,-1))),mul(A1,add(B0,B1)))
    r1=add(scale(I,q),scale(F,-1))
    r2=add(scale(A1,second_square_sign*alpha),scale(sprime,-1))
    result=exact.verify_sos_identity(algebra,q,F,[(1/(2*q),r1),(1/(2*q),r2)])
    in_basis=all(sum(letter[0]=='A' for letter in word)<=1
                 and sum(letter[0]=='B' for letter in word)<=1
                 for poly in (r1,r2) for word in poly)
    c=q/4
    d=sp.sqrt(1-c*c)
    z=alpha/(2*c)
    rho=sp.Matrix([[(1+z)/2,0,0,d/(2*c)],[0,0,0,0],
                   [0,0,0,0],[d/(2*c),0,0,(1-z)/2]])
    Z=sp.diag(1,-1);X=sp.Matrix([[0,1],[1,0]])
    bell=alpha*sp.kronecker_product(Z,sp.eye(2))+2*c*sp.kronecker_product(Z,Z)+2*d*sp.kronecker_product(X,X)
    attained=(rho==rho.T and sp.simplify(sp.trace(rho)-1)==0
              and (rho*rho-rho).applyfunc(sp.simplify)==sp.zeros(4)
              and sp.simplify(sp.trace(rho*bell)-q)==0)
    result.update({'source':'Bamps and Pironio, PRA 91, 052111 (2015), Eq. (27)',
                   'range':'0 <= alpha < 2',
                   'squares_in_1_plus_AB':in_basis, 'attaining_strategy':attained,
                   'weight_positive':'1/(2*sqrt(8+2*alpha**2)) > 0',
                   'strategy_reality':'1-c^2=(4-alpha^2)/8 > 0 on the stated range'})
    if not in_basis or not attained:
        result['status']='FAIL'
        result['failures'].append('basis or attaining-strategy check failed')
    return result


def verify_aq_block_restriction():
    algebra=Scenario(2,2,2,2).algebra()
    I=exact.sym_identity()
    P=lambda party,question,outcome:exact.sym_projector(algebra,party,question,outcome)
    mul=lambda x,y:exact.sym_multiply(algebra,x,y)
    dag=lambda x:exact.sym_dagger(algebra,x)
    bob=[I,P('B',0,0),P('B',1,0)]
    polynomials=[I,P('A',0,0),P('A',1,0),bob[1],bob[2]]
    polynomials += [mul(P('A',x,0),bob[y+1]) for x in (0,1) for y in (0,1)]
    words=[next(iter(p)) for p in polynomials]
    symbols={}
    def moment(poly):
        value=0
        for word,coefficient in poly.items():
            if word not in symbols:symbols[word]=sp.Symbol('m'+str(len(symbols)))
            value+=coefficient*symbols[word]
        return value
    gamma=sp.Matrix(9,9,lambda i,j:moment(mul(dag(polynomials[i]),polynomials[j])))
    blocks=[];maps=[]
    for x in (0,1):
        for a in (0,1):
            Aa=P('A',x,a)
            rows=[mul(Aa,w) for w in bob]
            T=sp.Matrix([[poly.get(word,0) for word in words] for poly in rows])
            phi=T*gamma*T.T
            expected=sp.Matrix(3,3,lambda i,j:moment(mul(Aa,mul(dag(bob[i]),bob[j]))))
            assert (phi-expected).applyfunc(sp.expand)==sp.zeros(3)
            blocks.append(phi);maps.append(T.tolist())
    bob_gram=sp.Matrix(3,3,lambda i,j:moment(mul(dag(bob[i]),bob[j])))
    assert (blocks[0]+blocks[1]-bob_gram).applyfunc(sp.expand)==sp.zeros(3)
    assert (blocks[2]+blocks[3]-bob_gram).applyfunc(sp.expand)==sp.zeros(3)
    return {'status':'PASS','failures':[],'blocks_checked':4,
            'congruence_maps':[[[str(v) for v in row] for row in T] for T in maps],
            'basis':[algebra.word_label(w) for w in words]}


def verify_stored():
    stored=json.loads(ARTIFACT.read_text(encoding='utf-8'))
    report=verify_family()
    blocks=verify_aq_block_restriction()
    if stored.get('square_polys')!=report['square_polys']:
        report['status']='FAIL';report['failures'].append('stored BP15 squares differ from verified identity')
    if stored.get('aq_restriction')!=blocks:
        report['status']='FAIL';report['failures'].append('stored outcome congruences differ from verified maps')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    if args.generate:
        result=verify_family();result['aq_restriction']=verify_aq_block_restriction()
        if result['status']!='PASS':raise RuntimeError(result)
        ARTIFACT.parent.mkdir(parents=True,exist_ok=True)
        ARTIFACT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:
        from unittest.mock import patch
        import cvxpy as cp
        with patch.object(cp.Problem,'solve',side_effect=RuntimeError('SDP calls disabled')):
            result=verify_stored()
    print(json.dumps({'status':result['status'],'failures':result['failures'],
                      'squares_in_1_plus_AB':result['squares_in_1_plus_AB'],
                      'attaining_strategy':result['attaining_strategy']},indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
