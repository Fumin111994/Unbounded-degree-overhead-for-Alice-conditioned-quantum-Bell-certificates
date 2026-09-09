"""Exact verification of a continuous level-three tilted-CHSH certificate.

Checks the ORIGINAL free-moment identity and Bernstein matrix positivity
on a whole rational parameter interval. No sampling or SDP solver is used.
"""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import sys
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npa2 import exact
from run_m6_closure import exact_tilted_functional

U=sp.Symbol('u')
ARTIFACT=ROOT/'artifacts/prl/interval/level3_interval.json'


def load_certificate():
    return json.loads(ARTIFACT.read_text(encoding='utf-8'))


def _expression(raw):
    value=sp.sympify(raw,locals={'u':U})
    if value.free_symbols-set([U]) or value.has(sp.Float):
        raise ValueError('Require exact rational functions in u')
    numerator,denominator=sp.cancel(value).as_numer_denom()
    sp.Poly(numerator,U,domain=sp.QQ)
    sp.Poly(denominator,U,domain=sp.QQ)
    return sp.cancel(value)


def _matrix(raw,shape):
    result=sp.Matrix([[_expression(x) for x in row] for row in raw])
    if result.shape!=shape:raise ValueError('Incorrect matrix shape')
    return result


def _bernstein_coefficients(poly,lo,hi,n):
    # Exact conversion of p(lo+(hi-lo)*z) to the degree-n Bernstein basis.
    power=[poly.nth(i) for i in range(n+1)]
    shifted=[(hi-lo)**j*sum(power[i]*sp.binomial(i,j)*lo**(i-j)
                            for i in range(j,n+1)) for j in range(n+1)]
    return [sum(shifted[j]*sp.binomial(k,j)/sp.binomial(n,j)
                for j in range(k+1)) for k in range(n+1)]


def check_no_poles(expressions,lo,hi):
    denominators=set()
    for expression in expressions:
        denominator=sp.cancel(expression).as_numer_denom()[1]
        denominators.add(sp.Poly(denominator,U,domain=sp.QQ).monic().as_expr())
    for denominator in denominators:
        poly=sp.Poly(denominator,U,domain=sp.QQ)
        coefficients=_bernstein_coefficients(poly,lo,hi,poly.degree())
        if not (all(v>0 for v in coefficients) or all(v<0 for v in coefficients)):
            raise ValueError('A certificate denominator is not certified nonzero on the interval')
    return len(denominators)


def verify_quantum_kernels(kernels,lo,hi):
    """Rebuild word maps and certify their rank and nullspaces on J."""
    if len(kernels)!=4:
        raise ValueError('Require four quantum kernel matrices')
    c=(U**2-4*U+8)/(U**2-8)
    alpha=-2*(U**2-8*U+8)/(U**2-8)
    projectors=[sp.Matrix([[1+c,1],[1-c*c,1-c]])/2,
                sp.Matrix([[1+c,-1],[-(1-c*c),1-c]])/2]
    vectors=[sp.Matrix([1,0]),sp.Matrix([0,1]),
             sp.Matrix([1,c-alpha/2]),sp.Matrix([1,-c+alpha/2])]
    words=exact_tilted_functional(Fraction(0)).scenario.algebra().words(3,party='B')
    minors=[]
    for K,v in zip(kernels,vectors):
        columns=[]
        for word in words:
            column=v
            for _,question,_ in reversed(word):
                column=(projectors[question]*column).applyfunc(sp.cancel)
            columns.append(column)
        V=sp.Matrix.hstack(*columns)
        minor=sp.factor(V[:,:2].det())
        if minor==0:raise ValueError('Quantum word map is rank deficient')
        # Absence of poles of 1/minor certifies absence of zeros of minor.
        check_no_poles(list(V)+[1/minor],lo,hi)
        if K.shape!=(7,5) or K[2:,:]!=sp.eye(5):
            raise ValueError('Invalid full-rank quantum kernel basis')
        if any(sp.cancel(value)!=0 for value in V*K):
            raise ValueError('Stored matrix does not span the quantum kernel')
        minors.append(str(minor))
    return {'quantum_word_maps_checked':4,'quantum_word_map_rank':2,
            'kernel_dimension':5,'rank_minor_factors':minors}


def check_bernstein_psd(numerators,lo,hi):
    checked=0
    positive_definite=0
    try:
        if not lo<hi:raise ValueError('Empty parameter interval')
        polynomials=[[sp.Poly(z,U,domain=sp.QQ) for z in N] for N in numerators]
        degree=max(p.degree() for P in polynomials for p in P)
        if degree>20:raise ValueError('Unexpected polynomial degree')
        for block,(N,P) in enumerate(zip(numerators,polynomials)):
            if N.shape!=(5,5) or N!=N.T:raise ValueError('Numerator is not symmetric 5 by 5')
            coefficients=[_bernstein_coefficients(p,lo,hi,degree) for p in P]
            for j in range(degree+1):
                B=sp.Matrix(5,5,lambda r,c:coefficients[5*r+c][j])
                checked+=1
                if not exact.is_psd_exact(B):
                    return {'status':'FAIL','failures':[f'Bernstein matrix {block}:{j} is not PSD'],
                            'bernstein_matrices_checked':checked}
                # For a PSD matrix, nonsingularity is equivalent to PD.
                if B.to_DM().det()!=0:positive_definite+=1
        return {'status':'PASS','failures':[],'polynomial_degree':int(degree),
                'bernstein_matrices_checked':checked,
                'bernstein_positive_definite_matrices':positive_definite}
    except (ValueError,TypeError,sp.PolynomialError) as exc:
        return {'status':'FAIL','failures':[str(exc)],'bernstein_matrices_checked':checked}


def verify_certificate(payload):
    failures=[]
    try:
        if payload['schema_version']!=1 or payload['level']!=3 or payload['parameter']!='u':
            raise ValueError('Unsupported interval certificate')
        lo,hi=map(sp.Rational,payload['parameter_interval'])
        if not 4<lo<hi:raise ValueError('Require 4 < lower < upper')
        c=(U**2-4*U+8)/(U**2-8)
        alpha=-2*(U**2-8*U+8)/(U**2-8)
        q=4*c
        if sp.cancel(_expression(payload['alpha'])-alpha)!=0:
            raise ValueError('Incorrect tilt parameterization')
        if sp.cancel(_expression(payload['q'])-q)!=0:
            raise ValueError('Incorrect exact quantum bound')
        if sp.cancel(q*q-8-2*alpha*alpha)!=0:
            raise ValueError('Quantum-bound relation fails')
        # alpha'(u)=-16*((u-2)^2+4)/(u^2-8)^2 < 0 for u>4.
        low_alpha,high_alpha=sp.cancel(alpha.subs(U,hi)),sp.cancel(alpha.subs(U,lo))
        if not 0<low_alpha<=sp.Rational(13,10)<sp.Rational(3,2)<=high_alpha<2:
            raise ValueError('Certified interval does not contain [13/10,3/2]')
        D=U*(U-4)*(U-2)*(U**2-8)**4*(U**2-4*U+8)**3
        if sp.cancel(_expression(payload['positive_denominator'])-D)!=0:
            raise ValueError('Incorrect positive denominator')
        # Every displayed factor of D is strictly positive on u>4.
        if len(payload['K'])!=4 or len(payload['H_numerators'])!=4:
            raise ValueError('Require four kernel and numerator matrices')
        kernels=[_matrix(m,(7,5)) for m in payload['K']]
        numerators=[_matrix(m,(5,5)) for m in payload['H_numerators']]
        for K in kernels:
            if K[2:,:]!=sp.eye(5):raise ValueError('Kernel basis lacks the claimed full rank')
        kernel_report=verify_quantum_kernels(kernels,lo,hi)
        f0=exact_tilted_functional(Fraction(0))
        f1=exact_tilted_functional(Fraction(1))
        all_names,A,b0=exact.onesided_identity_system(f0,3,Fraction(0),False)
        _,_,bq=exact.onesided_identity_system(f0,3,Fraction(1),False)
        _,_,ba=exact.onesided_identity_system(f1,3,Fraction(0),False)
        names=payload['names'];raw=payload['u']
        if len(names)!=len(raw) or len(names)!=len(set(names)):
            raise ValueError('Certificate name/value mismatch')
        required={name for name in all_names if name.startswith('S_') or name=='nu'}
        if not required<=set(names)<=set(all_names):
            raise ValueError('Missing or unrecognized identity coefficient')
        values={name:_expression(value) for name,value in zip(names,raw)}
        denominators_checked=check_no_poles(
            list(values.values())+[z for K in kernels for z in K],lo,hi)
        vector=sp.Matrix([values.get(name,0) for name in all_names])
        residual=A*vector-b0-q*(bq-b0)-alpha*(ba-b0)
        if any(sp.cancel(r)!=0 for r in residual):
            raise ValueError('Original free-moment identity fails')
        blocks,_,_,_=exact.unpack_onesided_unknowns(names,[values[name] for name in names])
        for block,key in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            S=sp.Matrix(7,7,lambda i,j:blocks[key][(min(i,j),max(i,j))])
            difference=S-kernels[block]*numerators[block]*kernels[block].T/D
            if any(sp.cancel(r)!=0 for r in difference):
                raise ValueError('Stored Gram is not K N K^T / D')
        positivity=check_bernstein_psd(numerators,lo,hi)
        if positivity['status']!='PASS':return positivity
        return {'status':'PASS','failures':[],
                **kernel_report,
                'parameter_interval':[str(lo),str(hi)],
                'alpha_interval':[str(low_alpha),str(high_alpha)],
                'closure_interval_contains':['13/10','3/2'],
                'identity_coefficients_checked':A.rows,
                'bernstein_matrices_checked':positivity['bernstein_matrices_checked'],
                'bernstein_positive_definite_matrices':positivity['bernstein_positive_definite_matrices'],
                'identity_verification_field':'QQ(u); zero numerator polynomial for each free-moment coefficient',
                'polynomial_degree':positivity['polynomial_degree'],
                'distinct_denominators_checked':denominators_checked,
                'scope':'Level-three exact closure; sharp cost also uses independently checked standard SOS and level-two tangent fan.'}
    except (KeyError,ValueError,TypeError,IndexError,sp.PolynomialError) as exc:
        failures.append(str(exc))
    return {'status':'FAIL','failures':failures}


def verify_all():
    return verify_certificate(load_certificate())


if __name__=='__main__':
    from unittest.mock import patch
    import cvxpy as cp
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with patch.object(cp.Problem,'solve',side_effect=RuntimeError('SDP calls disabled')):
        result=verify_all()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
