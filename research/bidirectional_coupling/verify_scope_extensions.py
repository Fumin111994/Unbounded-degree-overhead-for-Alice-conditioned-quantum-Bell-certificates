"""Exact identities for mixed-direction membership and bridge-strength scaling.

No numerical optimization, sampled parameter tests, or floating-point
positivity decisions. Universal arguments are in the research proof fragments.
"""
from pathlib import Path
import json

import sympy as sp

from verify_bidirectional_coupling import add, scale, multiply


def verify():
    checks=[]
    def check(ok,name):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    def zero(expr,name):
        check(sp.factor(expr)==0,name)
    unit={((),()):sp.S.One}
    a=[{((i,),()):sp.S.One} for i in range(2)]
    b=[{((),(i,)):sp.S.One} for i in range(2)]
    alpha,q,c=sp.symbols('alpha q c',real=True)
    relation=q*q-8-2*alpha*alpha
    v=add(scale(unit,q/2),scale(b[0],1-alpha/2),scale(b[1],1+alpha/2),
          scale(add(multiply(b[0],b[1]),multiply(b[1],b[0])),(4-alpha*alpha)/(4*q)))
    blocks=[add(v,scale(b[0],-2),scale(b[1],-2)),v,
            add(scale(unit,q),scale(v,-1),scale(b[0],-alpha),scale(b[1],2)),
            add(scale(unit,q),scale(v,-1),scale(b[0],2-alpha))]
    decomposition={}
    for i,p in enumerate(blocks):
        projector=scale(add(unit,scale(a[i//2],(-1)**i)),sp.Rational(1,2))
        decomposition=add(decomposition,multiply(projector,p))
    target=add(scale(unit,q),scale(b[0],-alpha),
               scale(multiply(a[0],add(b[0],b[1])),-1),
               scale(multiply(a[1],add(b[0],scale(b[1],-1))),-1))
    check(not add(decomposition,scale(target,-1)),'opposite-tilt full free-algebra identity')
    v0=q/2+(4-alpha**2)*c/(2*q)
    vz=1-alpha/2+(1+alpha/2)*c
    vx=1+alpha/2
    reps=[(v0,vz-2-2*c,vx-2),(v0,vz,vx),
          (q-v0,-vz-alpha+2*c,-vx+2),(q-v0,-vz+2-alpha,-vx)]
    for i,(s,z,x) in enumerate(reps):
        residual=s*s-z*z-x*x*(1-c*c)-c*c*(4-alpha**2)**2/(4*q*q)
        numerator=sp.together(residual).as_numer_denom()[0]
        zero(sp.rem(sp.expand(numerator),relation,q),f'opposite block {i}: exact determinant for every Jordan parameter')
    zero(sp.rem(q*q-(4-alpha**2)-(4+3*alpha**2),relation,q),
         'positive scalar-part bound for every Jordan block')
    for conditioned,other,name in [(a[0],b[0],'Alice'),(b[0],a[0],'Bob')]:
        summed={}
        for sign in (-1,1):
            f=add(unit,scale(other,-sign))
            summed=add(summed,scale(multiply(add(unit,scale(conditioned,sign)),multiply(f,f)),sp.Rational(1,4)))
        expected=add(unit,scale(multiply(a[0],b[0]),-1))
        check(not add(summed,scale(expected,-1)),f'bridge: {name}-conditioned degree-one SOS identity')
    e,r,eta=sp.symbols('e r eta',positive=True)
    aa=2-e
    w=aa+2+2*r*e
    zero((8+2*aa*aa)-(aa+2)**2-e**2,'strict subcritical quantum advantage identity')
    zero(w-q-2*eta*r-(2*r*(e-eta)-(q-(aa+2))),
         'optimized product-lift upper gap equals 2r(e-eta)-(q-alpha-2)')
    # t -> Q(t) is 1-Lipschitz because every tilted marginal is in [-1,1].
    t=sp.symbols('t',real=True)
    zero((8+2*t*t)-(2*t)**2-2*(4-t*t),
         'tilted support derivative at most one on [0,2]')
    zero(sp.sqrt(8+2*2**2)-(2+2),'two tilted-support branches agree at 2')

    # Improved product lift at a variable bridge ratio eta=c*epsilon.
    z,C=sp.symbols('z C',positive=True)
    rr=z*z/(8+2*z*z)
    ee=C*z*z
    effective=2-ee+c*ee*(1-2*rr)
    delta=4-ee+2*rr*ee+sp.sqrt(8+2*effective**2)-2*sp.sqrt(8+2*(2-ee)**2)-c*ee
    coefficient=sp.series(delta,z,0,5).removeO().expand().coeff(z,4)
    zero(coefficient-C*(2*(1-c)-(1+2*c-c*c)*C)/8,
         'all-ratio optimized witness asymptotic coefficient')
    zero((2-effective)/ee-(1-c+2*c*rr),
         'effective tilt remains below 2 for every 0<c<1')
    zero((1+2*c-c*c)-(2-(1-c)**2),'positive denominator for every 0<c<1')
    zero((2*(1-c)/(1+2*c-c*c)).subs(c,sp.Rational(1,8))-sp.Rational(112,79),
         'improved liminf constant squared at c=1/8 is 112/79')
    zero((2*(1-c)/(1+2*c-c*c)).subs(c,sp.Rational(9,10))-sp.Rational(20,199),
         'liminf constant squared at c=9/10 is 20/199')

    bias=sp.symbols('bias',real=True)
    first_order_bias=1-(1-c)*e/4
    stationarity=2*bias/sp.sqrt(2-bias*bias)-(2-e)-c*e*bias
    zero(sp.diff(stationarity,bias).subs({e:0,bias:1})-4,
         'independent-resource optimum is nondegenerate for all fixed ratios')
    zero(sp.series(stationarity.subs(bias,first_order_bias),e,0,2).removeO(),
         'independent-resource optimum bias is 1-(1-c)e/4+O(e squared)')
    independent=2*(2-e)*first_order_bias+4*sp.sqrt(2-first_order_bias**2)+c*e*first_order_bias**2
    beta=2*sp.sqrt(8+2*(2-e)**2)+c*e
    loss=sp.series(beta-independent,e,0,3).removeO().expand()
    zero(loss-c*(2-c)*e*e/4,'independent-resource penalty equals c(2-c)e squared/4 at leading order')
    advantage=sp.series(beta-(8-2*e+c*e),e,0,3).removeO()
    zero(advantage-e*e/4,'full nonlocal advantage equals e squared/4 at leading order')
    zero((c*(2-c)).subs(c,sp.Rational(9,10))-sp.Rational(99,100),
         'c=9/10 accounts for 99/100 of nonlocal advantage at critical scale')
    return dict(status='PASS',checks=len(checks),passed_checks=checks,
                mixed_gap_membership='O_A1 + O_B1 for all nonnegative eta',
                product_lift_no_go='for the fixed Fejer first assemblage, epsilon <= eta cannot separate',
                bridge_ratio_range='0 < c < 1, eta=c*epsilon',
                improved_liminf='sqrt(2*(1-c)/(1+2*c-c**2))',
                independent_penalty_fraction='c*(2-c)',
                limitations=['No fixed-positive-eta all-level theorem',
                             'The weakly coupled family has mixed degree one',
                             'No compiled-protocol security consequence is established'])


if __name__=='__main__':
    report=verify()
    Path(__file__).with_name('scope_extensions_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='passed_checks'},indent=2))
