"""Symbolic audit of the relative-accuracy corollary; no SDP optimization."""
from pathlib import Path
from itertools import product
import json
import sympy as S

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def exact(expr,name):
    assert S.simplify(expr)==0, name
    checks.append(name)

eps,c,r,z,C,rho=S.symbols('eps c r z C rho',positive=True)
q=lambda x:S.sqrt(8+2*x*x)
alpha=2-eps
h=1-2*r
t=alpha+c*eps*h
V=2*(q(alpha)-alpha-2)
D=4-h*eps+q(t)-(2*q(alpha)+c*eps)
x=S.symbols('x',real=True)
exact(q(2-x)-(4-x)-x*x/(q(2-x)+4-x),'rationalized quantum-gap identity')
exact(S.expand(4-h*eps+(t+2)-2*(alpha+2)-c*eps-2*(1-c)*r*eps),
      'linear part of stable witness identity')
exact(S.limit(V/eps**2,eps,0)-S.Rational(1,4),'advantage quadratic coefficient')
exact(S.diff(D,eps).subs(eps,0)-2*(1-c)*r,'fixed-level linear witness coefficient')

# Parameter z=1/k turns the joint limit into an ordinary Taylor coefficient.
joint=D.subs({eps:C*z*z,r:z*z/(8+2*z*z)},simultaneous=True)
leading=S.expand(S.series(joint,z,0,5).removeO()).coeff(z,4)
a=1+2*c-c*c
exact(leading-C*(2*(1-c)-a*C)/8,'joint inverse-degree asymptotic coefficient')
ratio=(1-c)/C-a/2
exact(leading/(C*C/4)-ratio,'witness error on the advantage scale')
threshold=2*(1-c)/(a+2*rho)
exact(ratio.subs(C,threshold)-rho,'relative-tolerance threshold')
exact(S.diff(ratio,C)+(1-c)/C**2,'strictly decreasing threshold function')

# Bell score is bilinear in (eps,c); checking rectangle vertices proves
# the proposed local upper bound on 0<=eps<=2 and 0<=c<=1.
for values in product([-1,1],repeat=8):
    A0,A1,A2,A3,B0,B1,B2,B3=values
    F=alpha*A0+A0*(B0+B1)+A1*(B0-B1)
    T=alpha*B2+A2*(B2+B3)+A3*(B2-B3)
    deficit=S.expand(2*(alpha+2)+c*eps-(F+T+c*eps*A0*B2))
    assert all(deficit.subs({eps:u,c:v})>=0 for u,v in product([0,2],[0,1]))
checks.append('local bound on all 256 deterministic assignments throughout the parameter rectangle')
exact(2*(alpha+2)+c*eps-(alpha+2+alpha+2+c*eps),'attaining local assignment')

words={()}
front={()}
for k in range(1,6):
    front={w+(j,) for w in front for j in range(4) if not w or j!=w[-1]}
    words|=front
    assert len(words)==2*3**k-1
checks.append('four-involution conditional word count through levels 1 to 5')
standard={()}|{(i,) for i in range(8)}
for i,j in product(range(8),repeat=2):
    if i==j: word=()
    elif i>=4 and j<4: word=(j,i)
    else: word=(i,j)
    standard.add(word)
assert len(standard)==49
checks.append('49 independent standard words through total degree two')
assert S.Rational(19,10)>30/S.sqrt(287)
checks.append('figure epsilon range satisfies the degree-upper-bound hypothesis')
report={'status':'PASS','checks':checks,
        'scope':'Symbolic identities and constants supporting Supplementary Corollary on relative accuracy; the text provides the analytic proof.'}
(ROOT/'relative_precision_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
