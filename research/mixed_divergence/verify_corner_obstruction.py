"""Exact symbolic check of a physical obstruction on the cubic corner path.

This verifies a necessary quantum-positivity condition, not an SOS upper
bound or a mixed-degree divergence claim. Uses SymPy, never an SDP solver.
"""
from pathlib import Path
import json
import sympy as sp
HERE=Path(__file__).resolve().parent
t,s,h=sp.symbols('t s h',real=True)
def reduce_scalar(p):return sp.rem(sp.expand(p),s*s-t*(1-t),s).expand()
def reduce_matrix(M):return M.applyfunc(reduce_scalar)
I=sp.eye(2);P=sp.diag(0,1);R=sp.Matrix([[t,s],[s,1-t]])
assert reduce_matrix(R*R-R)==sp.zeros(2)
X=sp.kronecker_product(P,I);Y=sp.kronecker_product(I,P)
U=sp.kronecker_product(R,I);V=sp.kronecker_product(I,R);J=sp.eye(4)
H0=(X-Y)**2+2*X*(J-V)+2*Y*(J-U)+U*V
A0=J-2*X;A1=J-2*U;B0=J-2*Y;B1=J-2*V
G=(1-t)*(A0+B0)-(sp.Rational(1,2)+h)*(A1+B1)+A0*B0+A0*B1+A1*B0-sp.Rational(1,2)*A1*B1
assert reduce_matrix(((sp.Rational(7,2)-2*t)*J-G)/2-(H0-t*(X+Y)+h*(J-U-V)))==sp.zeros(4)
psi=sp.Matrix([-(1-t)*(3+2*t),2*t*s,2*t*s,t*(3-2*t)])
def expectation(M):return sp.factor(reduce_scalar((psi.T*M*psi)[0]))
values={'H0':expectation(H0),'S':expectation(X+Y),'T':expectation(J-U-V)}
assert sp.expand(values['H0']-4*t**3*(3-2*t))==0
assert sp.expand(values['S']-2*t**2*(9-8*t))==0
assert sp.expand(values['T']-(3-4*t)**2)==0
gap=expectation(H0-t*(X+Y)+h*(J-U-V))
assert sp.expand(gap-(3-4*t)*(h*(3-4*t)-2*t**3))==0
assert reduce_matrix(U*V*psi)==sp.zeros(4,1)
out=dict(status='PASS',domain='0 < delta < 3/4; t=delta; s=sqrt(t(1-t))',
         exact_expectations={k:str(v) for k,v in values.items()},exact_gap=str(gap),
         necessary_condition='h >= 2 delta^3/(3-4 delta)',
         scope='Necessary for the proposed local target to be quantum-valid. No sufficient condition, fixed SOS degree, or mixed divergence is inferred.')
(HERE/'corner_obstruction_report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
