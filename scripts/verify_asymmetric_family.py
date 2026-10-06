"""Exact audits of uniform conditioned closure for asymmetric tilted CHSH.

The universal proof, including the critical endpoint, is in the supplement.
This checker verifies its symbolic identities and exact Bernstein positivity
ingredients. It neither samples a parameter grid nor invokes an SDP solver.
"""

import json
import sympy as sp


def verify():
    checks = []

    def zero(expr, name):
        if sp.factor(expr) != 0:
            raise AssertionError(name)
        checks.append(name)

    l, alpha, q, a, c = sp.symbols('l alpha q a c', positive=True)
    # Work modulo the exact quantum-value relation; denominators are positive.
    def quantum_zero(expr, name):
        numerator = sp.together(expr).as_numer_denom()[0]
        zero(sp.rem(sp.expand(numerator), q**2-(l*l+1)*(alpha*alpha+4), q), name)

    aq = l*q/(2*(l*l+1))
    b = alpha/(2*l)
    D = (c*c-a*a)**2
    H = l*(a+c*c/a)+alpha*(3*c-c**3/a**2)/(2*a)
    rp = (alpha*(c+2*a)-2*l*a*a)/(2*a**3*(c+a)**2)
    rm = (alpha*(c-2*a)-2*l*a*a)/(2*a**3*(c-a)**2)
    for sign in (-1, 1):
        zero(H.subs(c, sign*a)-(2*l*a+sign*alpha),
             f'Hermite contact value {sign}')
        zero(sp.diff(H, c).subs(c, sign*a)-sign*2*l,
             f'Hermite contact derivative {sign}')
    zero(alpha+2*l*c-H-D*rp, 'positive absolute-value remainder')
    zero(-alpha-2*l*c-H-D*rm, 'negative absolute-value remainder')
    zero(rp.subs(c, -b)+l/a**3, 'right remainder at kink')
    zero(rm.subs(c, -b)+l/a**3, 'left remainder at kink')
    zero(sp.diff(rp, c)-(-alpha*(c+a)+4*l*a*(a-b))/(2*a**3*(c+a)**3),
         'right derivative and denominator gap')
    zero(sp.diff(rm, c)+(alpha*(a-c)+4*l*a*(a+b))/(2*a**3*(a-c)**3),
         'left derivative and denominator gap')
    outer = (q-alpha-2*l*c)*(q+alpha-2*l*c)-4*(1-c*c)
    inner = (q-alpha-2*l*c)*(q-alpha+2*l*c)-4*(1-c*c)
    quantum_zero(outer-4*(l*l+1)*(c-aq)**2, 'outer determinant perfect square')
    zero(inner-((q-alpha)**2-4+4*(1-l*l)*c*c), 'inner determinant')
    zero(inner.subs(c,b)-outer.subs(c,b), 'determinant branches meet')
    quantum_zero(q*q-(alpha+2*l)**2-(2-l*alpha)**2,
                 'quantum violation numerator')
    quantum_zero(1-aq*aq-(4-l*l*alpha*alpha)/(4*(l*l+1)),
                 'distance of contact from spectral endpoint')
    zero(aq.subs({alpha:2/l,q:2*(l+1/l)})-1, 'critical contact a=1')
    zero((aq-b).subs({alpha:2/l,q:2*(l+1/l)})-(1-1/l**2),
         'critical kink-contact gap')
    # d/(1-a^2)^2 is bounded below even as both numerator and denominator vanish.
    zero(16*(l*l+1)**2/(16*4*(l+1/l))-l*(l*l+1)/4,
         'uniform endpoint-relative diagonal constant')
    u, v = sp.symbols('u v', nonnegative=True)
    zero(2*u*u+2*v*v-(u-v)**2-(u+v)**2,
         'squared-distance triangle inequality identity')
    z = sp.symbols('z', nonnegative=True)
    # For z=|c| in [0,1], 4(1-z)-(1-z^2)^2 >= 0 by nonnegative power factors.
    zero(4*(1-z)-(1-z*z)**2-(1-z)*(3-z+z*z+z**3),
         'endpoint distance bound factorization')
    zero(l*(l*l+1)/4-l/2-l*(l*l-1)/4,
         'diagonal lower bound coefficient for l>=1')
    # Bernstein coefficients on z in [0,1] prove the remaining cubic is >=0.
    poly = sp.Poly(3-z+z*z+z**3,z)
    bern = [sum(poly.nth(j)*sp.binomial(i,j)/sp.binomial(3,j)
                for j in range(i+1)) for i in range(4)]
    if bern != [3, sp.Rational(8,3), sp.Rational(8,3), 4]:
        raise AssertionError('endpoint-distance Bernstein positivity')
    checks.append('endpoint-distance Bernstein positivity')
    # Positive slack in da/dalpha < db/dalpha over 0<=alpha<=2/l.
    quantum_zero((q*q-l**4*alpha**2)
                 -(4*(l*l+1)+(l*l+1-l**4)*alpha**2),
                 'contact-gap monotonicity numerator')
    zero((4*(l*l+1)+(l*l+1-l**4)*alpha**2).subs(alpha,2/l)
         -4*(2+1/l**2), 'contact-gap monotonicity at critical tilt')
    # Attaining qubit state: Z_A, X_A; B_{0,1}=a Z +/- sqrt(1-a^2) X.
    r = q/(l*l+1)
    quantum_zero(alpha**2+4*(1-aq*aq)-r*r, 'Schmidt bias normalization')
    quantum_zero(2*l*aq+(alpha**2+4*(1-aq*aq))/r-q,
                 'attaining quantum value')
    gamma, eta, tau, dd, qp, qm = sp.symbols('gamma eta tau D qp qm', positive=True)
    zero((qp-2*tau*dd)*(qm-2*tau*dd)-qp*qm
         +2*tau*dd*(qp+qm)-4*tau*tau*dd*dd,
         'weighted matrix determinant perturbation')
    zero((eta-4*q*tau).subs(tau,eta/(8*q))-eta/2,
         'continuous-separator determinant margin')
    zero((eta-6*q*tau).subs(tau,eta/(8*q))-eta/4,
         'polynomial-separator determinant margin')
    zero((gamma-3*tau).subs(tau,gamma/4)-gamma/4,
         'polynomial-separator diagonal margin')
    quantum_zero(q-alpha**2*(l*l+1)/q-4*(l*l+1)/q,
                 'weighted randomness tangent statistic')
    quantum_zero((alpha*(l*l+1)/q)**2
                 -(l*l+1-(4*(l*l+1)/q)**2/4),
                 'weighted randomness bias curve')

    # A direct degree-two conditioned certificate when l^2 >= 3 uses P=H.
    zero(H-alpha-2*l*c-(c-a)**2*(2*l*a*a-alpha*(c+2*a))/(2*a**3),
         'cubic separator positive-outcome factorization')
    zero(H+alpha+2*l*c-(c+a)**2*(2*l*a*a-alpha*(c-2*a))/(2*a**3),
         'cubic separator negative-outcome factorization')
    qq = 2*(l*l+1)*a/l
    qh = qq-H
    determinant = sp.expand(qh*qh.subs(c,-c)-4*(1-c*c))
    determinant = determinant.subs(alpha**2,4*(l*l+1)*a*a/l**2-4)
    numerator = (a**4*(l*l+2)**2-4*a*a*l*l
                 +(l*l-a*a*(l*l+1))*c*c)
    zero(determinant-D*numerator/(a**6*l*l),
         'cubic separator determinant contact factorization')
    zero(qh+qh.subs(c,-c)-2*(a*a*(l*l+2)-l*l*c*c)/(a*l),
         'cubic separator positive trace')
    t, A, h = sp.symbols('t A h',positive=True)
    Amin = t/(t+1)
    f = A*A*(t+2)**2-A*(5*t+1)+t
    derivative = (2*t**3+3*t*t+2*t-1)/(t+1)
    zero(f-((t+2)**2*(A-Amin)**2+derivative*(A-Amin)+t**4/(t+1)**2),
         'cubic determinant numerator positive expansion')
    zero((2*t**3+3*t*t+2*t-1).subs(t,h+1)-(2*h**3+9*h*h+14*h+6),
         'cubic determinant derivative positive for t>=1')
    # The scalar requirement is worst at alpha=2/l; a=1 there.
    zero((alpha*(1+2*a)/(2*l*a*a)).subs({alpha:2/l,a:1})-3/l**2,
         'degree-two threshold l^2>=3')
    zero(sp.diff(alpha/(alpha*alpha+4),alpha)
         -(4-alpha*alpha)/(alpha*alpha+4)**2,
         'monotonic scalar requirement first term')
    zero(sp.diff(alpha/sp.sqrt(alpha*alpha+4),alpha)
         -4/(alpha*alpha+4)**sp.Rational(3,2),
         'monotonic scalar requirement second term')
    return {'status':'PASS','failures':[], 'number_of_checks':len(checks),
            'checks':checks,
            'scope':'exact symbolic proof audits; universal inequalities and '
                    'conditioned SOS conversion are established analytically'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
