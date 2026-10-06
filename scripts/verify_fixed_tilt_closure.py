"""Symbolic audits of the fixed-tilt polynomial-separator existence proof.

No finite sampling is used as positivity proof, and no SDP solver is called.
The Bernstein construction is effective but its degree estimate is very loose.
"""
import json
import sympy as sp


def verify_fixed_tilt_closure():
    c, a, alpha = sp.symbols('c a alpha', real=True)
    H = a+c*c/a+alpha*(3*c-c**3/a**2)/(2*a)
    D = (c*c-a*a)**2
    Rplus = (alpha*(c+2*a)-2*a*a)/(2*a**3*(c+a)**2)
    Rminus = (alpha*(c-2*a)-2*a*a)/(2*a**3*(c-a)**2)
    checks = []

    def zero(expr, name):
        if sp.factor(expr) != 0:
            raise AssertionError(name)
        checks.append(name)

    zero(H.subs(c, a)-2*a-alpha, 'value at positive contact')
    zero(H.subs(c, -a)-2*a+alpha, 'value at negative contact')
    zero(sp.diff(H, c).subs(c, a)-2, 'derivative at positive contact')
    zero(sp.diff(H, c).subs(c, -a)+2, 'derivative at negative contact')
    zero(alpha+2*c-H-D*Rplus, 'positive absolute-value branch')
    zero(-alpha-2*c-H-D*Rminus, 'negative absolute-value branch')
    zero(Rplus.subs(c, -alpha/2)+1/a**3, 'right branch at the kink')
    zero(Rminus.subs(c, -alpha/2)+1/a**3, 'left branch at the kink')
    zero(sp.diff(Rplus, c)-(-alpha*(c+a)+4*a*(a-alpha/2))
         /(2*a**3*(c+a)**3), 'right derivative for Lipschitz bound')
    zero(sp.diff(Rminus, c)+(alpha*(a-c)+4*a*(a+alpha/2))
         /(2*a**3*(a-c)**3), 'left derivative for Lipschitz bound')
    outer = (4*a-alpha-2*c)*(4*a+alpha-2*c)-4*(1-c*c)
    zero(outer-8*(c-a)**2-(8*a*a-alpha*alpha-4),
         'outer determinant modulo the quantum-value identity')
    zero((4*a-alpha-2*c)*(4*a-alpha+2*c)-4*(1-c*c)
         -((4*a-alpha)**2-4), 'inner determinant')
    x, y, u, v = sp.symbols('x y u v', real=True)
    zero((x-u)*(y-v)-x*y+u*y+v*x-u*v, 'two-sided determinant perturbation')
    q, d = sp.symbols('q d', real=True)
    zero(((q-alpha)**2-4).subs(q, d+alpha+2)-d*(d+4),
         'positive inner margin from the Bell violation')
    eta, tau = sp.symbols('eta tau', positive=True)
    zero((eta-6*q*tau).subs(tau, eta/(8*q))-eta/4,
         'strict determinant safety margin')
    zero((d-3*tau*a**4).subs(tau, d/(4*a**4))-d/4,
         'strict diagonal safety margin')

    # Check the singular scales of the explicit, deliberately crude degree bound.
    e = sp.symbols('e', positive=True)
    ae = sp.sqrt(8+2*(2-e)**2)/4
    de = 4*ae-(2-e)-2
    Le = ((2-e)+4*ae)/(2*ae**3*(ae-(2-e)/2)**2)
    if sp.limit(de/e**2, e, 0) != sp.Rational(1, 8):
        raise AssertionError('Bell margin asymptotic')
    if sp.limit(Le*e**2, e, 0) != 48:
        raise AssertionError('Lipschitz asymptotic')
    checks.extend(['Bell margin asymptotic', 'Lipschitz asymptotic'])
    aq = sp.sqrt(8+2*alpha*alpha)
    zero(sp.diff(aq, alpha, 2)-16/aq**3, 'curvature for the chord bound')
    J = sp.symbols('J', real=True)
    zero(J-(J/(4*y))*y-(J/(4*x))*x-J/2,
         'pointwise weighted-envelope safety margin')

    # Exact scalar checks at rational tilts. These audit constants, not the proof.
    constants = []
    for tilt in [sp.Rational(0), sp.Rational(1), sp.Rational(3, 2),
                 sp.Rational(19, 10), sp.Rational(199, 100)]:
        aa = sp.sqrt(8+2*tilt*tilt)/4
        qq = 4*aa
        dd = qq-tilt-2
        mm = (qq-tilt)**2-4
        ss = aa-tilt/2
        for number in [dd, mm, ss]:
            if number.is_positive is not True:
                raise AssertionError('Nonpositive safety constant')
        constants.append(str(tilt))
    return {'status': 'PASS', 'checks': checks, 'number_of_checks': len(checks),
            'exact_parameter_audits': constants,
            'scope': 'symbolic identities; universal inequalities, extension, '
                     'and finite-degree existence are proved in the supplement',
            'matching_upper_bound_scope': 'not established by this auxiliary '
             'Bernstein audit; see verify_matching_upper.py and the supplement'}


if __name__ == '__main__':
    print(json.dumps(verify_fixed_tilt_closure(), indent=2))
