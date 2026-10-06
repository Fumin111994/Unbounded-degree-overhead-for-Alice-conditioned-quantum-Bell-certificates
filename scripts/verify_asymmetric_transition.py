"""Exact audits of the sharp asymmetric-weight transition.

The supplement proves the compactness/Markov lower bound and the universal
inequalities. This checker audits the algebra, rational safety constants,
and complete pole-cancelled polynomials; it never solves an SDP.
"""
import json
from math import isqrt
import sympy as sp


def interval_nonnegative(poly, left=-1, right=1):
    """Exact sign decision on a rational interval, using Sturm counts.

    No floats, grid, or claimed SOS factorization is used. Interior roots of
    a nonnegative nonzero polynomial must have even multiplicity. Once this
    holds, one rational nonroot fixes the sign throughout the open interval.
    Endpoint roots are stripped before counting interior roots.
    """
    poly = sp.Poly(poly, domain=sp.QQ)
    if poly.is_zero:
        return True
    if poly.eval(left) < 0 or poly.eval(right) < 0:
        return False
    x = poly.gen
    reduced = poly
    for endpoint in (left, right):
        while reduced.eval(endpoint) == 0:
            reduced = reduced.exquo(sp.Poly(x-endpoint, x, domain=sp.QQ))
    for factor, multiplicity in reduced.sqf_list()[1]:
        if multiplicity % 2 and factor.count_roots(left, right):
            return False
    for j in range(1, poly.degree()+3):
        value = poly.eval(left+(right-left)*sp.Rational(j, poly.degree()+3))
        if value:
            return bool(value > 0)
    raise AssertionError('No nonroot found for a nonzero polynomial')


def audit_four_blocks(P, q, alpha, lam, a):
    """Build the actual scalar diagonals and 2x2 determinants in c.

    Returns exact global positivity decisions, not evaluations of the
    factorization formulas used by the analytic construction.
    """
    c = P.gen
    expressions = {
        'positive_outcome': P.as_expr()-alpha-2*lam*c,
        'negative_outcome': P.as_expr()+alpha+2*lam*c,
        'second_question_diagonal': q-P.as_expr(),
        'second_question_determinant':
            (q-P.as_expr())*(q-P.as_expr().subs(c, -c))-4*(1-c*c),
    }
    for name, expression in expressions.items():
        poly = sp.Poly(expression, c, domain=sp.QQ)
        # Divide only by known nonnegative squared contact factors. This
        # reduces exact arithmetic without weakening the interval decision.
        if name == 'positive_outcome':
            poly = poly.exquo(sp.Poly((c-a)**2,c))
        elif name == 'negative_outcome':
            poly = poly.exquo(sp.Poly((c+a)**2,c))
        elif name == 'second_question_determinant':
            poly = poly.exquo(sp.Poly((c*c-a*a)**2,c))
            assert all(power[0] % 2 == 0 for power, _ in poly.terms())
            even = sp.Poly(sum(coeff*c**(power[0]//2)
                               for power, coeff in poly.terms()),c)
            if not interval_nonnegative(even,0,1):
                raise AssertionError(f'global block positivity: {name}')
            continue
        if not interval_nonnegative(poly):
            raise AssertionError(f'global block positivity: {name}')
    for sign in (-1, 1):
        assert P.eval(sign*a) == 2*lam*a+sign*alpha
        assert P.diff().eval(sign*a) == sign*2*lam
    return {'all_four_blocks_psd_on_closed_interval': True,
            'method': 'exact square-free decomposition and Sturm root counts',
            'contact_values_and_derivatives': True}


def verify():
    z, t, v, r, c, s = sp.symbols('z t v r c s', real=True)
    checks = []

    def zero(expr, name):
        if sp.factor(expr) != 0:
            raise AssertionError(name)
        checks.append(name)

    # Positive and negative controls include repeated and endpoint roots.
    for expr, expected in [(c*c, True), (-c*c, False), (c, False),
                           (1-c*c, True), (c*c-sp.Rational(1,101)**2, False),
                           ((1-c)**3*(1+c)**2, True), (sp.S.Zero, True)]:
        assert interval_nonnegative(sp.Poly(expr, c)) == expected
    checks.append('Sturm sign checker positive and negative controls')

    h = (t + 1 - t*t*v*v)/t
    p = 1+z*z+v*(3*z-z**3)+r*(z*z-1)**2
    q = 2*(t+1)/t
    C2 = r*r
    C1 = 2*r-2*r*r-v*v
    C0 = r*r-2*r-4*r/t+4*v*v+1
    zero(p-2*(v+z)-(z-1)**2*(1-v*(z+2)+r*(z+1)**2),
         'quartic positive-branch factor')
    zero(p+2*(v+z)-(z+1)**2*(1-v*(z-2)+r*(z-1)**2),
         'quartic negative-branch factor')
    zero((q-p)*(q-p.subs(z,-z))-4/t*(h-z*z)
         -(z*z-1)**2*(C2*z**4+C1*z*z+C0),
         'quartic determinant factor')
    rr = v*v/(4*(1-v))
    zero((4*C2*C0-C1*C1).subs(r,rr)
         -v**5*(t*v*v-(5*t+1)*v+4*t)/(4*t*(1-v)**3),
         'determinant discriminant margin')
    zero((1-v-v*v/(4*r)).subs(r,rr), 'first scalar square margin')
    zero((1+v-v*v/(4*r)).subs(r,rr)-2*v, 'second scalar square margin')
    zero((v*v-(5+1/t)*v+4).subs(v,1/t)-(4-5/t),
         'sharp quartic weight threshold')
    assert (v*v-6*v+4).subs(v,sp.Rational(3,4)) == sp.Rational(1,16)
    checks.append('compact-tilt quartic margin')

    g, lam, T0, T1 = sp.symbols('g lam T0 T1', real=True)
    # E=2 lambda g (1-c) T; contacts E(-1)=4 lambda g, E'(-1)=-4 lambda.
    zero(2*lam*g*(-1+2*(sp.Rational(1,2)-1/g))+4*lam,
         'normalized endpoint derivative for Markov bound')
    zero(((2-g)/g).subs(g,1-1/t)-(t+1)/(t-1),
         'Markov lower bound in weight variable')
    zero((t+1)/(t-1)-9-8*(sp.Rational(5,4)-t)/(t-1),
         'necessary level-two threshold')
    # Illustrates why a numerical tolerance/grid is not exact uniform closure.
    lower_examples = []
    for lval, expected in [(sp.Rational(51,50),5),(sp.Rational(101,100),6),
                           (sp.Rational(201,200),8)]:
        rad = (lval*lval+1)/(lval*lval-1)
        k = 1
        while (2*k-1)**2 < rad:
            k += 1
        assert k == expected
        lower_examples.append({'lambda':str(lval), 'uniform_level_at_least':k})
    checks.append('three exact uniform-degree lower bounds')

    vv = (1-s)*z+1+s
    pr = 2*(1-s+z)+4*s**3*(z-1)**2/vv**2
    hh = (t+1-t*t*(1-s)**2)/t
    L = 1+2*s-2*s**3+s**4
    J = ((q-pr)*(q-pr.subs(z,-z))-4/t*(hh-z*z))*vv**2*vv.subs(z,-z)**2
    correction = 4*(t-1)/t*(1-s)**2*(1+4*s+5*s*s+2*s**3-(1-s)**2*z*z)
    zero(J-(z*z-1)**2*(8*s*s*L+correction),
         'asymmetric rational determinant and nonnegative correction')
    zero(1+4*s+5*s*s+2*s**3-(1-s)**2*(1+2*s-s*s)
         -(4*s+9*s*s-2*s**3+s**4), 'correction lower bound')
    zero((1+2*s-s*s)-hh-(t-1)*(1/t+(1-s)**2),
         'spectral interval containment')
    H = 1+z*z+(1-s)*(3*z-z**3)
    R = (1-s)**2*((1-s)*z+1+2*s)/vv**2
    zero(pr-H-(z*z-1)**2*R, 'scaled rational Hermite remainder')
    Fbound = sum(sp.Rational(4,3)**(2*j)*5**(2*j-2)/sp.factorial(2*j)
                 for j in range(1,6))
    assert Fbound == sp.Rational(496881496,33480783) > 14
    assert 198**2*9 < 14**2*32**2*2  # 3(198/(32 sqrt(2))-1)<11.
    assert sp.Rational(26)*sp.Rational(8,3)**2*sp.Rational(27,350)**3 \
        == sp.Rational(454896,5359375) < sp.Rational(1,2)
    checks.append('scale-five kernel rational safety margins')

    # At t=5/4 and a=1 divide each block by lambda. The resulting
    # diagonals and determinant are rational, even though lambda is not.
    tv, rhov = sp.Rational(5,4), sp.Rational(4,5)
    pv = sp.Poly(p.subs({z:c, v:rhov, r:rr.subs(v,rhov)}), c)
    qv = 2*(tv+1)/tv
    for expression in [pv.as_expr()-2*(rhov+c), pv.as_expr()+2*(rhov+c),
                       qv-pv.as_expr(),
                       (qv-pv.as_expr())*(qv-pv.as_expr().subs(c,-c))
                       -4/tv*(1-c*c)]:
        assert interval_nonnegative(sp.Poly(expression,c))
    checks.append('sharp threshold: independent global four-block Sturm audit')

    # A rational point on (lambda*rho)^2+(1/a)^2=1+1/lambda^2,
    # obtained using a rational secant through the critical point.
    linter, slope = sp.Rational(11,10), -sp.Rational(9,10)
    v0 = 1/linter
    vinter = (v0*(slope*slope-1)-2*slope)/(1+slope*slope)
    ainter = 1/(1+slope*(vinter-v0))
    rhointer = vinter/linter
    assert 0 < ainter < 1 and sp.Rational(3,4) < rhointer < 1/linter**2
    parameters = [(sp.Rational(11,10),sp.S.One,1-sp.Rational(10,11)**2),
                  (sp.Rational(101,100),sp.S.One,1-sp.Rational(100,101)**2),
                  (linter,ainter,1-rhointer)]
    reports = []
    for lval, aval, sval in parameters:
        alphaval = 2*lval*aval*(1-sval)
        qval = 2*aval*(lval*lval+1)/lval
        assert qval*qval == (lval*lval+1)*(alphaval*alphaval+4)
        gamma = aval*(1+sval)/(1-sval)
        dist = gamma-1
        n = isqrt(int(sp.ceiling(25/dist)))
        while n*n < 25/dist:
            n += 1
        assert (n-1)**2 < 25/dist <= n*n
        up = sp.Poly(c+gamma,c,domain=sp.QQ)
        f, rem = sp.div(sp.Poly(1-sp.chebyshevt(n,-c),c,domain=sp.QQ),
                       sp.Poly(n*n*(1+c),c,domain=sp.QQ))
        assert rem.is_zero
        F = f.eval(-gamma)
        K = (f**3).mul_ground(1/F**3)
        kp = K.diff().eval(-gamma)
        A, B = lval*(1-sval)/aval**2, lval*sval/aval
        C = A-B*kp
        M = B/dist**2+C/dist
        num = up.mul_ground(A)+sp.Poly(B,c)-K*(sp.Poly(B,c)+up.mul_ground(C)-(up**2).mul_ground(M))
        Pi, rem = sp.div(num,up**2)
        assert rem.is_zero and Pi.degree() <= 3*(n-1)
        hp = sp.Poly(lval*(aval+c*c/aval)
                     +alphaval*(3*c-c**3/aval**2)/(2*aval),c)
        P = hp+sp.Poly((c*c-aval*aval)**2,c)*Pi
        for sign in (-1,1):
            assert P.eval(sign*aval) == 2*lval*aval+sign*alphaval
            assert P.diff().eval(sign*aval) == sign*2*lval
        block_audit = (audit_four_blocks(P,qval,alphaval,lval,aval) if n <= 8
                       else {'contact_values_and_derivatives':True,
                             'scope':'Global PSD follows from the analytic error budget; '
                                     'direct Sturm audit is performed on the two degree-25 cases.'})
        assert F > 14 and 0 < -dist*kp < 11 and dist*M < 26*lval
        reports.append({'lambda':str(lval),'alpha':str(alphaval),'a':str(aval),'N':n,
                        'degree':P.degree(),'exact_pole_cancellation':True,
                        'block_audit':block_audit})
    checks.append('three rational pole cancellations and contacts; two global PSD audits')
    return {'status':'PASS','number_of_checks':len(checks),'checks':checks,
            'markov_lower_examples':lower_examples,'polynomial_audits':reports,
            'scope':'Exact algebra audits; compactness, symmetry, Markov inequality '
                    'and universal positivity estimates are proved in the supplement.'}


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
