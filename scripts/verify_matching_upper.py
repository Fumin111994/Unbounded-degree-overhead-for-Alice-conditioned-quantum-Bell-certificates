"""Exact algebra audits for the square-root upper-bound construction.

The universal inequalities and the dihedral SOS extension are proved in the
supplement. This script checks symbolic identities, rational safety constants,
and actual polynomial denominator cancellation at rational parameters.
No optimizer, floating-point feasibility check, or sampled positivity is used.
"""
import json
from math import isqrt
import sympy as sp


def verify_matching_upper():
    s, z = sp.symbols('s z', real=True)
    h = 1+2*s-s*s
    v = (1-s)*z+1+s
    p = 2*(1-s+z)+4*s**3*(z-1)**2/v**2
    H = 1+z*z+(1-s)*(3*z-z**3)
    R = (1-s)**2*((1-s)*z+1+2*s)/v**2
    checks = []

    def zero(expr, name):
        if sp.factor(expr) != 0:
            raise AssertionError(name)
        checks.append(name)

    zero(p-H-(z*z-1)**2*R, 'rational Hermite remainder')
    zero((p-2*(1-s+z))*v**2-4*s**3*(z-1)**2,
         'first rational positivity factor')
    zero((p+2*(1-s+z))*v**2
         -4*(z+1)**2*((1-s)**2*z+1+s-s*s),
         'second rational positivity factor')
    zero(((4-p)*(4-p.subs(z, -z))-4*(h-z*z))*v**2*v.subs(z, -z)**2
         -8*s*s*(z*z-1)**2*(s**4-2*s**3+2*s+1),
         'rational determinant factor')
    zero(1-h*(1-s)**2-s*s*(2-s)**2, 'pole lower-distance comparison')
    zero(1+s-s*s-(1-s)**2*(1+s)-s*(2-s*s),
         'positive linear factor lower bound')
    zero((1+s)**2-h-2*s*s, 'spectral endpoint comparison')
    zero((4-p.subs(z, 0))*(1+s)**2
         -2*(1+3*s+3*s*s-s**3), 'positive central diagonal')

    a, c, g = sp.symbols('a c g', positive=True)
    gamma = a*(1+s)/(1-s)
    rr = R.subs(z, c/a)/a**3
    zero(rr-(1-s)/a**2/(c+gamma)-s/a/(c+gamma)**2,
         'two-pole partial fractions')
    zero((8*a*a*s*s*(s**4-2*s**3+2*s+1)
          /((1-s)**4*(c+gamma)**2*(gamma-c)**2))
         -(8*s*s*(s**4-2*s**3+2*s+1)
           /(a*a*v.subs(z,c/a)**2*v.subs(z,-c/a)**2)),
         'determinant after dimensional rescaling')

    A, B, u, kp, t = sp.symbols('A B u kp t', real=True)
    C = A-B*kp
    Kjet = 1+kp*u
    residual = Kjet*(B+C*u-sp.Symbol('M')*u*u)
    zero(residual.subs(u, 0)-B, 'residual value at cancelled pole')
    zero(sp.diff(residual, u).subs(u, 0)-A,
         'residual derivative at cancelled pole')
    zero((B/t**2+C/t)*u*u-B-C*u
         -(u-t)*(B*(u+t)/t**2+C*u/t),
         'one-sided error factor for nonnegative slope')
    zero(h*(1-s/2)**2-1-s*(4-11*s+6*s*s-s**3)/4,
         'global pole upper-distance identity')
    zero(h*(1-s+2*s*s)**2-1-s**3*(8-9*s+12*s*s-4*s**3),
         'endpoint pole upper-distance identity')
    zero(1+s-(1-s)*h-(3*s*s-s**3), 'endpoint lambda bound identity')
    # Positive polynomial brackets on the full stated sigma intervals,
    # using their unfavorable coefficients at the right endpoint.
    assert 4-sp.Rational(11,4)-sp.Rational(1,64) > 0
    assert 8-sp.Rational(9,16)-sp.Rational(4,16**3) > 0
    assert 1-sp.Rational(13,16)-sp.Rational(12,16**2) > 0
    checks.append('uniform rational parameter margins')
    # The two cosh estimates reduce to these rational endpoint inequalities.
    for scale, tmax in [(sp.Rational(4,3),sp.Rational(3,4)),
                        (sp.Rational(7,5),sp.Rational(6,49))]:
        assert scale**2/(2*(1-scale**2*tmax/12)) <= 1
    checks.append('hyperbolic argument lower-bound constants')
    F_global = sum(sp.Rational(4,3)**(2*j)*4**(2*j-2)/sp.factorial(2*j)
                   for j in range(1,6))
    F_endpoint = sum(sp.Rational(7,5)**(2*j)*3**(2*j-2)/sp.factorial(2*j)
                     for j in range(1,6))
    assert F_global == sp.Rational(5297547256,837019575) > sp.Rational(25,4)
    assert F_endpoint == sp.Rational(224174937917,62500000000) > sp.Rational(7,2)
    checks.append('two exact Chebyshev normalization lower bounds')
    assert sum(sp.Rational(16,3)**j/sp.factorial(j) for j in range(5)) > 65
    assert sum(sp.Rational(21,5)**j/sp.factorial(j) for j in range(10)) > 65
    assert 495**2 < 2*352**2 and 121 < 128
    checks.append('exterior kernel derivative constants')
    # C>0 because Kprime<0; the error is bounded by M*K, not (M+C/u)*K.
    assert 2+2*(1+8) == 20
    assert sp.Rational(9,8)*(2+sp.Rational(9,2)) == sp.Rational(117,16)
    assert 80*sp.Rational(9,50)**3 == sp.Rational(1458,3125) < sp.Rational(1,2)
    assert (sp.Rational(117,16)*sp.Rational(7,6)**2*sp.Rational(22,63)**3
            == sp.Rational(17303,40824) < sp.Rational(1,2))
    checks.append('global and endpoint weighted error safety margins')

    x = sp.symbols('x', real=True)
    for n in [1, 2, 3, 7]:
        f = sp.div(sp.Poly(1-sp.chebyshevt(n, -x), x),
                   sp.Poly(n*n*(1+x), x))
        if not f[1].is_zero or f[0].eval(-1) != 1:
            raise AssertionError('Fejer polynomial division')
    checks.append('removable Fejer denominator in four exact examples')

    reports = []
    # Rational points on 2a^2-b^2=1. Build both global-kernel examples
    # and an independent example for the improved endpoint kernel.
    for parameter, scale in [(sp.Rational(1,4),4),(sp.Rational(1,10),4),
                             (sp.Rational(1,32),3)]:
        den = 2+4*parameter+parameter**2
        aa = (2+2*parameter+parameter**2)/den
        bb = (2-parameter**2)/den
        ss = 1-bb/aa
        gg = aa*(1+ss)/(1-ss)
        tt = gg-1
        if not (0 < ss <= sp.Rational(1, 4) and ss <= tt <= 2*ss):
            raise AssertionError('Parameter range')
        if scale == 3:
            assert ss <= sp.Rational(1,16) and tt <= sp.Rational(7,6)*ss
        target = sp.Rational(scale**2)/tt
        n = isqrt(int(sp.ceiling(target)))
        while n*n < target:
            n += 1
        assert (n-1)**2 < target <= n*n
        upoly = sp.Poly(x+gg, x, domain=sp.QQ)
        f, rem = sp.div(sp.Poly(1-sp.chebyshevt(n, -x), x, domain=sp.QQ),
                        sp.Poly(n*n*(1+x), x, domain=sp.QQ))
        assert rem.is_zero
        FF = f.eval(-gg)
        K = (f**3).mul_ground(1/FF**3)
        kprime = K.diff().eval(-gg)
        AA, BB = (1-ss)/aa**2, ss/aa
        CC = AA-BB*kprime
        assert kprime < 0 and CC > 0
        MM = BB/tt**2+CC/tt
        numerator = upoly.mul_ground(AA)+sp.Poly(BB,x)-K*(
            sp.Poly(BB,x)+upoly.mul_ground(CC)-(upoly**2).mul_ground(MM))
        Tpoly, rem = sp.div(numerator, upoly**2)
        if not rem.is_zero or Tpoly.degree() > 3*(n-1):
            raise AssertionError('Double pole was not cancelled')
        HH = sp.Poly(aa+x*x/aa+bb/aa*(3*x-x**3/aa**2), x)
        DD = sp.Poly((x*x-aa*aa)**2, x)
        P = HH+DD*Tpoly
        for sign in [-1, 1]:
            cc = sign*aa
            if P.eval(cc) != 2*aa+sign*2*bb or P.diff().eval(cc) != sign*2:
                raise AssertionError('Exact contact preservation')
        if scale == 4:
            assert FF > sp.Rational(25,4) and -kprime*tt < 8
            assert MM*tt < 20
        else:
            assert FF > sp.Rational(7,2) and -kprime*tt < sp.Rational(9,2)
            assert MM*tt < sp.Rational(117,16)
        assert P.degree() <= 3*n+1
        reports.append({'alpha': str(2*bb), 's': str(ss), 'pole_distance': str(tt),
                        'kernel_scale': scale, 'N': n, 'polynomial_degree': P.degree(),
                        'native_level': (P.degree()+1)//2,
                        'pole_cancellation': 'EXACT', 'contact_checks': 'EXACT'})
    checks.append('three full rational polynomial constructions')
    stat = sp.symbols('stat', positive=True)
    bias = sp.sqrt(2-stat**2/4)
    zero(sp.diff(bias, stat, 2)+1/(2*bias**3), 'randomness curvature')
    eps = sp.symbols('eps', positive=True)
    q = sp.sqrt(8+2*stat**2)
    assert sp.diff(q,stat,2).subs(stat,2) == sp.Rational(1,4)
    assert sp.diff(bias,stat,2).subs(stat,2) == -sp.Rational(1,2)
    assert sp.limit((8/sp.sqrt(8+2*(2-eps)**2)-2)/eps,eps,0,dir='+') == sp.Rational(1,2)
    assert sp.Rational(9**4,32) == sp.Rational(6561,32)
    checks.append('endpoint curvature and limiting precision constants')
    return {'status': 'PASS', 'checks': checks, 'number_of_checks': len(checks),
            'polynomial_audits': reports,
            'scope': 'exact symbolic and rational audits; the universal matching '
                     'upper bound follows from the analytic proof in the supplement'}


if __name__ == '__main__':
    print(json.dumps(verify_matching_upper(), indent=2))
