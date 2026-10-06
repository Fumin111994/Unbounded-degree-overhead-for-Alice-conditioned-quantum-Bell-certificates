"""Solver-free audits of the opposite-tilt certificate and structural criterion.

The all-parameter determinant and quotient identities are checked exactly.
The approximation theorem is proved in the Supplemental Material; the last
checks audit its Bernstein variance and its application to tilted CHSH.
Finite symbolic checks are not substituted for the universal proof.
"""
import json

import sympy as sp


def verify():
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    alpha, q, c = sp.symbols("alpha q c", real=True)

    def reduce_q(expr):
        numerator = sp.together(expr).as_numer_denom()[0]
        return sp.rem(sp.expand(numerator), q*q-8-2*alpha*alpha, q) == 0

    v0 = q/2 + (4-alpha**2)*c/(2*q)
    vz = 1-alpha/2 + (1+alpha/2)*c
    vx = 1+alpha/2
    blocks = [
        (v0, vz-2-2*c, vx-2),
        (v0, vz, vx),
        (q-v0, -vz-alpha+2*c, -vx+2),
        (q-v0, -vz+2-alpha, -vx),
    ]
    target = c*c*(4-alpha*alpha)**2/(4*q*q)
    for i, (scalar, z, x) in enumerate(blocks):
        check(reduce_q(scalar*scalar-z*z-x*x*(1-c*c)-target),
              f"block {i}: all-parameter determinant")
    check(reduce_q(q/2-(4-alpha*alpha)/(2*q)-(4+3*alpha*alpha)/(2*q)),
          "uniform positive scalar-part lower bound")

    # Compare the I, A0, and A1 coefficients separately. The block vectors
    # are coefficients of I, Z, sqrt(1-c^2) X in the Bob representation.
    b = [sp.Matrix(t) for t in blocks]
    coefficients = [(b[0]+b[1]+b[2]+b[3])/2,
                    (b[0]-b[1])/2, (b[2]-b[3])/2]
    expected = [sp.Matrix([q, -alpha, 0]),
                sp.Matrix([0, -(1+c), -1]),
                sp.Matrix([0, -(1-c), 1])]
    check(all(sp.simplify(e) == 0 for lhs, rhs in zip(coefficients, expected)
              for e in lhs-rhs), "opposite-tilt certificate identity")

    # E[(-1+2J/n-t)^2]=(1-t^2)/n for J~Bin(n,(1+t)/2).
    t, n = sp.symbols("t n", real=True, positive=True)
    p = (1+t)/2
    first_moment = n*p
    second_moment = n*p*(1-p)+(n*p)**2
    variance = sp.expand(4*second_moment/n**2
                         -4*(1+t)*first_moment/n+(1+t)**2)
    check(sp.simplify(variance-(1-t*t)/n) == 0,
          "Bernstein approximation variance identity")

    # For the tilted-CHSH application, E=2 tau D and the two Q diagonals
    # have sum at most 2q. This gives determinant slack at least
    # (eta-4q tau)D; tau<=eta/(8q) makes it at least eta D/2.
    eta, tau, w, qa, qb, slack = sp.symbols("eta tau w qa qb slack")
    perturbed = sp.expand((qa-2*tau*w)*(qb-2*tau*w)-qa*qb+slack)
    expected_slack = slack-2*tau*w*(qa+qb)+4*tau*tau*w*w
    check(sp.expand(perturbed-expected_slack) == 0,
          "weighted separator determinant perturbation identity")
    check(sp.simplify((eta-4*q*tau).subs(tau, eta/(8*q))-eta/2) == 0,
          "contact margin for the compact tilted family")

    # The fixed repair is a support consequence of the published BP15 SOS.
    # Check its factors in the original raw-PVM quotient, not sampled matrices.
    from standard_tilted_sos import exact, Scenario
    algebra = Scenario(2, 2, 2, 2).algebra()
    add, scale = exact.sym_add, exact.sym_scale
    mul = lambda v, w: exact.sym_multiply(algebra, v, w)
    ident = exact.sym_identity()
    a0, a1 = [exact.sym_observable(algebra, 'A', x) for x in (0, 1)]
    b0, b1 = [exact.sym_observable(algebra, 'B', y) for y in (0, 1)]
    bell_s = add(mul(a0, add(b0, b1)), mul(a1, add(b0, scale(b1, -1))))
    bell_sp = add(mul(a0, add(b0, scale(b1, -1))), mul(a1, add(b0, b1)))
    basis = [ident, a0, a1, bell_s, bell_sp]
    check(all(not sp.sympify(v).free_symbols for poly in basis for v in poly.values()),
          "five-row repair support is parameter independent")
    check(max(len(word) for poly in basis for word in poly) == 2,
          "fixed repair rows have total word length at most two")
    repair_alpha = sp.Symbol('repair_alpha', nonnegative=True)
    repair_q = sp.sqrt(8+2*repair_alpha**2)
    bell_f = add(scale(a0, repair_alpha), bell_s)
    factors = [add(scale(ident, repair_q), scale(bell_f, -1)),
               add(scale(a1, repair_alpha), scale(bell_sp, -1))]
    report = exact.verify_sos_identity(
        algebra, repair_q, bell_f, [(1/(2*repair_q), v) for v in factors])
    check(report['status'] == 'PASS', "BP15 repair identity in the raw-PVM quotient")
    products = [mul(exact.sym_dagger(algebra, v), w) for v in basis for w in basis]
    check(any(len({letter[1] for letter in word if letter[0] == 'A'}) == 2
              for poly in products for word in poly),
          "repair block explicitly includes mixed-Alice-question moments")
    return {"status": "PASS", "failures": [], "checks": checks,
            "verification": "exact symbolic identities; no SDP solver"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
