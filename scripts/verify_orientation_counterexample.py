"""Exact research check: no Alice marginal does not imply level-one exactness.

This result accompanies the working manuscript, not the archived public
release. Run with Python and SymPy; no SDP solver is used.

For binary projective observables define
    G = B0 - 3 B1 + A0 (7 B0 + 14 B1) + A1 (11 B0 - 5 B1).
We prove omega_Q(G) = 29, whereas the Alice-conditioned level-one value
is at least 145519/5000 = 29.1038. This disproves universal LEVEL-ONE
exactness without Alice marginals. It does not disprove a uniform finite
degree bound for the full class of such Bell functionals.

Quantum proof: Jordan's lemma reduces each pair of reflections to real
two-dimensional blocks, including scalar limiting blocks. Put A0=B0=Z,
A1=u Z+a X, B1=c Z+b X, with a^2=1-u^2 and b^2=1-c^2. For the resulting
Bell matrix H we verify (H^2-d I)^2=4 D I. Hence every eigenvalue lambda
satisfies lambda^2 <= d+2 sqrt(D). The explicit d obeys d<=733<29^2
on [-1,1]^2. A rational Bernstein certificate proves
(29^2-d)^2-4D >= 0 on this square, so ||H||<=29. A deterministic strategy
attains 29. The same argument applies blockwise to arbitrary dimensions.

The conditioned witness consists of four rational positive definite
3x3 moment matrices on (I,B0,B1). Equal diagonals enforce B_y^2=I;
the two outcome sums coincide and have unit diagonal. Sylvester's
criterion and the Bell value are checked in exact rational arithmetic.
"""

from itertools import product
from contextlib import redirect_stdout
from io import StringIO

import sympy as sp


def require(condition, description):
    if not condition:
        raise AssertionError(description)
    print("PASS:", description)


def verify_quantum_bound():
    u, c, a, b = sp.symbols("u c a b", real=True)
    eye = sp.eye(2)
    z = sp.diag(1, -1)
    x = sp.Matrix([[0, 1], [1, 0]])
    alice = [z, u * z + a * x]
    bob = [z, c * z + b * x]
    h = (sp.kronecker_product(eye, bob[0] - 3 * bob[1])
         + sp.kronecker_product(alice[0], 7 * bob[0] + 14 * bob[1])
         + sp.kronecker_product(alice[1], 11 * bob[0] - 5 * bob[1]))

    metric = sp.Matrix([[1, c], [c, 1]])
    p, r, t = [sp.Matrix(v) for v in ([1, -3], [7, 14], [11, -5])]

    def dot(v, w):
        return (v.T * metric * w)[0]

    d = sp.expand(dot(p, p) + dot(r, r) + dot(t, t) + 2 * u * dot(r, t))
    big_d = sp.expand(dot(p, r)**2 + dot(p, t)**2
                      + 2 * u * dot(p, r) * dot(p, t)
                      + (1-u**2) * (1-c**2)
                      * sp.det(sp.Matrix.hstack(r, t))**2)

    def reduce_squares(expr):
        expr = sp.rem(sp.expand(expr), a**2 + u**2 - 1, a)
        return sp.expand(sp.rem(expr, b**2 + c**2 - 1, b))

    residual = (h*h - d*sp.eye(4))**2 - 4*big_d*sp.eye(4)
    require(all(reduce_squares(e) == 0 for e in residual),
            "Jordan-block spectral identity (H^2-d I)^2 = 4 D I")
    require(d == 238*u*c + 14*u + 80*c + 401,
            "d <= 238+14+80+401 = 733 < 29^2 on the square")

    # D is nonnegative on the square: its first three terms equal
    # (dot(p,r)+u*dot(p,t))^2+(1-u^2)*dot(p,t)^2.
    nonnegative_form = ((dot(p, r) + u*dot(p, t))**2
                        + (1-u**2)*dot(p, t)**2
                        + (1-u**2)*(1-c**2)
                        * sp.det(sp.Matrix.hstack(r, t))**2)
    require(sp.expand(big_d - nonnegative_form) == 0,
            "D is a sum of nonnegative terms for |u|,|c| <= 1")

    residual_bound = sp.expand((29**2 - d)**2 - 4*big_d)
    sx, sy = sp.symbols("sx sy")
    poly = sp.Poly(residual_bound.subs({u: 2*sx-1, c: 2*sy-1}), sx, sy)
    require(poly.degree(sx) <= 2 and poly.degree(sy) <= 2,
            "the bound polynomial has bidegree at most (2,2)")
    coeff = sp.Matrix(3, 3, lambda i, j: sum(
        poly.coeff_monomial(sx**a0 * sy**b0)
        * sp.binomial(i, a0) / sp.binomial(2, a0)
        * sp.binomial(j, b0) / sp.binomial(2, b0)
        for a0 in range(i+1) for b0 in range(j+1)))
    basis = lambda i, v: sp.binomial(2, i) * v**i * (1-v)**(2-i)
    reconstructed = sum(coeff[i, j]*basis(i, sx)*basis(j, sy)
                        for i in range(3) for j in range(3))
    require(sp.expand(reconstructed - poly.as_expr()) == 0,
            "exact Bernstein coefficient reconstruction")

    left = sp.Matrix([[1, 0, 0], [sp.Rational(1, 2), sp.Rational(1, 2), 0],
                      [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)]])
    right = sp.Matrix([[sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)],
                       [0, sp.Rational(1, 2), sp.Rational(1, 2)], [0, 0, 1]])
    for matrix, substitution in [(left, sx/2), (right, (sx+1)/2)]:
        for column in range(3):
            refined = sum(matrix[row, column]*basis(row, sx) for row in range(3))
            require(sp.expand(refined - basis(column, substitution)) == 0,
                    "exact Bernstein half-interval subdivision identity")

    cells = [coeff]
    for _ in range(2):
        cells = [l*m*r.T for m in cells for l in (left, right) for r in (left, right)]
    require(all(min(m) >= 0 for m in cells),
            "all 144 Bernstein coefficients on the 16 subrectangles are nonnegative")
    print("Subrectangle minima:", [str(min(m)) for m in cells])

    def deterministic(a0, a1, b0, b1):
        return b0 - 3*b1 + a0*(7*b0+14*b1) + a1*(11*b0-5*b1)

    require(max(deterministic(*v) for v in product((-1, 1), repeat=4)) == 29,
            "a deterministic strategy attains the exact quantum bound 29")


def verify_conditioned_witness():
    blocks = [sp.Matrix(m)/10000 for m in [
        [[1002, 992, 992], [992, 1002, 992], [992, 992, 1002]],
        [[8998, -6148, -8680], [-6148, 8998, 4232], [-8680, 4232, 8998]],
        [[2430, 2258, -1979], [2258, 2430, -1344], [-1979, -1344, 2430]],
        [[7570, -7414, -5709], [-7414, 7570, 6568], [-5709, 6568, 7570]],
    ]]
    for i, m in enumerate(blocks):
        require(m == m.T and m[0, 0] == m[1, 1] == m[2, 2],
                f"block {i}: all degree-two involution moment identities")
        require(all(m[:j, :j].det() > 0 for j in (1, 2, 3)),
                f"block {i}: positive definite by exact leading principal minors")
    common = blocks[0] + blocks[1]
    require(common == blocks[2] + blocks[3] and common[0, 0] == 1,
            "outcome sums are equal and normalized")
    coef = [1, -3, 7, 14, 11, -5]
    value = (sum(coef[j]*common[0, j+1] for j in range(2))
             + sum(coef[2+2*x+y]*(blocks[2*x]-blocks[2*x+1])[0, y+1]
                   for x in range(2) for y in range(2)))
    require(value == sp.Rational(145519, 5000) and value > 29,
            "rational level-one value 145519/5000 = 29.1038 > 29")


def verify():
    output = StringIO()
    with redirect_stdout(output):
        verify_quantum_bound()
        verify_conditioned_witness()
    return {"status": "PASS", "failures": [],
            "checks": [line for line in output.getvalue().splitlines()
                       if line.startswith("PASS:")],
            "verification": "exact rational and symbolic; no SDP solver"}


if __name__ == "__main__":
    verify_quantum_bound()
    verify_conditioned_witness()
    print("ALL EXACT CHECKS PASSED. No numerical solver or sampling used.")
