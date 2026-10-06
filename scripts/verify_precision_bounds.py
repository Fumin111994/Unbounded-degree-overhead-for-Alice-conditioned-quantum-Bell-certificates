"""Solver-free algebraic audits for the finite-precision revision.

The universal proofs are in the supplement. Finite matrix checks audit the
translation convention and simultaneous outcome sums, not all-level validity.
This checker is new in the September 24 working revision, not the cited release.
"""
import json
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from higher_level_witness import reduced, words, fejer_blocks


def multiply(a, b):
    return reduced(a + b)


def inverse(a):
    return tuple(reversed(a))


def audit_extension(moment, k):
    """Audit a larger Gram as an exact sum of compressed original Grams."""
    native = words(k)
    # Rational window: universal positivity does not require the sine window.
    c = {w: sp.Integer(k + 1 - len(w)) for w in native}
    norm = sum(t*t for t in c.values())
    larger = words(k + 2)
    original = sp.Matrix([[moment(multiply(inverse(u), v)) for v in native]
                          for u in native])

    def rounded(g):
        overlap = sum(c[u] * c.get(multiply(u, g), 0) for u in native)
        return sp.simplify(overlap * moment(g) / norm) if overlap else sp.S.Zero

    direct = sp.Matrix([[rounded(multiply(inverse(u), v)) for v in larger]
                        for u in larger])
    translations = {multiply(w, inverse(g)) for w in native for g in larger}
    decomposed = sp.zeros(len(larger))
    index = {w: i for i, w in enumerate(native)}
    for x in translations:
        embedding = sp.zeros(len(native), len(larger))
        for j, g in enumerate(larger):
            u = multiply(x, g)
            if u in index:
                embedding[index[u], j] = c[u]
        decomposed += embedding.T * original * embedding / norm
    if direct != decomposed:
        raise AssertionError('Translation decomposition failed')
    if direct != direct.H:
        raise AssertionError('Hermiticity failed')
    if direct[0, 0] != moment(()):
        raise AssertionError('Outcome mass changed')
    return direct


def verify_precision_bounds():
    checks = []

    def zero(expression, name):
        if sp.simplify(expression) != 0:
            raise AssertionError(name)
        checks.append(name)

    r, e = sp.symbols('r e', positive=True)
    q = sp.sqrt(8 + 2*(2-e)**2)
    w = 4 - (1-2*r)*e
    zero((w*w-q*q).subs(e, 8*r) - 64*r*r*(1-2*r)**2,
         'exact witness squared gap')
    zero(sp.series(q-(4-e), e, 0, 4).removeO() - e**2/8-e**3/32,
         'local violation expansion')
    zero(sp.series(1-(4-e)/q, e, 0, 4).removeO() - e**2/32-e**3/64,
         'white noise expansion')
    zero(sp.series((w-q).subs(e, 8*r), r, 0, 4).removeO() - 8*r**2+16*r**3,
         'witness absolute gap expansion')
    gq = (1 + sp.sqrt(1-4*r-4*r*r))/2
    zero(sp.series(1-r-gq, r, 0, 4).removeO() - 2*r*r-4*r**3,
         'guessing deficit expansion')
    zero(sp.series(sp.log((1-r)/gq)/sp.log(2), r, 0, 4).removeO()
         - (2*r*r+6*r**3)/sp.log(2), 'entropy deficit expansion')
    z = sp.symbols('z', positive=True)
    rz = z*z/(8+2*z*z)
    zero(sp.limit(((w-q).subs(e, 8*r)).subs(r, rz)/z**4, z, 0)-sp.Rational(1, 8),
         'asymptotic lower constant')
    zero(sp.Rational(8, 100)*sp.Rational(16, 25)-sp.Rational(32, 625),
         'uniform lower constant')
    j, theta = sp.symbols('j theta', real=True)
    zero(sp.expand_trig(sp.sin(j*theta)+sp.sin((j+2)*theta)
                        -2*sp.cos(theta)*sp.sin((j+1)*theta)),
         'symbolic sine eigenvector recurrence')

    # Independent graph construction: right multiplication by either generator.
    for k in (1, 2, 3):
        native = words(k)
        theta_k = sp.pi/(2*k+2)
        c = {u: sp.sin((k+1-len(u))*theta_k)/sp.sqrt(k+1) for u in native}
        zero(sum(a*a for a in c.values())-1, f'sine normalization k={k}')
        for y in (0, 1):
            overlap = sum(c[u]*c.get(multiply(u, (y,)), 0) for u in native)
            zero(overlap-sp.cos(theta_k), f'generator contraction k={k} y={y}')

    # All four truncated separating blocks, including a block that has no
    # compatible unmodified all-level completion with the other three.
    for k in (1, 2, 3):
        native = words(k)
        output = []
        for block in fejer_blocks(k):
            moments = {}
            for i, u in enumerate(native):
                for j, v in enumerate(native):
                    moments[multiply(inverse(u), v)] = block[i, j]
            output.append(audit_extension(lambda g: moments[g], k))
        if output[0]+output[1] != output[2]+output[3]:
            raise AssertionError('Common outcome sum failed')
        checks.append(f'four-block translation decomposition k={k}')

    # Complex moments expose orientation errors hidden by real witnesses.
    b = [sp.diag(1, -1), sp.Matrix([[sp.Rational(3, 5), sp.Rational(4, 5)],
                                   [sp.Rational(4, 5), -sp.Rational(3, 5)]])]
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    densities = [(sp.eye(2)+y)/4, (sp.eye(2)-y)/4,
                 (sp.eye(2)+b[0])/4, (sp.eye(2)-b[0])/4]
    output = []
    for density in densities:
        def moment(g):
            operator = sp.eye(2)
            for letter in g:
                operator *= b[letter]
            return sp.trace(density*operator)
        output.append(audit_extension(moment, 2))
    if output[0]+output[1] != output[2]+output[3]:
        raise AssertionError('Complex common outcome sum failed')
    checks.append('complex four-block orientation and common sums')
    return {'status': 'PASS', 'checks': checks, 'number_of_checks': len(checks),
            'scope': 'symbolic identities and finite exact decomposition audits; '
                     'universal proof in the supplement, no SDP solver used'}


if __name__ == '__main__':
    print(json.dumps(verify_precision_bounds(), indent=2))
