"""Solver-free verification of exact tilted-CHSH optimal-face certificates.

The verifier expands the ORIGINAL block-moment identity. It does not use
the generator's reduced affine system, fitted margins, or PASS flags.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from npa2 import exact
from run_m6_closure import exact_tilted_functional

EXPECTED = {('1', 2), ('5/4', 2), ('41/32', 2), ('3/2', 3), ('13/10', 3)}
ARTIFACTS = ROOT / 'artifacts/prl/quantum_face'


def strategy_data(alpha, level):
    """Exact optimal density matrix, Bell operator, and conditional word maps."""
    alpha = sp.Rational(alpha)
    if not 0 <= alpha < 2 or level not in (2, 3):
        raise ValueError('Require 0 <= alpha < 2 and level 2 or 3')
    c = sp.sqrt(8 + 2 * alpha**2) / 4
    d = sp.sqrt(1 - c**2)
    z = alpha / (2 * c)
    t = sp.sqrt(1 - z**2)
    identity = sp.eye(2)
    Z = sp.diag(1, -1)
    X = sp.Matrix([[0, 1], [1, 0]])
    projectors = [(identity + c*Z + d*X)/2, (identity + c*Z - d*X)/2]
    functional = exact_tilted_functional(Fraction(alpha))
    words = functional.scenario.algebra().words(level, party='B')
    matrices = []
    for word in words:
        matrix = identity
        for _, y, _ in word:
            matrix = matrix * projectors[y]
        matrices.append(matrix.applyfunc(sp.simplify))
    vectors = [sp.Matrix([1, 0]), sp.Matrix([0, 1]),
               sp.Matrix([1, (1-z)/t]), sp.Matrix([1, -(1-z)/t])]
    maps = [sp.Matrix.hstack(*[matrix*v for matrix in matrices]).applyfunc(sp.simplify)
            for v in vectors]
    rho = sp.Matrix([[(1+z)/2, 0, 0, t/2], [0, 0, 0, 0],
                     [0, 0, 0, 0], [t/2, 0, 0, (1-z)/2]])
    bell = (alpha*sp.kronecker_product(Z, identity)
            + 2*c*sp.kronecker_product(Z, Z) + 2*d*sp.kronecker_product(X, X))
    return functional, 4*c, maps, rho, bell


def _matrix(raw):
    matrix = sp.Matrix([[sp.sympify(entry) for entry in row] for row in raw])
    if any(entry.free_symbols or entry.is_real is not True or entry.is_finite is not True
           for entry in matrix):
        raise ValueError('Matrix entries must be finite exact real constants')
    if any(entry.has(sp.Float) for entry in matrix):
        raise ValueError('Floating-point certificate entries are not exact')
    return matrix


def verify_certificate(payload):
    failures = []
    try:
        alpha, level = sp.Rational(payload['alpha']), int(payload['level'])
        functional, q, maps, rho, bell = strategy_data(alpha, level)
        bound = sp.sympify(payload['bound'])
        if sp.simplify(bound - q) != 0:
            failures.append('bound is not the exact quantum optimum')
        # Real symmetric idempotent rho with trace 1 is a PSD pure state.
        if rho != rho.T or sp.simplify(sp.trace(rho)-1) != 0:
            failures.append('quantum state is not normalized and symmetric')
        if (rho*rho-rho).applyfunc(sp.simplify) != sp.zeros(4):
            failures.append('quantum state is not an exact projector')
        if sp.simplify(sp.trace(rho*bell)-q) != 0:
            failures.append('quantum strategy does not attain q')

        data = exact.onesided_certificate_unknowns(functional, level, False)
        names, values = payload['names'], [sp.sympify(v) for v in payload['u']]
        if len(names) != len(values) or len(set(names)) != len(names):
            raise ValueError('certificate name/value mismatch or duplicate name')
        expected_names = set(data['unknown_names'])
        if not set(names) <= expected_names:
            raise ValueError('unknown certificate variable')
        required = {name for name in expected_names if name.startswith('S_') or name == 'nu'}
        if not required <= set(names):
            raise ValueError('missing Gram entry or normalization multiplier')
        if any(v.free_symbols or v.has(sp.Float) or v.is_real is not True for v in values):
            raise ValueError('certificate coefficients must be exact real constants')
        blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, values)
        if len(payload['K']) != 4 or len(payload['H']) != 4:
            raise ValueError('exactly four Gram factors are required')
        n = len(data['words'])
        positive_definite = []
        for index, key in enumerate(data['block_names']):
            K, H = _matrix(payload['K'][index]), _matrix(payload['H'][index])
            if K.shape != (n, n-2) or H.shape != (n-2, n-2):
                raise ValueError('incorrect kernel or reduced Gram dimension')
            if H != H.T or not exact.is_psd_exact(H):
                failures.append(f'{key}: reduced Gram is not symmetric PSD')
            # An exact rational lower bound avoids expensive primitive-element
            # construction for a determinant with large algebraic coefficients.
            positive_definite.append(exact.is_psd_exact(H-sp.eye(n-2)/10**8))
            if K.rank() != n-2 or (maps[index]*K).applyfunc(sp.simplify) != sp.zeros(2, n-2):
                failures.append(f'{key}: kernel basis does not span the optimal quantum kernel')
            S = sp.Matrix(n, n, lambda i,j: blocks[key][(min(i,j),max(i,j))])
            if (S-K*H*K.T).applyfunc(sp.simplify) != sp.zeros(n):
                failures.append(f'{key}: stored Gram differs from K H K^T')
        residuals = exact._onesided_identity_residuals(
            functional, level, bound, blocks, lam, nu, mu, data['symbols'],
            data['entry_key'], data['block_names'], data['words'],
            data['algebra'], data['scenario'])
        if any(sp.simplify(r) != 0 for r in residuals):
            failures.append('original free-moment certificate identity fails')
        return {'status': 'FAIL' if failures else 'PASS', 'failures': failures,
                'alpha': str(alpha), 'level': level, 'bound': str(q),
                'quantum_block_rank': 2, 'reduced_gram_size': n-2,
                'reduced_grams_positive_definite': all(positive_definite),
                'identity_coefficients_checked': len(residuals)}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {'status': 'FAIL', 'failures': failures + [f'malformed certificate: {exc}']}


def verify_all():
    records = []
    found = set()
    for path in sorted(ARTIFACTS.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        found.add((payload['alpha'], payload['level']))
        records.append({'file': path.name, **verify_certificate(payload)})
    failures = [r['file'] for r in records if r['status'] != 'PASS']
    if any(not r.get('reduced_grams_positive_definite', False) for r in records):
        failures.append('stored reduced Gram matrices lack the stated exact positive margin')
    if found != EXPECTED or len(records) != len(EXPECTED):
        failures.append('expected exactly the five stated certificates')
    return {'status': 'FAIL' if failures else 'PASS', 'failures': failures, 'records': records}


if __name__ == '__main__':
    from unittest.mock import patch
    import cvxpy as cp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with patch.object(cp.Problem, 'solve', side_effect=RuntimeError('SDP calls disabled')):
        report = verify_all()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
