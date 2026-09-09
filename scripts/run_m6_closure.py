#!/usr/bin/env python3
"""M6: exact level-two closure/non-closure certificates for tilted CHSH.

Two sides of the critical tilt:

1. ``closure``: for rational alpha below the threshold, prove
   omega_os^2(alpha) <= sqrt(8 + 2 alpha^2) (= the quantum value, hence
   equality) by an exact nice degree-2 dual certificate whose bound is the
   ALGEBRAIC number sqrt(8 + 2 alpha^2).  The certificate is built by solving
   the identity with a symbolic bound t and substituting the algebraic value;
   PSD is checked exactly over the quadratic field.

2. ``nonclosure``: for rational alpha above the threshold, prove
   omega_os^2(alpha) > sqrt(8 + 2 alpha^2) by an exact one-sided level-2
   primal witness whose (rational) objective exceeds the algebraic bound
   (exact comparison by squaring).

Solvers are used only to find candidates; verification is exact.
Output: artifacts/m6/closure/<alpha_tag>.json, artifacts/m6/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

import numpy as np
import sympy as sp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from npa2 import exact  # noqa: E402
from npa2.bell import BellFunctional, Scenario, tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from run_m3_exactify import find_onesided_witness  # noqa: E402

T = sp.Symbol("t")


def exact_tilted_functional(alpha: Fraction) -> BellFunctional:
    signs = {0: Fraction(1), 1: Fraction(-1)}
    joint = {}
    for (x, y), c in {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1}.items():
        for a in (0, 1):
            for b in (0, 1):
                joint[(x, y, a, b)] = Fraction(c) * signs[a] * signs[b]
    alice_local = {(0, a): alpha * signs[a] for a in (0, 1)}
    return BellFunctional(
        name=f"tilted_chsh_exact_alpha_{alpha}",
        scenario=Scenario(2, 2, 2, 2),
        alice_local=alice_local,
        joint=joint,
    )


def quantum_alg(alpha: Fraction):
    return sp.sqrt(8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2)


def encode_fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# Closure side: exact nice dual certificate at the algebraic quantum bound
# ---------------------------------------------------------------------------


def symbolic_identity_system(functional: BellFunctional, level: int):
    """A, b0, c such that the certificate identity is A z = b0 + t*c."""

    names, A, b0 = exact.onesided_identity_system(functional, level, Fraction(0), False)
    _, _, b1 = exact.onesided_identity_system(functional, level, Fraction(1), False)
    return names, A, b0, b1 - b0


def find_closure_certificate(alpha: Fraction, solver: str, max_denominator: int) -> dict:
    functional = exact_tilted_functional(alpha)
    float_functional = tilted_chsh(float(alpha))
    model = build_onesided_problem(
        float_functional, 2, enforce_probability_positivity=False
    )
    summary, arrays = solve_problem(model, solver)
    numeric_value = summary["value"]
    q_alg = quantum_alg(alpha)
    if abs(numeric_value - float(q_alg)) > 1e-5:
        raise RuntimeError(
            f"one-sided L2 not at quantum at alpha={alpha}: "
            f"{numeric_value} vs {float(q_alg)}"
        )

    names, A, b0, c = symbolic_identity_system(functional, 2)
    minv = (A * A.T).inv()
    z0 = A.T * minv * b0
    z1 = A.T * minv * c
    null = A.nullspace()
    N = sp.Matrix.hstack(*null) if null else sp.Matrix.zeros(A.cols, 0)

    # numeric optimal dual vector at the numeric optimum t* ~ quantum
    cons_dual = {
        item.name: np.asarray(item.constraint.dual_value)
        for item in model.constraints
        if item.constraint.dual_value is not None
    }
    numeric_vector: list[float] = []
    for name in names:
        parts = name.split(":")
        if name.startswith("S_"):
            x, a = parts[0][2:].split("_")
            numeric_vector.append(
                float(arrays[f"dual_psd__Phi_x{x}_a{a}"][int(parts[1]), int(parts[2])])
            )
        elif name.startswith("L_"):
            numeric_vector.append(
                -float(cons_dual[f"consistency:x{parts[0][2:]}"][int(parts[1]), int(parts[2])])
            )
        elif name == "nu":
            numeric_vector.append(-float(cons_dual["normalization"]))
        else:
            raise ValueError(name)

    z_num = sp.Matrix([exact.rationalize(v, max_denominator) for v in numeric_vector])
    t_star = sp.Rational(numeric_value).limit_denominator(10**9)
    # fit the null-space coordinates of the numeric dual around z_p(t*)
    target = z_num - (z0 + t_star * z1)
    if N.cols:
        w = (N.T * N).inv() * N.T * target
        z_t = z0 + T * z1 + N * w
    else:
        z_t = z0 + T * z1

    # exact identity check with symbolic t
    s_blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, list(z_t))
    residuals = exact._onesided_identity_residuals(
        functional, 2, T, s_blocks, lam, nu, mu,
        exact.onesided_certificate_unknowns(functional, 2, False)["symbols"],
        exact.onesided_certificate_unknowns(functional, 2, False)["entry_key"],
        exact.onesided_certificate_unknowns(functional, 2, False)["block_names"],
        exact.onesided_certificate_unknowns(functional, 2, False)["words"],
        exact.onesided_certificate_unknowns(functional, 2, False)["algebra"],
        exact.onesided_certificate_unknowns(functional, 2, False)["scenario"],
    )
    residual_failures = [str(r) for r in residuals if sp.simplify(r) != 0]

    # substitute the algebraic quantum bound and check PSD exactly.
    # Direct PSD check fails because the optimal dual sits on the boundary;
    # instead use the limit argument: S(t) is affine in t with RATIONAL
    # coefficients, so if
    #   (i)  S(t1) is strictly PSD at a rational t1 > q, and
    #   (ii) det S(t) (a polynomial over QQ) has no roots strictly above q
    #        in (r_lo, t1], except possibly a root AT q itself
    #        (boundary point, allowed),
    # then S(t) is positive definite on (q, t1] and S(q) is PSD by closedness.
    failures = list(residual_failures)
    q = q_alg
    m = sp.Rational(8) + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2
    q_float = float(q)
    t1 = sp.Rational(int(np.ceil(q_float * 10**6)) + 1, 10**6)
    r_lo = sp.Rational(int(np.floor(q_float * 10**4)), 10**4)
    t1 = max(t1, r_lo + sp.Rational(1, 10**6))
    minpoly = sp.Poly(T**2 - m, T)

    def isolated_roots_above_q(det_poly: sp.Poly) -> int:
        """Number of real roots of det_poly strictly above q in [r_lo, t1]."""

        # divide out the (T^2 - m) factor: a root exactly at q is allowed
        quotient = det_poly
        for _ in range(5):
            q2, r2 = sp.div(quotient, minpoly)
            if sp.simplify(r2.as_expr()) != 0:
                break
            quotient = q2
        if sp.expand(quotient.as_expr()) == 0:
            return -1  # determinant identically zero
        q_root = sp.CRootOf(T**2 - m, 1)
        violations = 0
        for root in sp.Poly(quotient.as_expr(), T).all_roots(radicals=False):
            if not root.is_real:
                continue
            if root > q_root and root >= r_lo and root <= t1:
                violations += 1
        return violations

    for (x, a), entries in s_blocks.items():
        n = 5
        matrix_t = sp.zeros(n)
        matrix_t1 = sp.zeros(n)
        for (i, j), value in entries.items():
            matrix_t[i, j] = sp.simplify(value)
            matrix_t[j, i] = matrix_t[i, j]
            v1 = sp.simplify(value.subs(T, t1))
            matrix_t1[i, j] = v1
            matrix_t1[j, i] = v1
        tag = f"S block {(x, a)}"
        if not exact.is_psd_exact(matrix_t1):
            failures.append(f"{tag} not PSD at rational t1={t1}")
            continue
        det_expr = sp.expand(matrix_t.det())
        if det_expr == 0:
            failures.append(f"{tag}: determinant identically zero in t")
            continue
        det_poly = sp.Poly(det_expr, T)
        violations = isolated_roots_above_q(det_poly)
        if violations != 0:
            failures.append(f"{tag}: determinant roots above q ({violations})")
    return {
        "kind": "one-sided level-2 closure certificate at algebraic bound",
        "alpha": str(alpha),
        "bound": str(q_alg),
        "numeric_value": numeric_value,
        "rational_t1": str(t1),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


# ---------------------------------------------------------------------------
# Non-closure side: exact primal witness above the algebraic bound
# ---------------------------------------------------------------------------


def find_nonclosure_witness(alpha: Fraction, solver: str,
                            max_denominator: int) -> dict:
    functional = exact_tilted_functional(alpha)
    # epsilon must satisfy (1 - eps) * omega_os^2 > quantum(alpha); the
    # reference-point lift handles the rationalization boundary error.
    witness = find_onesided_witness(
        functional, solver, max_denominator, epsilon=Fraction(1, 100000)
    )
    q_alg = quantum_alg(alpha)
    objective = Fraction(witness["verification"]["objective_exact"])
    obj_alg = sp.Rational(objective.numerator, objective.denominator)
    above = sp.simplify(obj_alg**2 - (8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2)) > 0 and objective > 0
    failures = list(witness["verification"]["failures"])
    if not above:
        failures.append(f"witness objective {objective} not above {q_alg}")
    witness.update(
        {
            "kind": "one-sided level-2 non-closure witness",
            "alpha": str(alpha),
            "bound": str(q_alg),
            "objective": str(objective),
            "objective_float": float(objective),
            "objective_above_bound": bool(above),
            "failures": failures,
            "status": "PASS" if not failures else "FAIL",
        }
    )
    return witness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument("--witness-max-denominator", type=int, default=10**8)
    parser.add_argument("--closure-alphas", nargs="+", default=["5/4", "41/32"])
    parser.add_argument("--nonclosure-alphas", nargs="+", default=["13/10", "3/2"])
    args = parser.parse_args()

    results = {"closure": [], "nonclosure": []}
    failures: list[str] = []
    for text in args.closure_alphas:
        alpha = Fraction(text)
        print(f"closure alpha={alpha} ...", flush=True)
        result = find_closure_certificate(alpha, args.solver, args.max_denominator)
        print(f"  status={result['status']} failures={result['failures'][:2]}", flush=True)
        results["closure"].append(result)
        if result["status"] != "PASS":
            failures.append(f"closure at {alpha} failed")
    for text in args.nonclosure_alphas:
        alpha = Fraction(text)
        print(f"nonclosure alpha={alpha} ...", flush=True)
        result = find_nonclosure_witness(alpha, args.solver, args.witness_max_denominator)
        print(f"  status={result['status']} failures={result['failures'][:2]}", flush=True)
        results["nonclosure"].append(result)
        if result["status"] != "PASS":
            failures.append(f"nonclosure at {alpha} failed")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps({"results": results, "status": "PASS" if not failures else "FAIL",
                    "failures": failures}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS" if not failures else "FAIL",
                      "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
