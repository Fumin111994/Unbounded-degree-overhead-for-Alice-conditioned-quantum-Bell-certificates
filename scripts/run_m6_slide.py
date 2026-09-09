#!/usr/bin/env python3
"""Historical M6 sliding attempt (invalid target, retained for audit).

The proposed B' < q certificate contradicts the attainable quantum value.
The attempt now fails before invoking any solver. Its failure is not
evidence of high algebraic degree or nonexistence of a certificate at q.
Use run_quantum_face.py for exact optimal-bound certificates instead.

Given an exact rational dual certificate z* at bound B = q + eps (epsilon
closure), try to extend it DOWNWARD through an affine family

    z(t) = z* + (t - B) D,     A D = c   (the identity's t-direction)

such that every S block of z(t) stays PSD on a rational interval [B', B]
with B' < q.  Then S(q) is PSD by convexity, proving omega_os^2 <= q exactly.

The direction space of A D = c is solved exactly (rational particular +
null space); the small sub-SDP  S*_i - eps dS_i(w) >= 0  is solved
numerically (cvxpy), rationalized, and then verified exactly (LDL).  If the
sub-SDP is infeasible, the slide does not reach q and we report that.

Output: artifacts/m6/slide.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from npa2 import exact  # noqa: E402
from npa2.bell import tilted_chsh  # noqa: E402
from run_exact_level1 import find_onesided_dual  # noqa: E402
from run_m6_closure import exact_tilted_functional, quantum_alg  # noqa: E402


def rational_s_blocks(payload, names):
    cert = payload["certificate"]
    blocks: dict[tuple[int, int], dict[tuple[int, int], Fraction]] = {}
    for name, value in cert.items():
        parts = name.split(":")
        if name.startswith("S_"):
            x, a = parts[0][2:].split("_")
            num, _, den = value.partition("/")
            blocks.setdefault((int(x), int(a)), {})[
                (int(parts[1]), int(parts[2]))
            ] = Fraction(int(num), int(den or "1"))
    return blocks


def slide_attempt(alpha: Fraction, eps: Fraction, solver: str,
                  max_denominator: int) -> dict:
    functional = exact_tilted_functional(alpha)
    q = quantum_alg(alpha)
    q_float = float(q)
    # rational bounds bracketing q
    B = Fraction(int(np.ceil((q_float) * 10**9)), 10**9) + eps
    Bp = Fraction(int(np.floor(q_float * 10**9)), 10**9) - eps
    slide = B - Bp  # rational, 2 eps + rounding

    if sp.Rational(Bp) < q:
        return {"alpha": str(alpha), "status": "FAIL",
                "reason_code": "INVALID_BELOW_QUANTUM_TARGET",
                "failures": ["B' < q is incompatible with the attainable quantum value; "
                             "the historical sliding target is mathematically infeasible."]}

    # exact rational certificate at B (raw cone, level 2)
    payload = find_onesided_dual(
        solver, max_denominator, positivity=False, functional=functional,
        bound=B, level=2,
    )
    if payload["verification"]["status"] != "PASS":
        return {"alpha": str(alpha), "status": "FAIL",
                "failures": ["epsilon certificate failed"]}

    names, A, b0 = exact.onesided_identity_system(functional, 2, Fraction(0), False)
    _, _, b1 = exact.onesided_identity_system(functional, 2, Fraction(1), False)
    c = b1 - b0  # identity: A z = b0 + t * c
    minv = (A * A.T).inv()
    D0 = A.T * minv * c                      # particular direction
    null = A.nullspace()
    N = sp.Matrix.hstack(*null) if null else sp.Matrix.zeros(A.cols, 0)
    if N.cols == 0:
        return {"alpha": str(alpha), "status": "FAIL",
                "failures": ["no identity null space"]}

    # certificate point z* at bound B
    cert = payload["certificate"]

    def _dec(s: str) -> Fraction:
        num, _, den = s.partition("/")
        return Fraction(int(num), int(den or "1"))

    z_star = [_dec(cert[name]) for name in names]

    # numerical sub-SDP over the null-space coordinates w:
    # find D = D0 + N w minimizing the worst sliding depth, i.e. maximize rho
    # such that  S*_i - slide * dS_i(w)  >=  0  (we then check rho >= 1).
    s_idx = [k for k, name in enumerate(names) if name.startswith("S_")]
    blocks = sorted({(int(names[k].split(":")[0][2:].split("_")[0]),
                      int(names[k].split(":")[0][2:].split("_")[1]))
                     for k in s_idx})
    n_block = 5
    w = cp.Variable(N.cols)
    D_expr = sp.Matrix([D0[k] for k in range(A.cols)]) + N * sp.Matrix(
        [sp.Symbol(f"w_{i}") for i in range(N.cols)]
    )
    constraints = []
    for (x, a) in blocks:
        S_star = np.zeros((n_block, n_block))
        dS_entries = {}
        for k in s_idx:
            parts = names[k].split(":")
            bx, ba = int(parts[0][2:].split("_")[0]), int(parts[0][2:].split("_")[1])
            if (bx, ba) != (x, a):
                continue
            i, j = int(parts[1]), int(parts[2])
            S_star[i, j] = float(z_star[k])
            S_star[j, i] = float(z_star[k])
            dS_entries[(i, j)] = D_expr[k]
        # dS as cvxpy expression
        dS = cp.Constant(np.zeros((n_block, n_block)))
        for (i, j), expr in dS_entries.items():
            coeffs = [float(expr.coeff(sp.Symbol(f"w_{m}"))) for m in range(N.cols)]
            const = float(expr.subs({sp.Symbol(f"w_{m}"): 0 for m in range(N.cols)}))
            entry = const + sum(coeffs[m] * w[m] for m in range(N.cols))
            basis = np.zeros((n_block, n_block))
            basis[i, j] = 1.0
            if j != i:
                basis[j, i] = 1.0
            dS = dS + cp.Constant(basis) * entry
        constraints.append(
            cp.Constant(S_star) - slide.numerator / slide.denominator * dS >> 0
        )
    sub = cp.Problem(cp.Minimize(0), constraints)
    sub.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
              tol_feas=1e-10, max_iter=2000)
    report: dict = {
        "alpha": str(alpha),
        "B": f"{B.numerator}/{B.denominator}",
        "Bp": f"{Bp.numerator}/{Bp.denominator}",
        "slide": f"{slide.numerator}/{slide.denominator}",
        "sub_sdp_status": sub.status,
    }
    if sub.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        report.update(status="FAIL",
                      failures=["slide direction sub-SDP infeasible: no affine "
                                "family reaches below q"])
        return report

    # rationalize w and verify EXACTLY: A D = c (exact by construction) and
    # S*_i - slide * dS_i >= 0 (exact LDL)
    w_rat = [Fraction(float(v)).limit_denominator(max_denominator) for v in np.ravel(w.value)]
    D_exact = D0 + N * sp.Matrix(w_rat)
    failures: list[str] = []
    for (x, a) in blocks:
        M = sp.zeros(n_block)
        for k in s_idx:
            parts = names[k].split(":")
            bx, ba = int(parts[0][2:].split("_")[0]), int(parts[0][2:].split("_")[1])
            if (bx, ba) != (x, a):
                continue
            i, j = int(parts[1]), int(parts[2])
            value = sp.Rational(z_star[k]) - sp.Rational(slide) * D_exact[k]
            M[i, j] = sp.simplify(value)
            M[j, i] = M[i, j]
        if not exact.is_psd_exact(M):
            failures.append(f"slid block {(x, a)} not PSD at B'={Bp}")
    # also verify D is a valid direction: A D = c exactly
    direction_residual = A * D_exact - c
    if any(r != 0 for r in direction_residual):
        failures.append("direction does not satisfy A D = c exactly")
    report.update(
        status="PASS" if not failures else "FAIL",
        failures=failures,
        statement=(
            f"affine certificate family valid on [{Bp}, {B}] covering "
            f"q = {q}: S(q) PSD by convexity, hence omega_os^2({alpha}) <= q"
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument("--alphas", nargs="+", default=["5/4", "41/32"])
    parser.add_argument("--epsilon", default="1/1000000")
    args = parser.parse_args()

    results = []
    failures: list[str] = []
    for text in args.alphas:
        alpha = Fraction(text)
        print(f"slide alpha={alpha} ...", flush=True)
        result = slide_attempt(alpha, Fraction(args.epsilon), args.solver,
                               args.max_denominator)
        print(f"  {result['status']} {result.get('failures', [])[:2]}", flush=True)
        results.append(result)
        if result["status"] != "PASS":
            failures.append(f"slide at {alpha} failed")
    args.output.mkdir(parents=True, exist_ok=True)
    out = {"schema_version": 1, "results": results,
           "status": "PASS" if not failures else "FAIL", "failures": failures}
    (args.output / "slide.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"], "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
