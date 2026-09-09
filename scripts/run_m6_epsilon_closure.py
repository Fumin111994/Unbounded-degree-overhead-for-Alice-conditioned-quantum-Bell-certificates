#!/usr/bin/env python3
"""M6 fallback: exact closure certificates up to an explicit rational epsilon.

For rational alpha below the critical tilt, certify with solver-free exact
arithmetic

    omega_os^2(alpha) <= B,  where B is rational and B < q(alpha) + eps,

so the one-sided level-2 value is pinned to the explicit interval
[q(alpha), B] with B - q(alpha) < eps (exact comparison over the quadratic
field).  This is the fallback if the exact boundary certificate (bound
exactly q(alpha)) cannot be produced.

Output: artifacts/m6/epsilon_closure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exact_level1 import find_onesided_dual  # noqa: E402
from run_m6_closure import exact_tilted_functional, quantum_alg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument("--alphas", nargs="+", default=["5/4", "41/32"])
    parser.add_argument("--epsilon", default="1/1000000")
    args = parser.parse_args()

    eps = Fraction(args.epsilon)
    results = []
    failures: list[str] = []
    for text in args.alphas:
        alpha = Fraction(text)
        functional = exact_tilted_functional(alpha)
        q_alg = quantum_alg(alpha)
        # rational bound: q + eps rounded up to the next 1e-9
        bound = Fraction(float(q_alg)).limit_denominator(10**9) + eps
        payload = find_onesided_dual(
            args.solver,
            args.max_denominator,
            positivity=False,
            functional=functional,
            bound=bound,
            level=2,
        )
        status = payload["verification"]["status"]
        # exact check: bound < q + 2 eps  <=>  (bound - 2 eps)^2 < q^2
        rest = sp.Rational(bound.numerator, bound.denominator) - 2 * sp.Rational(
            eps.numerator, eps.denominator
        )
        within = sp.simplify(rest**2 - (8 + 2 * sp.Rational(
            alpha.numerator, alpha.denominator) ** 2)) < 0 and rest > 0
        record = {
            "alpha": str(alpha),
            "bound_rational": f"{bound.numerator}/{bound.denominator}",
            "bound_float": float(bound),
            "epsilon": str(eps),
            "bound_within_q_plus_2eps": bool(within),
            "verification": payload["verification"],
            "certificate": payload["certificate"],
            "solver_used_to_find": payload["solver_used_to_find"],
            "numeric_value": payload["numeric_value"],
        }
        print(f"alpha={alpha}: bound={float(bound):.9f} within_eps={within} {status}",
              flush=True)
        results.append(record)
        if status != "PASS" or not within:
            failures.append(f"epsilon-closure at {alpha} failed")

    args.output.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": 2,
        "kind": "closure certificates up to explicit rational epsilon",
        "results": results,
        "statement": (
            "omega_os^2(alpha) in [q(alpha), q(alpha) + 2 eps] with eps = "
            f"{eps}, exact rational certificates; the lower bound is the "
            "structural relaxation/quantum argument.  Full certificate "
            "payloads (S blocks, L, nu) are stored per alpha for "
            "solver-free re-verification"
        ),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (args.output / "epsilon_closure.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
