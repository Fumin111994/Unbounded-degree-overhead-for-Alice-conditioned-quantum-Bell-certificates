#!/usr/bin/env python3
"""M6: exact one-sided LEVEL-3 certificates for tilted CHSH, tight bounds.

Regenerates artifacts/m6/o3_certificates.json at alpha = 3/2 and 13/10 with
rational bounds B certified to satisfy B - q(alpha) < 1e-6 EXACTLY (rational
square comparison against q^2 = 8 + 2 alpha^2), fixing the previous version
whose bounds landed at 1.000067e-6 and 1.000313e-6 above q.

For each alpha the script tries eps from a tightening schedule until the
exact certificate verifies and the exact comparison B - q < 1e-6 holds:

    B - q < 1e-6  <=>  B - 1e-6 <= 0  or  (B - 1e-6)^2 < 8 + 2 alpha^2 ,

both sides rational.  The full certificate payload (S blocks, L, nu) is
stored, matching the previous artifact's structure.

No solver is used for verification; the solver only seeds the rational
candidate.  Output: artifacts/m6/o3_certificates.json
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
from run_m6_closure import exact_tilted_functional  # noqa: E402

GAP = Fraction(1, 10**6)  # certified budget: B - q < GAP


def certified_gap_ok(bound: Fraction, alpha: Fraction) -> bool:
    """Exact check bound - q < 1e-6 via squaring (both sides rational)."""
    rest = sp.Rational(bound.numerator, bound.denominator) - sp.Rational(
        GAP.numerator, GAP.denominator
    )
    if rest <= 0:
        return True
    q2 = 8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2
    return bool(sp.simplify(rest**2 - q2) < 0)


def certified_above_q(bound: Fraction, alpha: Fraction) -> bool:
    """Exact check bound > q (so [q, B] is a genuine pin interval)."""
    b = sp.Rational(bound.numerator, bound.denominator)
    q2 = 8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2
    return bool(sp.simplify(b**2 - q2) > 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument("--alphas", nargs="+", default=["3/2", "13/10"])
    parser.add_argument(
        "--eps-schedule",
        nargs="+",
        default=["9/10000000", "95/100000000", "99/100000000"],
        help="tightening schedule for the rational bound offset",
    )
    args = parser.parse_args()

    results = []
    failures: list[str] = []
    for text in args.alphas:
        alpha = Fraction(text)
        functional = exact_tilted_functional(alpha)
        q_float = float(sp.sqrt(8 + 2 * float(alpha) ** 2))
        chosen = None
        attempts = []
        for eps_text in args.eps_schedule:
            eps = Fraction(eps_text)
            bound = Fraction(
                int(sp.ceiling(q_float * 10**9)), 10**9
            ) + eps
            payload = find_onesided_dual(
                args.solver,
                args.max_denominator,
                positivity=False,
                functional=functional,
                bound=bound,
                level=3,
            )
            status = payload["verification"]["status"]
            gap_ok = certified_gap_ok(bound, alpha)
            above_q = certified_above_q(bound, alpha)
            attempts.append({
                "epsilon": str(eps),
                "bound": f"{bound.numerator}/{bound.denominator}",
                "verification": status,
                "bound_minus_q_certified_below_1e-6": gap_ok,
                "bound_certified_above_q": above_q,
            })
            print(f"alpha={alpha} eps={eps}: verify={status} "
                  f"gap_ok={gap_ok} above_q={above_q}", flush=True)
            if status == "PASS" and gap_ok and above_q:
                chosen = (bound, eps, payload)
                break
        if chosen is None:
            failures.append(f"{alpha}: no epsilon schedule entry succeeded")
            continue
        bound, eps, payload = chosen
        record = {
            "alpha": str(alpha),
            "level": 3,
            "bound": f"{bound.numerator}/{bound.denominator}",
            "epsilon": str(eps),
            "bound_float": float(bound),
            "bound_minus_q_certified_below_1e-6": True,
            "bound_certified_above_q": True,
            "verification": payload["verification"],
            "certificate": payload["certificate"],
            "attempts": attempts,
        }
        results.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": 2,
        "kind": "exact one-sided level-3 certificates, tight rational bounds",
        "results": results,
        "statement": (
            "omega_os^3(alpha) <= B with rational B certified EXACTLY to "
            "satisfy B - q(alpha) < 1e-6 (rational square comparison), at "
            "the listed points only; certificates are exact rational "
            "one-sided level-3 dual certificates, verified without solvers"
        ),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (args.output / "o3_certificates.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"], "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
