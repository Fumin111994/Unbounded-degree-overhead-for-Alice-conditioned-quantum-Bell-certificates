#!/usr/bin/env python3
"""M2.6: exact one-sided upper certificates on a rational anchor grid, plus
the exact covering argument for the level-one separation on an interval.

For each anchor alpha_i (dyadic rational) the script produces an exact
rational dual certificate omega_os^1(alpha_i) <= U_i (same machinery as
scripts/run_exact_level1.py, raw CFNZ Definition 4.1 cone).  The covering
argument then extends the separation to all alpha in an interval:

- omega_os^1(alpha) is CONVEX in alpha: it is the support function of the
  one-sided level-1 moment body composed with the affine map alpha -> F_alpha.
  Hence between anchors the chord of the U_i is an upper bound, and it lies
  strictly below the linear function 2 sqrt(2) + alpha because it does so at
  both endpoints (checked exactly over Q(sqrt(2))).
- LIPSCHITZ: |E_A0| <= 1 on the one-sided body (normalization + block PSD
  force 0 <= m_{0,0}(I) <= 1), so omega_os^1 has Lipschitz constant 1 in
  alpha.  Below the first anchor alpha_0 this covers alpha > alpha_lo :=
  (U_0 + alpha_0 - 2 sqrt(2))/2; above the last anchor it covers everything
  up to 2 because U_last < 2 sqrt(2) + alpha_last.

Comparisons with 2 sqrt(2) + alpha are exact: for rational r > 0,
r < 2 sqrt(2)  iff  r^2 < 8, checked with Fraction arithmetic.

Output: artifacts/m26/anchors/{anchor_*.json, coverage.json}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exact_level1 import find_onesided_dual  # noqa: E402
from npa2 import exact  # noqa: E402
from npa2.bell import Scenario, tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402


def onesided_chsh_endpoint_certificate() -> dict:
    """Exact certificate omega_os^1(0) <= 2 sqrt(2).

    The classic degree-1 CHSH SOS certificate is already nice: each square
    involves only one Alice question.  Verified word-level in the quotient
    algebra; the matching lower bound omega_os^1(0) >= 2 sqrt(2) is the
    usual relaxation argument (quantum strategies embed at every level).
    """

    import sympy as sp

    algebra = Scenario(2, 2, 2, 2).algebra()
    a0 = exact.sym_observable(algebra, "A", 0)
    a1 = exact.sym_observable(algebra, "A", 1)
    b0 = exact.sym_observable(algebra, "B", 0)
    b1 = exact.sym_observable(algebra, "B", 1)
    chsh = exact.sym_add(
        exact.sym_multiply(algebra, a0, b0),
        exact.sym_multiply(algebra, a0, b1),
        exact.sym_multiply(algebra, a1, b0),
        exact.sym_scale(exact.sym_multiply(algebra, a1, b1), sp.Integer(-1)),
    )
    sqrt2 = sp.sqrt(2)
    squares = [
        (
            1 / sqrt2,
            exact.sym_add(a0, exact.sym_scale(exact.sym_add(b0, b1), -1 / sqrt2)),
        ),
        (
            1 / sqrt2,
            exact.sym_add(
                a1,
                exact.sym_scale(
                    exact.sym_add(b0, exact.sym_scale(b1, sp.Integer(-1))),
                    -1 / sqrt2,
                ),
            ),
        ),
    ]
    report = exact.verify_sos_identity(algebra, 2 * sqrt2, chsh, squares)
    # Niceness: every Alice generator in a square must use one question.
    nice_failures = []
    for _, poly in squares:
        questions = {g[1] for word in poly for g in word if g[0] == "A"}
        if len(questions) > 1:
            nice_failures.append(f"square mixes Alice questions {questions}")
    report["nice"] = not nice_failures
    report["nice_failures"] = nice_failures
    report["lower_bound_note"] = (
        "omega_os^1(0) >= 2 sqrt(2): quantum strategies embed at every level"
    )
    return report

DEFAULT_ANCHORS = [
    Fraction(1, 16),
    Fraction(1, 8),
    Fraction(1, 4),
    Fraction(3, 8),
    Fraction(1, 2),
    Fraction(3, 4),
    Fraction(1),
    Fraction(5, 4),
    Fraction(3, 2),
    Fraction(7, 4),
]


def separated_exactly(U: Fraction, alpha: Fraction) -> bool:
    """Exact check U < 2 sqrt(2) + alpha over Q(sqrt(2))."""

    rest = U - alpha
    return rest > 0 and rest * rest < 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m26" / "anchors"
    )
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="reload existing anchor artifacts and rebuild only coverage.json",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    anchors = sorted(DEFAULT_ANCHORS)
    records = []
    failures: list[str] = []

    for alpha in anchors:
        functional = tilted_chsh(float(alpha))
        if args.coverage_only:
            payload = json.loads(
                (args.output / f"anchor_{alpha.numerator}_{alpha.denominator}.json")
                .read_text(encoding="utf-8")
            )
            bound = Fraction(payload["bound"])
            numeric = payload["numeric_value"]
            # Re-verify the stored certificate solver-free; never trust the
            # stored status field.
            names = list(payload["certificate"].keys())
            values = [Fraction(payload["certificate"][n]) for n in names]
            s_blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, values)
            payload["verification"] = exact.verify_onesided_dual(
                functional, payload["level"], bound, s_blocks, lam, nu, mu,
                enforce_probability_positivity=payload.get(
                    "enforce_probability_positivity", False),
            )
        else:
            # provisional bound: 1e-6 above the numeric optimum
            numeric = solve_problem(
                build_onesided_problem(
                    functional, 1, enforce_probability_positivity=False
                ),
                args.solver,
            )[0]["value"]
            bound = Fraction(math.ceil(numeric * 10**6), 10**6)
            payload = find_onesided_dual(
                args.solver,
                args.max_denominator,
                positivity=False,
                functional=functional,
                bound=bound,
            )
            (args.output / f"anchor_{alpha.numerator}_{alpha.denominator}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        status = payload["verification"]["status"]
        strict = separated_exactly(bound, alpha)
        tag = f"alpha={alpha}"
        if status != "PASS":
            failures.append(f"{tag}: certificate verification failed")
        if not strict:
            failures.append(f"{tag}: no strict separation at anchor")
        record = {
            "alpha": str(alpha),
            "numeric_value": numeric,
            "bound": str(bound),
            "bound_float": float(bound),
            "verification": status,
            "strictly_below_standard": strict,
        }
        records.append(record)
        print(f"{tag}: os<={bound} ({float(bound):.6f}) "
              f"vs std={2.0**1.5 + float(alpha):.6f} {status} strict={strict}",
              flush=True)

    first, last = records[0], records[-1]
    # Exact endpoint: omega_os^1(0) = 2 sqrt(2), certified by the nice CHSH
    # SOS certificate below (+ relaxation argument for the lower bound).
    endpoint = onesided_chsh_endpoint_certificate()
    if endpoint["status"] != "PASS" or not endpoint["nice"]:
        failures.append("alpha=0 endpoint certificate failed")

    # Chord from the exact endpoint (0, 2 sqrt(2)) to the first anchor:
    # convexity bounds omega below the chord; separation needs chord slope
    # (U_0 - 2 sqrt(2)) / alpha_0 < 1, i.e. U_0 - alpha_0 < 2 sqrt(2).
    U0 = Fraction(first["bound"])
    alpha0 = Fraction(first["alpha"])
    endpoint_chord_ok = separated_exactly(U0, alpha0)
    if not endpoint_chord_ok:
        failures.append("chord from alpha=0 endpoint has slope >= 1")

    coverage = {
        "schema_version": 1,
        "endpoint_alpha0": {
            "value": "2*sqrt(2)",
            "certificate": endpoint,
        },
        "anchors": records,
        "convexity_note": (
            "omega_os^1 is a support function of the one-sided level-1 body "
            "composed with the affine alpha-dependence of the tilted-CHSH "
            "functional, hence convex; chords from the exact endpoint "
            "(0, 2 sqrt(2)) and between consecutive anchors are exact upper "
            "bounds and lie strictly below 2 sqrt(2) + alpha because they do "
            "at both endpoints (checked exactly over Q(sqrt(2)))."
        ),
        "lipschitz_note": (
            "Lipschitz constant 1 from |E_A0| <= 1 on the body; covers alpha "
            "above the last anchor up to 2 because U_last < 2 sqrt(2) + alpha_last."
        ),
        "endpoint_chord_slope_below_one": endpoint_chord_ok,
        "covered_interval": "(0, 2)",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (args.output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": coverage["status"],
                      "covered_interval": coverage["covered_interval"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
