#!/usr/bin/env python3
"""M2.6: exact level-one value of tilted CHSH for the standard hierarchy.

Proves, with symbolic-alpha exact arithmetic and no SDP solver:

    omega_std^1(tilted CHSH, alpha) = 2 sqrt(2) + alpha   for all 0 <= alpha < 2

for the raw CFNZ Eq. (2.7) cone (no probability-positivity constraints).

- Lower bound: the alpha-independent moment matrix RAW_WITNESS (PSD with exact
  eigenvalues 0 (x3), (7 + sqrt(2) +/- sqrt(28 + 10 sqrt(2)))/4) attains
  objective 2 sqrt(2) + alpha for every alpha.
- Upper bound: the degree-1 SOS certificate

    (2 sqrt(2) + alpha) - F_alpha
      = (1/sqrt(2)) (A0 - (B0 + B1)/sqrt(2))^2
      + (1/sqrt(2)) (A1 - (B0 - B1)/sqrt(2))^2
      + 2 alpha (I - P_{A0:0})^2

  where A_x = 2 P_{A_x:0} - I, B_y = 2 P_{B_y:0} - I, verified as a word-level
  identity in the PVM quotient algebra with alpha symbolic.  All square
  coefficients are nonnegative for alpha >= 0.

Output: artifacts/m26/family/standard_family.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2 import exact  # noqa: E402
from npa2.bell import Scenario  # noqa: E402

ALPHA = sp.Symbol("alpha", nonnegative=True)
SQRT2 = sp.sqrt(2)

# Alpha-independent level-1 moment matrix attaining 2 sqrt(2) + alpha, in the
# word order [I, A0:0, A1:0, B0:0, B1:0].  Eigenvalues: 0 (x3) and
# (7 + sqrt(2) +/- sqrt(28 + 10 sqrt(2)))/4 > 0.
_s = (2 + SQRT2) / 4
_u = (1 + SQRT2) / 4
_h = sp.Rational(1, 2)
_q = sp.Rational(1, 4)
RAW_WITNESS = sp.Matrix(
    [
        [sp.Integer(1), sp.Integer(1), _h, _s, _s],
        [sp.Integer(1), sp.Integer(1), _h, _s, _s],
        [_h, _h, _h, _u, _q],
        [_s, _s, _u, _s, _u],
        [_s, _s, _q, _u, _s],
    ]
)


def tilted_polynomial(alpha) -> exact.SymPoly:
    """Game polynomial alpha * E_A0 + CHSH in the reduced PVM quotient."""

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
    return exact.sym_add(exact.sym_scale(a0, alpha), chsh)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m26" / "family"
    )
    args = parser.parse_args()

    algebra = Scenario(2, 2, 2, 2).algebra()
    functional_poly = tilted_polynomial(ALPHA)

    # --- Upper bound: parametric degree-1 SOS certificate -------------------
    a0 = exact.sym_observable(algebra, "A", 0)
    a1 = exact.sym_observable(algebra, "A", 1)
    b0 = exact.sym_observable(algebra, "B", 0)
    b1 = exact.sym_observable(algebra, "B", 1)
    squares = [
        (
            1 / SQRT2,
            exact.sym_add(
                a0, exact.sym_scale(exact.sym_add(b0, b1), -1 / SQRT2)
            ),
        ),
        (
            1 / SQRT2,
            exact.sym_add(
                a1,
                exact.sym_scale(
                    exact.sym_add(b0, exact.sym_scale(b1, sp.Integer(-1))),
                    -1 / SQRT2,
                ),
            ),
        ),
        (
            2 * ALPHA,
            exact.sym_add(
                exact.sym_identity(),
                exact.sym_scale(
                    exact.sym_projector(algebra, "A", 0, 0), sp.Integer(-1)
                ),
            ),
        ),
    ]
    bound = 2 * SQRT2 + ALPHA
    sos_report = exact.verify_sos_identity(algebra, bound, functional_poly, squares)

    # --- Lower bound: alpha-independent exact witness -----------------------
    witness_psd = RAW_WITNESS.is_positive_semidefinite
    words = algebra.words(1)
    expected_order = [(), (("A", 0, 0),), (("A", 1, 0),), (("B", 0, 0),), (("B", 1, 0),)]
    order_ok = list(words) == expected_order
    # Moment consistency: entry (i, j) must equal the representative entry.
    from npa2.model import NPAProblem  # noqa: F401  (documentation anchor)
    from npa2.standard import build_standard_problem
    from npa2.bell import tilted_chsh

    model = build_standard_problem(
        tilted_chsh(0.5), 1, enforce_probability_positivity=False
    )
    representatives = model.representatives
    consistency_failures: list[str] = []
    for i, left in enumerate(words):
        left_adjoint = algebra.dagger(left)
        for j in range(i, len(words)):
            product_word = algebra.multiply_words(left_adjoint, words[j])
            if product_word is None:
                if RAW_WITNESS[i, j] != 0:
                    consistency_failures.append(f"zero word at ({i},{j})")
                continue
            key = algebra.moment_key(product_word)
            rep_i, rep_j = representatives[key]
            if sp.simplify(RAW_WITNESS[i, j] - RAW_WITNESS[rep_i, rep_j]) != 0:
                consistency_failures.append(f"inconsistency at ({i},{j})")
    if RAW_WITNESS[representatives[()][0], representatives[()][1]] != 1:
        consistency_failures.append("normalization violated")

    # Objective with symbolic alpha: <F_alpha, witness>.
    objective = sp.Integer(0)
    for word, coeff in functional_poly.items():
        key = algebra.moment_key(word)
        i, j = representatives[key]
        objective += coeff * RAW_WITNESS[i, j]
    objective = sp.simplify(objective)
    objective_ok = sp.simplify(objective - bound) == 0

    failures = (
        sos_report["failures"]
        + consistency_failures
        + ([] if witness_psd else ["witness not PSD"])
        + ([] if objective_ok else [f"witness objective {objective} != {bound}"])
        + ([] if order_ok else ["unexpected word order"])
    )
    report = {
        "schema_version": 1,
        "statement": (
            "standard level-1 value of tilted CHSH (raw CFNZ Eq. (2.7) cone) "
            "is exactly 2*sqrt(2) + alpha for all 0 <= alpha < 2"
        ),
        "upper_bound_sos": sos_report,
        "witness": {
            "matrix": [[str(RAW_WITNESS[i, j]) for j in range(5)] for i in range(5)],
            "psd_exact": bool(witness_psd),
            "eigenvalues_exact": [str(e) for e in RAW_WITNESS.eigenvals()],
            "objective_symbolic": str(objective),
            "moment_consistency_failures": consistency_failures,
        },
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "standard_family.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "failures": failures,
                      "objective": str(objective)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
