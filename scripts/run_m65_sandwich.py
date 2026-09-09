#!/usr/bin/env python3
"""M7 (reviewer sandwich): exact strictness of Q-tilde^beh subset O_1^beh.

Produces an exact rational one-sided level-one primal witness at alpha=1/4
whose objective exceeds the (almost-)quantum value sqrt(8 + 2 alpha^2):
omega_1^os(F_{1/4}) >= rational > sqrt(8.125).  Since the almost-quantum
value of tilted CHSH equals the quantum value (Bamps-Pironio), this
certifies O_1^beh strictly contains Q-tilde^beh.

Output: artifacts/m65/sandwich.json
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

from npa2 import exact  # noqa: E402
from run_m3_exactify import find_onesided_witness  # noqa: E402
from run_m6_closure import exact_tilted_functional  # noqa: E402

ALPHA = Fraction(1, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m65")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**8)
    args = parser.parse_args()

    functional = exact_tilted_functional(ALPHA)
    witness = find_onesided_witness(
        functional, args.solver, args.max_denominator,
        epsilon=Fraction(1, 1000), level=1,
    )
    report = witness["verification"]
    objective = Fraction(report["objective_exact"])
    q = sp.sqrt(8 + 2 * sp.Rational(1, 4) ** 2)
    above = sp.simplify(
        sp.Rational(objective.numerator, objective.denominator) ** 2
        - (sp.Rational(8) + 2 * sp.Rational(1, 4) ** 2)
    ) > 0 and objective > 0
    failures = list(report["failures"])
    if not above:
        failures.append(f"witness objective {objective} not above {q}")

    out = {
        "schema_version": 1,
        "kind": "one-sided level-1 primal witness above almost-quantum",
        "alpha": "1/4",
        "objective": f"{objective.numerator}/{objective.denominator}",
        "objective_float": float(objective),
        "almost_quantum_value": str(q),
        "objective_above_almost_quantum": bool(above),
        "witness": witness,
        "verification": report,
        "statement": (
            "omega_1^os(F_{1/4}) >= rational objective > sqrt(8 + 2 alpha^2) "
            "= almost-quantum value, so O_1^beh strictly contains "
            "Q-tilde^beh"
        ),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sandwich.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"],
                      "objective_float": float(objective),
                      "q_float": float(q)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
