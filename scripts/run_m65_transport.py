#!/usr/bin/env python3
"""M6.5 G3(iii): transport control — exact one-sided L2 certificate for a
functional where the standard level-2 certificate DOES transport.

For a TRANSPORT_EXAMPLE candidate, produce an exact rational one-sided
level-2 dual certificate with bound B = omega_std^2 + eps (eps = 1e-6):
proving omega_os^2(G) <= omega_std^2(G) + eps exactly, i.e. the two cones
agree on this functional up to 1e-6 (contrast with the separated candidates).

Output: artifacts/m65/transport_control.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402
from run_exact_level1 import find_onesided_dual  # noqa: E402
from run_m3_exactify import rationalize_functional  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "artifacts" / "m3" / "search" / "results.json",
    )
    parser.add_argument("--index", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m65")
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()

    report_in = json.loads(args.results.read_text(encoding="utf-8"))
    record = next(r for r in report_in["records"] if r["index"] == args.index)
    if record["flag"] != "TRANSPORT_EXAMPLE":
        raise RuntimeError(f"candidate {args.index} is not a transport example")
    functional = rationalize_functional(record)

    std2 = solve_problem(
        build_standard_problem(functional, 2, enforce_probability_positivity=False),
        args.solver,
    )[0]["value"]
    os2 = solve_problem(
        build_onesided_problem(functional, 2, enforce_probability_positivity=False),
        args.solver,
    )[0]["value"]
    bound = Fraction(int(np.ceil(os2 * 10**6)) + 1, 10**6)
    payload = find_onesided_dual(
        args.solver, 10**6, positivity=False, functional=functional,
        bound=bound, level=2,
    )
    eps = bound - Fraction(std2).limit_denominator(10**9)
    out = {
        "schema_version": 1,
        "candidate": f"{args.results.parent.name}_{args.index}",
        "std2": std2,
        "os2": os2,
        "bound": f"{bound.numerator}/{bound.denominator}",
        "bound_minus_std2": str(eps),
        "verification": payload["verification"]["status"],
        "verification_failures": payload["verification"]["failures"],
        "statement": (
            f"omega_os^2(G) <= {float(bound):.9f} <= omega_std^2(G) + "
            f"{float(eps):.2e}: the certificate transports up to eps "
            "(contrast with the separated candidates)"
        ),
        "certificate": payload["certificate"],
        "status": "PASS" if payload["verification"]["status"] == "PASS" else "FAIL",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transport_control.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: out[k] for k in ("std2", "os2", "bound",
                                          "bound_minus_std2", "status")},
                     indent=2))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
