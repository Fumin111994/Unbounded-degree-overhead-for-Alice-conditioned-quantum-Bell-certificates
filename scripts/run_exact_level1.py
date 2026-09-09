#!/usr/bin/env python3
"""M2.5 exactification of the tilted-CHSH alpha=1/2 native-level-one values.

Produces exact, solver-independent certificates for two variants:

``positivity`` (the M2 implementation, with redundant joint-probability
nonnegativity constraints):

1. a rational primal witness for the standard-PVM level-1 relaxation with
   objective exactly 31/10;
2. a rational dual certificate for the one-sided-PVM level-1 relaxation with
   bound exactly 596609/200000 = 2.983045.

``raw`` (the literal CFNZ definitions, Eq. (2.7) and Definition 4.1, with no
probability-positivity constraints):

3. an exact primal witness over Q(sqrt(2)) for the standard level-1 relaxation
   with objective exactly 1/2 + 2*sqrt(2);
4. a rational dual certificate for the one-sided level-1 relaxation with the
   same bound 596609/200000.

The SDP solver is used only to *find* the dual candidates (and to confirm the
witness values numerically); every verification step runs in exact arithmetic.
``--verify-only`` reloads the exported artifacts and re-verifies them without
any solver.

Output: artifacts/m25/exact/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2 import exact  # noqa: E402
from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402

ALPHA = 0.5
STANDARD_WITNESS_VALUE = Fraction(31, 10)
ONESIDED_BOUND = Fraction(596609, 200000)  # 2.983045
RAW_WITNESS_VALUE = sp.Rational(1, 2) + 2 * sp.sqrt(2)

# Exact standard level-1 witness for the raw cone (no probability
# positivity), in the word order [I, A0:0, A1:0, B0:0, B1:0].  It saturates
# Alice's marginal (m(A0) = 1) and attains the Tsirelson-optimal correlator
# part; the eigenvalues are 0 (multiplicity 3) and
# (7 + sqrt(2) +/- sqrt(28 + 10 sqrt(2)))/4 >= 0.
_s = (2 + sp.sqrt(2)) / 4
_u = (1 + sp.sqrt(2)) / 4
_h = sp.Rational(1, 2)
_q = sp.Rational(1, 4)
RAW_WITNESS_MATRIX = [
    [sp.Integer(1), sp.Integer(1), _h, _s, _s],
    [sp.Integer(1), sp.Integer(1), _h, _s, _s],
    [_h, _h, _h, _u, _q],
    [_s, _s, _u, _s, _u],
    [_s, _s, _q, _u, _s],
]


def encode(value) -> str:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return str(sp.nsimplify(value) if not value.is_number else value)


def decode(text: str):
    num, sep, den = text.partition("/")
    if sep and den.isdigit() and num.lstrip("-").isdigit():
        return Fraction(int(num), int(den))
    return sp.sympify(text)


# ---------------------------------------------------------------------------
# Standard primal witness
# ---------------------------------------------------------------------------


def find_standard_witness(solver: str, max_denominator: int) -> dict:
    functional = tilted_chsh(ALPHA)
    model = build_standard_problem(functional, 1)
    summary, arrays = solve_problem(model, solver)
    matrix = arrays["primal__Gamma_std"]
    n = len(model.words)
    entries = {
        (i, j): exact.rationalize(matrix[i, j], max_denominator)
        for i in range(n)
        for j in range(i, n)
    }
    report = exact.verify_standard_witness(functional, 1, entries)
    return {
        "schema_version": 1,
        "kind": "standard_pvm primal witness (with probability positivity)",
        "functional": functional.name,
        "level": 1,
        "enforce_probability_positivity": True,
        "solver_used_to_find": solver,
        "numeric_value": summary["value"],
        "gamma_entries": {f"{i}:{j}": encode(v) for (i, j), v in entries.items()},
        "verification": report,
    }


def find_standard_witness_raw(solver: str) -> dict:
    functional = tilted_chsh(ALPHA)
    model = build_standard_problem(
        functional, 1, enforce_probability_positivity=False
    )
    summary, _ = solve_problem(model, solver)
    n = len(model.words)
    entries = {
        (i, j): RAW_WITNESS_MATRIX[i][j] for i in range(n) for j in range(i, n)
    }
    report = exact.verify_standard_witness(
        functional, 1, entries, enforce_probability_positivity=False
    )
    return {
        "schema_version": 1,
        "kind": "standard_pvm primal witness (raw CFNZ Eq. (2.7) cone)",
        "functional": functional.name,
        "level": 1,
        "enforce_probability_positivity": False,
        "solver_used_to_find": ("constructed analytically; " + solver +
                                " only confirms the numeric optimum"),
        "numeric_value": summary["value"],
        "gamma_entries": {f"{i}:{j}": encode(v) for (i, j), v in entries.items()},
        "verification": report,
    }


def verify_standard_artifact(payload: dict) -> dict:
    functional = tilted_chsh(ALPHA)
    entries = {
        tuple(int(p) for p in key.split(":")): decode(value)
        for key, value in payload["gamma_entries"].items()
    }
    return exact.verify_standard_witness(
        functional,
        payload["level"],
        entries,
        enforce_probability_positivity=payload["enforce_probability_positivity"],
    )


# ---------------------------------------------------------------------------
# One-sided dual certificate
# ---------------------------------------------------------------------------


def find_onesided_dual(
    solver: str,
    max_denominator: int,
    *,
    positivity: bool,
    functional=None,
    bound: Fraction | None = None,
    level: int = 1,
) -> dict:
    functional = functional if functional is not None else tilted_chsh(ALPHA)
    bound = bound if bound is not None else ONESIDED_BOUND
    model = build_onesided_problem(
        functional, level, enforce_probability_positivity=positivity
    )
    summary, arrays = solve_problem(model, solver)
    numeric_value = summary["value"]
    if float(bound) <= numeric_value:
        raise RuntimeError(
            f"bound {float(bound)} is not above the numeric optimum "
            f"{numeric_value}"
        )
    cons_dual = {
        item.name: np.asarray(item.constraint.dual_value)
        for item in model.constraints
        if item.constraint.dual_value is not None
    }
    names, A, b = exact.onesided_identity_system(
        functional, level, bound, positivity
    )

    # Sign conventions matching CVXPY's dual report for this maximization:
    # PSD duals and probability duals enter with +, consistency and
    # normalization duals with -.
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
                -float(
                    cons_dual[f"consistency:x{parts[0][2:]}"][
                        int(parts[1]), int(parts[2])
                    ]
                )
            )
        elif name == "nu":
            numeric_vector.append(-float(cons_dual["normalization"]))
        else:
            numeric_vector.append(
                float(
                    cons_dual[
                        f"probability:{parts[1]}:{parts[2]}:{parts[3]}:{parts[4]}"
                    ]
                )
            )

    # Rationalize, then correct exactly onto the identity's solution space by
    # the minimum-norm exact perturbation.
    z = sp.Matrix([exact.rationalize(v, max_denominator) for v in numeric_vector])
    residual = A * z - b
    correction = A.T * (A * A.T).inv() * residual
    z_exact = z - correction

    s_blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, list(z_exact))
    report = exact.verify_onesided_dual(
        functional, level, bound, s_blocks, lam, nu, mu,
        enforce_probability_positivity=positivity,
    )
    return {
        "schema_version": 1,
        "kind": (
            "one_sided_pvm dual upper certificate "
            + ("(with probability positivity)" if positivity
               else "(raw CFNZ Definition 4.1 cone)")
        ),
        "functional": functional.name,
        "alpha": str(functional.metadata.get("alpha", "")),
        "level": level,
        "enforce_probability_positivity": positivity,
        "solver_used_to_find": solver,
        "numeric_value": numeric_value,
        "bound": encode(bound),
        "certificate": {name: encode(Fraction(v)) for name, v in zip(names, z_exact)},
        "verification": report,
    }


def verify_onesided_artifact(payload: dict) -> dict:
    alpha = payload.get("alpha") or ALPHA
    functional = tilted_chsh(float(alpha))
    names = list(payload["certificate"].keys())
    values = [decode(payload["certificate"][name]) for name in names]
    s_blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, values)
    return exact.verify_onesided_dual(
        functional,
        payload["level"],
        decode(payload["bound"]),
        s_blocks,
        lam,
        nu,
        mu,
        enforce_probability_positivity=payload["enforce_probability_positivity"],
    )


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m25" / "exact"
    )
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--witness-max-denominator", type=int, default=10**3)
    parser.add_argument("--dual-max-denominator", type=int, default=10**6)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="reload artifacts and re-verify without calling any solver",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "standard_witness": args.output / "standard_witness.json",
        "onesided_dual": args.output / "onesided_dual.json",
        "standard_witness_raw": args.output / "standard_witness_raw.json",
        "onesided_dual_raw": args.output / "onesided_dual_raw.json",
    }

    if args.verify_only:
        payloads = {
            key: json.loads(path.read_text(encoding="utf-8"))
            for key, path in paths.items()
        }
        for key in ("standard_witness", "standard_witness_raw"):
            payloads[key]["verification"] = verify_standard_artifact(payloads[key])
        for key in ("onesided_dual", "onesided_dual_raw"):
            payloads[key]["verification"] = verify_onesided_artifact(payloads[key])
    else:
        payloads = {
            "standard_witness": find_standard_witness(
                args.solver, args.witness_max_denominator
            ),
            "onesided_dual": find_onesided_dual(
                args.solver, args.dual_max_denominator, positivity=True
            ),
            "standard_witness_raw": find_standard_witness_raw(args.solver),
            "onesided_dual_raw": find_onesided_dual(
                args.solver, args.dual_max_denominator, positivity=False
            ),
        }
        for key, payload in payloads.items():
            paths[key].write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )

    failures: list[str] = []
    for key, payload in payloads.items():
        for failure in payload["verification"]["failures"]:
            failures.append(f"{key}: {failure}")

    witness_value = Fraction(payloads["standard_witness"]["verification"]["objective_exact"])
    raw_witness_value = sp.sympify(
        payloads["standard_witness_raw"]["verification"]["objective_exact"]
    )
    if witness_value != STANDARD_WITNESS_VALUE:
        failures.append(
            f"standard witness objective {witness_value} != {STANDARD_WITNESS_VALUE}"
        )
    if sp.simplify(raw_witness_value - RAW_WITNESS_VALUE) != 0:
        failures.append(
            f"raw witness objective {raw_witness_value} != {RAW_WITNESS_VALUE}"
        )

    summary = {
        "schema_version": 1,
        "functional": f"tilted_chsh_alpha_{ALPHA:g}",
        "level": 1,
        "positivity_variant": {
            "standard_primal_witness_value": str(witness_value),
            "one_sided_dual_bound": str(ONESIDED_BOUND),
            "exact_separation_gap": str(witness_value - ONESIDED_BOUND),
            "exact_separation_gap_float": float(witness_value - ONESIDED_BOUND),
        },
        "raw_variant": {
            "standard_primal_witness_value": str(raw_witness_value),
            "standard_primal_witness_float": float(raw_witness_value),
            "one_sided_dual_bound": str(ONESIDED_BOUND),
            "exact_separation_gap_float": float(raw_witness_value - sp.Rational(
                ONESIDED_BOUND.numerator, ONESIDED_BOUND.denominator)),
        },
        "statement": (
            "tilted-CHSH alpha=1/2 at native level 1: standard-PVM value >= "
            "31/10 (M2 cone, with probability positivity) resp. >= 1/2 + "
            "2 sqrt(2) (raw CFNZ Eq. (2.7) cone), while the one-sided-PVM "
            "value <= 596609/200000 = 2.983045 in both variants; all bounds "
            "exact and verified without any SDP solver"
        ),
        "verifications": {
            key: payload["verification"] for key, payload in payloads.items()
        },
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "positivity": summary["positivity_variant"],
        "raw": summary["raw_variant"],
        "status": summary["status"],
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
