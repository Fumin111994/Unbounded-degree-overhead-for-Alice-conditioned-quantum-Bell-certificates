#!/usr/bin/env python3
"""M3/M5: exactify the level-two cone separation for an M3 candidate.

For a shortlisted candidate (default: search_s31337 #308) with rationalized
coefficients this produces, with solver-free exact verification:

1. an exact standard level-2 DUAL certificate  eta*I - F = Tr(S Gamma) + ...
   proving omega_std^2 <= eta;
2. an exact one-sided level-2 PRIMAL witness with objective > eta, proving
   omega_os^2 > eta;

so the Bell-gap polynomial  eta*I - F_G  lies in D_2 but not in O_2: the
certificate cones are separated at native level two (D_2 not subset O_2).

The solver is used only to find candidates; verification is exact rational
arithmetic.  Output: artifacts/m3/exact/<tag>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from npa2 import exact  # noqa: E402
from npa2.bell import BellFunctional, Scenario  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def rationalize_functional(record: dict, max_denominator: int = 10**3) -> BellFunctional:
    signs = {0: Fraction(1), 1: Fraction(-1)}
    corr = {
        (int(k[0]), int(k[1])): Fraction(v).limit_denominator(max_denominator)
        for k, v in record["corr"].items()
    }
    amarg = {
        int(k): Fraction(v).limit_denominator(max_denominator)
        for k, v in record["alice_marginals"].items()
    }
    bmarg = {
        int(k): Fraction(v).limit_denominator(max_denominator)
        for k, v in record["bob_marginals"].items()
    }
    joint: dict[tuple[int, int, int, int], Fraction] = {}
    for (x, y), c in corr.items():
        for a in (0, 1):
            for b in (0, 1):
                joint[(x, y, a, b)] = c * signs[a] * signs[b]
    alice_local = {
        (x, a): c * signs[a] for x, c in amarg.items() for a in (0, 1)
    }
    bob_local = {(y, b): c * signs[b] for y, c in bmarg.items() for b in (0, 1)}
    return BellFunctional(
        name=f"{record.get('flag', 'candidate')}_rationalized",
        scenario=Scenario(2, 2, 2, 2),
        alice_local=alice_local,
        bob_local=bob_local,
        joint=joint,
    )


def encode(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def decode(text: str) -> Fraction:
    num, _, den = text.partition("/")
    return Fraction(int(num), int(den or "1"))


def solve_value(functional, builder, level, solver):
    summary, _ = solve_problem(builder(functional, level, enforce_probability_positivity=False), solver)
    return summary["value"]


def find_standard_dual(functional: BellFunctional, eta: Fraction, solver: str,
                       max_denominator: int) -> dict:
    model = build_standard_problem(functional, 2, enforce_probability_positivity=False)
    summary, arrays = solve_problem(model, solver)
    numeric_value = summary["value"]
    if float(eta) <= numeric_value:
        raise RuntimeError(f"eta {float(eta)} not above numeric std2 {numeric_value}")
    names, A, b, unpack = exact.standard_identity_system(functional, 2, eta, False)

    cons_dual = {
        item.name: np.asarray(item.constraint.dual_value)
        for item in model.constraints
        if item.constraint.dual_value is not None
    }
    s_dual = arrays["dual_psd__Gamma_std"]

    def numeric_vector(sign_nu: float) -> list[float]:
        vec = []
        for name in names:
            parts = name.split(":")
            if parts[0] == "S":
                vec.append(float(s_dual[int(parts[1]), int(parts[2])]))
            elif name == "nu":
                vec.append(sign_nu * float(cons_dual["normalization"]))
            else:
                raise ValueError(name)
        return vec

    # sign detection on the identity at the numeric optimum
    t0 = Fraction(numeric_value).limit_denominator(10**9)
    names0, A0, b0, _ = exact.standard_identity_system(functional, 2, t0, False)
    best = None
    for sign in (1.0, -1.0):
        vec = numeric_vector(sign)
        residual = A0 * sp.Matrix([Fraction(v) for v in vec]) - b0
        norm = max(abs(float(x)) for x in residual)
        if best is None or norm < best[0]:
            best = (norm, sign, vec)
    if best[0] > 1e-4:
        raise RuntimeError(f"could not match CVXPY dual conventions: {best[0]}")
    vec = best[2]

    z = sp.Matrix([exact.rationalize(v, max_denominator) for v in vec])
    residual = A * z - b
    correction = A.T * (A * A.T).inv() * residual
    z_exact = z - correction
    s_entries, nu, mu = unpack(list(z_exact))
    report = exact.verify_standard_dual(functional, 2, eta, s_entries, nu)
    return {
        "kind": "standard level-2 dual upper certificate (raw cone)",
        "solver_used_to_find": solver,
        "numeric_value": numeric_value,
        "bound": encode(eta),
        "certificate": {
            **{f"S:{i}:{j}": encode(v) for (i, j), v in s_entries.items()},
            "nu": encode(nu),
        },
        "verification": report,
    }


def uniform_reference_point(words) -> dict[tuple[int, int], Fraction]:
    """Exact feasible one-sided reference point: uniform mixture over the
    four deterministic Bob strategies, identical for every Alice block.

    Each block is Phi0 = (1/8) sum_e u_e u_e^T with u_e the moment vector of
    a deterministic Bob strategy.  At level 1 (3 Bob words) the four u_e
    span R^3 and Phi0 is strictly positive definite.  At level 2 (5 Bob
    words) deterministic strategies have B0 B1 = B1 B0, so the u_e span
    only the 4-dimensional consistent subspace and Phi0 is PSD with the
    rank-1 kernel (0,0,0,1,-1): the admixture is an exact FEASIBLE
    reference point, not a strictly feasible one; the final witness PSD is
    in any case verified exactly a posteriori.  Normalization:
    sum_a Phi0[I,I] = 1; consistency: blocks are identical across (x, a).
    """

    # Level 2: word indices I, B0:0, B1:0, B0:0 B1:0, B1:0 B0:0.  The two
    # orderings share one moment, so consistent matrices always have
    # (0,0,0,1,-1) in their kernel; the four strategy vectors span the
    # consistent subspace, hence Phi0 is PSD with exactly that kernel.
    # Level 1: word indices I, B0:0, B1:0; the four vectors span R^3.
    n = len(words)
    strategy_vectors = []
    for e0 in (0, 1):
        for e1 in (0, 1):
            joint = (1 - e0) * (1 - e1)
            if n == 5:
                strategy_vectors.append(
                    (Fraction(1), Fraction(1 - e0), Fraction(1 - e1),
                     Fraction(joint), Fraction(joint)))
            elif n == 3:
                strategy_vectors.append(
                    (Fraction(1), Fraction(1 - e0), Fraction(1 - e1)))
            else:
                raise ValueError("uniform reference point implemented for "
                                 "levels 1 and 2 only")
    matrix = sp.zeros(n)
    for vec in strategy_vectors:
        for i in range(n):
            for j in range(n):
                matrix[i, j] += vec[i] * vec[j]
    return matrix / 8


def find_onesided_witness(functional: BellFunctional, solver: str,
                          max_denominator: int,
                          epsilon: Fraction = Fraction(1, 1000),
                          level: int = 2) -> dict:
    model = build_onesided_problem(functional, level, enforce_probability_positivity=False)
    summary, _ = solve_problem(model, solver)
    algebra = functional.scenario.algebra()
    words = model.words
    reference = uniform_reference_point(words)
    A = functional.scenario.alice_outcomes
    X = functional.scenario.alice_questions
    block_entries: dict[tuple[int, int], dict[tuple[int, int], Fraction]] = {}
    numeric_blocks = {
        name: np.asarray(variable.value, dtype=float)
        for name, variable in model.matrices.items()
    }
    # Rationalize per representative moment with EXACT cross-block
    # consistency: rationalize the block sums T(k) = sum_a Phi_{0,a}(k) and
    # give the last-outcome block the residual; for the identity key the sum
    # is exactly 1 (normalization + consistency).
    key_values: dict[tuple[int, int], dict] = {}
    for x in range(X):
        for a in range(A):
            name = f"Phi_x{x}_a{a}"
            representatives = model.block_representatives[name]
            matrix = numeric_blocks[name]
            if a < A - 1:
                key_values[(x, a)] = {
                    key: exact.rationalize(matrix[i, j], max_denominator)
                    for key, (i, j) in representatives.items()
                }
            else:
                entries_for_block = {}
                for key, (i, j) in representatives.items():
                    if key == ():
                        # normalization + consistency force the identity
                        # moment of every block sum to be exactly 1
                        total = Fraction(1)
                    else:
                        total = exact.rationalize(
                            sum(
                                numeric_blocks[f"Phi_x0_a{a2}"][i2, j2]
                                for a2 in range(A)
                                for i2, j2 in [model.block_representatives[f"Phi_x0_a{a2}"][key]]
                            ),
                            max_denominator,
                        )
                    entries_for_block[key] = total - sum(
                        key_values[(x, a2)][key] for a2 in range(A - 1)
                    )
                key_values[(x, a)] = entries_for_block

    for x in range(X):
        for a in range(A):
            entries: dict[tuple[int, int], Fraction] = {}
            for i in range(len(words)):
                left_adjoint = algebra.dagger(words[i])
                for j in range(i, len(words)):
                    product_word = algebra.multiply_words(left_adjoint, words[j])
                    if product_word is None:
                        base = Fraction(0)
                    else:
                        base = key_values[(x, a)][algebra.moment_key(product_word)]
                    # mix with the exact feasible reference point to lift the
                    # rationalized blocks off the PSD boundary
                    entries[(i, j)] = (1 - epsilon) * base + epsilon * Fraction(
                        reference[i, j].numerator, reference[i, j].denominator
                    )
            block_entries[(x, a)] = entries
    report = exact.verify_onesided_witness(functional, level, block_entries)
    return {
        "kind": "one-sided level-2 primal witness (raw cone)",
        "solver_used_to_find": solver,
        "numeric_value": summary["value"],
        "block_entries": {
            f"{x}:{a}:{i}:{j}": encode(v)
            for (x, a), entries in block_entries.items()
            for (i, j), v in entries.items()
        },
        "verification": report,
    }


def rebuild_functional(payload: dict) -> BellFunctional:
    rf = payload["rationalized_functional"]
    return BellFunctional(
        name=payload["candidate_tag"] + "_rationalized",
        scenario=Scenario(2, 2, 2, 2),
        alice_local={
            tuple(int(p) for p in k.split(":")): decode(v)
            for k, v in rf["alice_local"].items()
        },
        bob_local={
            tuple(int(p) for p in k.split(":")): decode(v)
            for k, v in rf["bob_local"].items()
        },
        joint={
            tuple(int(p) for p in k.split(":")): decode(v)
            for k, v in rf["joint"].items()
        },
    )


def verify_artifact(payload: dict) -> dict:
    """Solver-free re-verification of an exported exactification artifact."""

    functional = rebuild_functional(payload)
    eta = decode(payload["eta"])
    cert = payload["standard_dual"]["certificate"]
    s_entries = {
        (int(parts[1]), int(parts[2])): decode(v)
        for name, v in cert.items()
        if name.startswith("S:")
        for parts in [name.split(":")]
    }
    dual_report = exact.verify_standard_dual(
        functional, 2, eta, s_entries, decode(cert["nu"])
    )
    block_entries: dict[tuple[int, int], dict[tuple[int, int], Fraction]] = {}
    for key, value in payload["onesided_witness"]["block_entries"].items():
        x, a, i, j = (int(p) for p in key.split(":"))
        block_entries.setdefault((x, a), {})[(i, j)] = decode(value)
    witness_report = exact.verify_onesided_witness(functional, 2, block_entries)
    witness_objective = Fraction(witness_report["objective_exact"])
    failures = dual_report["failures"] + witness_report["failures"]
    if not witness_objective > eta:
        failures.append(f"witness objective {witness_objective} not > eta {eta}")
    return {
        "dual": dual_report,
        "witness": witness_report,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "artifacts" / "m3" / "search_s31337" / "results.json",
    )
    parser.add_argument("--index", type=int, default=308)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m3" / "exact"
    )
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--functional-max-denominator", type=int, default=10**3)
    parser.add_argument("--dual-max-denominator", type=int, default=10**6)
    parser.add_argument("--witness-max-denominator", type=int, default=10**8)
    parser.add_argument(
        "--witness-epsilon", default="1/1000",
        help="reference-point mixing ratio; shrink when the witness margin is tight",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="reload the artifact and re-verify without calling any solver",
    )
    args = parser.parse_args()

    tag = args.tag or f"{args.results.parent.name}_{args.index}"
    if args.verify_only:
        path = args.output / f"{tag}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = verify_artifact(payload)
        print(json.dumps({"status": report["status"],
                          "failures": report["failures"]}, indent=2))
        return 0 if report["status"] == "PASS" else 1

    report_in = json.loads(args.results.read_text(encoding="utf-8"))
    record = next(r for r in report_in["records"] if r["index"] == args.index)
    functional = rationalize_functional(record, args.functional_max_denominator)

    # recompute the hierarchy values for the rationalized functional
    std2 = solve_value(functional, build_standard_problem, 2, args.solver)
    os2 = solve_value(functional, build_onesided_problem, 2, args.solver)
    os3 = solve_value(functional, build_onesided_problem, 3, args.solver)
    print(f"rationalized: std2={std2:.9f} os2={os2:.9f} os3={os3:.9f}")
    if not (std2 < os2 and abs(std2 - os3) < 1e-5):
        raise RuntimeError("rationalized functional lost the separation")

    # eta strictly between std2 and os2, on a coarse rational grid
    eta = Fraction(math.floor((std2 + os2) / 2 * 10**4), 10**4)
    if not (std2 < float(eta) < os2):
        eta = Fraction(math.ceil(std2 * 10**4) + 1, 10**4)
    print(f"eta = {eta} = {float(eta):.6f}")

    dual = find_standard_dual(functional, eta, args.solver, args.dual_max_denominator)
    print("std dual:", dual["verification"]["status"], dual["verification"]["failures"][:2])
    witness = find_onesided_witness(functional, args.solver, args.witness_max_denominator,
                                    epsilon=Fraction(args.witness_epsilon))
    print("os witness:", witness["verification"]["status"],
          witness["verification"]["failures"][:2],
          "objective:", witness["verification"]["objective_exact"])

    witness_objective = Fraction(witness["verification"]["objective_exact"])
    failures = dual["verification"]["failures"] + witness["verification"]["failures"]
    if not witness_objective > eta:
        failures.append(f"witness objective {witness_objective} not > eta {eta}")

    out = {
        "schema_version": 1,
        "candidate_tag": tag,
        "rationalized_functional": {
            "alice_local": {f"{x}:{a}": encode(v) for (x, a), v in functional.alice_local.items()},
            "bob_local": {f"{y}:{b}": encode(v) for (y, b), v in functional.bob_local.items()},
            "joint": {f"{x}:{y}:{a}:{b}": encode(v) for (x, y, a, b), v in functional.joint.items()},
        },
        "numeric_values_rationalized": {"std2": std2, "os2": os2, "os3": os3},
        "eta": encode(eta),
        "standard_dual": dual,
        "onesided_witness": witness,
        "statement": (
            f"eta*I - F_G with eta={eta} lies in D_2 (exact standard level-2 "
            f"SOS certificate) but not in O_2 (exact one-sided level-2 primal "
            f"witness has value {witness_objective} > eta): D_2 and O_2 are "
            "separated at native level two for this functional"
        ),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": out["status"], "eta": str(eta),
                      "witness_objective": str(witness_objective)}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
