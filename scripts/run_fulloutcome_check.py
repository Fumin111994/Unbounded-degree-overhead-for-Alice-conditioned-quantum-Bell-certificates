#!/usr/bin/env python3
"""M2.5 full-outcome cross-check: reduced vs full-outcome moment bodies.

Optimal SDP solutions are not unique, so raw optimal moment matrices cannot be
compared entry-wise.  Instead this script compares the two implementations on
the level where they must agree if they describe the same moment body: for
every Bell coordinate probe (each joint probability p(a,b|x,y), each Alice
marginal m(A_{a|x}) and each Bob marginal m(B_{b|y})) it computes the minimum
and the maximum over both moment bodies and compares the optima.  It also
compares the optima of the actual control functionals.

Output: artifacts/m25/fulloutcome/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import BellFunctional, control_suite  # noqa: E402
from npa2.fulloutcome import (  # noqa: E402
    build_onesided_problem_full,
    build_standard_problem_full,
)
from npa2.model import NPAProblem  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def probe_functionals(functional: BellFunctional) -> list[BellFunctional]:
    """One BellFunctional per unit probe on a single Bell coordinate."""

    scenario = functional.scenario
    probes: list[BellFunctional] = []
    for x in range(scenario.alice_questions):
        for y in range(scenario.bob_questions):
            for a in range(scenario.alice_outcomes):
                for b in range(scenario.bob_outcomes):
                    probes.append(
                        BellFunctional(
                            name=f"probe_p_{x}_{y}_{a}_{b}",
                            scenario=scenario,
                            joint={(x, y, a, b): 1.0},
                        )
                    )
    for x in range(scenario.alice_questions):
        for a in range(scenario.alice_outcomes):
            probes.append(
                BellFunctional(
                    name=f"probe_mA_{x}_{a}",
                    scenario=scenario,
                    alice_local={(x, a): 1.0},
                )
            )
    for y in range(scenario.bob_questions):
        for b in range(scenario.bob_outcomes):
            probes.append(
                BellFunctional(
                    name=f"probe_mB_{y}_{b}",
                    scenario=scenario,
                    bob_local={(y, b): 1.0},
                )
            )
    return probes


def solve_with_fallback(model: NPAProblem, solvers: list[str]):
    """Try each solver in order; return (solver_used, value) or (None, None)."""

    for solver in solvers:
        try:
            summary, _ = solve_problem(model, solver)
        except Exception:
            continue
        if summary["status"] in ("optimal", "optimal_inaccurate"):
            return solver, summary["value"]
    return None, None


def optimum_pair(
    functional: BellFunctional,
    level: int,
    builder,
    solvers: list[str],
    sign: float,
):
    """Max of sign*<functional> over the moment body; None on solver failure."""

    probe = BellFunctional(
        name=functional.name + ("_min" if sign < 0 else "_max"),
        scenario=functional.scenario,
        constant=sign * functional.constant,
        alice_local={k: sign * v for k, v in functional.alice_local.items()},
        bob_local={k: sign * v for k, v in functional.bob_local.items()},
        joint={k: sign * v for k, v in functional.joint.items()},
    )
    _, value = solve_with_fallback(builder(probe, level), solvers)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "m25" / "fulloutcome",
    )
    parser.add_argument("--solvers", nargs="+", default=["CLARABEL", "SCS"])
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--b3-levels",
        nargs="+",
        type=int,
        default=[1],
        help="levels for the 3-outcome b3 control (heavier full-outcome SDP)",
    )
    parser.add_argument("--tol", type=float, default=5e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builders = {
        "standard": (build_standard_problem, build_standard_problem_full),
        "one_sided": (build_onesided_problem, build_onesided_problem_full),
    }
    records: list[dict] = []
    failures: list[str] = []
    for control in control_suite():
        levels = args.b3_levels if control.name == "b3_game" else args.levels
        probes = probe_functionals(control)
        for level in levels:
            for hierarchy, (reduced_builder, full_builder) in builders.items():
                tag = f"{control.name} {hierarchy} L{level}"
                objectives = [("functional", control, 1.0)] + [
                    (probe.name, probe, sign)
                    for probe in probes
                    for sign in (1.0, -1.0)
                ]
                worst_diff = 0.0
                worst_probe = None
                solver_failures = 0
                for probe_name, probe, sign in objectives:
                    red = optimum_pair(
                        probe, level, reduced_builder, args.solvers, sign
                    )
                    full = optimum_pair(
                        probe, level, full_builder, args.solvers, sign
                    )
                    if red is None or full is None:
                        solver_failures += 1
                        continue
                    diff = abs(red - full)
                    if diff > worst_diff:
                        worst_diff = diff
                        worst_probe = probe_name
                records.append(
                    {
                        "functional": control.name,
                        "hierarchy": hierarchy,
                        "level": level,
                        "probes_compared": len(objectives),
                        "solver_failures": solver_failures,
                        "worst_optimum_diff": worst_diff,
                        "worst_probe": worst_probe,
                    }
                )
                print(
                    f"{tag}: probes={len(objectives)} worst_diff={worst_diff:.3e} "
                    f"({worst_probe})",
                    flush=True,
                )
                if solver_failures:
                    failures.append(f"{tag}: {solver_failures} solver failures")
                if worst_diff > args.tol:
                    failures.append(
                        f"{tag}: probe optimum diff {worst_diff:.3e} at {worst_probe}"
                    )

    report = {
        "schema_version": 1,
        "scope": (
            "reduced vs full-outcome moment bodies compared through the optima of "
            "every Bell-coordinate probe; excludes the omitted-outcome quotient "
            "as a source of any level-one discrepancy"
        ),
        "tolerance": args.tol,
        "records": records,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"status={report['status']}")
    for failure in failures:
        print("FAIL:", failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
