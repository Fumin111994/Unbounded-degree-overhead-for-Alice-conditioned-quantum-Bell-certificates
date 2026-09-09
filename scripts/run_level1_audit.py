#!/usr/bin/env python3
"""Audit the native-level-one convention instead of assuming equality.

This script intentionally keeps the standard PVM quotient and the CFNZ
one-sided PVM quotient separate.  It records their tilted-CHSH optima with two
solvers, plus a native-level-two control at alpha=1/2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "m2" / "level1_audit",
    )
    parser.add_argument("--solvers", nargs="+", default=["CLARABEL", "SCS"])
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict] = []
    for alpha in args.alphas:
        functional = tilted_chsh(alpha)
        for solver in args.solvers:
            standard, _ = solve_problem(build_standard_problem(functional, 1), solver)
            one_sided, _ = solve_problem(build_onesided_problem(functional, 1), solver)
            records.append(
                {
                    "alpha": alpha,
                    "solver": solver,
                    "analytic_value": functional.analytic_value,
                    "standard_pvm_level_1": standard["value"],
                    "one_sided_pvm_level_1": one_sided["value"],
                    "standard_minus_one_sided": standard["value"] - one_sided["value"],
                    "standard_status": standard["status"],
                    "one_sided_status": one_sided["status"],
                }
            )

    # Level two is a control for the functional and algebra implementation,
    # not evidence that the native-level-one cones are equal.
    functional = tilted_chsh(0.5)
    level_two: list[dict] = []
    for solver in args.solvers:
        standard, _ = solve_problem(build_standard_problem(functional, 2), solver)
        one_sided, _ = solve_problem(build_onesided_problem(functional, 2), solver)
        level_two.append(
            {
                "solver": solver,
                "analytic_value": functional.analytic_value,
                "standard_pvm_level_2": standard["value"],
                "one_sided_pvm_level_2": one_sided["value"],
            }
        )

    failures: list[str] = []
    grouped: dict[float, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["alpha"]].append(record)
    for alpha, group in grouped.items():
        for key in ("standard_pvm_level_1", "one_sided_pvm_level_1"):
            spread = max(row[key] for row in group) - min(row[key] for row in group)
            if spread > 2e-5:
                failures.append(f"alpha={alpha}: {key} solver spread {spread}")
        mean_gap = sum(row["standard_minus_one_sided"] for row in group) / len(group)
        if alpha == 0.0 and abs(mean_gap) > 2e-5:
            failures.append(f"CHSH baseline mismatch {mean_gap}")
        if alpha > 0.0 and mean_gap <= 1e-4:
            failures.append(f"alpha={alpha}: expected reproducible diagnostic gap, got {mean_gap}")
    for row in level_two:
        for key in ("standard_pvm_level_2", "one_sided_pvm_level_2"):
            if abs(row[key] - row["analytic_value"]) > 2e-5:
                failures.append(f"level-two control {row['solver']} {key} failed")

    report = {
        "schema_version": 1,
        "scope": "numerical finite-level convention audit; not an exact cone separation",
        "records": records,
        "level_two_alpha_0_5_control": level_two,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
