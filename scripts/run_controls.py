#!/usr/bin/env python3
"""Run and export the M2 standard/one-sided NPA control suite."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cvxpy
import numpy
import scipy
import sympy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import control_suite  # noqa: E402
from npa2.export import sha256_file, write_solution  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402
from npa2.verify import verify_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "m2" / "canonical",
        help="Artifact directory",
    )
    parser.add_argument(
        "--solvers", nargs="+", default=["CLARABEL", "SCS"], help="CVXPY solver names"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Optional functional names to run"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [
        functional
        for functional in control_suite()
        if args.only is None or functional.name in set(args.only)
    ]
    if not selected:
        raise SystemExit("No controls selected.")

    index: list[dict] = []
    for functional in selected:
        for hierarchy, builder in (
            ("standard_pvm", build_standard_problem),
            ("one_sided_pvm", build_onesided_problem),
        ):
            level_key = f"{hierarchy.removesuffix('_pvm')}_control_level"
            level = int(functional.metadata[level_key])
            for solver in args.solvers:
                print(f"solving {functional.name} {hierarchy} level={level} solver={solver}")
                model = builder(functional, level)
                summary, arrays = solve_problem(model, solver)
                run_dir = write_solution(args.output, summary, arrays)
                index.append(
                    {
                        "functional": functional.name,
                        "hierarchy": hierarchy,
                        "level": level,
                        "solver": solver,
                        "status": summary["status"],
                        "value": summary["value"],
                        "analytic_value": summary["analytic_value"],
                        "analytic_error": summary["analytic_error"],
                        "artifact": str(run_dir.relative_to(args.output)),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    verification = verify_artifacts(args.output)
    (args.output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "cvxpy": cvxpy.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "sympy": sympy.__version__,
        },
        "solvers": args.solvers,
        "random_seed": None,
        "deterministic_instance_generation": True,
        "run_count": len(index),
        "index_sha256": sha256_file(args.output / "index.json"),
        "verification_sha256": sha256_file(args.output / "verification.json"),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))
    return 0 if verification["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
