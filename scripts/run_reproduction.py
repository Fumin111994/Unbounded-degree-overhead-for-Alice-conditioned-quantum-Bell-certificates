#!/usr/bin/env python3
"""M7: reproduction of all project checks, in two commands.

Command 1 (default, solver-free): re-verifies every stored exact
certificate from the artifacts alone --- affine identities
coefficient-wise, exact rational LDL* PSD tests, consistency,
normalization, and all certified comparisons.  No SDP solver is called
(cvxpy.Problem.solve is patched to raise inside verify_certificates.py);
a missing artifact or missing certificate payload is a FAIL.

    python scripts/run_reproduction.py

Command 2 (--full, solver-based numerical regression): additionally runs
the unit-test suite (some tests call CLARABEL/SCS) and re-runs the
floating-point controls, searches, and audits.

    python scripts/run_reproduction.py --full

Output: artifacts/reproduction/report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PY = sys.executable


def step(name: str, command: list[str], timeout: int = 7200) -> dict:
    start = time.time()
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )
    return {
        "name": name,
        "command": " ".join(command),
        "returncode": result.returncode,
        "seconds": round(time.time() - start, 1),
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "tail": (result.stdout + result.stderr).strip().splitlines()[-3:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also run solver-based tests, controls, audits")
    parser.add_argument("--skip-slow", action="store_true",
                        help="pass --skip-slow to verify_certificates.py")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "reproduction")
    args = parser.parse_args()

    verify_cmd = [PY, "scripts/verify_certificates.py"]
    if args.skip_slow:
        verify_cmd.append("--skip-slow")

    # Command 1: pure certificate re-verification (no solver anywhere).
    steps = [
        ("certificate_checks", verify_cmd),
        ("m2_artifact_verifier", [PY, "scripts/verify_results.py"]),
        ("m25_exact_verify_only",
         [PY, "scripts/run_exact_level1.py", "--verify-only"]),
        ("m26_standard_family", [PY, "scripts/run_level1_family.py"]),
        ("m26_anchors_coverage",
         [PY, "scripts/run_level1_anchors.py", "--coverage-only"]),
        ("m3_exact_verify_only",
         [PY, "scripts/run_m3_exactify.py", "--verify-only"]),
    ]
    if args.full:
        # Command 2: numerical regression (solvers allowed).
        steps += [
            ("unit_tests", [PY, "-m", "unittest", "discover", "-s", "tests"]),
            ("m2_controls", [PY, "scripts/run_controls.py"]),
            ("m2_level1_audit", [PY, "scripts/run_level1_audit.py"]),
            ("m25_fulloutcome", [PY, "scripts/run_fulloutcome_check.py"]),
            ("m6_closure", [PY, "scripts/run_m6_closure.py"]),
            ("m6_epsilon_closure", [PY, "scripts/run_m6_epsilon_closure.py"]),
            ("m65_tangent_fan", [PY, "scripts/run_m65_tangent_fan.py"]),
            ("m65_transport", [PY, "scripts/run_m65_transport.py"]),
        ]

    results = [step(name, cmd) for name, cmd in steps]
    failures = [r["name"] for r in results if r["status"] != "PASS"]
    report = {
        "schema_version": 2,
        "full": args.full,
        "results": results,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for r in results:
        print(f"{r['name']:32s} {r['status']}  ({r['seconds']}s)")
    print("status:", report["status"])
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
