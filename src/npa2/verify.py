"""Solver-independent checks on exported numerical control artifacts."""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(run_dir: Path, failures: list[str]) -> None:
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.exists():
        failures.append(f"{run_dir.name}: missing checksums.sha256")
        return
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        actual = _sha256(run_dir / name)
        if actual != expected:
            failures.append(f"{run_dir.name}: checksum mismatch for {name}")


def verify_artifacts(output_root: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    index_path = output_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        summary_paths = [output_root / row["artifact"] / "summary.json" for row in index]
    else:
        summary_paths = sorted(output_root.glob("*/summary.json"))
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        run_dir = summary_path.parent
        _verify_checksums(run_dir, failures)
        arrays = np.load(run_dir / "raw_primal_dual.npz")
        for name, stats in summary["matrix_stats"].items():
            matrix = arrays[f"primal__{name}"]
            min_eigenvalue = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0])
            if min_eigenvalue < -2e-5:
                failures.append(f"{run_dir.name}: primal {name} min eig {min_eigenvalue}")
        for name in summary["psd_dual_stats"]:
            matrix = arrays[f"dual_psd__{name}"]
            min_eigenvalue = float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0])
            if min_eigenvalue < -2e-5:
                failures.append(f"{run_dir.name}: dual {name} min eig {min_eigenvalue}")
        tolerance = 2e-5 if summary["solver"] == "SCS" else 2e-6
        if summary["analytic_value"] is not None and abs(summary["analytic_error"]) > tolerance:
            failures.append(
                f"{run_dir.name}: analytic error {summary['analytic_error']} > {tolerance}"
            )
        if summary["max_constraint_violation"] > tolerance:
            failures.append(
                f"{run_dir.name}: constraint violation "
                f"{summary['max_constraint_violation']} > {tolerance}"
            )

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary["functional"], summary["hierarchy"], summary["level"])].append(
            summary
        )
    comparisons: list[dict[str, Any]] = []
    for key, group in grouped.items():
        values = [item["value"] for item in group]
        spread = max(values) - min(values)
        comparisons.append(
            {
                "functional": key[0],
                "hierarchy": key[1],
                "level": key[2],
                "solvers": [item["solver"] for item in group],
                "spread": spread,
            }
        )
        if len(group) > 1 and spread > 2e-5:
            failures.append(f"{key}: solver spread {spread} > 2e-5")

    # Cross-hierarchy agreement is declared per control functional.  It is not
    # treated as a universal same-native-level theorem.
    by_functional_solver: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for summary in summaries:
        if summary["functional_metadata"].get("expect_cross_hierarchy_match", False):
            by_functional_solver[(summary["functional"], summary["solver"])][
                summary["hierarchy"]
            ] = summary["value"]
    cross_hierarchy_comparisons: list[dict[str, Any]] = []
    for key, values in by_functional_solver.items():
        if {"standard_pvm", "one_sided_pvm"}.issubset(values):
            difference = values["standard_pvm"] - values["one_sided_pvm"]
            cross_hierarchy_comparisons.append(
                {"functional": key[0], "solver": key[1], "difference": difference}
            )
            if abs(difference) > 2e-5:
                failures.append(f"{key}: declared control hierarchy difference {difference}")

    return {
        "schema_version": 1,
        "run_count": len(summaries),
        "solver_comparisons": comparisons,
        "cross_hierarchy_comparisons": cross_hierarchy_comparisons,
        "failure_count": len(failures),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
