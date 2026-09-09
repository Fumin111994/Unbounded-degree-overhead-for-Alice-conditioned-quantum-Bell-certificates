"""Solver execution and numerical certificate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cvxpy as cp
import numpy as np

from .model import NPAProblem


@dataclass(frozen=True)
class SolverConfig:
    name: str
    options: dict[str, Any]


DEFAULT_SOLVERS = {
    "CVXOPT": SolverConfig(
        "CVXOPT",
        {
            "abstol": 1e-9,
            "reltol": 1e-9,
            "feastol": 1e-9,
            "max_iters": 300,
            "refinement": 2,
        },
    ),
    "SCS": SolverConfig(
        "SCS",
        {
            "eps": 1e-7,
            "max_iters": 200_000,
            "normalize": True,
            "acceleration_lookback": 20,
            "verbose": False,
        },
    ),
    "CLARABEL": SolverConfig(
        "CLARABEL",
        {
            "tol_gap_abs": 1e-9,
            "tol_gap_rel": 1e-9,
            "tol_feas": 1e-9,
            "max_iter": 1_000,
            "verbose": False,
        },
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def solve_problem(model: NPAProblem, solver: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if solver not in DEFAULT_SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}.")
    config = DEFAULT_SOLVERS[solver]
    value = model.problem.solve(solver=config.name, **config.options)
    arrays: dict[str, np.ndarray] = {}

    matrix_stats: dict[str, Any] = {}
    for name, variable in model.matrices.items():
        matrix = np.asarray(variable.value, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        arrays[f"primal__{name}"] = matrix
        matrix_stats[name] = {
            "shape": list(matrix.shape),
            "min_eigenvalue": float(eigenvalues[0]),
            "max_eigenvalue": float(eigenvalues[-1]),
            "numerical_rank_1e-7": int(np.sum(eigenvalues > 1e-7)),
        }

    psd_dual_stats: dict[str, Any] = {}
    for name, constraint in model.psd_constraints.items():
        dual = np.asarray(constraint.dual_value, dtype=float)
        dual = 0.5 * (dual + dual.T)
        eigenvalues, eigenvectors = np.linalg.eigh(dual)
        arrays[f"dual_psd__{name}"] = dual
        positive = eigenvalues > 1e-10
        factor = (
            np.diag(np.sqrt(eigenvalues[positive])) @ eigenvectors[:, positive].T
            if np.any(positive)
            else np.zeros((0, dual.shape[0]))
        )
        arrays[f"dual_factor__{name}"] = factor
        psd_dual_stats[name] = {
            "shape": list(dual.shape),
            "min_eigenvalue": float(eigenvalues[0]),
            "max_eigenvalue": float(eigenvalues[-1]),
            "factor_rows": int(factor.shape[0]),
        }

    max_violation = 0.0
    constraint_manifest: list[dict[str, Any]] = []
    for index, item in enumerate(model.constraints):
        try:
            violation_array = np.asarray(item.constraint.violation(), dtype=float)
            violation = float(np.max(np.abs(violation_array))) if violation_array.size else 0.0
        except Exception:
            violation = float("nan")
        if np.isfinite(violation):
            max_violation = max(max_violation, violation)
        dual_value = item.constraint.dual_value
        dual_key = None
        if dual_value is not None:
            dual_key = f"constraint_dual__{index:05d}"
            arrays[dual_key] = np.asarray(dual_value)
        constraint_manifest.append(
            {
                "index": index,
                "name": item.name,
                "violation": violation,
                "dual_array_key": dual_key,
            }
        )

    analytic = model.functional.analytic_value
    summary = {
        "schema_version": 1,
        "functional": model.functional.name,
        "hierarchy": model.hierarchy,
        "level": model.level,
        "solver": solver,
        "status": model.problem.status,
        "value": float(value),
        "analytic_value": analytic,
        "analytic_error": None if analytic is None else float(value - analytic),
        "max_constraint_violation": max_violation,
        "word_count": len(model.words),
        "word_labels": [model.functional.scenario.algebra().word_label(w) for w in model.words],
        "matrix_stats": matrix_stats,
        "psd_dual_stats": psd_dual_stats,
        "constraints": constraint_manifest,
        "model_metadata": model.metadata,
        "functional_metadata": dict(model.functional.metadata),
        "source": model.functional.source,
        "solver_stats": {
            "solver_name": model.problem.solver_stats.solver_name,
            "solve_time": model.problem.solver_stats.solve_time,
            "setup_time": model.problem.solver_stats.setup_time,
            "num_iters": model.problem.solver_stats.num_iters,
            "extra_stats": _json_safe(model.problem.solver_stats.extra_stats),
        },
    }
    return _json_safe(summary), arrays
