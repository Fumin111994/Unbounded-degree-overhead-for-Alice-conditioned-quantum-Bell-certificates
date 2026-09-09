"""M2.6 KKT probe, task 1: symmetric reduction of the one-sided level-1 SDP.

The full one-sided level-1 problem (CFNZ Definition 4.1) has four 3x3 PSD
blocks Phi_{a|x} indexed by Bob words {I, B0, B1}.  Numerics show an optimal
face with the symmetric ansatz

    Phi_00 = [[p, q0, q0], [q0, q0, r0], [q0, r0, q0]]
    Phi_01 = [[1-p, q1, q1], [q1, q1, r1], [q1, r1, q1]]
    Phi_10 = [[1/2, v, w], [v, v, z], [w, z, w]]
    Phi_11 = Phi_10 with B0 <-> B1 swapped

under which the CFNZ consistency constraints reduce to

    q0 + q1 = v + w,   r0 + r1 = 2 z

and the tilted-CHSH objective F(alpha) = alpha*E_A0 + CHSH reduces to

    F = alpha*(2p - 1) + 4*(q0 - q1) - 4p + 2 + 4*(v - w).

This script solves the 8-variable reduced SDP and the full SDP with
build_onesided_problem(..., 1, enforce_probability_positivity=False) on the
m26 grid and checks agreement to 1e-6.  Idempotent: re-running overwrites
artifacts/m26/kkt/probe.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402

GRID_PATH = Path("artifacts/m26/grid/grid.json")
OUT_PATH = Path("artifacts/m26/kkt/probe.json")
TOL = 1e-6

SOLVER_OPTS = dict(
    tol_gap_abs=1e-10,
    tol_gap_rel=1e-10,
    tol_feas=1e-10,
    max_iter=4000,
)

VAR_NAMES = ["p", "q0", "r0", "q1", "r1", "v", "w", "z"]


def solve_reduced(alpha: float):
    """Solve the 8-variable symmetric reduction; return (value, variables)."""
    p, q0, r0, q1, r1, v, w, z = cp.Variable(8)
    M0 = cp.bmat([[p, q0, q0], [q0, q0, r0], [q0, r0, q0]])
    M1 = cp.bmat([[1 - p, q1, q1], [q1, q1, r1], [q1, r1, q1]])
    M2 = cp.bmat([[0.5, v, w], [v, v, z], [w, z, w]])
    objective = alpha * (2 * p - 1) + 4 * (q0 - q1) - 4 * p + 2 + 4 * (v - w)
    problem = cp.Problem(
        cp.Maximize(objective),
        [M0 >> 0, M1 >> 0, M2 >> 0, q0 + q1 == v + w, r0 + r1 == 2 * z],
    )
    problem.solve(solver="CLARABEL", **SOLVER_OPTS)
    values = {name: float(var.value) for name, var in zip(VAR_NAMES, (p, q0, r0, q1, r1, v, w, z))}
    return float(problem.value), values


def solve_full(alpha: float) -> float:
    model = build_onesided_problem(
        tilted_chsh(alpha), 1, enforce_probability_positivity=False
    )
    summary, _ = solve_problem(model, "CLARABEL")
    return float(summary["value"])


def main() -> None:
    grid = json.loads(GRID_PATH.read_text())
    alphas = [record["alpha"] for record in grid["records"]]
    records = []
    worst = 0.0
    for alpha in alphas:
        full_value = solve_full(alpha)
        reduced_value, variables = solve_reduced(alpha)
        diff = abs(full_value - reduced_value)
        worst = max(worst, diff)
        records.append(
            {
                "alpha": alpha,
                "full": full_value,
                "reduced": reduced_value,
                "abs_diff": diff,
                "variables": variables,
            }
        )
        print(
            f"alpha={alpha:6.4f}  full={full_value:.10f}  reduced={reduced_value:.10f}"
            f"  |diff|={diff:.2e}  {'OK' if diff < TOL else 'FAIL'}"
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"tolerance": TOL, "records": records}, indent=2))
    print(f"\nworst |diff| = {worst:.3e}  (tolerance {TOL:.0e})")
    print(f"wrote {OUT_PATH}")
    if worst >= TOL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
