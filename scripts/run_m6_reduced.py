#!/usr/bin/env python3
"""M6 plan B: symmetry-reduced one-sided level-2 model for tilted CHSH.

The one-sided level-2 problem for tilted CHSH is invariant under
sigma = (Bob question swap) o (Alice x=1 outcome flip), so an optimal
symmetric solution exists (convexity).  This script builds the reduced SDP
in the symmetric moment coordinates and checks it against the full model.

Symmetric blocks (Bob words [I, B0, B1, B0B1, B1B0], moments
p=m(I), u=m(B0)=m(B1), r=m(B0B1), s3=m(B0B1B0)=m(B1B0B1), s4=m(B0B1B0B1)):

    P(p,u,r,s3,s4) =
    [[p, u, u, r,  r ],
     [u, u, r, r,  s3],
     [u, r, u, s3, r ],
     [r, r, s3, s3, s4],
     [r, s3, r, s4, s3]]

Blocks (0,0) and (0,1) use P; blocks (1,0) use

    Q(pu, v0, v1, z, w30, w31, w4) =
    [[pu, v0, v1, z,  z ],
     [v0, v0, z,  z,  w30],
     [v1, z,  v1, w31, z ],
     [z,  z,  w31, w31, w4],
     [z,  w30, z,  w4,  w30]]

and block (1,1) = swap(Q).  Consistency sum_a Phi_{0a} = sum_a Phi_{1a}:
    p + p' = 2 pu,  u + u' = v0 + v1,  r + r' = 2 z,
    s3 + s3' = w30 + w31,  s4 + s4' = 2 w4.
Normalization: p + p' = 1.
Objective: F = alpha (p - p') + 4 (u - u') - 2 (p - p') + 4 (v0 - v1).

Output: artifacts/m6/reduced_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402


def build_reduced(alpha: float) -> cp.Problem:
    p, u, r, s3, s4 = cp.Variable(5)
    pp, up, rp, s3p, s4p = cp.Variable(5)
    pu, v0, v1, z, w30, w31, w4 = cp.Variable(7)

    def P(p_, u_, r_, s3_, s4_):
        return cp.bmat(
            [
                [p_, u_, u_, r_, r_],
                [u_, u_, r_, r_, s3_],
                [u_, r_, u_, s3_, r_],
                [r_, r_, s3_, s3_, s4_],
                [r_, s3_, r_, s4_, s3_],
            ]
        )

    def Q(pu_, v0_, v1_, z_, w30_, w31_, w4_):
        return cp.bmat(
            [
                [pu_, v0_, v1_, z_, z_],
                [v0_, v0_, z_, z_, w30_],
                [v1_, z_, v1_, w31_, z_],
                [z_, z_, w31_, w31_, w4_],
                [z_, w30_, z_, w4_, w30_],
            ]
        )

    blocks = {
        "00": P(p, u, r, s3, s4),
        "01": P(pp, up, rp, s3p, s4p),
        "10": Q(pu, v0, v1, z, w30, w31, w4),
        "11": Q(pu, v1, v0, z, w31, w30, w4),
    }
    constraints = [block >> 0 for block in blocks.values()]
    constraints += [
        p + pp == 2 * pu,
        u + up == v0 + v1,
        r + rp == 2 * z,
        s3 + s3p == w30 + w31,
        s4 + s4p == 2 * w4,
        p + pp == 1,
    ]
    objective = alpha * (p - pp) + 4 * (u - up) - 2 * (p - pp) + 4 * (v0 - v1)
    return cp.Problem(cp.Maximize(objective), constraints), blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 1.0, 1.25, 1.3, 1.5])
    args = parser.parse_args()

    records = []
    for alpha in args.alphas:
        functional = tilted_chsh(alpha)
        full = solve_problem(
            build_onesided_problem(functional, 2, enforce_probability_positivity=False),
            "CLARABEL",
        )[0]["value"]
        problem, _ = build_reduced(alpha)
        reduced = problem.solve(solver="CLARABEL", tol_gap_abs=1e-9,
                                tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=2000)
        records.append(
            {"alpha": alpha, "full": full, "reduced": float(reduced),
             "diff": abs(full - reduced)}
        )
        print(f"alpha={alpha:5.2f} full={full:.9f} reduced={reduced:.9f} diff={abs(full-reduced):.2e}",
              flush=True)

    worst = max(r["diff"] for r in records)
    status = "PASS" if worst < 1e-5 else "FAIL"
    (args.output / "reduced_check.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "reduced_check.json").write_text(
        json.dumps({"records": records, "worst_diff": worst, "status": status},
                   indent=2),
        encoding="utf-8",
    )
    print(f"worst={worst:.2e} status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
