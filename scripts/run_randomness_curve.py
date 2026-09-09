"""Numerical illustration only: two-branch quantum-side-information relaxation.

The proof and exact point witness do not use these floating-point SDP values.
"""
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from npa2.bell import chsh
from npa2.standard import build_standard_problem
from npa2.onesided import build_onesided_problem


def guessing_problem(hierarchy, level, two_branches=True):
    builder = build_standard_problem if hierarchy == "std" else build_onesided_problem
    constraints, masses, scores, objectives = [], [], [], []
    for e in ((1, -1) if two_branches else (1,)):
        model = builder(chsh(), level, enforce_probability_positivity=False)
        constraints.extend(item.constraint for item in model.constraints if item.name != "normalization")
        if hierarchy == "std":
            M = next(iter(model.matrices.values()))
            mass = M[0, 0]
            aindex = model.words.index((("A", 0, 0),))
            bias = 2*M[0, aindex]-mass
        else:
            m0 = model.matrices["Phi_x0_a0"][0, 0]
            m1 = model.matrices["Phi_x0_a1"][0, 0]
            mass, bias = m0+m1, m0-m1
        masses.append(mass)
        scores.append(model.problem.objective.expr)
        objectives.append((mass+e*bias)/2)
    score = cp.Parameter(name="observed_chsh")
    constraints.extend([sum(masses) == 1, sum(scores) == score])
    return cp.Problem(cp.Maximize(sum(objectives)), constraints), score


def solve(problem, parameter, score):
    parameter.value = score
    value = problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                          tol_feas=1e-10, max_iter=400)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Numerical illustration failed: {problem.status}")
    violation = max(float(np.max(np.abs(c.violation()))) for c in problem.constraints)
    return float(value), {"status": problem.status, "maximum_constraint_violation": violation}


def main():
    low, high = 8/np.sqrt(12.5), 8/np.sqrt(11.38)
    scores = sorted(set(np.linspace(low, high, 31).tolist()+[2.3]))
    programs = {name: guessing_problem(h, k) for name, h, k in
                [("std2", "std", 2), ("os2", "os", 2), ("os3", "os", 3)]}
    single, parameter = guessing_problem("os", 2, False)
    records = []
    for s in scores:
        row = {"chsh": s, "quantum_guess": (1+np.sqrt(2-s*s/4))/2, "diagnostics": {}}
        for name, (problem, p) in programs.items():
            row[name], row["diagnostics"][name] = solve(problem, p, s)
        row["os2_single_branch"], _ = solve(single, parameter, s)
        records.append(row)
        print(f"s={s:.8f} gap={row['os2']-row['quantum_guess']:.8g}", flush=True)
    checks = {
        "std2_max_abs_error": max(abs(r["std2"]-r["quantum_guess"]) for r in records),
        "os3_max_abs_error": max(abs(r["os3"]-r["quantum_guess"]) for r in records),
        "branch_reduction_max_abs_error": max(abs(r["os2"]-r["os2_single_branch"]) for r in records),
        "os2_min_positive_gap": min(r["os2"]-r["quantum_guess"] for r in records),
    }
    if any(checks[key] > 2e-6 for key in checks if key != "os2_min_positive_gap"):
        raise RuntimeError(f"Numerical cross-check failed: {checks}")
    out = ROOT / "artifacts/prl/randomness/curve.json"
    out.write_text(json.dumps({"method": "Numerical illustration; two unnormalized guessing branches; raw PVM",
                               "checks": checks, "records": records}, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
