#!/usr/bin/env python3
"""M3: search for new (2,2,2,2) Bell benchmarks with standard finite-level closure.

Samples general Bell functionals

    F = sum_xy c_xy E_xy + sum_x a_x E_Ax + sum_y b_y E_By

in correlator/marginal form and computes the raw-cone hierarchy values
omega_std^2, omega_std^3, omega_os^2, omega_os^3 plus the classical value
(brute force over the 16 deterministic strategies).

Flags (tolerances in --help):

- ``closure``:     |std2 - os3| small  (standard level-2 has closed at omega_qc)
- ``separation``:  closure and os2 - std2 large  (one-sided level-2 not closed)
- ``transport``:   closure and |os2 - std2| small (level-2 certificate transports)
- ``novel``:       not XOR (some marginal nonzero) and not tilted-like
                   (CHSH correlator + at most one nonzero marginal component)

Output: artifacts/m3/search/results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import correlator_functional  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def classical_value(corr: dict, amarg: dict, bmarg: dict) -> float:
    best = -np.inf
    for a0, a1, b0, b1 in itertools.product([1.0, -1.0], repeat=4):
        avec = {0: a0, 1: a1}
        bvec = {0: b0, 1: b1}
        value = sum(c * avec[x] * bvec[y] for (x, y), c in corr.items())
        value += sum(c * avec[x] for x, c in amarg.items())
        value += sum(c * bvec[y] for y, c in bmarg.items())
        best = max(best, value)
    return float(best)


def novelty(corr: dict, amarg: dict, bmarg: dict, tol: float = 1e-9) -> dict:
    marginals = [v for v in list(amarg.values()) + list(bmarg.values()) if abs(v) > tol]
    is_xor = len(marginals) == 0
    entries = [abs(v) for v in corr.values()]
    chsh_correlator = max(entries) - min(entries) < tol
    tilted_like = chsh_correlator and len(marginals) <= 1
    return {
        "is_xor": is_xor,
        "chsh_correlator": chsh_correlator,
        "tilted_like": tilted_like,
        "novel": not is_xor and not tilted_like,
        "nonzero_marginals": len(marginals),
    }


def solve_with_fallback(model, solvers=("CLARABEL", "SCS")):
    for solver in solvers:
        try:
            summary, _ = solve_problem(model, solver)
        except Exception:
            continue
        if summary["status"] in ("optimal", "optimal_inaccurate"):
            return summary["value"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m3" / "search")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--closure-tol", type=float, default=1e-5)
    parser.add_argument("--separation-tol", type=float, default=1e-4)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    records = []
    for index in range(args.count):
        corr = {(x, y): float(rng.uniform(-1, 1)) for x in (0, 1) for y in (0, 1)}
        scale = max(abs(v) for v in corr.values())
        corr = {k: v / scale for k, v in corr.items()}
        marginal_scale = float(rng.uniform(0.0, 0.5))
        amarg = {x: float(rng.uniform(-1, 1)) * marginal_scale for x in (0, 1)}
        bmarg = {y: float(rng.uniform(-1, 1)) * marginal_scale for y in (0, 1)}
        # randomly zero out some marginals to sample different structural types
        for table in (amarg, bmarg):
            for key in list(table):
                if rng.random() < 0.3:
                    table[key] = 0.0

        nov = novelty(corr, amarg, bmarg)
        functional = correlator_functional(
            f"candidate_{index}", corr, amarg, bmarg
        )
        std2 = solve_with_fallback(build_standard_problem(functional, 2, enforce_probability_positivity=False))
        std3 = solve_with_fallback(build_standard_problem(functional, 3, enforce_probability_positivity=False))
        os2 = solve_with_fallback(build_onesided_problem(functional, 2, enforce_probability_positivity=False))
        os3 = solve_with_fallback(build_onesided_problem(functional, 3, enforce_probability_positivity=False))
        classical = classical_value(corr, amarg, bmarg)

        record = {
            "index": index,
            "corr": {f"{x}{y}": v for (x, y), v in corr.items()},
            "alice_marginals": amarg,
            "bob_marginals": bmarg,
            "classical": classical,
            "std2": std2,
            "std3": std3,
            "os2": os2,
            "os3": os3,
            **nov,
        }
        values = [std2, std3, os2, os3]
        if any(v is None for v in values):
            record["flag"] = "solver_failure"
        else:
            closure = abs(std2 - os3) < args.closure_tol
            nonlocal_q = os3 > classical + 1e-6
            record["nonlocal"] = bool(nonlocal_q)
            if closure and nonlocal_q and nov["novel"]:
                if os2 - std2 > args.separation_tol:
                    record["flag"] = "SEPARATION_CANDIDATE"
                elif abs(os2 - std2) < args.closure_tol:
                    record["flag"] = "TRANSPORT_EXAMPLE"
                else:
                    record["flag"] = "closure_ambiguous"
            elif closure:
                record["flag"] = "closure_not_novel"
            else:
                record["flag"] = "no_l2_closure"
        records.append(record)
        if record["flag"].startswith(("SEPARATION", "TRANSPORT")):
            print(f"[{index}] {record['flag']} std2={std2:.7f} os2={os2:.7f} os3={os3:.7f}", flush=True)

    flags = {}
    for record in records:
        flags[record["flag"]] = flags.get(record["flag"], 0) + 1
    report = {
        "schema_version": 1,
        "seed": args.seed,
        "count": args.count,
        "closure_tol": args.closure_tol,
        "separation_tol": args.separation_tol,
        "flag_counts": flags,
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(flags, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
