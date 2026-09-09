#!/usr/bin/env python3
"""M3: verify a shortlisted candidate (dual solvers, full-outcome, strategy).

For a candidate record from artifacts/m3/search*/results.json:

1. re-solves std L2/L3 and os L2/L3 (raw cone) with CLARABEL and SCS;
2. re-solves std L2 and os L2/L3 with the full-outcome implementation
   (quotient cross-check);
3. maximizes the functional over explicit two-qubit strategies
   |psi> = cos t |00> + sin t |11>, projective measurements on the Bloch
   sphere (differential evolution + polish), to confirm omega_q = omega_std2.

Output: artifacts/m3/candidates/<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import correlator_functional  # noqa: E402
from npa2.fulloutcome import (  # noqa: E402
    build_onesided_problem_full,
    build_standard_problem_full,
)
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def solve_with_fallback(model, solvers):
    for solver in solvers:
        try:
            summary, _ = solve_problem(model, solver)
        except Exception:
            continue
        if summary["status"] in ("optimal", "optimal_inaccurate"):
            return summary["value"]
    return None


def bloch(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def strategy_value(params: np.ndarray, corr, amarg, bmarg) -> float:
    """Value of the functional on |psi> = cos t |00> + sin t |11> with
    projective qubit measurements (Bloch directions per question)."""

    t = params[0]
    c, s = np.cos(t), np.sin(t)
    avec = {0: bloch(params[1], params[2]), 1: bloch(params[3], params[4])}
    bvec = {0: bloch(params[5], params[6]), 1: bloch(params[7], params[8])}

    def correl(u: np.ndarray, v: np.ndarray) -> float:
        # <psi| (u.sigma) x (v.sigma) |psi> for c|00> + s|11>
        return u[0] * v[0] * (2 * c * s) - u[1] * v[1] * (2 * c * s) + u[2] * v[2]

    def local(u: np.ndarray) -> float:
        # <psi| sigma_u x I |psi> = <psi| I x sigma_u |psi> = u_z (c^2 - s^2)
        return u[2] * (c * c - s * s)

    value = 0.0
    for (x, y), coeff in corr.items():
        value += coeff * correl(avec[x], bvec[y])
    for x, coeff in amarg.items():
        value += coeff * local(avec[x])
    for y, coeff in bmarg.items():
        value += coeff * local(bvec[y])
    return float(value)


def maximize_strategy(corr, amarg, bmarg) -> dict:
    def objective(params):
        return -strategy_value(params, corr, amarg, bmarg)

    bounds = [(0.0, np.pi / 2)] + [(0.0, np.pi), (0.0, 2 * np.pi)] * 4
    best = (np.inf, None)
    rng = np.random.default_rng(0)
    # Plain differential evolution can get stuck in the product-state basin
    # (t = 0); multistart Nelder-Mead from entangled points avoids that.
    for _ in range(24):
        x0 = np.concatenate(
            [[rng.uniform(0.2, 1.35)], rng.uniform(0, np.pi, 4),
             rng.uniform(0, 2 * np.pi, 4)]
        )
        res = minimize(
            objective, x0, method="Nelder-Mead",
            options={"xatol": 1e-13, "fatol": 1e-14, "maxiter": 8000},
        )
        if res.fun < best[0]:
            best = (res.fun, res.x)
    for seed in (1, 101):
        res = differential_evolution(
            objective, bounds, seed=seed, tol=1e-12, polish=True,
            maxiter=3000, popsize=30,
        )
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return {"value": -best[0], "params": [float(p) for p in best[1]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m3" / "candidates"
    )
    parser.add_argument("--solvers", nargs="+", default=["CLARABEL", "SCS"])
    args = parser.parse_args()

    report_in = json.loads(args.results.read_text(encoding="utf-8"))
    record = next(r for r in report_in["records"] if r["index"] == args.index)
    corr = {(int(k[0]), int(k[1])): v for k, v in record["corr"].items()}
    amarg = {int(k): v for k, v in record["alice_marginals"].items()}
    bmarg = {int(k): v for k, v in record["bob_marginals"].items()}
    tag = args.tag or f"{args.results.parent.name}_{args.index}"
    functional = correlator_functional(f"candidate_{tag}", corr, amarg, bmarg)

    values: dict[str, dict] = {}
    for solver in args.solvers:
        values[solver] = {
            "std2": solve_with_fallback(
                build_standard_problem(
                    functional, 2, enforce_probability_positivity=False
                ),
                (solver,),
            ),
            "std3": solve_with_fallback(
                build_standard_problem(
                    functional, 3, enforce_probability_positivity=False
                ),
                (solver,),
            ),
            "os2": solve_with_fallback(
                build_onesided_problem(
                    functional, 2, enforce_probability_positivity=False
                ),
                (solver,),
            ),
            "os3": solve_with_fallback(
                build_onesided_problem(
                    functional, 3, enforce_probability_positivity=False
                ),
                (solver,),
            ),
        }
    values["full"] = {
        "std2_full": solve_with_fallback(
            build_standard_problem_full(
                functional, 2, enforce_probability_positivity=False
            ),
            tuple(args.solvers),
        ),
        "os2_full": solve_with_fallback(
            build_onesided_problem_full(
                functional, 2, enforce_probability_positivity=False
            ),
            tuple(args.solvers),
        ),
        "os3_full": solve_with_fallback(
            build_onesided_problem_full(
                functional, 3, enforce_probability_positivity=False
            ),
            tuple(args.solvers),
        ),
    }
    strategy = maximize_strategy(corr, amarg, bmarg)

    reference_std2 = values[args.solvers[0]]["std2"]
    out = {
        "schema_version": 1,
        "candidate_tag": tag,
        "functional": {
            "corr": record["corr"],
            "alice_marginals": record["alice_marginals"],
            "bob_marginals": record["bob_marginals"],
        },
        "classical": record["classical"],
        "novelty": {
            k: record[k]
            for k in (
                "is_xor",
                "chsh_correlator",
                "tilted_like",
                "novel",
                "nonzero_marginals",
                "nonlocal",
            )
        },
        "hierarchy_values": values,
        "strategy": {"value": float(strategy["value"]),
                     "params": [float(p) for p in strategy["params"]]},
        "strategy_matches_std2": bool(
            reference_std2 is not None
            and abs(strategy["value"] - reference_std2) < 1e-6
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"candidate_{tag}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
