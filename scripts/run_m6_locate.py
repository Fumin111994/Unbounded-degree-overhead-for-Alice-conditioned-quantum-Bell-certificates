#!/usr/bin/env python3
"""M6: locate the critical tilt alpha* for level-two one-sided closure.

Computes omega_os^2(alpha) on a grid and compares with the quantum value
sqrt(8 + 2 alpha^2); refines the threshold by bisection; tries to identify
alpha* algebraically.  Output: artifacts/m6/locate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def values(alpha: float) -> dict:
    functional = tilted_chsh(alpha)
    os2 = solve_problem(
        build_onesided_problem(functional, 2, enforce_probability_positivity=False),
        "CLARABEL",
    )[0]["value"]
    std2 = solve_problem(
        build_standard_problem(functional, 2, enforce_probability_positivity=False),
        "CLARABEL",
    )[0]["value"]
    return {"alpha": alpha, "os2": os2, "std2": std2,
            "quantum": float(np.sqrt(8 + 2 * alpha * alpha)),
            "gap": os2 - float(np.sqrt(8 + 2 * alpha * alpha))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    args = parser.parse_args()

    grid = [1.0 + 0.05 * k for k in range(0, 11)]
    records = [values(a) for a in grid]
    for r in records:
        print(f"alpha={r['alpha']:.2f} os2={r['os2']:.9f} quantum={r['quantum']:.9f} gap={r['gap']:+.2e}", flush=True)

    # bisection on gap = 0 between the bracketing grid points
    bracket = None
    for left, right in zip(records, records[1:]):
        if left["gap"] <= 1e-7 < right["gap"]:
            bracket = (left["alpha"], right["alpha"])
            break
    refined = None
    if bracket:
        lo, hi = bracket
        for _ in range(22):
            mid = (lo + hi) / 2
            if values(mid)["gap"] <= 1e-7:
                lo = mid
            else:
                hi = mid
        refined = (lo, hi)
        mid = (lo + hi) / 2
        print(f"alpha* in ({lo:.10f}, {hi:.10f}); midpoint value: {values(mid)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "locate.json").write_text(
        json.dumps({"records": records, "bracket": refined}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
