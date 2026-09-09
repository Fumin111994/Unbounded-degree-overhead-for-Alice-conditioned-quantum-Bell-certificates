#!/usr/bin/env python3
"""M2.6 numeric grid: level-one values of tilted CHSH as a function of alpha."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402
from npa2.standard import build_standard_problem  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "m26" / "grid"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9],
    )
    args = parser.parse_args()

    records = []
    for alpha in args.alphas:
        functional = tilted_chsh(alpha)
        std = solve_problem(
            build_standard_problem(
                functional, 1, enforce_probability_positivity=False
            ),
            "CLARABEL",
        )[0]["value"]
        os_ = solve_problem(
            build_onesided_problem(
                functional, 1, enforce_probability_positivity=False
            ),
            "CLARABEL",
        )[0]["value"]
        records.append(
            {
                "alpha": alpha,
                "standard_raw": std,
                "two_sqrt2_plus_alpha": 2.0**1.5 + alpha,
                "std_matches_formula": abs(std - (2.0**1.5 + alpha)) < 1e-6,
                "one_sided_raw": os_,
                "quantum": (8.0 + 2.0 * alpha * alpha) ** 0.5,
                "gap": std - os_,
            }
        )
        print(
            f"alpha={alpha:6.3f} std={std:.8f} 2sqrt2+a={2.0**1.5 + alpha:.8f} "
            f"os={os_:.8f} gap={std - os_:.8f}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "grid.json").write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
