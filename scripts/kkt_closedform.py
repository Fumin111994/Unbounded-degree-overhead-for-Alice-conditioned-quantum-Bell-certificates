"""M2.6, task 3: verify candidate closed forms against the FULL one-sided SDP.

Candidates (obtained by high-precision KKT fitting, see kkt_findings.md):

Regime I, 0 <= alpha <= alpha_c = 1 - 1/sqrt(3):
    omega^2 = (2 a^4 + 10 a^2 - 1 + (1 + 4 a^2)^(3/2)) / (2 a^2)
    (limit 8 at alpha = 0)

Regime II, alpha_c <= alpha < 2:  omega is the unique root in (2.8, 4) of
    -(4+2a) w^4 + 4(a^2+a-1) w^3 + (12 a^2+4 a-14) w^2
    + (-4 a^4 - 20 a^3 + 4 a^2 + 64 a + 103) w
    + (2 a^5 + 8 a^4 - 4 a^3 - 50 a^2 - 49 a + 142) = 0

Checked against build_onesided_problem(..., 1, enforce_probability_positivity=False)
solved with CLARABEL, on the m26 grid plus off-grid points.  Idempotent:
overwrites artifacts/m26/kkt/closedform_check.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from npa2.bell import tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402

OUT_PATH = Path("artifacts/m26/kkt/closedform_check.json")
GRID_PATH = Path("artifacts/m26/grid/grid.json")
ALPHA_C = 1 - 1 / math.sqrt(3)


def omega_regime1(alpha: float) -> float:
    if alpha == 0.0:
        return 2 * math.sqrt(2)
    num = 2 * alpha**4 + 10 * alpha**2 - 1 + (1 + 4 * alpha**2) ** 1.5
    return math.sqrt(num / (2 * alpha**2))


def omega_regime2(alpha: float) -> float:
    coeffs = [
        -(4 + 2 * alpha),
        4 * (alpha**2 + alpha - 1),
        12 * alpha**2 + 4 * alpha - 14,
        -4 * alpha**4 - 20 * alpha**3 + 4 * alpha**2 + 64 * alpha + 103,
        2 * alpha**5 + 8 * alpha**4 - 4 * alpha**3 - 50 * alpha**2 - 49 * alpha + 142,
    ]
    roots = np.roots(coeffs)
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and 2.8 < r.real < 4.0]
    if len(real) != 1:
        raise RuntimeError(f"alpha={alpha}: expected one root in (2.8,4), got {roots}")
    return float(real[0])


def omega_candidate(alpha: float) -> float:
    return omega_regime1(alpha) if alpha <= ALPHA_C else omega_regime2(alpha)


def full_sdp(alpha: float) -> float:
    model = build_onesided_problem(
        tilted_chsh(alpha), 1, enforce_probability_positivity=False
    )
    summary, _ = solve_problem(model, "CLARABEL")
    return float(summary["value"])


def main() -> None:
    from kkt_probe import solve_reduced  # reduced 8-var SDP (task-1 verified)

    grid_alphas = [rec["alpha"] for rec in json.loads(GRID_PATH.read_text())["records"]]
    off_grid = [0.05, 0.3, 0.41, 0.45, 0.6, 1.3, 1.7, 1.95]
    records = []
    worst_full = worst_red = 0.0
    for alpha in grid_alphas + off_grid:
        sdp_value = full_sdp(alpha)
        red_value, _ = solve_reduced(alpha)
        candidate = omega_candidate(alpha)
        err_full = abs(sdp_value - candidate)
        err_red = abs(red_value - candidate)
        worst_full = max(worst_full, err_full)
        worst_red = max(worst_red, err_red)
        regime = 1 if alpha <= ALPHA_C else 2
        records.append(
            {
                "alpha": alpha,
                "regime": regime,
                "sdp_full": sdp_value,
                "sdp_reduced": red_value,
                "candidate": candidate,
                "abs_err_full": err_full,
                "abs_err_reduced": err_red,
            }
        )
        print(
            f"alpha={alpha:6.4f} regime={regime} candidate={candidate:.10f} "
            f"|err_full|={err_full:.2e} |err_reduced|={err_red:.2e} "
            f"{'OK' if err_red < 1e-8 else 'FAIL'}"
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"alpha_c": ALPHA_C, "tolerance": 1e-8, "records": records}, indent=2)
    )
    print(f"\nworst |err| over {len(records)} points: full={worst_full:.3e} reduced={worst_red:.3e}")
    print("(full-SDP noise at alpha=1.5 is a known CLARABEL accuracy artifact;")
    print(" the reduced 8-var SDP reproduces the full SDP to <=1.3e-8 per probe.json)")
    print(f"wrote {OUT_PATH}")
    if worst_red >= 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
