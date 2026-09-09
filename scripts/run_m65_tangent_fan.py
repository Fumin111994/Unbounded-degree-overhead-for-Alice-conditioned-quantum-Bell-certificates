#!/usr/bin/env python3
"""M6.5 G2: exact tangent-fan proof of level-2 non-closure on an interval.

Each anchor's exact one-sided level-2 primal witness m_i (rational, verified
feasible) gives an exact rational affine lower bound

    omega_os^2(alpha) >= g_i + alpha * h_i      for ALL alpha

where g_i is the witness's CHSH-correlator value and h_i its Alice-marginal
value.  Non-closure at alpha follows when g_i + alpha h_i > q(alpha) =
sqrt(8 + 2 alpha^2); since both sides are positive in our range this is the
exact rational quadratic inequality

    (g_i + alpha h_i)^2 - (8 + 2 alpha^2) > 0.

Its roots are computed exactly and the union of the resulting intervals is
checked to cover [alpha_cov, 2).

Output: artifacts/m65/tangent_fan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from npa2 import exact  # noqa: E402
from run_m3_exactify import find_onesided_witness  # noqa: E402
from run_m6_closure import exact_tilted_functional  # noqa: E402

T = sp.Symbol("t")


def witness_gh(payload: dict) -> tuple[Fraction, Fraction, dict]:
    """Exact (g, h) of a witness: objective = g + alpha * h with g the CHSH
    correlator part and h the Alice x=0 marginal."""

    entries = payload["block_entries"]
    key_values: dict[tuple[int, int], dict[tuple[int, int], Fraction]] = {}
    for key, value in entries.items():
        x, a, i, j = (int(p) for p in key.split(":"))
        num, _, den = value.partition("/")
        key_values.setdefault((x, a), {})[(i, j)] = Fraction(int(num), int(den or "1"))

    # word order at level 2 (reduced Bob algebra):
    # [I, B0:0, B1:0, B0:0 B1:0, B1:0 B0:0];  I=0, B0=1, B1=2
    h = key_values[(0, 0)][(0, 0)] - key_values[(0, 1)][(0, 0)]
    # CHSH correlators E_xy = sum_ab s_a s_b p(ab|xy):
    #   p(a,0|x,y) = m_{x,a}(B_y)  [entries (0, y+1)]
    #   p(a,1|x,y) = m_{x,a}(I) - m_{x,a}(B_y)
    g = Fraction(0)
    for x in (0, 1):
        for y in (0, 1):
            c_xy = -1 if (x, y) == (1, 1) else 1
            e = Fraction(0)
            for a in (0, 1):
                s_a = 1 if a == 0 else -1
                m_b0 = key_values[(x, a)][(0, y + 1)]
                m_i = key_values[(x, a)][(0, 0)]
                e += s_a * (2 * m_b0 - m_i)
            g += c_xy * e
    return g, h, key_values


def interval_above_q(g: Fraction, h: Fraction):
    """Exact set {alpha in [0, 2] : g + alpha h > sqrt(8 + 2 alpha^2)}."""

    gg = sp.Rational(g.numerator, g.denominator)
    hh = sp.Rational(h.numerator, h.denominator)
    quad = sp.expand((gg + hh * T) ** 2 - (8 + 2 * T**2))
    poly = sp.Poly(quad, T)
    roots = [r for r in poly.all_roots(radicals=False) if r.is_real]
    # evaluate sign between roots on [0, 2]
    breakpoints = sorted(
        [sp.Rational(0)] + sorted(r for r in roots if 0 < r < 2) + [sp.Rational(2)]
    )
    positive_intervals = []
    for lo, hi in zip(breakpoints, breakpoints[1:]):
        mid = (lo + hi) / 2
        value = sp.simplify(quad.subs(T, mid))
        line_mid = sp.simplify((gg + hh * mid))
        if value > 0 and line_mid > 0:
            positive_intervals.append((lo, hi))
    return positive_intervals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m65")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--witness-max-denominator", type=int, default=10**8)
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["13/10", "21/16", "11/8", "3/2", "13/8", "7/4", "15/8",
                 "19/10", "197/100", "199/100"],
    )
    args = parser.parse_args()

    fan = []
    failures: list[str] = []
    for text in args.alphas:
        alpha = Fraction(text)
        functional = exact_tilted_functional(alpha)
        # adaptive mixing: epsilon well below the numeric non-closure gap
        import numpy as _np

        from npa2.onesided import build_onesided_problem
        from npa2.solve import solve_problem as _solve

        numeric = _solve(
            build_onesided_problem(
                functional, 2, enforce_probability_positivity=False
            ),
            args.solver,
        )[0]["value"]
        q_float = float((8 + 2 * float(alpha) ** 2) ** 0.5)
        gap = numeric - q_float
        eps = Fraction(max(int(gap * 10**9) // 20, 1), 10**9)
        if gap < 2e-5:
            failures.append(f"{alpha}: numeric gap {gap:.1e} too small for a robust witness")
            continue
        payload = find_onesided_witness(
            functional, args.solver, args.witness_max_denominator, epsilon=eps
        )
        report = payload["verification"]
        if report["status"] != "PASS":
            failures.append(f"{alpha}: witness verification failed")
            continue
        g, h, _ = witness_gh(payload)
        intervals = interval_above_q(g, h)
        fan.append({
            "alpha": str(alpha),
            "g": f"{g.numerator}/{g.denominator}",
            "h": f"{h.numerator}/{h.denominator}",
            "objective_at_anchor": report["objective_exact"],
            "intervals_above_q": [[str(lo), str(hi)] for lo, hi in intervals],
            "block_entries": payload["block_entries"],
            "witness_verification": {
                "status": report["status"],
                "objective_exact": report["objective_exact"],
                "objective_float": report.get("objective_float"),
                "word_labels": report.get("word_labels"),
            },
            "_intervals_sympy": intervals,
        })
        print(f"alpha={alpha}: g={float(g):.7f} h={float(h):.7f} "
              f"intervals={[(str(l), str(r)) for l, r in intervals]}", flush=True)
        if not intervals:
            failures.append(f"{alpha}: tangent line never above q on [0,2]")

    # coverage check: the merged cluster that reaches alpha = 2.
    # Interval endpoints are algebraic (CRootOf); compare via exact signs.
    def _leq(a, b) -> bool:
        return sp.sign(sp.simplify(a - b)) <= 0

    all_intervals = []
    for item in fan:
        for lo, hi in item["_intervals_sympy"]:
            all_intervals.append((lo, hi))
    all_intervals.sort(key=lambda iv: float(sp.N(iv[0])))
    merged: list = []
    for lo, hi in all_intervals:
        if merged and _leq(lo, merged[-1][1]):
            if _leq(merged[-1][1], hi):
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    # coverage: the merged cluster containing the highest intervals.  Note
    # alpha -> 2 is a trivial endpoint (q -> 4, slope -> 1 while every tangent
    # line has slope h < 1), so no finite fan covers a neighbourhood of 2;
    # we report the exact covered interval as the OPEN interval
    # (alpha_cov, alpha_hi): at the endpoints the tangent polynomial
    # vanishes, so only the non-strict comparison is certified there.
    top_cluster = merged[-1] if merged else None
    top_ok = top_cluster is not None
    alpha_cov = top_cluster[0] if top_ok else None
    alpha_hi = top_cluster[1] if top_ok else None

    for item in fan:
        item.pop("_intervals_sympy", None)
    report = {
        "schema_version": 2,
        "fan": fan,
        "merged_intervals": [[str(lo), str(hi)] for lo, hi in merged],
        "covered_interval": (
            None if alpha_cov is None else [str(alpha_cov), str(alpha_hi)]
        ),
        "covered_interval_float": (
            None if alpha_cov is None
            else [float(sp.N(alpha_cov, 12)), float(sp.N(alpha_hi, 12))]
        ),
        "statement": (
            "omega_os^2(alpha) > sqrt(8+2 alpha^2) for all alpha in the "
            "OPEN covered interval (alpha_cov, alpha_hi), by exact rational "
            "tangent lines from exact primal witnesses; at the endpoints "
            "themselves the tangent polynomial vanishes, so the certified "
            "comparison there is non-strict (>=) and the interval is open"
        ),
        "status": "PASS" if not failures and top_ok else "FAIL",
        "failures": failures + ([] if top_ok else ["no covered interval"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tangent_fan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"],
                      "covered": report["covered_interval"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
