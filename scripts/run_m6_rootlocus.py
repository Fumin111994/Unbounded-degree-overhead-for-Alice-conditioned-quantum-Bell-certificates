#!/usr/bin/env python3
"""M6: root-locus analysis of the exact rational epsilon-certificate family.

The exact rational dual certificate z* at bound B (epsilon closure) extends
affinely to z(t) = z* + (t - B) D for any direction D with A D = c.  If every
S block's determinant has its largest root <= q in (-inf, t1], then S(t) is
PSD on (q, B] and S(q) is PSD by closedness, proving omega_os^2 <= q exactly.

This script checks the min-norm direction D = A^+ c: compute per block the
largest real root of det S*(t) and compare it with q using exact CRootOf
arithmetic.  Any root strictly above q means this family proves closure only
up to that algebraic slack.

Every root is stored as a full exact description: the determinant polynomial
over QQ, the CRootOf (minpoly + root index), and a certified rational
isolating interval (sympy Poly.intervals, refined to width <= 2^-60).  All
comparisons (root > q, root < q + s*) are exact CRootOf arithmetic; s* is
certified, not floated.

Output: artifacts/m6/rootlocus.json
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
from run_exact_level1 import find_onesided_dual  # noqa: E402
from run_m6_closure import exact_tilted_functional, quantum_alg  # noqa: E402

T = sp.Symbol("t")
SSTAR = sp.Rational(1, 50_000_000)  # 2e-8: certified slack budget
ISO_EPS = sp.Rational(1, 2**60)     # isolating interval width budget


def rational_str(value) -> str:
    value = sp.Rational(value)
    return f"{value.p}/{value.q}"


def exact_root_record(poly: sp.Poly, root, q_root) -> dict:
    """Exact data for one real root: isolation + exact comparisons."""
    intervals = poly.intervals(eps=ISO_EPS)
    lo = hi = None
    for (a, b), _mult in intervals:
        if sp.Rational(a) <= root <= sp.Rational(b):
            lo, hi = sp.Rational(a), sp.Rational(b)
            break
    if lo is None:
        raise RuntimeError("no isolating interval contains the root")
    return {
        "crootof": str(root),
        "isolating_interval": [rational_str(lo), rational_str(hi)],
        "interval_width": rational_str(hi - lo),
        "above_q": bool(root > q_root),
        "below_q_plus_sstar": bool(root < q_root + SSTAR),
        "float30": str(sp.N(root, 30)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "m6")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-denominator", type=int, default=10**6)
    parser.add_argument("--alphas", nargs="+", default=["5/4", "41/32"])
    parser.add_argument("--epsilon", default="1/1000000")
    args = parser.parse_args()

    results = []
    failures: list[str] = []
    for text in args.alphas:
        alpha = Fraction(text)
        functional = exact_tilted_functional(alpha)
        q = quantum_alg(alpha)
        q_float = float(q)
        eps = Fraction(args.epsilon)
        B = Fraction(int(np.ceil(q_float * 10**9)), 10**9) + eps

        payload = find_onesided_dual(
            args.solver, args.max_denominator, positivity=False,
            functional=functional, bound=B, level=2,
        )
        if payload["verification"]["status"] != "PASS":
            failures.append(f"{alpha}: epsilon certificate failed")
            continue
        cert = payload["certificate"]

        def dec(s: str) -> Fraction:
            num, _, den = s.partition("/")
            return Fraction(int(num), int(den or "1"))

        names, A, b0 = exact.onesided_identity_system(functional, 2, Fraction(0), False)
        _, _, b1 = exact.onesided_identity_system(functional, 2, Fraction(1), False)
        c = b1 - b0
        z1 = A.T * (A * A.T).inv() * c
        z_star = sp.Matrix([dec(cert[name]) for name in names])

        # family z(t) = z* + (t - B) z1 ; S blocks affine in t over QQ
        q_root = sp.CRootOf(T**2 - (8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2), 1)
        s_idx = [k for k, name in enumerate(names) if name.startswith("S_")]
        block_keys = sorted({
            (int(names[k].split(":")[0][2:].split("_")[0]),
             int(names[k].split(":")[0][2:].split("_")[1])) for k in s_idx
        })
        alpha_report: dict = {"alpha": str(alpha),
                              "B": f"{B.numerator}/{B.denominator}",
                              "q": str(q),
                              "sstar_budget": rational_str(SSTAR),
                              "blocks": {}}
        max_slack_root = None
        for (x, a) in block_keys:
            matrix_t = sp.zeros(5)
            for k in s_idx:
                parts = names[k].split(":")
                bx, ba = (int(p) for p in parts[0][2:].split("_"))
                if (bx, ba) != (x, a):
                    continue
                i, j = int(parts[1]), int(parts[2])
                value = sp.Rational(z_star[k]) + (T - sp.Rational(B)) * z1[k]
                matrix_t[i, j] = value
                matrix_t[j, i] = value
            det_expr = sp.expand(matrix_t.det())
            if det_expr == 0:
                alpha_report["blocks"][f"{x}:{a}"] = {"det": "identically zero"}
                failures.append(f"{alpha} block {(x,a)}: det identically zero")
                continue
            poly = sp.Poly(det_expr, T)
            roots = poly.all_roots(radicals=False)
            real_roots = [r for r in roots if r.is_real]
            real_above = [r for r in real_roots if r > q_root]
            largest = max(real_roots) if real_roots else None
            record = {
                "det_coefficients_ascending": [
                    str(c) for c in sp.Poly(det_expr, T).all_coeffs()[::-1]
                ],
                "n_real_roots": len(real_roots),
                "n_roots_above_q": len(real_above),
            }
            if largest is None:
                record["largest_root"] = None
            else:
                record["largest_root"] = exact_root_record(poly, largest, q_root)
                if max_slack_root is None or largest > max_slack_root:
                    max_slack_root = largest
            alpha_report["blocks"][f"{x}:{a}"] = record
        if max_slack_root is not None:
            alpha_report["max_root_over_blocks"] = {
                "crootof": str(max_slack_root),
                "above_q": bool(max_slack_root > q_root),
                "below_q_plus_sstar": bool(max_slack_root < q_root + SSTAR),
                "float30": str(sp.N(max_slack_root, 30)),
            }
            if not (max_slack_root < q_root + SSTAR):
                failures.append(
                    f"{alpha}: largest det root not below q + 2e-8 (exact)"
                )
        results.append(alpha_report)
        print(f"alpha={alpha}: largest det root "
              f"{sp.N(max_slack_root, 12) if max_slack_root is not None else None} "
              f"vs q={sp.N(q_root, 12)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": 2,
        "statement": (
            "for each alpha, the exact rational certificate family S(t) at "
            "bound B has every block determinant's largest real root below "
            "q + 2e-8 (exact CRootOf comparisons), pinning omega_os^2(alpha) "
            "in [q, q + 2e-8]; roots are stored with minpoly (CRootOf) and "
            "certified rational isolating intervals of width <= 2^-60"
        ),
        "results": results,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (args.output / "rootlocus.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"], "failures": failures[:4]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
