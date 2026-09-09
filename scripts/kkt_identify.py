"""M2.6, task 2/3: bivariate polynomial identification of omega_os^1(alpha).

Computes high-precision KKT values omega(alpha) on dense rational grids
(separately for regime I: alpha < alpha_c = 1 - 1/sqrt(3), and regime II:
alpha > alpha_c), then fits an exact polynomial relation

    P(omega, alpha) = sum_{j<=J, k<=K} c_{jk} omega^j alpha^k = 0

by high-precision null-space + continued-fraction rational reconstruction,
and verifies the candidate on all computed points.  For regime I the ansatz
is in y = omega^2 (biquadratic structure seen in algdep).

Point cache: artifacts/m26/kkt/identify_points.json (reused across runs).
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kkt_refine import solve_alpha  # noqa: E402

CACHE_PATH = Path("artifacts/m26/kkt/identify_points.json")
ALPHA_C = 1 - 1 / mp.sqrt(3)
DPS = 110


# ---------------------------------------------------------------------------
# Point computation with cache
# ---------------------------------------------------------------------------


def get_points(alphas: list[Fraction]) -> dict[str, str]:
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    for a in alphas:
        key = f"{a.numerator}/{a.denominator}"
        if key in cache and cache[key].get("dps", 0) >= DPS - 10:
            continue
        omega, regime, _, sdp_value = solve_alpha(mp.mpf(a.numerator) / a.denominator, dps=DPS)
        cache[key] = {
            "omega": mp.nstr(omega, DPS - 10),
            "regime": regime,
            "dps": DPS - 10,
            "sdp_value": sdp_value,
        }
        print(f"  computed alpha={key} regime={regime} omega={mp.nstr(omega, 25)}", flush=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return cache


# ---------------------------------------------------------------------------
# Exact polynomial fitting
# ---------------------------------------------------------------------------


def rational_reconstruct(x, max_q=10**12):
    """Continued-fraction reconstruction of a rational from an mpf."""
    f = Fraction(str(mp.nstr(x, mp.mp.dps - 5)))
    bound = f.limit_denominator(max_q)
    if abs(mp.mpf(bound.numerator) / bound.denominator - x) < mp.mpf(10) ** (-(mp.mp.dps // 3)):
        return bound
    return None


def fit_polynomial(points, yfun, J, K):
    """Fit sum c_{jk} y^j alpha^k = 0.  Returns (coeff dict, max residual)."""
    cols = [(j, k) for j in range(J + 1) for k in range(K + 1)]
    n = len(cols)
    if len(points) < n + 2:
        return None, None
    rows = []
    for a, om in points:
        y = yfun(om)
        rows.append([y**j * a**k for (j, k) in cols])
    M = mp.matrix(rows)
    colscale = [max(abs(M[i, j]) for i in range(M.rows)) for j in range(n)]
    # square subsystem on rows spread across the range, c[last] = 1
    idx = [round(i * (len(points) - 1) / (n - 2)) for i in range(n - 1)] if n > 1 else []
    Asq = mp.matrix(n - 1, n - 1)
    bsq = mp.matrix(n - 1, 1)
    for r, i in enumerate(idx):
        for j in range(n - 1):
            Asq[r, j] = M[i, j] / colscale[j]
        bsq[r, 0] = -M[i, n - 1] / colscale[n - 1]
    try:
        sol = mp.lu_solve(Asq, bsq)
    except Exception:
        return None, None
    v = [sol[i, 0] / colscale[i] for i in range(n - 1)] + [mp.mpf(1) / colscale[n - 1]]
    # normalize by the largest entry and rational-reconstruct
    scale = max(v, key=abs)
    rationals = []
    for x in v:
        q = rational_reconstruct(x / scale)
        if q is None:
            return None, None
        rationals.append(q)
    # clear denominators -> integer coefficients
    from math import lcm

    L = 1
    for q in rationals:
        L = lcm(L, q.denominator)
    coeffs = {cols[i]: int(rationals[i] * L) for i in range(n)}
    # verify on all points
    worst = mp.mpf(0)
    for a, om in points:
        y = yfun(om)
        res = abs(sum(mp.mpf(c) * y**j * a**k for (j, k), c in coeffs.items()))
        scale_y = max(abs(y), mp.mpf(1)) ** J
        worst = max(worst, res / scale_y)
    return coeffs, worst


def print_poly(coeffs, yname):
    terms = []
    for (j, k), c in sorted(coeffs.items()):
        if c == 0:
            continue
        mon = ""
        if j:
            mon += f"{yname}^{j}" if j > 1 else yname
        if k:
            mon += f"*a^{k}" if k > 1 else ("*a" if j or True else "a")
            if j:
                pass
        if not mon:
            mon = "1"
        terms.append(f"{'+' if c > 0 else '-'} {abs(c)}{mon if mon == '1' else '*' + mon}")
    return "P = " + " ".join(terms)


def main() -> None:
    mp.mp.dps = DPS

    regime1_alphas = [Fraction(k, 64) for k in range(0, 27)] + [
        Fraction(1, 5), Fraction(3, 10), Fraction(2, 5), Fraction(5, 12), Fraction(7, 20)
    ]
    regime2_alphas = [Fraction(k, 64) for k in range(28, 122, 2)] + [
        Fraction(19, 10), Fraction(1, 2), Fraction(3, 4), Fraction(5, 4), Fraction(3, 2)
    ]
    print("computing regime-I points...")
    cache = get_points(sorted(set(regime1_alphas)))
    print("computing regime-II points...")
    cache = get_points(sorted(set(regime2_alphas)))

    def points_for(alphas):
        pts = []
        for a in sorted(set(alphas)):
            key = f"{a.numerator}/{a.denominator}"
            om = mp.mpf(cache[key]["omega"])
            pts.append((mp.mpf(a.numerator) / a.denominator, om))
        return pts

    pts1 = points_for(regime1_alphas)
    pts2 = points_for(regime2_alphas)

    print(f"\nregime I: {len(pts1)} points, fitting in y = omega^2 ...")
    for K in (1, 2, 3, 4, 5, 6):
        coeffs, worst = fit_polynomial(pts1, lambda om: om**2, J=2, K=K)
        if coeffs is None:
            print(f"  J=2, K={K}: no rational null vector (or underdetermined)")
            continue
        print(f"  J=2, K={K}: max residual {mp.nstr(worst, 3)}")
        if worst < mp.mpf(10) ** (-(DPS // 3)):
            print("   ", print_poly(coeffs, "y"))
            break

    print(f"\nregime II: {len(pts2)} points, fitting in omega ...")
    for J, K in ((4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (3, 4), (3, 5), (3, 6), (5, 4), (6, 3)):
        coeffs, worst = fit_polynomial(pts2, lambda om: om, J=J, K=K)
        if coeffs is None:
            print(f"  J={J}, K={K}: no rational null vector (or underdetermined)")
            continue
        print(f"  J={J}, K={K}: max residual {mp.nstr(worst, 3)}")
        if worst < mp.mpf(10) ** (-(DPS // 3)):
            print("   ", print_poly(coeffs, "om"))
            break


if __name__ == "__main__":
    main()
