"""M2.6 KKT probe, task 2: 50-digit refinement of the KKT/stationary system.

Two regimes of the reduced SDP (see kkt_probe.py for the ansatz), identified
numerically:

Regime I (0 <= alpha <= alpha_c, alpha_c ~ 0.4226):
    all three blocks rank 2; q1, r1 nonzero.  Eliminating q0 = v+w-q1 and
    r0 = 2z-r1, the active-set KKT system in (p, q1, r1, v, w, z) with
    multipliers (m0, m1, m2) is

        d0 := p*(q0+r0) - 2*q0^2 = 0          (det Phi_00 = 0 branch)
        d1 := (1-p)*(q1+r1) - 2*q1^2 = 0      (det Phi_01 = 0 branch)
        d2 := det M2 = 0
        grad F = m0 grad d0 + m1 grad d1 + m2 grad d2

    with F = (2*alpha-4)*p + 8*v - 8*q1  (+ const 2-alpha).

Regime II (alpha >= alpha_c):
    q1 = r1 = 0 (Phi_01 = (1-p) E00, rank 1), p at its lower boundary
    p = 2 s^2 / t with s = v+w, t = v+w+2z.  Remaining concave maximization
    in (v, s, t):  f = 8v - 2*(4-2*alpha)*s^2/t  s.t.  M2 PSD, and at the
    optimum g := det M2 = 0 with grad f parallel to grad g.

omega_os^1(alpha) = 2 - alpha + (stationary value of the shift-invariant part).

Equations are generated with sympy, lambdified to mpmath, and solved with
mpmath.findroot starting from a CLARABEL solution of the reduced SDP.
Idempotent: batch results are overwritten on each run.

Usage:
    python scripts/kkt_refine.py 0.5            # single point, 50 dps
    python scripts/kkt_refine.py --batch        # rational grid -> JSON
    python scripts/kkt_refine.py --alpha-c      # locate the regime boundary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OUT_PATH = Path("artifacts/m26/kkt/refined.json")

# ---------------------------------------------------------------------------
# Symbolic equation construction
# ---------------------------------------------------------------------------

alpha_sym = sp.Symbol("alpha")


def _det_m2(v, w, z):
    M2 = sp.Matrix([[sp.Rational(1, 2), v, w], [v, v, z], [w, z, w]])
    return sp.expand(M2.det())


def build_regime1():
    p, q1, r1, v, w, z, m0, m1, m2 = sp.symbols("p q1 r1 v w z m0 m1 m2")
    q0 = v + w - q1
    r0 = 2 * z - r1
    d0 = sp.expand(p * (q0 + r0) - 2 * q0**2)
    d1 = sp.expand((1 - p) * (q1 + r1) - 2 * q1**2)
    d2 = _det_m2(v, w, z)
    F = (2 * alpha_sym - 4) * p + 8 * v - 8 * q1
    unknowns = (p, q1, r1, v, w, z, m0, m1, m2)
    eqs = [d0, d1, d2]
    for x in (p, q1, r1, v, w, z):
        eqs.append(
            sp.expand(
                sp.diff(F, x)
                - m0 * sp.diff(d0, x)
                - m1 * sp.diff(d1, x)
                - m2 * sp.diff(d2, x)
            )
        )
    omega_expr = 2 - alpha_sym + F
    return unknowns, eqs, omega_expr


def build_regime2():
    v, s, t, mu = sp.symbols("v s t mu")
    w = s - v
    z = (t - s) / 2
    g = _det_m2(v, w, z)
    f = 8 * v - 2 * (4 - 2 * alpha_sym) * s**2 / t
    unknowns = (v, s, t, mu)
    eqs = [g]
    for x in (v, s, t):
        eqs.append(sp.expand((sp.diff(f, x) - mu * sp.diff(g, x)) * t**2))
    omega_expr = 2 - alpha_sym + f
    return unknowns, eqs, omega_expr


# Lambdified at import time (cheap).
_R1 = build_regime1()
_R2 = build_regime2()
_R1_FUNS = [sp.lambdify(_R1[0] + (alpha_sym,), e, "mpmath") for e in _R1[1]]
_R1_OMEGA = sp.lambdify(_R1[0] + (alpha_sym,), _R1[2], "mpmath")
_R2_FUNS = [sp.lambdify(_R2[0] + (alpha_sym,), e, "mpmath") for e in _R2[1]]
_R2_OMEGA = sp.lambdify(_R2[0] + (alpha_sym,), _R2[2], "mpmath")


def _lstsq(cols, rhs):
    """Least-squares solution of [c1|...|ck] mu = rhs with mpmath matrices."""
    A = mp.matrix(list(zip(*cols)))
    b = mp.matrix(rhs)
    res = mp.lu_solve((A.T * A), A.T * b)
    if isinstance(res, mp.matrix):
        return [res[i, 0] for i in range(res.rows)]
    return [res]


def refine_regime1(alpha, start9, dps=50):
    """start9 = (p, q1, r1, v, w, z, m0, m1, m2) initial guess."""
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        a = mp.mpf(alpha)
        p0, q10, r10, v0, w0, z0 = (mp.mpf(x) for x in start9[:6])
        if all(x == 0 for x in start9[6:]):
            start9 = (p0, q10, r10, v0, w0, z0) + tuple(
                _r1_mult_start((p0, q10, r10, v0, w0, z0), a)
            )

        def system(*args):
            return [fn(*args, a) for fn in _R1_FUNS]

        root = mp.findroot(system, tuple(mp.mpf(x) for x in start9), tol=mp.mpf(10) ** (-(dps - 10)))
        omega = _R1_OMEGA(*root, a)
        names = ("p", "q1", "r1", "v", "w", "z", "m0", "m1", "m2")
        return omega, dict(zip(names, root))
    finally:
        mp.mp.dps = old


def _r1_mult_start(point6, a):
    """Least-squares (m0, m1, m2) at a regime-I point, central differences."""
    gradF = [_value_diff("F", point6, i, a) for i in range(6)]
    cols = [[_value_diff(d, point6, i, a) for i in range(6)] for d in ("d0", "d1", "d2")]
    return _lstsq(cols, gradF)


def _det_m2_num(v, w, z):
    return (
        mp.mpf("0.5") * (v * w - z**2)
        - v * (v * w - v * z)
        + w * (v * z - w * v)
    )


def _value_diff(which, point6, i, a):
    h = mp.mpf(10) ** (-(mp.mp.dps // 3))
    lo = [mp.mpf(x) for x in point6]; lo[i] -= h
    hi = [mp.mpf(x) for x in point6]; hi[i] += h

    def value(coords):
        pp, qq1, rr1, vv, ww, zz = coords
        qq0 = vv + ww - qq1
        rr0 = 2 * zz - rr1
        if which == "F":
            return (2 * a - 4) * pp + 8 * vv - 8 * qq1
        if which == "d0":
            return pp * (qq0 + rr0) - 2 * qq0**2
        if which == "d1":
            return (1 - pp) * (qq1 + rr1) - 2 * qq1**2
        return _det_m2_num(vv, ww, zz)

    return (value(hi) - value(lo)) / (2 * h)


def refine_regime2(alpha, start4, dps=50):
    """start4 = (v, s, t, mu) initial guess."""
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        a = mp.mpf(alpha)
        v0, s0, t0 = (mp.mpf(x) for x in start4[:3])
        if start4[3] == 0:
            start4 = (v0, s0, t0, _r2_mult_start((v0, s0, t0), a))

        def system(*args):
            return [fn(*args, a) for fn in _R2_FUNS]

        root = mp.findroot(system, tuple(mp.mpf(x) for x in start4), tol=mp.mpf(10) ** (-(dps - 10)))
        omega = _R2_OMEGA(*root, a)
        names = ("v", "s", "t", "mu")
        return omega, dict(zip(names, root))
    finally:
        mp.mp.dps = old


def _r2_mult_start(point3, a):
    def value(which, coords):
        vv, ss, tt = coords
        ww = ss - vv
        zz = (tt - ss) / 2
        if which == "f":
            return 8 * vv - 2 * (4 - 2 * a) * ss**2 / tt
        return _det_m2_num(vv, ww, zz)

    def diff(which, i):
        h = mp.mpf(10) ** (-(mp.mp.dps // 3))
        lo = list(point3); lo[i] -= h
        hi = list(point3); hi[i] += h
        return (value(which, hi) - value(which, lo)) / (2 * h)

    gradf = [diff("f", i) for i in range(3)]
    gradg = [diff("g", i) for i in range(3)]
    return _lstsq([gradg], gradf)[0]


# ---------------------------------------------------------------------------
# Double-precision starting points from the reduced SDP
# ---------------------------------------------------------------------------


def start_from_sdp(alpha: float):
    """Solve the reduced 8-var SDP with CLARABEL; return (regime, start dict)."""
    import cvxpy as cp

    p, q0, r0, q1, r1, v, w, z = cp.Variable(8)
    M0 = cp.bmat([[p, q0, q0], [q0, q0, r0], [q0, r0, q0]])
    M1 = cp.bmat([[1 - p, q1, q1], [q1, q1, r1], [q1, r1, q1]])
    M2 = cp.bmat([[0.5, v, w], [v, v, z], [w, z, w]])
    objective = alpha * (2 * p - 1) + 4 * (q0 - q1) - 4 * p + 2 + 4 * (v - w)
    problem = cp.Problem(
        cp.Maximize(objective),
        [M0 >> 0, M1 >> 0, M2 >> 0, q0 + q1 == v + w, r0 + r1 == 2 * z],
    )
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
        max_iter=4000,
    )
    pv, q0v, r0v, q1v, r1v, vv, wv, zv = (float(x.value) for x in (p, q0, r0, q1, r1, v, w, z))
    sdp_value = float(problem.value)
    if abs(q1v) > 1e-7 or abs(r1v) > 1e-7:
        regime = 1
        start = (pv, q1v, r1v, vv, wv, zv, 0.0, 0.0, 0.0)
    else:
        regime = 2
        start = (vv, q0v, q0v + 2 * zv, 0.0)
    return regime, start, sdp_value


def solve_alpha(alpha, dps=50, warm=None):
    """High-precision omega and KKT point at a given alpha.

    warm: optional (regime, start) used instead of an SDP solve.
    Returns (omega, regime, variables, sdp_value).
    """
    if warm is None:
        regime, start, sdp_value = start_from_sdp(float(alpha))
    else:
        regime, start = warm
        sdp_value = None
    if regime == 1:
        omega, variables = refine_regime1(alpha, start, dps=dps)
    else:
        omega, variables = refine_regime2(alpha, start, dps=dps)
    return omega, regime, variables, sdp_value


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------


def fmt(x, n=45):
    return mp.nstr(x, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", nargs="?", help="single alpha to refine")
    parser.add_argument("--batch", action="store_true", help="rational grid -> JSON")
    parser.add_argument("--alpha-c", action="store_true", help="locate regime boundary")
    parser.add_argument("--dps", type=int, default=50)
    args = parser.parse_args()

    if args.alpha is not None:
        alpha = mp.mpf(args.alpha)
        omega, regime, variables, sdp_value = solve_alpha(alpha, dps=args.dps)
        print(f"alpha = {mp.nstr(alpha, 20)}   regime = {regime}")
        if sdp_value is not None:
            print(f"SDP value      = {sdp_value:.12f}")
        print(f"omega (refined)= {fmt(omega)}")
        if sdp_value is not None:
            print(f"|refined - SDP|= {mp.nstr(abs(omega - sdp_value), 5)}")
        for key, val in variables.items():
            print(f"  {key:>3s} = {fmt(val, 40)}")
        return

    if args.alpha_c:
        # Secant iteration on q1(alpha) along the regime-I branch.
        mp.mp.dps = args.dps
        regime, start, _ = start_from_sdp(0.4)
        a_lo, a_hi = mp.mpf("0.42"), mp.mpf("0.43")

        def q1_of(a, start):
            omega, vars_ = refine_regime1(a, start, dps=args.dps)
            return vars_["q1"], vars_

        q_lo, vars_lo = q1_of(a_lo, start)
        start_hi = tuple(vars_lo[k] for k in ("p", "q1", "r1", "v", "w", "z", "m0", "m1", "m2"))
        q_hi, vars_hi = q1_of(a_hi, start_hi)
        print(f"q1({mp.nstr(a_lo,10)}) = {mp.nstr(q_lo,10)}")
        print(f"q1({mp.nstr(a_hi,10)}) = {mp.nstr(q_hi,10)}")
        prev = (a_lo, q_lo, vars_lo)
        curr = (a_hi, q_hi, vars_hi)
        for _ in range(12):
            a_new = curr[0] - curr[1] * (curr[0] - prev[0]) / (curr[1] - prev[1])
            start = tuple(curr[2][k] for k in ("p", "q1", "r1", "v", "w", "z", "m0", "m1", "m2"))
            q_new, vars_new = q1_of(a_new, start)
            print(f"q1({mp.nstr(a_new, 20)}) = {mp.nstr(q_new, 10)}")
            prev, curr = curr, (a_new, q_new, vars_new)
            if abs(q_new) < mp.mpf(10) ** (-(args.dps - 15)):
                break
        alpha_c = curr[0]
        print(f"\nalpha_c = {fmt(alpha_c)}")
        for deg in (2, 3, 4, 6):
            poly = mp.pslq([alpha_c**i for i in range(deg + 1)])
            if poly is not None:
                print(f"  algdep deg {deg}: {poly}")
        return

    if args.batch:
        rationals = [
            sp.Rational(0), sp.Rational(1, 16), sp.Rational(1, 8), sp.Rational(3, 16),
            sp.Rational(1, 4), sp.Rational(5, 16), sp.Rational(3, 8), sp.Rational(2, 5),
            sp.Rational(7, 16), sp.Rational(9, 20), sp.Rational(1, 2), sp.Rational(5, 8),
            sp.Rational(3, 4), sp.Rational(7, 8), sp.Rational(1), sp.Rational(9, 8),
            sp.Rational(5, 4), sp.Rational(11, 8), sp.Rational(3, 2), sp.Rational(13, 8),
            sp.Rational(7, 4), sp.Rational(15, 8), sp.Rational(19, 10),
        ]
        results = []
        warm = None
        for r in sorted(rationals):
            alpha = mp.mpf(r.numerator) / r.denominator
            omega, regime, variables, sdp_value = solve_alpha(alpha, dps=args.dps, warm=warm)
            warm = None  # warm starts are regime-fragile; SDP start is cheap enough
            results.append(
                {
                    "alpha": str(r),
                    "alpha_float": float(r),
                    "regime": regime,
                    "omega": mp.nstr(omega, args.dps - 5),
                    "sdp_value": sdp_value,
                    "variables": {k: mp.nstr(val, args.dps - 5) for k, val in variables.items()},
                }
            )
            print(f"alpha={str(r):>6s} regime={regime} omega={fmt(omega, 40)}")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps({"dps": args.dps, "results": results}, indent=2))
        print(f"wrote {OUT_PATH}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
