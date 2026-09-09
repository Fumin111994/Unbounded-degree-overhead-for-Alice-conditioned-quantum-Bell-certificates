"""M2.6: exact symbolic elimination for the regime-II (alpha >= alpha_c) branch.

Regime II reduces to a 2-variable rational optimization.  Parametrize the
rank-2 locus of M2 = [[1/2,v,w],[v,v,z],[w,z,w]] rationally by
u = tan(phi_1), r = tan(phi_2):

    v = (1+r^2)/D,  w = (1+u^2)/D,  z = (1+u*r)/D,  D = 2(1+u^2)(1+r^2)

so that s = v+w = (2+u^2+r^2)/D and t = s+2z = ((u+r)^2+4)/D, and

    f(u,r) = 8v - 2(4-2a) s^2/t,   omega = 2 - a + f.

We eliminate (u, r) from {f_u = 0, f_r = 0, omega = 2 - a + f} to obtain an
exact polynomial P(omega, alpha) = 0.  Currently the script prints the
factored resultant of the two stationarity equations (an octic in u that
factors as a quartic in u^2 with coefficients quadratic in alpha); the full
elimination to P(omega, alpha) was instead completed numerically by
high-precision fitting in scripts/kkt_identify.py.  Idempotent.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import sympy as sp

OUT_PATH = Path("artifacts/m26/kkt/regime2_elimination.json")


def main() -> None:
    u, r, a, om = sp.symbols("u r a om")
    D = 2 * (1 + u**2) * (1 + r**2)
    v = (1 + r**2) / D
    s = (2 + u**2 + r**2) / D
    t = ((u + r) ** 2 + 4) / D
    f = 8 * v - 2 * (4 - 2 * a) * s**2 / t

    t0 = time.time()
    eq_u = sp.together(sp.diff(f, u))
    eq_r = sp.together(sp.diff(f, r))
    pu = sp.Poly(sp.fraction(eq_u)[0], u, r)
    pr = sp.Poly(sp.fraction(eq_r)[0], u, r)
    print("deg eq_u:", pu.degree(u), pu.degree(r), " terms:", len(pu.terms()))
    print("deg eq_r:", pr.degree(u), pr.degree(r), " terms:", len(pr.terms()))
    print(f"derivative+to(clear) took {time.time()-t0:.1f}s", flush=True)

    # omega equation, cleared of denominators:
    # (om - 2 + a) * D * ((u+r)^2+4) - 8(1+r^2)((u+r)^2+4) + 2(4-2a)(2+u^2+r^2)^2 = 0
    U = (u + r) ** 2 + 4
    eq_om = sp.expand((om - 2 + a) * D * U - 8 * (1 + r**2) * U + 2 * (4 - 2 * a) * (2 + u**2 + r**2) ** 2)
    po = sp.Poly(eq_om, u, r)
    print("deg eq_om:", po.degree(u), po.degree(r), " terms:", len(po.terms()), flush=True)

    # Eliminate r: resultant of (eq_u, eq_r) then with eq_om.
    t1 = time.time()
    res_r = sp.resultant(pu, pr, r)
    print(f"resultant(eq_u, eq_r, r): deg u = {sp.Poly(res_r, u).degree()}, took {time.time()-t1:.1f}s", flush=True)
    res_r = sp.factor(res_r)
    print("factored:", res_r, flush=True)


if __name__ == "__main__":
    main()
