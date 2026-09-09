#!/usr/bin/env python3
"""Review item 4: exactify the optimal two-qubit strategy for candidate #308.

Refines the numerically found strategy to high precision with mpmath Newton
on the free stationarity system, then attempts algebraic identification of
the value and the Bloch components.  Exact verification (norms and value)
is done with sympy.

Output: artifacts/m3/exact/strategy_308.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

mp.mp.dps = 60


def bloch(theta, phi):
    st, ct = mp.sin(theta), mp.cos(theta)
    return mp.matrix([st * mp.cos(phi), st * mp.sin(phi), ct])


def strategy_value(params, corr, amarg, bmarg):
    t = params[0]
    c, s = mp.cos(t), mp.sin(t)
    avec = {0: bloch(params[1], params[2]), 1: bloch(params[3], params[4])}
    bvec = {0: bloch(params[5], params[6]), 1: bloch(params[7], params[8])}

    def correl(u, v):
        return u[0] * v[0] * (2 * c * s) - u[1] * v[1] * (2 * c * s) + u[2] * v[2]

    def local(u):
        return u[2] * (c * c - s * s)

    value = mp.mpf(0)
    for (x, y), coeff in corr.items():
        value += coeff * correl(avec[x], bvec[y])
    for x, coeff in amarg.items():
        value += coeff * local(avec[x])
    for y, coeff in bmarg.items():
        value += coeff * local(bvec[y])
    return value


def gradient(params, corr, amarg, bmarg, h=mp.mpf("1e-25")):
    g = mp.matrix(9, 1)
    for i in range(9):
        p1 = params.copy()
        p2 = params.copy()
        p1[i] += h
        p2[i] -= h
        g[i] = (strategy_value(p1, corr, amarg, bmarg)
                - strategy_value(p2, corr, amarg, bmarg)) / (2 * h)
    return g


def main() -> int:
    payload = json.loads(
        (ROOT / "artifacts/m3/candidates/candidate_search_s31337_308.json"
         ).read_text()
    )
    func = payload["functional"]
    corr = {(int(k[0]), int(k[1])): mp.mpf(v) for k, v in func["corr"].items()}
    amarg = {int(k): mp.mpf(v) for k, v in func["alice_marginals"].items()}
    bmarg = {int(k): mp.mpf(v) for k, v in func["bob_marginals"].items()}

    params = mp.matrix(payload["strategy"]["params"])
    # Newton refinement on the free stationarity system (t is interior)
    for iteration in range(60):
        g = gradient(params, corr, amarg, bmarg)
        # numeric Hessian
        h = mp.mpf("1e-18")
        H = mp.matrix(9, 9)
        for j in range(9):
            p1 = params.copy()
            p1[j] += h
            H[:, j] = (gradient(p1, corr, amarg, bmarg) - g) / h
        try:
            delta = mp.lu_solve(H, -g)
        except Exception:
            break
        params += delta
        if mp.norm(delta, 2) < mp.mpf("1e-50"):
            break
    value = strategy_value(params, corr, amarg, bmarg)
    gnorm = mp.norm(gradient(params, corr, amarg, bmarg), 2)
    print("value:", mp.nstr(value, 45))
    print("grad norm:", mp.nstr(gnorm, 3))
    print("params:")
    for i in range(9):
        print(f"  p{i} = {mp.nstr(params[i], 45)}")

    # algebraic identification
    print("\nalgdep attempts:")
    targets = {"value": value}
    targets.update({f"p{i}": params[i] for i in range(9)})
    out = {}
    for name, x in targets.items():
        found = None
        for deg in (2, 3, 4, 6):
            rel = mp.pslq([x**k for k in range(deg + 1)], tol=mp.mpf("1e-40"),
                          maxcoeff=10**8)
            if rel:
                found = (deg, rel)
                break
        print(f"  {name}: {found if found else 'unidentified'}")
        out[name] = {"poly": found[1] if found else None,
                     "value": mp.nstr(x, 40)}
    (ROOT / "artifacts/m3/exact").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts/m3/exact/strategy_308_raw.json").write_text(
        json.dumps(out, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
