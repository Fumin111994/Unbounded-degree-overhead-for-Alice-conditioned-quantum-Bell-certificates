#!/usr/bin/env python3
"""Uniform level-two separation on an l1 neighbourhood of G* (round-7 upgrade).

For the lead candidate G* of artifacts/m3/exact/search_s31337_308.json, with
exact one-sided level-two witness value w and eta = 501/200, this script
certifies: the exact margin w - eta exceeds 1/200, hence for r = 1/400 and
eta' = eta + r = 1003/400, EVERY coefficient vector theta in the
+/-1-observable basis (alice marginals, bob marginals, correlators) with
||theta - theta*||_1 <= r satisfies

    omega_std^2(G_theta) <= eta' < omega_os^2(G_theta),
    eta' I - B_{G_theta} in D_2 cap (O_3 setminus O_2).

The certificate is symbolic (SOS transport: I +/- X_j = (I +/- X_j)^2 / 2 for
the eight +/-1 monomials X_j, plus the G* certificates); this script verifies
the numerical core exactly over Q:

  1. the stored rationalized functional really lies in the +/-1
     parameterization (outcome-sign symmetry), so B_{G_theta} - B_{G*} =
     sum_j delta_j X_j with no constant term;
  2. the witness's eight expectation values <X_j> computed from the exact
     block entries all lie in [-1, 1] (diagonal entries nonnegative,
     normalization exact), and the value identity
     sum_j theta*_j <X_j> == w holds exactly (independent cross-check of the
     artifact);
  3. w - eta > 1/200 exactly, hence w - r > eta + r = 1003/400 exactly.

No SDP solver is called anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2.bell import BellFunctional, Scenario  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402


def decode(text: str) -> Fraction:
    num, _, den = str(text).partition("/")
    return Fraction(int(num), int(den or "1"))


def encode(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def compute_robustness(payload: dict, radius: Fraction) -> dict:
    """Pure solver-free computation of the robustness report.

    Returns theta_star, the eight exact witness expectation values, the
    value identity check, and the exact margin comparisons, plus failures.
    Used both by the artifact generator and by verify_certificates.py.
    """

    failures: list[str] = []

    rf = payload["rationalized_functional"]
    al = {tuple(int(t) for t in k.split(":")): decode(v)
          for k, v in rf["alice_local"].items()}
    bl = {tuple(int(t) for t in k.split(":")): decode(v)
          for k, v in rf["bob_local"].items()}
    jo = {tuple(int(t) for t in k.split(":")): decode(v)
          for k, v in rf["joint"].items()}

    # --- 1. +/-1 parameterization (outcome-sign symmetry) -------------------
    signs = {0: Fraction(1), 1: Fraction(-1)}
    for x in (0, 1):
        if al[(x, 1)] != -al[(x, 0)]:
            failures.append(f"alice marginal x={x} lacks outcome-sign symmetry")
    for y in (0, 1):
        if bl[(y, 1)] != -bl[(y, 0)]:
            failures.append(f"bob marginal y={y} lacks outcome-sign symmetry")
    for x in (0, 1):
        for y in (0, 1):
            c = jo[(x, y, 0, 0)]
            for a in (0, 1):
                for b in (0, 1):
                    if jo[(x, y, a, b)] != c * signs[a] * signs[b]:
                        failures.append(
                            f"correlator ({x},{y}) lacks outcome-sign symmetry"
                        )
    theta_star = {
        "a0": al[(0, 0)], "a1": al[(1, 0)],
        "b0": bl[(0, 0)], "b1": bl[(1, 0)],
        "c00": jo[(0, 0, 0, 0)], "c01": jo[(0, 1, 0, 0)],
        "c10": jo[(1, 0, 0, 0)], "c11": jo[(1, 1, 0, 0)],
    }

    # --- 2. witness expectation values from exact block entries -------------
    witness = payload["onesided_witness"]
    entries = witness["block_entries"]

    functional = BellFunctional(
        name=payload["candidate_tag"] + "_rationalized",
        scenario=Scenario(2, 2, 2, 2),
        alice_local=al, bob_local=bl, joint=jo,
    )
    model = build_onesided_problem(
        functional, 2, enforce_probability_positivity=False
    )
    algebra = functional.scenario.algebra()

    def block_value(x: int, a: int, word) -> Fraction:
        """Exact value phi_{x,a}(word) via the block representative."""
        reps = model.block_representatives[f"Phi_x{x}_a{a}"]
        i, j = reps[algebra.moment_key(word)]
        key = f"{x}:{a}:{min(i, j)}:{max(i, j)}"
        return decode(entries[key]) if key in entries else Fraction(0)

    exp: dict[str, Fraction] = {}
    for x in (0, 1):
        exp[f"A{x}"] = block_value(x, 0, ()) - block_value(x, 1, ())
    for y in (0, 1):
        word = (("B", y, 0),)
        # E_{B_y} = P_{B,y,0} - P_{B,y,1} = 2 P_{B,y,0} - I
        by_x0 = 2 * (block_value(0, 0, word) + block_value(0, 1, word)) - 1
        by_x1 = 2 * (block_value(1, 0, word) + block_value(1, 1, word)) - 1
        if by_x0 != by_x1:
            failures.append(f"bob marginal y={y} is x-dependent")
        exp[f"B{y}"] = by_x0
    for x in (0, 1):
        for y in (0, 1):
            word = (("B", y, 0),)
            # E_{xy} = E_{A_x} (2 P_{B,y,0} - I)
            exp[f"E{x}{y}"] = (
                2 * (block_value(x, 0, word) - block_value(x, 1, word))
                - exp[f"A{x}"]
            )

    for name, value in exp.items():
        if not (-1 <= value <= 1):
            failures.append(f"expectation {name} = {float(value)} outside [-1,1]")
    # normalization: sum_a phi_{x,a}(I) == 1 for each x
    for x in (0, 1):
        if block_value(x, 0, ()) + block_value(x, 1, ()) != 1:
            failures.append(f"normalization fails at x={x}")

    w = Fraction(witness["verification"]["objective_exact"])
    value_check = (
        theta_star["a0"] * exp["A0"] + theta_star["a1"] * exp["A1"]
        + theta_star["b0"] * exp["B0"] + theta_star["b1"] * exp["B1"]
        + theta_star["c00"] * exp["E00"] + theta_star["c01"] * exp["E01"]
        + theta_star["c10"] * exp["E10"] + theta_star["c11"] * exp["E11"]
    )
    if value_check != w:
        failures.append(
            "value identity mismatch: "
            f"{encode(value_check)} != {encode(w)}"
        )

    # --- 3. exact margin and radius -----------------------------------------
    eta = Fraction(payload["eta"])
    r = radius
    eta_prime = eta + r
    margin = w - eta
    if not margin > Fraction(1, 200):
        failures.append(f"margin w-eta = {float(margin)} not > 1/200")
    if not w - r > eta_prime:
        failures.append("w - r not > eta + r")

    return {
        "theta_star": theta_star,
        "exp": exp,
        "w": w,
        "eta": eta,
        "radius": r,
        "eta_prime": eta_prime,
        "margin": margin,
        "value_identity_ok": value_check == w,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/m3/exact/search_s31337_308.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/m3"))
    parser.add_argument("--radius", default="1/400")
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    computed = compute_robustness(payload, Fraction(args.radius))
    theta_star = computed["theta_star"]
    exp = computed["exp"]
    w = computed["w"]
    eta = computed["eta"]
    r = computed["radius"]
    eta_prime = computed["eta_prime"]
    margin = computed["margin"]
    failures = computed["failures"]
    value_check_ok = computed["value_identity_ok"]

    statement = (
        "uniform separation on an l1 neighbourhood: let theta* be the "
        "8-vector of +/-1-observable coefficients (alice marginals a, bob "
        "marginals b, correlators c) of G*, and let r = 1/400.  For every "
        "theta with ||theta - theta*||_1 <= r, eta' = eta + r = 1003/400 "
        "satisfies (i) eta' I - B_{G_theta} in D_2 by SOS transport of the "
        "standard level-2 certificate of G* (each perturbation term is a "
        "nonnegative multiple of (I +/- X_j)^2/2, a square of a word "
        "polynomial of length <= 2); (ii) eta' I - B_{G_theta} in O_3 by the "
        "same transport of the level-3 certificate (the squares use at most "
        "one Alice question, hence are nice); (iii) eta' I - B_{G_theta} "
        "not in O_2, since the one-sided level-2 witness of G* evaluates "
        "each X_j in [-1,1] (its behavior is nonsignaling), so its value "
        "on G_theta is >= w - ||delta||_1 >= w - r > eta'.  Equivalently "
        "omega_std^2(G_theta) <= eta' < omega_os^2(G_theta), and the "
        "certified upper-bound polynomial eta' I - B_{G_theta} has "
        "conversion degree exactly one level.  The exact enabling margin "
        "is w - eta = " + encode(margin) + " > 1/200.  This certifies the "
        "conversion degree of the explicit polynomial eta' I - B_{G_theta}; "
        "it does not claim that the perturbed games' quantum values close "
        "at level three."
    )

    report = {
        "schema_version": 1,
        "source": str(args.source),
        "theta_star": {k: encode(v) for k, v in theta_star.items()},
        "eta": encode(eta),
        "radius": encode(r),
        "eta_prime": encode(eta_prime),
        "eta_prime_float": float(eta_prime),
        "witness_value_exact": encode(w),
        "witness_value_float": float(w),
        "margin_exact": encode(margin),
        "margin_float": float(margin),
        "witness_expectations": {k: encode(v) for k, v in exp.items()},
        "witness_expectations_float": {k: float(v) for k, v in exp.items()},
        "value_identity_check": "PASS" if value_check_ok else "FAIL",
        "statement": statement,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "robust_family.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "margin_float": report["margin_float"],
                      "eta_prime": report["eta_prime"],
                      "failures": failures}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
