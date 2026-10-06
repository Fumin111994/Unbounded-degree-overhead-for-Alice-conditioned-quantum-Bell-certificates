#!/usr/bin/env python3
"""Solver-free re-verification of every stored exact certificate.

This command re-verifies stored certificates and closed-form families:
it rebuilds each affine identity and checks it
coefficient-wise, tests every Gram block PSD by exact rational LDL*,
checks consistency and normalization, and re-does every certified
comparison (strict inequalities via exact squaring, CRootOf comparisons,
root isolation from stored polynomials).  ``cvxpy.Problem.solve`` is
patched to raise for the whole run, so any hidden SDP call fails loudly;
a missing artifact or missing certificate payload is a FAIL, never a skip.

Usage:
    python scripts/verify_certificates.py              # all checks
    python scripts/verify_certificates.py --skip-slow  # skip level-3 PSD

Output: artifacts/reproduction/certificate_checks.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import sympy as sp
import cvxpy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from npa2 import exact  # noqa: E402
from run_m3_exactify import verify_artifact, rebuild_functional  # noqa: E402
from run_m6_closure import exact_tilted_functional  # noqa: E402
from run_m65_tangent_fan import witness_gh, interval_above_q  # noqa: E402
from run_robust_family import compute_robustness  # noqa: E402
from quantum_face import verify_all as check_quantum_face  # noqa: E402
from standard_tilted_sos import verify_stored as check_standard_tilted_sos  # noqa: E402
from quantum_interval import verify_all as check_quantum_interval  # noqa: E402
from randomness_cost import verify_all as check_randomness_cost  # noqa: E402
from higher_level_witness import verify_all as check_higher_level, verify_analytic_family, verify_fejer_family  # noqa: E402
from verify_precision_bounds import verify_precision_bounds  # noqa: E402
from verify_fixed_tilt_closure import verify_fixed_tilt_closure  # noqa: E402
from verify_matching_upper import verify_matching_upper  # noqa: E402
from verify_orientation_structure import verify as check_orientation_structure  # noqa: E402
from verify_orientation_counterexample import verify as check_orientation_counterexample  # noqa: E402
from verify_asymmetric_transition import verify as check_asymmetric_transition  # noqa: E402
from verify_asymmetric_family import verify as check_asymmetric_family  # noqa: E402

T = sp.Symbol("t")
SSTAR = sp.Rational(1, 50_000_000)
GAP1E6 = sp.Rational(1, 10**6)


def read(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def verify_dual(functional, payload) -> dict:
    names = list(payload["certificate"].keys())
    values = [Fraction(payload["certificate"][n]) for n in names]
    blocks, lam, nu, mu = exact.unpack_onesided_unknowns(names, values)
    return exact.verify_onesided_dual(
        functional, payload["level"], Fraction(payload["bound"]),
        blocks, lam, nu, mu,
        enforce_probability_positivity=payload.get(
            "enforce_probability_positivity", False),
    )


def verify_witness(functional, level, payload) -> dict:
    entries: dict = {}
    for key, value in payload["block_entries"].items():
        x, a, i, j = map(int, key.split(":"))
        entries.setdefault((x, a), {})[(i, j)] = Fraction(value)
    return exact.verify_onesided_witness(functional, level, entries)


def rat(expr) -> sp.Rational:
    return sp.Rational(expr)


def q_sq(alpha: Fraction) -> sp.Rational:
    a = sp.Rational(alpha.numerator, alpha.denominator)
    return sp.Rational(8) + 2 * a**2


def q_root_of(alpha: Fraction):
    return sp.CRootOf(T**2 - q_sq(alpha), 1)


def frac(text: str) -> Fraction:
    num, _, den = str(text).partition("/")
    return Fraction(int(num), int(den or "1"))


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def check_m3_exact(tag: str) -> dict:
    payload = read(f"artifacts/m3/exact/{tag}.json")
    return verify_artifact(payload)


def check_m3_o3() -> dict:
    lead = read("artifacts/m3/exact/search_s31337_308.json")
    payload = read("artifacts/m3/exact/o3_308.json")
    return verify_dual(rebuild_functional(lead), payload)


def check_m3_robust() -> dict:
    source = read("artifacts/m3/exact/search_s31337_308.json")
    computed = compute_robustness(source, Fraction(1, 400))
    failures = list(computed["failures"])
    if not computed["value_identity_ok"]:
        failures.append("value identity recomputation failed")
    # the stored artifact must agree with the recomputation
    robust = read("artifacts/m3/robust_family.json")
    if robust.get("value_identity_check") != "PASS":
        failures.append("stored robust_family.json not PASS")
    if frac(robust["witness_value_exact"]) != computed["w"]:
        failures.append("stored w disagrees with recomputation")
    if frac(robust["margin_exact"]) != computed["margin"]:
        failures.append("stored margin disagrees with recomputation")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_attainment_matrices() -> dict:
    """Exact checks of the Lemma 2.1 (attainment at level one) matrices."""
    failures = []
    gamma0 = sp.Matrix(
        [[4, 2, 2, 2, 2], [2, 2, 1, 1, 1], [2, 1, 2, 1, 1],
         [2, 1, 1, 2, 1], [2, 1, 1, 1, 2]]
    ) / 4
    schur = gamma0[1:, 1:] - gamma0[1:, 0] * gamma0[0, 1:] / gamma0[0, 0]
    if schur != sp.eye(4) / 4:
        failures.append("standard Gamma0 Schur complement != I4/4")
    if gamma0.det() != sp.Rational(1, 256):
        failures.append("standard Gamma0 det != 1/256")
    if not exact.is_psd_exact(gamma0):
        failures.append("standard Gamma0 not strictly PSD")
    phi = sp.Matrix([[4, 2, 2], [2, 2, 1], [2, 1, 2]]) / 8
    schur2 = phi[1:, 1:] - phi[1:, 0] * phi[0, 1:] / phi[0, 0]
    if schur2 != sp.eye(2) / 8:
        failures.append("one-sided Phi Schur complement != I2/8")
    if phi.det() != sp.Rational(1, 128):
        failures.append("one-sided Phi det != 1/128")
    if not exact.is_psd_exact(phi):
        failures.append("one-sided Phi not strictly PSD")
    if 2 * phi[0, 0] != 1:
        failures.append("one-sided outcome-sum normalization fails")
    expansion = sum(
        (sp.Matrix([1, e0, e1]) * sp.Matrix([1, e0, e1]).T
         for e0 in (0, 1) for e1 in (0, 1)),
        sp.zeros(3),
    )
    if phi != expansion / 8:
        failures.append("one-sided Phi != (1/8) sum_e u_e u_e^T")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def _require(condition, message, failures):
    if not condition:
        failures.append(message)


def check_m6_direct_pins() -> dict:
    failures = []
    expected = {
        "5/4": Fraction(416927004379, 125000000000),
        "41/32": Fraction(3359047968197, 1000000000000),
    }
    for tag, expected_bound in expected.items():
        path = f"artifacts/m6/direct_pin_{tag.replace('/', '_')}.json"
        try:
            payload = read(path)
        except FileNotFoundError:
            failures.append(f"missing artifact: {path}")
            continue
        alpha = Fraction(payload["alpha"])
        if payload.get("level") != 2:
            failures.append(f"{tag}: level != 2")
        if Fraction(payload["bound"]) != expected_bound:
            failures.append(f"{tag}: bound changed")
        if "certificate" not in payload or len(payload["certificate"]) == 0:
            failures.append(f"{tag}: missing certificate payload")
            continue
        report = verify_dual(exact_tilted_functional(alpha), payload)
        failures += [f"{tag}: {f}" for f in report["failures"]]
        t = sp.Rational(payload["bound"])
        a = sp.Rational(alpha.numerator, alpha.denominator)
        q2 = 8 + 2 * a**2
        if not sp.simplify(t**2 - q2) > 0:
            failures.append(f"{tag}: bound not above q (exact)")
        rest = t - SSTAR
        if not (rest > 0 and sp.simplify(rest**2 - q2) < 0):
            failures.append(f"{tag}: bound not below q + 2e-8 (exact)")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


EXPECTED_ANCHORS = ["1/16", "1/8", "1/4", "3/8", "1/2", "3/4", "1",
                    "5/4", "3/2", "7/4"]
EXPECTED_FAN_ALPHAS = ["13/10", "21/16", "11/8", "3/2", "13/8", "7/4",
                       "15/8", "19/10", "197/100", "199/100"]
EXPECTED_PIN_ALPHAS = ["5/4", "41/32"]
EXPECTED_O3_ALPHAS = ["13/10", "3/2"]
EXPECTED_NONCLOSURE_ALPHAS = ["13/10", "3/2"]


def check_m26_anchors() -> dict:
    failures = []
    paths = sorted((ROOT / "artifacts/m26/anchors").glob("anchor_*.json"))
    alphas = [p.stem.replace("anchor_", "").replace("_", "/") for p in paths]
    if sorted((Fraction(a) for a in alphas)) != [Fraction(a) for a in EXPECTED_ANCHORS]:
        failures.append(f"anchor set mismatch: {sorted(alphas, key=Fraction)}")
    if len(set(alphas)) != len(alphas):
        failures.append("duplicate anchor files")
    for path in paths:
        payload = json.loads(path.read_text())
        num, den = path.stem.replace("anchor_", "").split("_")
        alpha = Fraction(int(num), int(den))
        report = verify_dual(exact_tilted_functional(alpha), payload)
        failures += [f"{alpha}: {f}" for f in report["failures"]]
        bound = sp.Rational(payload["bound"])
        a = sp.Rational(alpha.numerator, alpha.denominator)
        # bound < 2 sqrt(2) + alpha  <=>  (bound - alpha)^2 < 8, bound > alpha
        if not (bound > a and sp.simplify((bound - a) ** 2 - 8) < 0):
            failures.append(f"{alpha}: bound not strictly below 2 sqrt(2) + alpha")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m26_endpoint() -> dict:
    from run_level1_anchors import onesided_chsh_endpoint_certificate
    report = onesided_chsh_endpoint_certificate()
    failures = [] if report["status"] == "PASS" and report.get("nice") else ["endpoint certificate failed"]
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m65_sandwich() -> dict:
    payload = read("artifacts/m65/sandwich.json")
    alpha = Fraction(payload["alpha"])
    report = verify_witness(exact_tilted_functional(alpha), 1, payload["witness"])
    failures = list(report["failures"])
    objective = sp.Rational(report["objective_exact"])
    a = sp.Rational(alpha.numerator, alpha.denominator)
    # objective > q(alpha)  <=>  objective > 0 and objective^2 > 8 + 2 alpha^2
    if not (objective > 0 and sp.simplify(objective**2 - (8 + 2 * a**2)) > 0):
        failures.append("sandwich witness objective not above almost-quantum value")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m6_nonclosure() -> dict:
    summary = read("artifacts/m6/summary.json")
    failures = []
    items = summary["results"]["nonclosure"]
    alphas = [item["alpha"] for item in items]
    if sorted((Fraction(a) for a in alphas)) != [Fraction(a) for a in EXPECTED_NONCLOSURE_ALPHAS]:
        failures.append(f"nonclosure alpha set mismatch: {alphas}")
    if not items:
        failures.append("empty nonclosure witness list")
    for item in items:
        alpha = Fraction(item["alpha"])
        report = verify_witness(exact_tilted_functional(alpha), 2, item)
        failures += [f"{alpha}: {f}" for f in report["failures"]]
        objective = sp.Rational(report["objective_exact"])
        a = sp.Rational(alpha.numerator, alpha.denominator)
        if not (objective > 0 and sp.simplify(objective**2 - (8 + 2 * a**2)) > 0):
            failures.append(f"{alpha}: witness objective not above q (exact)")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m6_epsilon_closure() -> dict:
    payload = read("artifacts/m6/epsilon_closure.json")
    failures = []
    results = payload["results"]
    alphas = [item["alpha"] for item in results]
    if sorted((Fraction(a) for a in alphas)) != [Fraction(a) for a in EXPECTED_PIN_ALPHAS]:
        failures.append(f"epsilon_closure alpha set mismatch: {alphas}")
    if not results:
        failures.append("empty results list")
    for item in results:
        if "certificate" not in item:
            failures.append(f"{item['alpha']}: missing certificate payload")
            continue
        alpha = Fraction(item["alpha"])
        dual_payload = {
            "certificate": item["certificate"],
            "level": 2,
            "bound": item["bound_rational"],
        }
        report = verify_dual(exact_tilted_functional(alpha), dual_payload)
        failures += [f"{alpha}: {f}" for f in report["failures"]]
        bound = sp.Rational(item["bound_rational"])
        a = sp.Rational(alpha.numerator, alpha.denominator)
        # bound - q < 2 eps  <=>  bound - 2 eps <= 0 or (bound - 2 eps)^2 < q^2
        rest = bound - sp.Rational(2, 10**6)
        if not (rest <= 0 or sp.simplify(rest**2 - (8 + 2 * a**2)) < 0):
            failures.append(f"{alpha}: bound not within q + 2 eps (exact)")
        # and bound above q: bound^2 > q^2
        if not sp.simplify(bound**2 - (8 + 2 * a**2)) > 0:
            failures.append(f"{alpha}: bound not above q (exact)")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m6_o3() -> dict:
    payload = read("artifacts/m6/o3_certificates.json")
    failures = []
    results = payload["results"]
    alphas = [item["alpha"] for item in results]
    if sorted((Fraction(a) for a in alphas)) != [Fraction(a) for a in EXPECTED_O3_ALPHAS]:
        failures.append(f"o3_certificates alpha set mismatch: {alphas}")
    if not results:
        failures.append("empty results list")
    for item in results:
        if "certificate" not in item:
            failures.append(f"{item['alpha']}: missing certificate payload")
            continue
        alpha = Fraction(item["alpha"])
        dual_payload = {
            "certificate": item["certificate"],
            "level": 3,
            "bound": item["bound"],
        }
        report = verify_dual(exact_tilted_functional(alpha), dual_payload)
        failures += [f"{alpha}: {f}" for f in report["failures"]]
        bound = sp.Rational(item["bound"])
        a = sp.Rational(alpha.numerator, alpha.denominator)
        rest = bound - GAP1E6
        if not (rest <= 0 or sp.simplify(rest**2 - (8 + 2 * a**2)) < 0):
            failures.append(f"{alpha}: bound not within q + 1e-6 (exact)")
        if not sp.simplify(bound**2 - (8 + 2 * a**2)) > 0:
            failures.append(f"{alpha}: bound not above q (exact)")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m65_tangent_fan() -> dict:
    payload = read("artifacts/m65/tangent_fan.json")
    failures = []
    fan = payload["fan"]
    fan_alphas = [item["alpha"] for item in fan]
    if sorted((Fraction(a) for a in fan_alphas)) != [Fraction(a) for a in EXPECTED_FAN_ALPHAS]:
        failures.append(f"fan alpha set mismatch: {fan_alphas}")
    if not fan:
        failures.append("empty fan")
    merged: list = []
    for item in fan:
        alpha = Fraction(item["alpha"])
        if "block_entries" not in item or not item["block_entries"]:
            failures.append(f"{alpha}: missing block_entries")
            continue
        report = verify_witness(exact_tilted_functional(alpha), 2, item)
        failures += [f"{alpha}: {f}" for f in report["failures"]]
        g, h, _ = witness_gh(item)
        if (f"{g.numerator}/{g.denominator}" != item["g"]
                or f"{h.numerator}/{h.denominator}" != item["h"]):
            failures.append(f"{alpha}: recomputed (g,h) mismatch")
        intervals = interval_above_q(g, h)
        for lo, hi in intervals:
            merged.append((lo, hi))
        # openness: the tangent polynomial vanishes at each interval endpoint
        gg, hh = sp.Rational(g.numerator, g.denominator), sp.Rational(h.numerator, h.denominator)
        quad = sp.expand((gg + hh * T) ** 2 - (8 + 2 * T**2))
        for lo, hi in intervals:
            for endpoint in (lo, hi):
                if endpoint not in (sp.Rational(0), sp.Rational(2)):
                    if sp.simplify(quad.subs(T, endpoint)) != 0:
                        failures.append(f"{alpha}: endpoint not a root (not open)")
    # merged coverage must match the stored covered interval
    stored = payload["covered_interval_float"]
    all_roots_sorted: list = []
    if stored is not None:
        def _leq(a, b) -> bool:
            return sp.sign(sp.simplify(a - b)) <= 0
        merged.sort(key=lambda iv: float(sp.N(iv[0])))
        clusters: list = []
        for lo, hi in merged:
            if clusters and _leq(lo, clusters[-1][1]):
                if _leq(clusters[-1][1], hi):
                    clusters[-1] = (clusters[-1][0], hi)
            else:
                clusters.append((lo, hi))
        top = clusters[-1]
        cov_lo, cov_hi = float(sp.N(top[0], 15)), float(sp.N(top[1], 15))
        if abs(cov_lo - stored[0]) > 1e-9 or abs(cov_hi - stored[1]) > 1e-9:
            failures.append("merged coverage does not match stored interval")
        if "OPEN" not in payload.get("statement", ""):
            failures.append("statement does not record the interval as open")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def check_m6_rootlocus() -> dict:
    payload = read("artifacts/m6/rootlocus.json")
    failures = []
    results = payload["results"]
    alphas = [r["alpha"] for r in results]
    if sorted((Fraction(a) for a in alphas)) != [Fraction(a) for a in EXPECTED_PIN_ALPHAS]:
        failures.append(f"rootlocus alpha set mismatch: {alphas}")
    if not results:
        failures.append("empty results list")
    expected_blocks = {"0:0", "0:1", "1:0", "1:1"}
    for result in results:
        alpha = Fraction(result["alpha"])
        q_root = q_root_of(alpha)
        maxima = []
        block_keys = set(result["blocks"].keys())
        if block_keys != expected_blocks:
            failures.append(f"{alpha}: block set mismatch: {sorted(block_keys)}")
        for block_key, block in result["blocks"].items():
            if "det_coefficients_ascending" not in block:
                failures.append(f"{alpha} {block_key}: missing det polynomial")
                continue
            coeffs = [sp.Rational(c) for c in block["det_coefficients_ascending"]]
            poly = sp.Poly(sum(c * T**k for k, c in enumerate(coeffs)), T)
            roots = [r for r in poly.all_roots(radicals=False) if r.is_real]
            if not roots:
                failures.append(f"{alpha} {block_key}: no real roots")
                continue
            largest = max(roots)
            maxima.append(largest)
            if not largest < q_root + SSTAR:
                failures.append(f"{alpha} {block_key}: root not below q + 2e-8")
            stored = block.get("largest_root") or {}
            if stored and abs(float(sp.N(largest, 20)) - float(stored["float30"])) > 1e-12:
                failures.append(f"{alpha} {block_key}: root mismatch vs stored")
        if maxima:
            top = max(maxima)
            stored_top = result.get("max_root_over_blocks", {})
            if bool(top > q_root) != stored_top.get("above_q"):
                failures.append(f"{alpha}: above_q mismatch")
            if bool(top < q_root + SSTAR) != stored_top.get("below_q_plus_sstar"):
                failures.append(f"{alpha}: below_q_plus_sstar mismatch")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


# ---------------------------------------------------------------------------


CHECKS = [
    ("asymmetric_sharp_weight_transition", False, check_asymmetric_transition),
    ("asymmetric_uniform_closure_audits", False, check_asymmetric_family),
    ("orientation_structure_audits", False, check_orientation_structure),
    ("bob_marginal_exact_counterexample", False, check_orientation_counterexample),
    ("matching_upper_bound_audits", False, verify_matching_upper),
    ("fixed_tilt_exact_closure_audits", False, verify_fixed_tilt_closure),
    ("finite_precision_rounding_audits", False, verify_precision_bounds),
    ("square_root_degree_fejer_family", False, verify_fejer_family),
    ("unbounded_conversion_analytic_family", False, verify_analytic_family),
    ("higher_level_rational_counterexamples", False, check_higher_level),
    ("randomness_cost_exact_witness", False, check_randomness_cost),
    ("quantum_interval_exact_closure", True, check_quantum_interval),
    ("standard_tilted_symbolic_sos", False, check_standard_tilted_sos),
    ("quantum_face_exact_closure", True, check_quantum_face),
    ("attainment_matrices", False, check_attainment_matrices),
    ("m3_exact_search_s31337_308", False, lambda: check_m3_exact("search_s31337_308")),
    ("m3_exact_search_s777_298", False, lambda: check_m3_exact("search_s777_298")),
    ("m3_o3_308", True, check_m3_o3),
    ("m3_robust_family", False, check_m3_robust),
    ("m26_anchors", False, check_m26_anchors),
    ("m26_chsh_endpoint", False, check_m26_endpoint),
    ("m65_sandwich", False, check_m65_sandwich),
    ("m6_nonclosure_witnesses", False, check_m6_nonclosure),
    ("m6_direct_pins", False, check_m6_direct_pins),
    ("m6_epsilon_closure", False, check_m6_epsilon_closure),
    ("m6_o3_certificates", True, check_m6_o3),
    ("m65_tangent_fan", True, check_m65_tangent_fan),
    ("m6_rootlocus", False, check_m6_rootlocus),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip checks marked slow (level-3 PSD, fan)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "reproduction")
    args = parser.parse_args(argv)

    records = []
    failures: list[str] = []
    start = time.time()
    with patch.object(cp.Problem, "solve",
                      side_effect=RuntimeError("SDP calls disabled")):
        for name, slow, fn in CHECKS:
            if slow and args.skip_slow:
                records.append({"name": name, "status": "SKIPPED"})
                continue
            t0 = time.time()
            try:
                report = fn()
            except FileNotFoundError as exc:
                report = {"status": "FAIL",
                          "failures": [f"missing artifact: {exc}"]}
            except Exception as exc:  # noqa: BLE001 - report, don't crash
                report = {"status": "FAIL",
                          "failures": [f"exception: {type(exc).__name__}: {exc}"]}
            status = report.get("status", "FAIL")
            records.append({
                "name": name,
                "status": status,
                "seconds": round(time.time() - t0, 1),
                "failures": report.get("failures", [])[:8],
            })
            print(f"{name:32s} {status}  ({records[-1]['seconds']}s)", flush=True)
            if status != "PASS":
                failures.append(name)

    out = {
        "schema_version": 1,
        "method": (
            "solver-free re-verification from stored artifacts: affine "
            "identities coefficient-wise, exact rational LDL* PSD tests, "
            "consistency/normalization, exact comparisons (squaring, "
            "CRootOf, root isolation); cvxpy.Problem.solve patched to raise"
        ),
        "skip_slow": args.skip_slow,
        "seconds": round(time.time() - start, 1),
        "records": records,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "certificate_checks.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": out["status"], "failures": failures,
                      "seconds": out["seconds"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
