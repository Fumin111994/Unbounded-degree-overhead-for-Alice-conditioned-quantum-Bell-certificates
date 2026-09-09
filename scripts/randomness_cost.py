"""Solver-free exact witness for a loss in CHSH randomness certification.

The continuum theorem is proved in the manuscript by concave duality.
This file independently certifies a quantitative example at CHSH = 23/10.
It checks actual moment blocks, not just stored scalar bounds.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import math
import sys

import cvxpy as cp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from npa2 import exact
from run_m6_closure import exact_tilted_functional
from run_m65_tangent_fan import witness_gh

ARTIFACT = ROOT / "artifacts/prl/randomness/cost.json"
FAN = ROOT / "artifacts/m65/tangent_fan.json"


def verify_blocks(entries):
    _, _, blocks = witness_gh({"block_entries": entries})
    return exact.verify_onesided_witness(exact_tilted_functional(Fraction(0)), 2, blocks)


@lru_cache(maxsize=16)
def quantum_blocks(u):
    """Construct physical two-qubit moments, then reduce them to rationals."""
    u = sp.Rational(u)
    alpha = -2 * (u*u - 8*u + 8) / (u*u - 8)
    q = 4 * (u*u - 4*u + 8) / (u*u - 8)
    if not 0 < alpha < 2 or q <= 0:
        raise ValueError("Parameter is outside the physical tilted-CHSH branch")
    c = q / 4
    d = sp.sqrt(1-c*c)
    z, t = alpha / (2*c), d / c
    if sp.simplify(z*z + t*t - 1) != 0:
        raise ValueError("State is not pure")
    projectors = [sp.Matrix([[1+c, d], [d, 1-c]]) / 2,
                  sp.Matrix([[1+c, -d], [-d, 1-c]]) / 2]
    rho = [sp.diag((1+z)/2, 0), sp.diag(0, (1-z)/2),
           sp.Matrix([[1+z, t], [t, 1-z]])/4,
           sp.Matrix([[1+z, -t], [-t, 1-z]])/4]
    words = exact_tilted_functional(Fraction(0)).scenario.algebra().words(2, party="B")
    operators = []
    for word in words:
        op = sp.eye(2)
        for _, question, _ in word:
            op = op * projectors[question]
        operators.append(op.applyfunc(sp.simplify))
    entries = {}
    for b, state in enumerate(rho):
        for i, left in enumerate(operators):
            for j in range(i, len(operators)):
                v = sp.simplify(sp.trace(state * left.T * operators[j]))
                entries[f"{b//2}:{b%2}:{i}:{j}"] = str(sp.Rational(v))
    return {"alpha": alpha, "q": q, "block_entries": entries}


def bob_flip_matrix(level):
    algebra = exact_tilted_functional(Fraction(0)).scenario.algebra()
    words = algebra.words(level, party="B")
    T = sp.zeros(len(words))
    for i, word in enumerate(words):
        poly = {(): Fraction(1)}
        for _, question, _ in word:
            poly = algebra.multiply_polynomials(poly, algebra.projector("B", question, 1))
        for w, coefficient in poly.items():
            T[i, words.index(w)] = sp.Rational(coefficient)
    if T*T != sp.eye(len(words)):
        raise ValueError("Outcome relabeling does not square to the identity")
    return T


def flipped_blocks(entries):
    T = bob_flip_matrix(2)
    result = {}
    for x in (0, 1):
        for a in (0, 1):
            M = sp.Matrix(5, 5, lambda i, j: sp.Rational(entries[f"{x}:{1-a}:{min(i,j)}:{max(i,j)}"]))
            M = T*M*T.T
            for i in range(5):
                for j in range(i, 5):
                    result[f"{x}:{a}:{i}:{j}"] = str(M[i, j])
    return result


def build_certificate():
    witness = next(w for w in json.loads(FAN.read_text())["fan"] if w["alpha"] == "11/8")
    u, observed = sp.Rational(111, 25), sp.Rational(23, 10)
    quantum = quantum_blocks(u)
    g, h, _ = witness_gh(witness)
    sq, hq, _ = witness_gh(quantum)
    weight = (sp.Rational(sq)-observed)/(sp.Rational(sq)-g)
    entries = {key: str(weight*sp.Rational(v)+(1-weight)*sp.Rational(quantum["block_entries"][key]))
               for key, v in witness["block_entries"].items()}
    _, mixed_bias, _ = witness_gh({"block_entries": entries})
    return {"schema_version": 1, "level": 2, "chsh": str(observed),
            "source_tilt": "11/8", "quantum_parameter": str(u),
            "witness_weight": str(weight), "block_entries": entries,
            "guess_probability": str((1+sp.Rational(mixed_bias))/2),
            "guess_gap_lower_bound": "3/5000", "entropy_ratio_lower_bound": "10007/10000",
            "entropy_loss_lower_bound_bits": "1/1000"}


def verify_certificate(payload):
    try:
        if payload["schema_version"] != 1 or payload["level"] != 2:
            raise ValueError("Unsupported randomness witness")
        observed = sp.Rational(payload["chsh"])
        # This artifact certifies the stated example, not an arbitrary branch.
        if observed != sp.Rational(23, 10) or payload["source_tilt"] != "11/8":
            raise ValueError("Incorrect observed CHSH value or source witness")
        u = sp.Rational(payload["quantum_parameter"])
        if u != sp.Rational(111, 25):
            raise ValueError("Incorrect quantum mixing parameter")
        witness = next(w for w in json.loads(FAN.read_text())["fan"] if w["alpha"] == payload["source_tilt"])
        quantum = quantum_blocks(u)
        for blockset in (witness["block_entries"], quantum["block_entries"], payload["block_entries"]):
            report = verify_blocks(blockset)
            if report["status"] != "PASS":
                raise ValueError("Moment feasibility fails: " + str(report))
        weight = sp.Rational(payload["witness_weight"])
        if not 0 < weight < 1:
            raise ValueError("Invalid convex mixing weight")
        for key, value in payload["block_entries"].items():
            expected = weight*sp.Rational(witness["block_entries"][key]) + (1-weight)*sp.Rational(quantum["block_entries"][key])
            if sp.Rational(value) != expected:
                raise ValueError("Stored moment does not equal the declared mixture")
        g, h, _ = witness_gh(payload)
        h = sp.Rational(h)
        if g != observed:
            raise ValueError("Mixture has the wrong CHSH expectation")
        guess = (1+h)/2
        if sp.Rational(payload["guess_probability"]) != guess:
            raise ValueError("Incorrect guessing objective")
        delta = sp.Rational(payload["guess_gap_lower_bound"])
        qbias_sq = 2-observed**2/4
        if not delta > 0 or not h-2*delta > 0 or not (h-2*delta)**2 > qbias_sq:
            raise ValueError("Claimed guessing-probability loss is not certified")
        ratio = sp.Rational(payload["entropy_ratio_lower_bound"])
        bits = sp.Rational(payload["entropy_loss_lower_bound_bits"])
        test = (1+h)/ratio-1
        if not ratio > 1 or not bits > 0 or not test > 0 or not test**2 > qbias_sq:
            raise ValueError("Entropy ratio does not follow from the witness")
        # log2(ratio) > bits, verified with integer powers, without floating logs.
        if not ratio**bits.q > sp.Integer(2)**bits.p:
            raise ValueError("Entropy lower bound fails the exact power comparison")
        flip = flipped_blocks(payload["block_entries"])
        if verify_blocks(flip)["status"] != "PASS":
            raise ValueError("Outcome relabeling breaks moment feasibility")
        gf, hf, _ = witness_gh({"block_entries": flip})
        if gf != g or hf != -h:
            raise ValueError("Outcome relabeling does not preserve CHSH and reverse bias")
        bob_flip_matrix(3)
        # The quantum optimum at this statistic lies in the proven OS3 interval.
        alpha_sq = (64/observed**2-8)/2
        if not sp.Rational(13,10)**2 < alpha_sq < sp.Rational(3,2)**2:
            raise ValueError("Observed statistic is outside the certified interval")
        exact_qguess = (1+sp.sqrt(qbias_sq))/2
        return {"status": "PASS", "failures": [], "chsh": str(observed),
                "moment_block_sets_checked": 4, "relabeling_levels_checked": [2, 3],
                "guess_probability_exact": str(guess),
                "quantum_guess_probability_exact": str(exact_qguess),
                "guess_gap_lower_bound": str(delta),
                "entropy_loss_lower_bound_bits": str(bits),
                "guess_gap_float_illustration": float(guess-exact_qguess),
                "entropy_loss_float_illustration": math.log2(float(guess/exact_qguess))}
    except (ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError) as exc:
        return {"status": "FAIL", "failures": [str(exc)]}


def verify_all():
    return verify_certificate(json.loads(ARTIFACT.read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="construct and verify the exact artifact")
    args = parser.parse_args()
    with patch.object(cp.Problem, "solve", side_effect=RuntimeError("SDP calls disabled")):
        payload = build_certificate() if args.write else json.loads(ARTIFACT.read_text())
        report = verify_certificate(payload)
        if args.write and report["status"] == "PASS":
            ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
            ARTIFACT.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
