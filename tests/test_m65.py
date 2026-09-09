"""M6.5 tests: mechanism criterion data, tangent fan, and M3 robustness."""

import json
import unittest
from fractions import Fraction
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
M65 = ROOT / "artifacts" / "m65"
M3EXACT = ROOT / "artifacts" / "m3" / "exact"


class TangentFanTests(unittest.TestCase):
    def test_tangent_fan_covers_interval(self) -> None:
        path = M65 / "tangent_fan.json"
        if not path.exists():
            self.skipTest("run scripts/run_m65_tangent_fan.py first")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS", report["failures"])
        lo, hi = report["covered_interval_float"]
        self.assertGreater(lo, 1.2842)   # above the critical tilt
        self.assertLess(lo, 1.35)
        self.assertGreater(hi, 1.99)
        self.assertLessEqual(hi, 2.0)
        for item in report["fan"]:
            self.assertTrue(item["intervals_above_q"])


class M3RobustnessTests(unittest.TestCase):
    def test_second_candidate_replication(self) -> None:
        path = M3EXACT / "search_s777_298.json"
        if not path.exists():
            self.skipTest("run run_m3_exactify.py for search_s777 #298 first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS", payload["failures"])

    def test_invariants_novelty(self) -> None:
        path = M65 / "invariants.json"
        if not path.exists():
            self.skipTest("run scripts/run_m65_invariants.py first")
        records = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(records)
        for record in records:
            self.assertFalse(record["chsh_type_correlator"])
            self.assertGreaterEqual(record["nonzero_marginals"], 2)
            self.assertTrue(record["novel"])

    def test_transport_control(self) -> None:
        path = M65 / "transport_control.json"
        if not path.exists():
            self.skipTest("run scripts/run_m65_transport.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        bound = Fraction(payload["bound"])
        self.assertLess(abs(bound - Fraction(payload["std2"]).limit_denominator(10**9)),
                        Fraction(2, 10**6))

    def test_sandwich_strictness_witness(self) -> None:
        path = M65 / "sandwich.json"
        if not path.exists():
            self.skipTest("run scripts/run_m65_sandwich.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS", payload["failures"])
        self.assertTrue(payload["objective_above_almost_quantum"])
        self.assertLess(2.850438562747845, payload["objective_float"])


if __name__ == "__main__":
    unittest.main()
