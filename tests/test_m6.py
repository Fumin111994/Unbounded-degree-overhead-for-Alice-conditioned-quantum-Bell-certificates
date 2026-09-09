"""M6 tests: level-two threshold artifacts (closure pins + non-closure witnesses)."""

import json
import unittest
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

M6 = ROOT / "artifacts" / "m6"


class M6ArtifactTests(unittest.TestCase):
    def test_epsilon_closure(self) -> None:
        path = M6 / "epsilon_closure.json"
        if not path.exists():
            self.skipTest("run scripts/run_m6_epsilon_closure.py first")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS", report["failures"])
        for record in report["results"]:
            self.assertTrue(record["bound_within_q_plus_2eps"])

    def test_nonclosure_witness_reverify(self) -> None:
        path = M6 / "summary.json"
        if not path.exists():
            self.skipTest("run scripts/run_m6_closure.py first")
        summary = json.loads(path.read_text(encoding="utf-8"))
        from npa2 import exact
        from run_m6_closure import exact_tilted_functional, quantum_alg

        for record in summary["results"]["nonclosure"]:
            alpha = Fraction(record["alpha"])
            functional = exact_tilted_functional(alpha)
            block_entries = {}
            for key, value in record["block_entries"].items():
                x, a, i, j = (int(p) for p in key.split(":"))
                num, _, den = value.partition("/")
                block_entries.setdefault((x, a), {})[(i, j)] = Fraction(
                    int(num), int(den or "1")
                )
            report = exact.verify_onesided_witness(functional, 2, block_entries)
            self.assertEqual(report["status"], "PASS", report["failures"][:3])
            objective = Fraction(report["objective_exact"])
            q = quantum_alg(alpha)
            import sympy as sp

            diff = sp.Rational(objective.numerator, objective.denominator) ** 2 - q**2
            self.assertTrue(sp.simplify(diff) > 0)

    def test_rootlocus_slack_small(self) -> None:
        path = M6 / "rootlocus.json"
        if not path.exists():
            self.skipTest("run scripts/run_m6_rootlocus.py first")
        report = json.loads(path.read_text(encoding="utf-8"))
        for alpha_report in report["results"]:
            alpha = Fraction(alpha_report["alpha"])
            q_float = float(8 + 2 * alpha**2) ** 0.5
            for block, info in alpha_report["blocks"].items():
                root = info.get("largest_root")
                self.assertIsNotNone(root)
                # certified exact comparison: largest det root < q + 2e-8
                self.assertTrue(root["below_q_plus_sstar"])
                slack = float(root["float30"]) - q_float
                self.assertLess(abs(slack), 1e-6)


if __name__ == "__main__":
    unittest.main()
