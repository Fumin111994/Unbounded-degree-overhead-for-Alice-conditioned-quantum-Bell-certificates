"""Independent moment checks and corruption tests for the randomness witness."""
import copy
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


class RandomnessCostTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("randomness_cost"),
                             "The exact randomness-cost verifier is missing")
        return importlib.import_module("randomness_cost")

    def test_quantum_moments_have_known_chsh_and_bias(self):
        m = self.module()
        data = m.quantum_blocks(sp.Rational(22, 5))
        self.assertEqual(data["alpha"], sp.Rational(98, 71))
        g, h, _ = m.witness_gh({"block_entries": data["block_entries"]})
        self.assertEqual(g, sp.Rational(142, 61))
        self.assertEqual(h, sp.Rational(49, 61))
        self.assertEqual(m.verify_blocks(data["block_entries"])["status"], "PASS")

    def test_exact_gap_with_solvers_disabled(self):
        m = self.module()
        with patch.object(cp.Problem, "solve", side_effect=RuntimeError("disabled")):
            result = m.verify_certificate(m.build_certificate())
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["guess_gap_lower_bound"], "3/5000")
        self.assertEqual(result["entropy_loss_lower_bound_bits"], "1/1000")

    def test_wrong_observed_statistic_is_rejected(self):
        m = self.module()
        data = m.build_certificate()
        data["chsh"] = "231/100"
        self.assertEqual(m.verify_certificate(data)["status"], "FAIL")

    def test_moment_corruption_is_rejected(self):
        m = self.module()
        data = copy.deepcopy(m.build_certificate())
        data["block_entries"]["0:0:0:1"] = "0"
        self.assertEqual(m.verify_certificate(data)["status"], "FAIL")

    def test_inflated_gap_is_rejected(self):
        m = self.module()
        data = m.build_certificate()
        data["guess_gap_lower_bound"] = "1/100"
        self.assertEqual(m.verify_certificate(data)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
