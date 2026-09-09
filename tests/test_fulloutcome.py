"""M2.5 tests: full-outcome implementation and exact level-one certificates."""

import json
import unittest
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npa2.bell import chsh, tilted_chsh
from npa2.fulloutcome import (
    FullPVMAlgebra,
    build_onesided_problem_full,
    build_standard_problem_full,
)
from npa2.onesided import build_onesided_problem
from npa2.solve import solve_problem
from npa2.standard import build_standard_problem

ROOT = Path(__file__).resolve().parents[1]
EXACT_DIR = ROOT / "artifacts" / "m25" / "exact"


class FullOutcomeAlgebraTests(unittest.TestCase):
    def test_all_outcomes_are_generators(self) -> None:
        algebra = FullPVMAlgebra(2, 2, 2, 2)
        self.assertEqual(len(algebra.generators()), 8)
        self.assertEqual(len(algebra.generators("B")), 4)

    def test_reduction_rules(self) -> None:
        algebra = FullPVMAlgebra(2, 2, 2, 2)
        a0 = ("A", 0, 0)
        a1 = ("A", 0, 1)
        b0 = ("B", 0, 0)
        # Idempotency, orthogonality, party commutation.
        self.assertEqual(algebra.reduce_word((a0, a0)), (a0,))
        self.assertIsNone(algebra.reduce_word((a0, a1)))
        self.assertEqual(algebra.reduce_word((b0, a0)), (a0, b0))

    def test_last_outcome_projector_is_explicit(self) -> None:
        algebra = FullPVMAlgebra(2, 2, 2, 2)
        poly = algebra.projector("B", 1, 1)
        self.assertEqual(poly, {(("B", 1, 1),): 1.0})


class FullOutcomeComparisonTests(unittest.TestCase):
    """Reduced and full-outcome moment bodies give the same optima."""

    def _check(self, functional, level, tol) -> None:
        reduced_std, _ = solve_problem(build_standard_problem(functional, level), "CLARABEL")
        full_std, _ = solve_problem(build_standard_problem_full(functional, level), "CLARABEL")
        reduced_os, _ = solve_problem(build_onesided_problem(functional, level), "CLARABEL")
        full_os, _ = solve_problem(build_onesided_problem_full(functional, level), "CLARABEL")
        self.assertLess(abs(reduced_std["value"] - full_std["value"]), tol)
        self.assertLess(abs(reduced_os["value"] - full_os["value"]), tol)

    def test_chsh_level_one(self) -> None:
        self._check(chsh(), 1, 1e-6)

    def test_tilted_level_one_gap_persists_without_quotient(self) -> None:
        functional = tilted_chsh(0.5)
        std, _ = solve_problem(build_standard_problem_full(functional, 1), "CLARABEL")
        one_sided, _ = solve_problem(build_onesided_problem_full(functional, 1), "CLARABEL")
        self.assertGreater(std["value"] - one_sided["value"], 0.1)

    def test_tilted_level_two(self) -> None:
        self._check(tilted_chsh(0.5), 2, 5e-6)


class ExactLevelOneTests(unittest.TestCase):
    """Solver-free re-verification of the exported exact certificates."""

    def _load(self, name: str) -> dict:
        path = EXACT_DIR / name
        if not path.exists():
            self.skipTest(f"{path} missing; run scripts/run_exact_level1.py first")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_standard_witness_artifact(self) -> None:
        from scripts.run_exact_level1 import verify_standard_artifact

        report = verify_standard_artifact(self._load("standard_witness.json"))
        self.assertEqual(report["status"], "PASS", report["failures"])
        self.assertEqual(Fraction(report["objective_exact"]), Fraction(31, 10))

    def test_onesided_dual_artifact(self) -> None:
        from scripts.run_exact_level1 import verify_onesided_artifact

        payload = self._load("onesided_dual.json")
        report = verify_onesided_artifact(payload)
        self.assertEqual(report["status"], "PASS", report["failures"])
        num, _, den = payload["bound"].partition("/")
        self.assertLess(Fraction(int(num), int(den)), Fraction(31, 10))

    def test_standard_witness_raw_artifact(self) -> None:
        import sympy as sp

        from scripts.run_exact_level1 import verify_standard_artifact

        report = verify_standard_artifact(self._load("standard_witness_raw.json"))
        self.assertEqual(report["status"], "PASS", report["failures"])
        objective = sp.sympify(report["objective_exact"])
        self.assertEqual(
            sp.simplify(objective - (sp.Rational(1, 2) + 2 * sp.sqrt(2))), 0
        )

    def test_onesided_dual_raw_artifact(self) -> None:
        import sympy as sp

        from scripts.run_exact_level1 import verify_onesided_artifact

        payload = self._load("onesided_dual_raw.json")
        report = verify_onesided_artifact(payload)
        self.assertEqual(report["status"], "PASS", report["failures"])
        num, _, den = payload["bound"].partition("/")
        bound = sp.Rational(int(num), int(den))
        self.assertLess(bound, sp.Rational(31, 10))
        self.assertLess(bound, sp.Rational(1, 2) + 2 * sp.sqrt(2))


if __name__ == "__main__":
    unittest.main()
