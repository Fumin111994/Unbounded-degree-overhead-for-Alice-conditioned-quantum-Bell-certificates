from math import isclose
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npa2.bell import b3_game, chsh, tilted_chsh
from npa2.onesided import build_onesided_problem
from npa2.solve import solve_problem
from npa2.standard import build_standard_problem


class ControlTests(unittest.TestCase):
    def test_control_definitions(self) -> None:
        self.assertTrue(isclose(chsh().analytic_value, 2.0**1.5))
        self.assertTrue(isclose(tilted_chsh(1.0).analytic_value, 10.0**0.5))
        game = b3_game()
        self.assertTrue(isclose(game.analytic_value, 5.0 / 6.0))
        for x in range(2):
            for y in range(2):
                winning_pairs = sum(
                    game.joint[(x, y, a, b)] > 0
                    for a in range(3)
                    for b in range(3)
                )
                self.assertEqual(winning_pairs, 3)
        self.assertEqual(chsh().metadata["standard_control_level"], 1)
        self.assertEqual(tilted_chsh(1.0).metadata["standard_control_level"], 2)
        self.assertEqual(tilted_chsh(1.5).metadata["one_sided_control_level"], 3)
        self.assertEqual(game.metadata["one_sided_control_level"], 2)

    def test_chsh_level_one_standard_and_one_sided(self) -> None:
        functional = chsh()
        standard, _ = solve_problem(build_standard_problem(functional, 1), "CLARABEL")
        one_sided, _ = solve_problem(build_onesided_problem(functional, 1), "CLARABEL")
        self.assertLess(abs(standard["analytic_error"]), 1e-7)
        self.assertLess(abs(one_sided["analytic_error"]), 1e-7)
        self.assertLess(abs(standard["value"] - one_sided["value"]), 1e-7)

    def test_tilted_level_two_standard_and_one_sided(self) -> None:
        functional = tilted_chsh(0.5)
        standard, _ = solve_problem(build_standard_problem(functional, 2), "CLARABEL")
        one_sided, _ = solve_problem(build_onesided_problem(functional, 2), "CLARABEL")
        self.assertLess(abs(standard["analytic_error"]), 5e-7)
        self.assertLess(abs(one_sided["analytic_error"]), 5e-7)
        self.assertLess(abs(standard["value"] - one_sided["value"]), 5e-7)


if __name__ == "__main__":
    unittest.main()
