import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npa2.algebra import PVMAlgebra


class AlgebraTests(unittest.TestCase):
    def test_cross_party_commutation_and_idempotence(self) -> None:
        algebra = PVMAlgebra(2, 2, 2, 2)
        a0 = ("A", 0, 0)
        b1 = ("B", 1, 0)
        self.assertEqual(algebra.reduce_word((b1, a0)), (a0, b1))
        self.assertEqual(algebra.reduce_word((a0, a0)), (a0,))

    def test_orthogonal_explicit_outcomes_reduce_to_zero(self) -> None:
        algebra = PVMAlgebra(1, 1, 3, 3)
        self.assertIsNone(algebra.reduce_word((("A", 0, 0), ("A", 0, 1))))

    def test_eliminated_projector_is_idempotent_and_orthogonal(self) -> None:
        algebra = PVMAlgebra(1, 1, 3, 3)
        omitted = algebra.projector("A", 0, 2)
        self.assertEqual(algebra.multiply_polynomials(omitted, omitted), omitted)
        explicit = algebra.projector("A", 0, 0)
        self.assertEqual(algebra.multiply_polynomials(explicit, omitted), {})

    def test_word_basis_is_deterministic_and_party_restricted(self) -> None:
        algebra = PVMAlgebra(2, 2, 2, 2)
        standard = algebra.words(2)
        bob = algebra.words(2, party="B")
        self.assertEqual(standard[0], ())
        self.assertEqual(bob[0], ())
        self.assertTrue(all(generator[0] == "B" for word in bob for generator in word))
        self.assertEqual(standard, algebra.words(2))


if __name__ == "__main__":
    unittest.main()
