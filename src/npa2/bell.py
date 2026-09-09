"""Bell scenarios and the M2 control functionals."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping

from .algebra import PVMAlgebra, Polynomial, add_polynomials, scale_polynomial


@dataclass(frozen=True)
class Scenario:
    alice_questions: int
    bob_questions: int
    alice_outcomes: int
    bob_outcomes: int

    def algebra(self) -> PVMAlgebra:
        return PVMAlgebra(
            self.alice_questions,
            self.bob_questions,
            self.alice_outcomes,
            self.bob_outcomes,
        )


@dataclass(frozen=True)
class BellFunctional:
    name: str
    scenario: Scenario
    constant: float = 0.0
    alice_local: Mapping[tuple[int, int], float] = field(default_factory=dict)
    bob_local: Mapping[tuple[int, int], float] = field(default_factory=dict)
    joint: Mapping[tuple[int, int, int, int], float] = field(default_factory=dict)
    analytic_value: float | None = None
    source: str = ""
    metadata: Mapping[str, float | str | int | bool] = field(default_factory=dict)

    def polynomial(self) -> Polynomial:
        algebra = self.scenario.algebra()
        result: Polynomial = {(): self.constant} if self.constant else {}
        for (question, outcome), coefficient in self.alice_local.items():
            result = add_polynomials(
                result,
                scale_polynomial(algebra.projector("A", question, outcome), coefficient),
            )
        for (question, outcome), coefficient in self.bob_local.items():
            result = add_polynomials(
                result,
                scale_polynomial(algebra.projector("B", question, outcome), coefficient),
            )
        for (x, y, a, b), coefficient in self.joint.items():
            product_poly = algebra.multiply_polynomials(
                algebra.projector("A", x, a), algebra.projector("B", y, b)
            )
            result = add_polynomials(result, scale_polynomial(product_poly, coefficient))
        return result


def correlator_functional(
    name: str,
    correlation_coefficients: Mapping[tuple[int, int], float],
    alice_coefficients: Mapping[int, float] | None = None,
    bob_coefficients: Mapping[int, float] | None = None,
    *,
    analytic_value: float | None = None,
    source: str = "",
    metadata: Mapping[str, float | str | int | bool] | None = None,
) -> BellFunctional:
    scenario = Scenario(2, 2, 2, 2)
    signs = {0: 1.0, 1: -1.0}
    joint: dict[tuple[int, int, int, int], float] = {}
    for (x, y), coefficient in correlation_coefficients.items():
        for a in range(2):
            for b in range(2):
                joint[(x, y, a, b)] = coefficient * signs[a] * signs[b]
    alice_local: dict[tuple[int, int], float] = {}
    for x, coefficient in (alice_coefficients or {}).items():
        for a in range(2):
            alice_local[(x, a)] = coefficient * signs[a]
    bob_local: dict[tuple[int, int], float] = {}
    for y, coefficient in (bob_coefficients or {}).items():
        for b in range(2):
            bob_local[(y, b)] = coefficient * signs[b]
    return BellFunctional(
        name=name,
        scenario=scenario,
        alice_local=alice_local,
        bob_local=bob_local,
        joint=joint,
        analytic_value=analytic_value,
        source=source,
        metadata=metadata or {},
    )


def chsh() -> BellFunctional:
    return correlator_functional(
        "chsh",
        {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -1.0},
        analytic_value=2.0 * sqrt(2.0),
        source="Tsirelson bound",
        metadata={
            "standard_control_level": 1,
            "one_sided_control_level": 1,
            "expect_cross_hierarchy_match": True,
        },
    )


def tilted_chsh(alpha: float) -> BellFunctional:
    if not 0.0 <= alpha < 2.0:
        raise ValueError("The standard tilted-CHSH formula is used for 0 <= alpha < 2.")
    return correlator_functional(
        f"tilted_chsh_alpha_{alpha:g}",
        {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -1.0},
        alice_coefficients={0: alpha},
        analytic_value=sqrt(8.0 + 2.0 * alpha * alpha),
        source="standard tilted-CHSH analytic bound",
        metadata={
            "alpha": alpha,
            "standard_control_level": 2,
            # This is a fixed control-suite choice, not a theorem about the
            # minimal level for every alpha.  Alpha=1.5 needs level 3 in the
            # implemented CFNZ PVM hierarchy; 0.5 and 1.0 close at level 2.
            "one_sided_control_level": 3 if abs(alpha - 1.5) < 1e-12 else 2,
            "expect_cross_hierarchy_match": True,
        },
    )


def b3_game() -> BellFunctional:
    """Three-answer CHSH/B3 game with uniform questions.

    Answers are exponents in Z_3.  The four predicates are those in CFNZ
    Appendix A.1.  Their unitary Bell polynomial has maximum 6, so the
    corresponding winning probability is (4 + 6) / 12 = 5/6.
    """

    scenario = Scenario(2, 2, 3, 3)
    joint: dict[tuple[int, int, int, int], float] = {}
    for x in range(2):
        for y in range(2):
            for a in range(3):
                for b in range(3):
                    if (x, y) in {(0, 0), (1, 0)}:
                        win = a == b
                    elif (x, y) == (0, 1):
                        win = (a + b) % 3 == 0
                    else:
                        win = (a + b) % 3 == 1
                    joint[(x, y, a, b)] = 0.25 if win else 0.0
    return BellFunctional(
        name="b3_game",
        scenario=scenario,
        joint=joint,
        analytic_value=5.0 / 6.0,
        source="CFNZ Appendix A.1, equations (A.2)--(A.6)",
        metadata={
            "unitary_polynomial_optimum": 6.0,
            "standard_control_level": 2,
            "one_sided_control_level": 2,
            "expect_cross_hierarchy_match": True,
        },
    )


def control_suite() -> tuple[BellFunctional, ...]:
    return (
        chsh(),
        tilted_chsh(0.5),
        tilted_chsh(1.0),
        tilted_chsh(1.5),
        b3_game(),
    )
