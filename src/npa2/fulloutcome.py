"""Full-outcome PVM algebra: no outcome is eliminated.

This is the independent cross-check required by M2.5.  Every outcome of every
measurement is an explicit generator.  The relation ``sum_a P_{q,a} = I``
cannot be expressed by word reduction, so it is imposed as explicit linear
constraints on the moment matrix:

    sum_a m(u^dagger P_{q,a} v) = m(u^dagger v)

for every pair of basis words ``u, v`` with ``|u| + |v| <= 2L - 1`` (longer
products fall outside the degree-``2L`` moment body and are correctly
omitted).  Together with the reduction rules (idempotency, orthogonality of
same-question outcomes, cross-party commutation) this is exactly the level-``L``
moment body of the quotient algebra with all outcomes present.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable, Mapping

import cvxpy as cp

from .algebra import Polynomial, _word_sort_key, clean_polynomial
from .bell import BellFunctional
from .model import (
    NPAProblem,
    NamedConstraint,
    add_moment_constraints,
    moment_expression,
)

Generator = tuple[str, int, int]
Word = tuple[Generator, ...]


@dataclass(frozen=True)
class FullPVMAlgebra:
    """Bipartite PVM algebra with every outcome kept as a generator."""

    alice_questions: int
    bob_questions: int
    alice_outcomes: int
    bob_outcomes: int

    def __post_init__(self) -> None:
        if min(
            self.alice_questions,
            self.bob_questions,
            self.alice_outcomes,
            self.bob_outcomes,
        ) < 1:
            raise ValueError("Question and outcome counts must be positive.")

    @property
    def identity(self) -> Word:
        return ()

    def generators(self, party: str | None = None) -> tuple[Generator, ...]:
        parties = (party,) if party is not None else ("A", "B")
        result: list[Generator] = []
        for current in parties:
            if current == "A":
                q_count, a_count = self.alice_questions, self.alice_outcomes
            elif current == "B":
                q_count, a_count = self.bob_questions, self.bob_outcomes
            else:
                raise ValueError(f"Unknown party {current!r}.")
            for question in range(q_count):
                for outcome in range(a_count):
                    result.append((current, question, outcome))
        return tuple(result)

    def reduce_word(self, raw_word: Iterable[Generator]) -> Word | None:
        """Return the canonical reduced word, or ``None`` for the zero word."""

        raw = tuple(raw_word)
        for party, question, outcome in raw:
            if party not in {"A", "B"}:
                raise ValueError(f"Unknown party {party!r}.")
            q_count = self.alice_questions if party == "A" else self.bob_questions
            a_count = self.alice_outcomes if party == "A" else self.bob_outcomes
            if not 0 <= question < q_count or not 0 <= outcome < a_count:
                raise ValueError(f"Generator {(party, question, outcome)!r} is out of range.")

        # Alice/Bob commute, so retain within-party order and move Alice left.
        ordered = tuple(g for g in raw if g[0] == "A") + tuple(
            g for g in raw if g[0] == "B"
        )
        reduced: list[Generator] = []
        for generator in ordered:
            if reduced and reduced[-1][:2] == generator[:2]:
                if reduced[-1][2] != generator[2]:
                    return None
                # P^2 = P.
                continue
            reduced.append(generator)
        return tuple(reduced)

    def dagger(self, word: Word) -> Word:
        reduced = self.reduce_word(reversed(word))
        if reduced is None:
            raise ValueError("The adjoint of a nonzero word cannot be zero.")
        return reduced

    def multiply_words(self, left: Word, right: Word) -> Word | None:
        return self.reduce_word(left + right)

    def moment_key(self, word: Word) -> Word:
        """Identify a word with its adjoint for real Bell objectives."""

        adjoint = self.dagger(word)
        return min(word, adjoint)

    def words(self, level: int, party: str | None = None) -> tuple[Word, ...]:
        if level < 0:
            raise ValueError("Level must be nonnegative.")
        alphabet = self.generators(party)
        current: set[Word] = {()}
        all_words: set[Word] = {()}
        for _ in range(level):
            next_words: set[Word] = set()
            for word, generator in product(current, alphabet):
                reduced = self.reduce_word(word + (generator,))
                if reduced is not None:
                    next_words.add(reduced)
                    all_words.add(reduced)
            current = next_words
        return tuple(sorted(all_words, key=_word_sort_key))

    def projector(self, party: str, question: int, outcome: int) -> Polynomial:
        outcome_count = self.alice_outcomes if party == "A" else self.bob_outcomes
        if not 0 <= outcome < outcome_count:
            raise ValueError("Outcome is out of range.")
        word = self.reduce_word(((party, question, outcome),))
        assert word is not None
        return {word: 1.0}

    def multiply_polynomials(
        self, left: Mapping[Word, float], right: Mapping[Word, float]
    ) -> Polynomial:
        result: Polynomial = {}
        for left_word, left_value in left.items():
            for right_word, right_value in right.items():
                word = self.multiply_words(left_word, right_word)
                if word is not None:
                    result[word] = result.get(word, 0.0) + left_value * right_value
        return clean_polynomial(result)

    @staticmethod
    def word_label(word: Word) -> str:
        if not word:
            return "I"
        return " ".join(f"{party}{question}:{outcome}" for party, question, outcome in word)


def functional_full_polynomial(functional: BellFunctional) -> Polynomial:
    """Bell polynomial with every outcome written as an explicit generator."""

    scenario = functional.scenario
    algebra = FullPVMAlgebra(
        scenario.alice_questions,
        scenario.bob_questions,
        scenario.alice_outcomes,
        scenario.bob_outcomes,
    )
    result: Polynomial = {(): functional.constant} if functional.constant else {}
    for (question, outcome), coefficient in functional.alice_local.items():
        for word, value in algebra.projector("A", question, outcome).items():
            result[word] = result.get(word, 0.0) + coefficient * value
    for (question, outcome), coefficient in functional.bob_local.items():
        for word, value in algebra.projector("B", question, outcome).items():
            result[word] = result.get(word, 0.0) + coefficient * value
    for (x, y, a, b), coefficient in functional.joint.items():
        product_poly = algebra.multiply_polynomials(
            algebra.projector("A", x, a), algebra.projector("B", y, b)
        )
        for word, value in product_poly.items():
            result[word] = result.get(word, 0.0) + coefficient * value
    return clean_polynomial(result)


def add_completeness_constraints(
    *,
    algebra: FullPVMAlgebra,
    matrix: cp.Variable,
    words: tuple[Word, ...],
    representatives: dict[Word, tuple[int, int]],
    prefix: str,
    named_constraints: list[NamedConstraint],
    parties: tuple[str, ...] = ("A", "B"),
) -> int:
    """Impose ``sum_a m(u^dagger P_{q,a} v) = m(u^dagger v)`` on the moment body.

    Only pairs with ``|u| + |v| <= max_body_degree - 1`` are constrained; the
    remaining products have degree above the moment body and do not exist as
    moments.  Returns the number of distinct constraints added.
    """

    max_body_degree = 2 * max(len(word) for word in words)

    def entry(word: Word | None) -> dict[tuple[int, int], float]:
        if word is None:
            return {}
        key = algebra.moment_key(word)
        if key not in representatives:
            raise ValueError(
                f"Moment {algebra.word_label(word)} is outside the moment body."
            )
        return {representatives[key]: 1.0}

    # First pass: collect every distinct raw constraint as an exact integer
    # row over the representative entries.  The raw set is highly redundant
    # (adjoint pairs, repeated products), and dependent equality rows make
    # interior-point KKT systems singular, so an exact row reduction selects
    # an independent subset before any CVXPY constraint is created.
    raw_rows: list[dict[tuple[int, int], Fraction]] = []
    seen: set[tuple] = set()
    for party in parties:
        q_count = algebra.alice_questions if party == "A" else algebra.bob_questions
        a_count = algebra.alice_outcomes if party == "A" else algebra.bob_outcomes
        for question in range(q_count):
            for left in words:
                for right in words:
                    if len(left) + len(right) > max_body_degree - 1:
                        continue
                    left_adjoint = algebra.dagger(left)
                    terms: dict[tuple[int, int], Fraction] = {}
                    for outcome in range(a_count):
                        product_word = algebra.reduce_word(
                            left_adjoint + ((party, question, outcome),) + right
                        )
                        for rep, coeff in entry(product_word).items():
                            terms[rep] = terms.get(rep, Fraction(0)) + Fraction(coeff)
                    base_word = algebra.reduce_word(left_adjoint + right)
                    for rep, coeff in entry(base_word).items():
                        terms[rep] = terms.get(rep, Fraction(0)) - Fraction(coeff)
                    terms = {rep: c for rep, c in terms.items() if c != 0}
                    if not terms:
                        continue
                    key = tuple(sorted(terms.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    raw_rows.append(terms)

    # Exact Gaussian elimination: keep only rows that raise the rank.
    basis: list[dict[tuple[int, int], Fraction]] = []
    for row in raw_rows:
        current = dict(row)
        for pivot, rest in basis:
            if pivot in current:
                factor = current.pop(pivot)  # basis rows are normalized
                for rep, coeff in rest.items():
                    current[rep] = current.get(rep, Fraction(0)) - factor * coeff
                current = {rep: c for rep, c in current.items() if c != 0}
        if current:
            pivot = min(current)
            pivot_coeff = current.pop(pivot)
            basis.append(
                (pivot, {rep: c / pivot_coeff for rep, c in current.items()})
            )

    for added, (pivot, rest) in enumerate(basis):
        expression: cp.Expression = matrix[pivot[0], pivot[1]]
        for (i, j), coeff in rest.items():
            expression = expression + float(coeff) * matrix[i, j]
        named_constraints.append(
            NamedConstraint(f"{prefix}:complete:{added}", expression == 0.0)
        )
    return len(basis)


def build_standard_problem_full(
    functional: BellFunctional,
    level: int,
    *,
    enforce_probability_positivity: bool = True,
) -> NPAProblem:
    """Standard NPA primal over the full-outcome PVM algebra."""

    if level < 1:
        raise ValueError("Bell objectives require standard level >= 1.")
    scenario = functional.scenario
    algebra = FullPVMAlgebra(
        scenario.alice_questions,
        scenario.bob_questions,
        scenario.alice_outcomes,
        scenario.bob_outcomes,
    )
    words = algebra.words(level)
    gamma = cp.Variable((len(words), len(words)), symmetric=True, name="Gamma_full")
    psd_constraint = gamma >> 0
    constraints: list[NamedConstraint] = [
        NamedConstraint("Gamma_full:psd", psd_constraint)
    ]
    representatives = add_moment_constraints(
        algebra=algebra,
        matrix=gamma,
        words=words,
        prefix="Gamma_full",
        named_constraints=constraints,
    )
    identity_i, identity_j = representatives[()]
    constraints.append(
        NamedConstraint("normalization", gamma[identity_i, identity_j] == 1.0)
    )
    completeness_count = add_completeness_constraints(
        algebra=algebra,
        matrix=gamma,
        words=words,
        representatives=representatives,
        prefix="Gamma_full",
        named_constraints=constraints,
    )

    if enforce_probability_positivity:
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        joint_poly = algebra.multiply_polynomials(
                            algebra.projector("A", x, a),
                            algebra.projector("B", y, b),
                        )
                        probability = moment_expression(
                            algebra=algebra,
                            matrix=gamma,
                            representatives=representatives,
                            polynomial=joint_poly,
                        )
                        constraints.append(
                            NamedConstraint(
                                f"probability:{x}:{y}:{a}:{b}", probability >= 0.0
                            )
                        )

    objective_expression = moment_expression(
        algebra=algebra,
        matrix=gamma,
        representatives=representatives,
        polynomial=functional_full_polynomial(functional),
    )
    problem = cp.Problem(
        cp.Maximize(objective_expression),
        [item.constraint for item in constraints],
    )
    return NPAProblem(
        hierarchy="standard_pvm_full",
        level=level,
        functional=functional,
        problem=problem,
        words=words,
        matrices={"Gamma_full": gamma},
        psd_constraints={"Gamma_full": psd_constraint},
        constraints=constraints,
        representatives=representatives,
        metadata={
            "native_level_meaning": "maximum total reduced word length",
            "definition": "standard dense NPA over the full-outcome bipartite PVM algebra",
            "completeness_constraints": completeness_count,
            "probability_positivity_added": enforce_probability_positivity,
        },
    )


def build_onesided_problem_full(
    functional: BellFunctional,
    level: int,
    *,
    enforce_probability_positivity: bool = True,
) -> NPAProblem:
    """One-sided CFNZ primal with the full-outcome Bob-PVM algebra.

    Alice's completeness is carried by the block labels and the
    x-independent block-sum consistency constraint, exactly as in the
    reduced implementation; only Bob's quotient changes.
    """

    if level < 1:
        raise ValueError("Bell objectives require one-sided level >= 1.")
    scenario = functional.scenario
    algebra = FullPVMAlgebra(
        scenario.alice_questions,
        scenario.bob_questions,
        scenario.alice_outcomes,
        scenario.bob_outcomes,
    )
    words = algebra.words(level, party="B")
    constraints: list[NamedConstraint] = []
    matrices: dict[str, cp.Variable] = {}
    psd_constraints: dict[str, cp.Constraint] = {}
    block_representatives: dict[str, dict] = {}

    for x in range(scenario.alice_questions):
        for a in range(scenario.alice_outcomes):
            name = f"Phi_x{x}_a{a}"
            matrix = cp.Variable((len(words), len(words)), symmetric=True, name=name)
            psd = matrix >> 0
            matrices[name] = matrix
            psd_constraints[name] = psd
            constraints.append(NamedConstraint(f"{name}:psd", psd))
            block_representatives[name] = add_moment_constraints(
                algebra=algebra,
                matrix=matrix,
                words=words,
                prefix=name,
                named_constraints=constraints,
            )

    def block_name(x: int, a: int) -> str:
        return f"Phi_x{x}_a{a}"

    completeness_total = 0
    for x in range(scenario.alice_questions):
        for a in range(scenario.alice_outcomes):
            name = block_name(x, a)
            completeness_total += add_completeness_constraints(
                algebra=algebra,
                matrix=matrices[name],
                words=words,
                representatives=block_representatives[name],
                prefix=name,
                named_constraints=constraints,
                parties=("B",),
            )

    reference_sum = sum(matrices[block_name(0, a)] for a in range(scenario.alice_outcomes))
    for x in range(1, scenario.alice_questions):
        current_sum = sum(
            matrices[block_name(x, a)] for a in range(scenario.alice_outcomes)
        )
        constraints.append(
            NamedConstraint(f"consistency:x{x}", current_sum == reference_sum)
        )

    norm_expression: cp.Expression = cp.Constant(0.0)
    for a in range(scenario.alice_outcomes):
        i, j = block_representatives[block_name(0, a)][()]
        norm_expression = norm_expression + matrices[block_name(0, a)][i, j]
    constraints.append(NamedConstraint("normalization", norm_expression == 1.0))

    def block_moment(x: int, a: int, bob_poly: Polynomial) -> cp.Expression:
        name = block_name(x, a)
        return moment_expression(
            algebra=algebra,
            matrix=matrices[name],
            representatives=block_representatives[name],
            polynomial=bob_poly,
        )

    objective_expression: cp.Expression = cp.Constant(functional.constant)
    for (x, a), coefficient in functional.alice_local.items():
        objective_expression = objective_expression + coefficient * block_moment(
            x, a, {(): 1.0}
        )
    for (y, b), coefficient in functional.bob_local.items():
        bob_poly = algebra.projector("B", y, b)
        marginal = sum(
            block_moment(0, a, bob_poly) for a in range(scenario.alice_outcomes)
        )
        objective_expression = objective_expression + coefficient * marginal
    for (x, y, a, b), coefficient in functional.joint.items():
        if coefficient:
            objective_expression = objective_expression + coefficient * block_moment(
                x, a, algebra.projector("B", y, b)
            )

    if enforce_probability_positivity:
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        probability = block_moment(x, a, algebra.projector("B", y, b))
                        constraints.append(
                            NamedConstraint(
                                f"probability:{x}:{y}:{a}:{b}", probability >= 0.0
                            )
                        )

    problem = cp.Problem(
        cp.Maximize(objective_expression),
        [item.constraint for item in constraints],
    )
    return NPAProblem(
        hierarchy="one_sided_pvm_full",
        level=level,
        functional=functional,
        problem=problem,
        words=words,
        matrices=matrices,
        psd_constraints=psd_constraints,
        constraints=constraints,
        block_representatives=block_representatives,
        metadata={
            "native_level_meaning": "maximum reduced Bob word length",
            "definition": "CFNZ Definition 4.1 over the full-outcome Bob-PVM algebra",
            "completeness_constraints_per_block": completeness_total
            // (scenario.alice_questions * scenario.alice_outcomes),
            "probability_positivity_added": enforce_probability_positivity,
        },
    )
