"""Reduced-word algebra for bipartite projective measurements.

One outcome per measurement is eliminated.  The omitted projector is
``I - sum(explicit projectors)``.  Cross-party generators commute, while
different questions on the same party remain noncommutative.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

Generator = tuple[str, int, int]
Word = tuple[Generator, ...]
Polynomial = dict[Word, float]


def _word_sort_key(word: Word) -> tuple:
    return (len(word), word)


def clean_polynomial(poly: Mapping[Word, float], tol: float = 0.0) -> Polynomial:
    return {word: float(value) for word, value in poly.items() if abs(value) > tol}


def add_polynomials(*polys: Mapping[Word, float]) -> Polynomial:
    result: Polynomial = {}
    for poly in polys:
        for word, value in poly.items():
            result[word] = result.get(word, 0.0) + float(value)
    return clean_polynomial(result)


def scale_polynomial(poly: Mapping[Word, float], scale: float) -> Polynomial:
    return clean_polynomial({word: scale * value for word, value in poly.items()})


@dataclass(frozen=True)
class PVMAlgebra:
    """Bipartite PVM quotient with the final outcome eliminated."""

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
                for outcome in range(a_count - 1):
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
            if not 0 <= question < q_count or not 0 <= outcome < a_count - 1:
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
        if outcome < outcome_count - 1:
            word = self.reduce_word(((party, question, outcome),))
            assert word is not None
            return {word: 1.0}
        result: Polynomial = {(): 1.0}
        for explicit in range(outcome_count - 1):
            word = self.reduce_word(((party, question, explicit),))
            assert word is not None
            result[word] = -1.0
        return result

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

