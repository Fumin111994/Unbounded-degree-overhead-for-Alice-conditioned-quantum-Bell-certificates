"""Shared model container and moment-matrix helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp

from .algebra import PVMAlgebra, Word
from .bell import BellFunctional


@dataclass
class NamedConstraint:
    name: str
    constraint: cp.Constraint


@dataclass
class NPAProblem:
    hierarchy: str
    level: int
    functional: BellFunctional
    problem: cp.Problem
    words: tuple[Word, ...]
    matrices: dict[str, cp.Variable]
    psd_constraints: dict[str, cp.Constraint]
    constraints: list[NamedConstraint]
    representatives: dict[Word, tuple[int, int]] | None = None
    block_representatives: dict[str, dict[Word, tuple[int, int]]] = field(
        default_factory=dict
    )
    metadata: dict[str, Any] = field(default_factory=dict)


def add_moment_constraints(
    *,
    algebra: PVMAlgebra,
    matrix: cp.Variable,
    words: tuple[Word, ...],
    prefix: str,
    named_constraints: list[NamedConstraint],
) -> dict[Word, tuple[int, int]]:
    representatives: dict[Word, tuple[int, int]] = {}
    for i, left in enumerate(words):
        left_adjoint = algebra.dagger(left)
        for j in range(i, len(words)):
            right = words[j]
            product_word = algebra.multiply_words(left_adjoint, right)
            if product_word is None:
                constraint = matrix[i, j] == 0.0
                named_constraints.append(
                    NamedConstraint(f"{prefix}:zero:{i}:{j}", constraint)
                )
                continue
            key = algebra.moment_key(product_word)
            if key not in representatives:
                representatives[key] = (i, j)
            else:
                rep_i, rep_j = representatives[key]
                if (i, j) != (rep_i, rep_j):
                    constraint = matrix[i, j] == matrix[rep_i, rep_j]
                    named_constraints.append(
                        NamedConstraint(f"{prefix}:equal:{i}:{j}", constraint)
                    )
    return representatives


def moment_expression(
    *,
    algebra: PVMAlgebra,
    matrix: cp.Variable,
    representatives: dict[Word, tuple[int, int]],
    polynomial: dict[Word, float],
) -> cp.Expression:
    expression: cp.Expression = cp.Constant(0.0)
    for word, coefficient in polynomial.items():
        key = algebra.moment_key(word)
        if key not in representatives:
            raise ValueError(
                f"Moment {algebra.word_label(word)} is outside the selected hierarchy level."
            )
        i, j = representatives[key]
        expression = expression + coefficient * matrix[i, j]
    return expression

