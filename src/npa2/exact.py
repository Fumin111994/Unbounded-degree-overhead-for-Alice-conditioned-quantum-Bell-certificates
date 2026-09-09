"""Exact, solver-free certificate verification for the M2.5 level-one results.

Everything in this module works over ``fractions.Fraction`` / ``sympy.Rational``
and never calls an SDP solver.  Two certificate types are supported:

1. ``verify_standard_witness``: an exact primal moment matrix for the standard
   PVM relaxation, proving that the standard level-``L`` value is at least the
   exact objective value of the witness.
2. ``verify_onesided_dual``: an exact dual (SOS-type) certificate for the
   one-sided relaxation, proving that the one-sided level-``L`` value is at
   most the exact bound ``t``.  The certificate is the affine identity

       t - F(m) = sum_b Tr(S_b Phi_b(m))
                  + sum_x Tr(Lambda_x (sum_a Phi_{x,a}(m) - sum_a Phi_{0,a}(m)))
                  + nu (sum_a m_{0,a,()} - 1)
                  + sum_p mu_p prob_p(m)

   in the free block moments ``m``, with S_b positive semidefinite and
   mu_p >= 0.  On every primal-feasible point the last three terms are
   nonnegative, so F(m) <= t.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

import numpy as np
import sympy as sp

from .bell import BellFunctional
from .onesided import build_onesided_problem
from .standard import build_standard_problem


def rationalize(value: float, max_denominator: int = 10**6) -> Fraction:
    return Fraction(float(value)).limit_denominator(max_denominator)


# ---------------------------------------------------------------------------
# Symbolic word polynomials (exact coefficients, possibly parametric)
# ---------------------------------------------------------------------------
#
# These helpers mirror the float polynomial arithmetic of algebra.py but keep
# exact sympy coefficients, so certificates can be verified with a symbolic
# parameter (e.g. the tilt angle alpha) standing free.

from .algebra import PVMAlgebra, Word  # noqa: E402

SymPoly = dict[Word, "sp.Expr"]


def sym_identity() -> SymPoly:
    return {(): sp.Integer(1)}


def sym_zero() -> SymPoly:
    return {}


def sym_clean(poly: SymPoly) -> SymPoly:
    return {word: sp.simplify(value) for word, value in poly.items()
            if sp.simplify(value) != 0}


def sym_add(*polys: SymPoly) -> SymPoly:
    result: SymPoly = {}
    for poly in polys:
        for word, value in poly.items():
            result[word] = result.get(word, sp.Integer(0)) + value
    return sym_clean(result)


def sym_scale(poly: SymPoly, scale) -> SymPoly:
    return sym_clean({word: scale * value for word, value in poly.items()})


def sym_projector(algebra: PVMAlgebra, party: str, question: int, outcome: int) -> SymPoly:
    """Projector P(party, question, outcome) with exact coefficients."""

    float_poly = algebra.projector(party, question, outcome)
    return {word: sp.Integer(int(value)) for word, value in float_poly.items()}


def sym_observable(algebra: PVMAlgebra, party: str, question: int) -> SymPoly:
    """Binary observable 2 P(party, question, 0) - I."""

    return sym_add(
        sym_scale(sym_projector(algebra, party, question, 0), sp.Integer(2)),
        {(): sp.Integer(-1)},
    )


def sym_multiply(algebra: PVMAlgebra, left: SymPoly, right: SymPoly) -> SymPoly:
    result: SymPoly = {}
    for left_word, left_value in left.items():
        for right_word, right_value in right.items():
            word = algebra.multiply_words(left_word, right_word)
            if word is not None:
                result[word] = result.get(word, sp.Integer(0)) + left_value * right_value
    return sym_clean(result)


def sym_dagger(algebra: PVMAlgebra, poly: SymPoly) -> SymPoly:
    return {algebra.dagger(word): value for word, value in poly.items()}


def verify_sos_identity(
    algebra: PVMAlgebra,
    bound,
    functional_poly: SymPoly,
    squares: list[tuple[object, SymPoly]],
) -> dict:
    """Verify ``bound * I - functional = sum_i c_i r_i^dagger r_i`` exactly.

    ``squares`` is a list of ``(c_i, r_i)`` with exact (possibly parametric)
    coefficients.  The identity is checked word by word in the quotient
    algebra; nonnegativity conditions on the ``c_i`` are reported but, when
    parametric, left for the caller to justify on the parameter range.
    """

    rhs: SymPoly = {}
    for coeff, poly in squares:
        square = sym_multiply(algebra, sym_dagger(algebra, poly), poly)
        rhs = sym_add(rhs, sym_scale(square, coeff))
    lhs = sym_add(
        sym_scale(sym_identity(), bound), sym_scale(functional_poly, sp.Integer(-1))
    )
    residual = sym_add(lhs, sym_scale(rhs, sp.Integer(-1)))
    residual = {word: sp.simplify(value) for word, value in residual.items()}
    failures = [
        f"word {algebra.word_label(word)}: residual {value}"
        for word, value in residual.items()
        if value != 0
    ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "square_coefficients": [str(coeff) for coeff, _ in squares],
        "square_polys": [
            {algebra.word_label(word): str(value) for word, value in poly.items()}
            for _, poly in squares
        ],
        "bound": str(bound),
    }


def _fraction(value) -> Fraction:
    """Exact Fraction for the dyadic floats used in Bell coefficients."""

    return Fraction(value) if isinstance(value, float) else Fraction(value)


def _sp(value):
    """Sympy-exact conversion accepting Fractions and symbolic expressions."""

    if isinstance(value, sp.Basic):
        return value
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    return sp.Rational(value)


def is_psd_exact(matrix) -> bool:
    """Exact PSD check for symmetric matrices over exact (possibly algebraic)
    fields, via LDL^T decomposition with zero-pivot handling.

    This avoids symbolic eigenvalue computations (cubics and beyond), which
    can be prohibitively slow on algebraic matrices.
    """

    n = matrix.rows
    if n != matrix.cols:
        raise ValueError("matrix must be square")
    M = [[sp.simplify(matrix[i, j]) for j in range(n)] for i in range(n)]
    L = [[sp.Integer(0)] * n for _ in range(n)]
    D = [sp.Integer(0)] * n
    for i in range(n):
        pivot = sp.simplify(
            M[i][i] - sum(L[i][k] ** 2 * D[k] for k in range(i))
        )
        sign = sp.sign(pivot)
        if sign == 0:
            # zero pivot: the whole remaining row/column must vanish
            for j in range(i + 1, n):
                residual = sp.simplify(
                    M[j][i]
                    - sum(L[j][k] * L[i][k] * D[k] for k in range(i))
                )
                if sp.sign(residual) != 0:
                    return False
            D[i] = sp.Integer(0)
            continue
        if sign < 0:
            return False
        D[i] = pivot
        for j in range(i + 1, n):
            L[j][i] = sp.simplify(
                (M[j][i] - sum(L[j][k] * L[i][k] * D[k] for k in range(i)))
                / pivot
            )
    return True


def exact_functional_polynomial(functional: BellFunctional) -> dict:
    """Game polynomial with exact coefficients (Fraction-preserving).

    Unlike ``BellFunctional.polynomial()`` this never passes through float
    arithmetic, so rationalized functionals keep their exact coefficients.
    """

    algebra = functional.scenario.algebra()

    def exact_poly(poly) -> dict:
        return {word: _fraction(value) for word, value in poly.items()}

    def multiply(left, right) -> dict:
        result: dict = {}
        for left_word, left_value in left.items():
            for right_word, right_value in right.items():
                word = algebra.multiply_words(left_word, right_word)
                if word is not None:
                    result[word] = result.get(word, Fraction(0)) + left_value * right_value
        return {word: value for word, value in result.items() if value != 0}

    result: dict = {}
    if functional.constant:
        result[()] = _fraction(functional.constant)
    for (question, outcome), coeff in functional.alice_local.items():
        for word, value in exact_poly(algebra.projector("A", question, outcome)).items():
            result[word] = result.get(word, Fraction(0)) + _fraction(coeff) * value
    for (question, outcome), coeff in functional.bob_local.items():
        for word, value in exact_poly(algebra.projector("B", question, outcome)).items():
            result[word] = result.get(word, Fraction(0)) + _fraction(coeff) * value
    for (x, y, a, b), coeff in functional.joint.items():
        product_poly = multiply(
            exact_poly(algebra.projector("A", x, a)),
            exact_poly(algebra.projector("B", y, b)),
        )
        for word, value in product_poly.items():
            result[word] = result.get(word, Fraction(0)) + _fraction(coeff) * value
    return {word: value for word, value in result.items() if value != 0}


# ---------------------------------------------------------------------------
# Standard primal witness
# ---------------------------------------------------------------------------


def verify_standard_witness(
    functional: BellFunctional,
    level: int,
    gamma_entries: Mapping[tuple[int, int], object],
    *,
    enforce_probability_positivity: bool = True,
) -> dict:
    """Verify an exact primal witness for the standard PVM relaxation.

    ``gamma_entries`` gives the exact value of every entry (i, j), i <= j, of
    the moment matrix in the word order of the level-``level`` standard model.
    Values may be ``Fraction`` or exact ``sympy`` expressions (e.g. involving
    ``sqrt(2)``).
    """

    model = build_standard_problem(
        functional, level,
        enforce_probability_positivity=enforce_probability_positivity,
    )
    algebra = functional.scenario.algebra()
    words = model.words
    representatives = model.representatives
    n = len(words)
    matrix = sp.zeros(n)
    for (i, j), value in gamma_entries.items():
        matrix[i, j] = sp.sympify(value)
        matrix[j, i] = matrix[i, j]

    failures: list[str] = []

    if not is_psd_exact(matrix):
        failures.append("moment matrix is not positive semidefinite")

    # Moment consistency and structural zeros.
    for i, left in enumerate(words):
        left_adjoint = algebra.dagger(left)
        for j in range(i, n):
            product_word = algebra.multiply_words(left_adjoint, words[j])
            if product_word is None:
                if sp.simplify(matrix[i, j]) != 0:
                    failures.append(f"zero word at ({i},{j}) has value {matrix[i, j]}")
                continue
            key = algebra.moment_key(product_word)
            rep_i, rep_j = representatives[key]
            if sp.simplify(matrix[i, j] - matrix[rep_i, rep_j]) != 0:
                failures.append(
                    f"moment inconsistency at ({i},{j}) vs representative "
                    f"({rep_i},{rep_j})"
                )

    identity_i, identity_j = representatives[()]
    if matrix[identity_i, identity_j] != 1:
        failures.append("normalization violated")

    scenario = functional.scenario
    min_probability = None
    if enforce_probability_positivity:
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        joint_poly = algebra.multiply_polynomials(
                            algebra.projector("A", x, a),
                            algebra.projector("B", y, b),
                        )
                        value = sp.Rational(0)
                        for word, coeff in joint_poly.items():
                            key = algebra.moment_key(word)
                            i, j = representatives[key]
                            value += sp.Rational(_fraction(coeff)) * matrix[i, j]
                        if min_probability is None or value < min_probability:
                            min_probability = value
                        if value < 0:
                            failures.append(
                                f"probability {x},{y},{a},{b} = {value} < 0"
                            )

    objective = sp.Rational(0)
    for word, coeff in exact_functional_polynomial(functional).items():
        key = algebra.moment_key(word)
        i, j = representatives[key]
        objective += sp.Rational(coeff) * matrix[i, j]

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "objective_exact": str(objective),
        "objective_float": float(objective),
        "min_probability_exact": str(min_probability),
        "word_labels": [algebra.word_label(w) for w in words],
    }


# ---------------------------------------------------------------------------
# One-sided dual certificate
# ---------------------------------------------------------------------------


def _onesided_model_data(
    functional: BellFunctional,
    level: int,
    enforce_probability_positivity: bool = True,
):
    model = build_onesided_problem(
        functional, level,
        enforce_probability_positivity=enforce_probability_positivity,
    )
    algebra = functional.scenario.algebra()
    scenario = functional.scenario
    words = model.words
    n = len(words)
    block_names = [
        (x, a)
        for x in range(scenario.alice_questions)
        for a in range(scenario.alice_outcomes)
    ]
    # entry_key[(x, a)][(i, j)] = moment key of word_i^dagger word_j, or None.
    entry_key: dict[tuple[int, int], dict[tuple[int, int], tuple | None]] = {}
    for x, a in block_names:
        table: dict[tuple[int, int], tuple | None] = {}
        for i in range(n):
            left_adjoint = algebra.dagger(words[i])
            for j in range(n):
                product_word = algebra.multiply_words(left_adjoint, words[j])
                table[(i, j)] = (
                    None if product_word is None else algebra.moment_key(product_word)
                )
        entry_key[(x, a)] = table
    return model, algebra, scenario, words, entry_key, block_names


def _onesided_identity_residuals(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    s_blocks: Mapping[tuple[int, int], Mapping[tuple[int, int], Fraction]],
    lam: Mapping[int, Mapping[tuple[int, int], Fraction]],
    nu: Fraction,
    mu: Mapping[tuple[int, int, int, int], Fraction],
    symbols: Mapping[tuple[int, int, tuple], sp.Symbol],
    entry_key,
    block_names,
    words,
    algebra,
    scenario,
) -> list:
    """Residuals of the certificate identity; empty/zero means verified."""

    n = len(words)

    def phi(x, a, i, j):
        key = entry_key[(x, a)][(i, j)]
        if key is None:
            return sp.Rational(0)
        return symbols[(x, a, key)]

    def block_poly_moment(x, a, poly):
        total = sp.Rational(0)
        for word, coeff in poly.items():
            total += sp.Rational(_fraction(coeff)) * symbols[(x, a, algebra.moment_key(word))]
        return total

    # Objective F(m).
    objective = sp.Rational(_fraction(functional.constant)) if functional.constant else sp.Rational(0)
    for (x, a), coeff in functional.alice_local.items():
        objective += sp.Rational(_fraction(coeff)) * symbols[(x, a, ())]
    for (y, b), coeff in functional.bob_local.items():
        bob_poly = algebra.projector("B", y, b)
        marginal = sum(
            block_poly_moment(0, a, bob_poly)
            for a in range(scenario.alice_outcomes)
        )
        objective += sp.Rational(_fraction(coeff)) * marginal
    for (x, y, a, b), coeff in functional.joint.items():
        if coeff:
            objective += sp.Rational(_fraction(coeff)) * block_poly_moment(
                x, a, algebra.projector("B", y, b)
            )

    rhs = sp.Rational(0)
    for x, a in block_names:
        s_entries = s_blocks[(x, a)]
        for i in range(n):
            for j in range(n):
                coeff = s_entries.get((i, j), s_entries.get((j, i)))
                if coeff:
                    rhs += _sp(coeff) * phi(x, a, i, j)
    for x, lam_entries in lam.items():
        for i in range(n):
            for j in range(n):
                # The consistency dual need not be symmetric: CVXPY treats the
                # matrix equality elementwise, so (i, j) and (j, i) carry
                # independent duals.
                coeff = lam_entries.get((i, j))
                if coeff:
                    term = sum(
                        phi(x, a, i, j) for a in range(scenario.alice_outcomes)
                    ) - sum(
                        phi(0, a, i, j) for a in range(scenario.alice_outcomes)
                    )
                    rhs += _sp(coeff) * term
    rhs += _sp(nu) * (
        sum(symbols[(0, a, ())] for a in range(scenario.alice_outcomes)) - 1
    )
    if mu:
        for (x, y, a, b), coeff in mu.items():
            if coeff:
                rhs += _sp(coeff) * block_poly_moment(
                    x, a, algebra.projector("B", y, b)
                )

    difference = sp.expand((sp.Rational(t) if isinstance(t, Fraction) else sp.sympify(t))
                           - objective - rhs)
    residuals = [sp.simplify(difference.subs({s: 0 for s in symbols.values()}))]
    for symbol in symbols.values():
        residuals.append(sp.expand(difference).coeff(symbol))
    return residuals


def onesided_certificate_unknowns(
    functional: BellFunctional,
    level: int,
    enforce_probability_positivity: bool = True,
) -> dict:
    """Describe the certificate unknowns and the linear identity system.

    Returns the symbols, and the exact rational system ``A z = b`` (with t
    folded into b) whose solutions make the identity hold.
    """

    model, algebra, scenario, words, entry_key, block_names = _onesided_model_data(
        functional, level, enforce_probability_positivity
    )
    n = len(words)
    symbols: dict[tuple[int, int, tuple], sp.Symbol] = {}
    for x, a in block_names:
        for key in {k for k in entry_key[(x, a)].values() if k is not None}:
            symbols[(x, a, key)] = sp.Symbol(f"m_{x}_{a}_{len(symbols)}")

    unknown_names: list[str] = []
    for x, a in block_names:
        for i in range(n):
            for j in range(i, n):
                unknown_names.append(f"S_{x}_{a}:{i}:{j}")
    for x in range(1, scenario.alice_questions):
        for i in range(n):
            for j in range(n):
                unknown_names.append(f"L_{x}:{i}:{j}")
    unknown_names.append("nu")
    if enforce_probability_positivity:
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        unknown_names.append(f"mu:{x}:{y}:{a}:{b}")

    return {
        "model": model,
        "algebra": algebra,
        "scenario": scenario,
        "words": words,
        "entry_key": entry_key,
        "block_names": block_names,
        "symbols": symbols,
        "unknown_names": unknown_names,
    }


def unpack_onesided_unknowns(names: list[str], values: list[Fraction]):
    s_blocks: dict[tuple[int, int], dict[tuple[int, int], Fraction]] = {}
    lam: dict[int, dict[tuple[int, int], Fraction]] = {}
    nu = Fraction(0)
    mu: dict[tuple[int, int, int, int], Fraction] = {}
    for name, value in zip(names, values):
        parts = name.split(":")
        if name.startswith("S_"):
            x, a = parts[0][2:].split("_")
            key = (int(x), int(a))
            s_blocks.setdefault(key, {})[(int(parts[1]), int(parts[2]))] = value
        elif name.startswith("L_"):
            x = int(parts[0][2:])
            lam.setdefault(x, {})[(int(parts[1]), int(parts[2]))] = value
        elif name == "nu":
            nu = value
        elif name.startswith("mu:"):
            mu[tuple(int(p) for p in parts[1:])] = value
        else:
            raise ValueError(f"Unknown certificate variable {name!r}")
    return s_blocks, lam, nu, mu


def onesided_identity_system(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    enforce_probability_positivity: bool = True,
):
    """Exact linear system ``A z = b`` for the certificate identity.

    ``z`` follows ``onesided_certificate_unknowns(...)["unknown_names"]`` and
    includes every certificate variable except the bound ``t``, which is a
    parameter folded into ``b``.  Returns (unknown_names, A, b) with exact
    rational entries.
    """

    data = onesided_certificate_unknowns(
        functional, level, enforce_probability_positivity
    )
    names = data["unknown_names"]

    def residuals_for(values: list[Fraction]) -> list:
        s_blocks, lam, nu, mu = unpack_onesided_unknowns(names, values)
        return _onesided_identity_residuals(
            functional,
            level,
            t,
            s_blocks,
            lam,
            nu,
            mu,
            data["symbols"],
            data["entry_key"],
            data["block_names"],
            data["words"],
            data["algebra"],
            data["scenario"],
        )

    zero = [Fraction(0)] * len(names)
    r0 = [sp.nsimplify(r, rational=True) for r in residuals_for(zero)]
    columns: list[list] = []
    for j in range(len(names)):
        basis = list(zero)
        basis[j] = Fraction(1)
        rj = [sp.nsimplify(r, rational=True) for r in residuals_for(basis)]
        columns.append([rj[k] - r0[k] for k in range(len(r0))])

    rows = len(r0)
    A = sp.Matrix(rows, len(names), lambda i, j: columns[j][i])
    b = -sp.Matrix(r0)
    return names, A, b


def verify_onesided_dual(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    s_blocks,
    lam,
    nu,
    mu,
    *,
    enforce_probability_positivity: bool = True,
) -> dict:
    """Verify an exact dual upper certificate for the one-sided relaxation."""

    data = onesided_certificate_unknowns(
        functional, level, enforce_probability_positivity
    )
    residuals = _onesided_identity_residuals(
        functional,
        level,
        t,
        s_blocks,
        lam,
        nu,
        mu,
        data["symbols"],
        data["entry_key"],
        data["block_names"],
        data["words"],
        data["algebra"],
        data["scenario"],
    )

    failures: list[str] = [
        f"identity residual {r}" for r in residuals if r != 0
    ]

    n = len(data["words"])
    min_eigeninfo: dict[str, float] = {}
    for key, entries in s_blocks.items():
        matrix = sp.zeros(n)
        for (i, j), value in entries.items():
            matrix[i, j] = sp.Rational(value.numerator, value.denominator)
            matrix[j, i] = matrix[i, j]
        if not is_psd_exact(matrix):
            failures.append(f"S block {key} is not positive semidefinite")
        # Reporting only: float eigenvalues of the exact matrix.
        numeric = np.array(matrix.tolist(), dtype=float)
        min_eigeninfo[str(key)] = float(np.linalg.eigvalsh(numeric)[0])
    for key, value in mu.items():
        if value < 0:
            failures.append(f"mu {key} = {value} < 0")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "bound_exact": str(t),
        "bound_float": float(t),
        "min_block_eigenvalue_exact": min_eigeninfo,
        "word_labels": [data["algebra"].word_label(w) for w in data["words"]],
    }


# ---------------------------------------------------------------------------
# Standard dual certificate and one-sided primal witness (M3)
# ---------------------------------------------------------------------------


def _standard_model_data(
    functional: BellFunctional,
    level: int,
    enforce_probability_positivity: bool = False,
):
    model = build_standard_problem(
        functional, level,
        enforce_probability_positivity=enforce_probability_positivity,
    )
    algebra = functional.scenario.algebra()
    words = model.words
    representatives = model.representatives
    n = len(words)
    entry_key: dict[tuple[int, int], tuple | None] = {}
    for i in range(n):
        left_adjoint = algebra.dagger(words[i])
        for j in range(i, n):
            product_word = algebra.multiply_words(left_adjoint, words[j])
            entry_key[(i, j)] = (
                None if product_word is None else algebra.moment_key(product_word)
            )
    return model, algebra, words, representatives, entry_key


def _standard_identity_residuals(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    s_entries: Mapping[tuple[int, int], Fraction],
    nu: Fraction,
    symbols: Mapping[tuple, sp.Symbol],
    entry_key,
    representatives,
    words,
    algebra,
    enforce_probability_positivity: bool = False,
    mu: Mapping[tuple[int, int, int, int], Fraction] | None = None,
) -> list:
    """Residuals of  t - F(m) = Tr(S Gamma(m)) + nu (m(()) - 1) + mu terms."""

    n = len(words)

    def gamma(i, j):
        key = entry_key[(i, j) if i <= j else (j, i)]
        if key is None:
            return sp.Rational(0)
        return symbols[key]

    objective = sp.Rational(0)
    for word, coeff in exact_functional_polynomial(functional).items():
        key = algebra.moment_key(word)
        if key not in symbols:
            raise ValueError(f"moment {algebra.word_label(word)} outside level")
        objective += sp.Rational(coeff) * symbols[key]

    rhs = sp.Rational(0)
    for i in range(n):
        for j in range(n):
            coeff = s_entries.get((i, j), s_entries.get((j, i)))
            if coeff:
                rhs += _sp(coeff) * gamma(i, j)
    identity_key = algebra.moment_key(())
    rhs += _sp(nu) * (symbols[identity_key] - 1)

    if enforce_probability_positivity and mu:
        scenario = functional.scenario
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        coeff = mu.get((x, y, a, b))
                        if not coeff:
                            continue
                        joint_poly = algebra.multiply_polynomials(
                            algebra.projector("A", x, a),
                            algebra.projector("B", y, b),
                        )
                        term = sp.Rational(0)
                        for word, c2 in joint_poly.items():
                            term += sp.Rational(_fraction(c2)) * symbols[
                                algebra.moment_key(word)
                            ]
                        rhs += _sp(coeff) * term

    difference = sp.expand((sp.Rational(t) if isinstance(t, Fraction) else sp.sympify(t))
                           - objective - rhs)
    residuals = [sp.simplify(difference.subs({s: 0 for s in symbols.values()}))]
    for symbol in symbols.values():
        residuals.append(sp.expand(difference).coeff(symbol))
    return residuals


def standard_identity_system(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    enforce_probability_positivity: bool = False,
):
    """Exact linear system ``A z = b`` for the standard dual identity.

    Unknowns: ``S:i:j`` (i <= j, symmetric PSD Gram dual) and ``nu``; when
    probability positivity is enforced, also ``mu:x:y:a:b``.
    """

    model, algebra, words, representatives, entry_key = _standard_model_data(
        functional, level, enforce_probability_positivity
    )
    n = len(words)
    symbols = {key: sp.Symbol(f"m_{k}") for k, key in enumerate(representatives)}
    unknown_names = [f"S:{i}:{j}" for i in range(n) for j in range(i, n)]
    unknown_names.append("nu")
    if enforce_probability_positivity:
        scenario = functional.scenario
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        unknown_names.append(f"mu:{x}:{y}:{a}:{b}")

    def unpack(values):
        s_entries: dict[tuple[int, int], Fraction] = {}
        nu = Fraction(0)
        mu: dict[tuple[int, int, int, int], Fraction] = {}
        for name, value in zip(unknown_names, values):
            parts = name.split(":")
            if parts[0] == "S":
                s_entries[(int(parts[1]), int(parts[2]))] = value
            elif name == "nu":
                nu = value
            else:
                mu[tuple(int(p) for p in parts[1:])] = value
        return s_entries, nu, mu

    def residuals_for(values):
        s_entries, nu, mu = unpack(values)
        return _standard_identity_residuals(
            functional, level, t, s_entries, nu, symbols, entry_key,
            representatives, words, algebra,
            enforce_probability_positivity, mu,
        )

    zero = [Fraction(0)] * len(unknown_names)
    r0 = [sp.nsimplify(r, rational=True) for r in residuals_for(zero)]
    columns = []
    for j in range(len(unknown_names)):
        basis = list(zero)
        basis[j] = Fraction(1)
        rj = [sp.nsimplify(r, rational=True) for r in residuals_for(basis)]
        columns.append([rj[k] - r0[k] for k in range(len(r0))])

    A = sp.Matrix(len(r0), len(unknown_names), lambda i, j: columns[j][i])
    b = -sp.Matrix(r0)
    return unknown_names, A, b, unpack


def verify_standard_dual(
    functional: BellFunctional,
    level: int,
    t: Fraction,
    s_entries,
    nu,
    mu=None,
    *,
    enforce_probability_positivity: bool = False,
) -> dict:
    """Verify an exact dual upper certificate for the standard relaxation."""

    model, algebra, words, representatives, entry_key = _standard_model_data(
        functional, level, enforce_probability_positivity
    )
    symbols = {key: sp.Symbol(f"m_{k}") for k, key in enumerate(representatives)}
    residuals = _standard_identity_residuals(
        functional, level, t, s_entries, nu, symbols, entry_key,
        representatives, words, algebra, enforce_probability_positivity, mu,
    )
    failures: list[str] = [
        f"identity residual {r}" for r in residuals if r != 0
    ]
    n = len(words)
    matrix = sp.zeros(n)
    for (i, j), value in s_entries.items():
        matrix[i, j] = sp.Rational(value.numerator, value.denominator)
        matrix[j, i] = matrix[i, j]
    if not is_psd_exact(matrix):
        failures.append("S is not positive semidefinite")
    numeric = np.array(matrix.tolist(), dtype=float)
    min_eig = float(np.linalg.eigvalsh(numeric)[0])
    if mu:
        for key, value in mu.items():
            if value < 0:
                failures.append(f"mu {key} = {value} < 0")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "bound_exact": str(t),
        "bound_float": float(t),
        "min_s_eigenvalue_float": min_eig,
        "word_labels": [algebra.word_label(w) for w in words],
    }


def verify_onesided_witness(
    functional: BellFunctional,
    level: int,
    block_entries: Mapping[tuple[int, int], Mapping[tuple[int, int], Fraction]],
    *,
    enforce_probability_positivity: bool = False,
) -> dict:
    """Verify an exact primal witness for the one-sided relaxation.

    ``block_entries[(x, a)]`` gives the exact value of every entry (i, j),
    i <= j, of the block Phi_{x,a} in the word order of the level-``level``
    one-sided model.
    """

    model = build_onesided_problem(
        functional, level,
        enforce_probability_positivity=enforce_probability_positivity,
    )
    algebra = functional.scenario.algebra()
    scenario = functional.scenario
    words = model.words
    n = len(words)
    block_reps = model.block_representatives

    failures: list[str] = []
    matrices: dict[tuple[int, int], sp.Matrix] = {}
    for x in range(scenario.alice_questions):
        for a in range(scenario.alice_outcomes):
            name = f"Phi_x{x}_a{a}"
            entries = block_entries.get((x, a), {})
            matrix = sp.zeros(n)
            for (i, j), value in entries.items():
                matrix[i, j] = sp.Rational(value.numerator, value.denominator)
                matrix[j, i] = matrix[i, j]
            matrices[(x, a)] = matrix
            if not is_psd_exact(matrix):
                failures.append(f"block {name} is not PSD")
            # moment consistency within the block
            representatives = block_reps[name]
            for i, left in enumerate(words):
                left_adjoint = algebra.dagger(left)
                for j in range(i, n):
                    product_word = algebra.multiply_words(left_adjoint, words[j])
                    if product_word is None:
                        if matrix[i, j] != 0:
                            failures.append(f"{name}: zero word at ({i},{j})")
                        continue
                    key = algebra.moment_key(product_word)
                    rep_i, rep_j = representatives[key]
                    if matrix[i, j] != matrix[rep_i, rep_j]:
                        failures.append(f"{name}: inconsistency at ({i},{j})")

    # cross-block consistency: sum_a Phi_{x,a} independent of x
    for x in range(1, scenario.alice_questions):
        diff = sum(
            (matrices[(x, a)] for a in range(scenario.alice_outcomes)),
            sp.zeros(n),
        ) - sum(
            (matrices[(0, a)] for a in range(scenario.alice_outcomes)),
            sp.zeros(n),
        )
        if any(diff[i, j] != 0 for i in range(n) for j in range(n)):
            failures.append(f"consistency violated at x={x}")

    identity_rep = block_reps["Phi_x0_a0"][()]
    normalization = sum(
        matrices[(0, a)][identity_rep[0], identity_rep[1]]
        for a in range(scenario.alice_outcomes)
    )
    if normalization != 1:
        failures.append(f"normalization violated: {normalization}")

    def block_poly_moment(x, a, poly):
        representatives = block_reps[f"Phi_x{x}_a{a}"]
        total = sp.Rational(0)
        for word, coeff in poly.items():
            i, j = representatives[algebra.moment_key(word)]
            total += sp.Rational(_fraction(coeff)) * matrices[(x, a)][i, j]
        return total

    min_probability = None
    if enforce_probability_positivity:
        for x in range(scenario.alice_questions):
            for y in range(scenario.bob_questions):
                for a in range(scenario.alice_outcomes):
                    for b in range(scenario.bob_outcomes):
                        value = block_poly_moment(x, a, algebra.projector("B", y, b))
                        if min_probability is None or value < min_probability:
                            min_probability = value
                        if value < 0:
                            failures.append(f"probability {x},{y},{a},{b} < 0")

    objective = sp.Rational(_fraction(functional.constant)) if functional.constant else sp.Rational(0)
    for (x, a), coeff in functional.alice_local.items():
        i, j = block_reps[f"Phi_x{x}_a{a}"][()]
        objective += sp.Rational(_fraction(coeff)) * matrices[(x, a)][i, j]
    for (y, b), coeff in functional.bob_local.items():
        bob_poly = algebra.projector("B", y, b)
        marginal = sum(
            block_poly_moment(0, a, bob_poly)
            for a in range(scenario.alice_outcomes)
        )
        objective += sp.Rational(_fraction(coeff)) * marginal
    for (x, y, a, b), coeff in functional.joint.items():
        if coeff:
            objective += sp.Rational(_fraction(coeff)) * block_poly_moment(
                x, a, algebra.projector("B", y, b)
            )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "objective_exact": str(objective),
        "objective_float": float(objective),
        "min_probability_exact": str(min_probability),
        "word_labels": [algebra.word_label(w) for w in words],
    }
