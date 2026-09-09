"""CFNZ one-sided NPA relaxation in the Bob-PVM quotient.

The block layout is also the projectivized moment presentation used in the
KPRSTXZ sparse-duality discussion.  We do not silently identify this finite
PVM quotient with every POVM/localizer indexing convention.
"""

from __future__ import annotations

import cvxpy as cp

from .algebra import Polynomial
from .bell import BellFunctional
from .model import (
    NPAProblem,
    NamedConstraint,
    add_moment_constraints,
    moment_expression,
)


def build_onesided_problem(
    functional: BellFunctional,
    level: int,
    *,
    enforce_probability_positivity: bool = True,
) -> NPAProblem:
    if level < 1:
        raise ValueError("Bell objectives require one-sided level >= 1.")
    scenario = functional.scenario
    algebra = scenario.algebra()
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

    # Strong nonsignaling / CFNZ consistency: sum_a Phi_{a|x} is independent of x.
    reference_sum = sum(matrices[block_name(0, a)] for a in range(scenario.alice_outcomes))
    for x in range(1, scenario.alice_questions):
        current_sum = sum(
            matrices[block_name(x, a)] for a in range(scenario.alice_outcomes)
        )
        consistency = current_sum == reference_sum
        constraints.append(NamedConstraint(f"consistency:x{x}", consistency))

    identity_rep = block_representatives[block_name(0, 0)][()]
    norm_expression: cp.Expression = cp.Constant(0.0)
    for a in range(scenario.alice_outcomes):
        i, j = block_representatives[block_name(0, a)][()]
        norm_expression = norm_expression + matrices[block_name(0, a)][i, j]
    normalization = norm_expression == 1.0
    constraints.append(NamedConstraint("normalization", normalization))

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
                        constraint = probability >= 0.0
                        constraints.append(
                            NamedConstraint(f"probability:{x}:{y}:{a}:{b}", constraint)
                        )

    problem = cp.Problem(
        cp.Maximize(objective_expression),
        [item.constraint for item in constraints],
    )
    return NPAProblem(
        hierarchy="one_sided_pvm",
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
            "definition": "CFNZ Definition 4.1 in the reduced Bob-PVM quotient",
            "projectivized_dual_interpretations": ["KPRSTXZ sparse", "CFNZ nice"],
            "finite_level_warning": (
                "A POVM/localizer hierarchy needs an explicit finite-level "
                "dilation map before it is identified with this PVM quotient."
            ),
            "probability_positivity_added": enforce_probability_positivity,
            "identity_representative": list(identity_rep),
        },
    )
