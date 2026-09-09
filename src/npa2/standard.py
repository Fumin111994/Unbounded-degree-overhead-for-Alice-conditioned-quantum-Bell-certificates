"""Standard dense NPA relaxation in the bipartite PVM quotient."""

from __future__ import annotations

import cvxpy as cp

from .bell import BellFunctional
from .model import (
    NPAProblem,
    NamedConstraint,
    add_moment_constraints,
    moment_expression,
)


def build_standard_problem(
    functional: BellFunctional,
    level: int,
    *,
    enforce_probability_positivity: bool = True,
) -> NPAProblem:
    if level < 1:
        raise ValueError("Bell objectives require standard level >= 1.")
    algebra = functional.scenario.algebra()
    words = algebra.words(level)
    gamma = cp.Variable((len(words), len(words)), symmetric=True, name="Gamma_std")
    psd_constraint = gamma >> 0
    constraints: list[NamedConstraint] = [
        NamedConstraint("Gamma_std:psd", psd_constraint)
    ]
    representatives = add_moment_constraints(
        algebra=algebra,
        matrix=gamma,
        words=words,
        prefix="Gamma_std",
        named_constraints=constraints,
    )
    identity_i, identity_j = representatives[()]
    normalization = gamma[identity_i, identity_j] == 1.0
    constraints.append(NamedConstraint("normalization", normalization))

    if enforce_probability_positivity:
        scenario = functional.scenario
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
                        constraint = probability >= 0.0
                        constraints.append(
                            NamedConstraint(f"probability:{x}:{y}:{a}:{b}", constraint)
                        )

    objective_expression = moment_expression(
        algebra=algebra,
        matrix=gamma,
        representatives=representatives,
        polynomial=functional.polynomial(),
    )
    problem = cp.Problem(
        cp.Maximize(objective_expression),
        [item.constraint for item in constraints],
    )
    return NPAProblem(
        hierarchy="standard_pvm",
        level=level,
        functional=functional,
        problem=problem,
        words=words,
        matrices={"Gamma_std": gamma},
        psd_constraints={"Gamma_std": psd_constraint},
        constraints=constraints,
        representatives=representatives,
        metadata={
            "native_level_meaning": "maximum total reduced word length",
            "definition": "standard dense NPA over the reduced bipartite PVM algebra",
            "probability_positivity_added": enforce_probability_positivity,
        },
    )
