# Reproduction guide

Run every command from the repository root with the data archive extracted.
The default validation is independent of the original working directory.

## Exact checks

```text
python scripts/run_reproduction.py
```

This performs the 21-group core checker and five independent legacy controls.
No SDP solver is used by this default workflow; floating-point exported controls
are rechecked numerically and are not promoted to exact certificates.
The full exact checker does not sample the interval identity: it reduces the
rational functions coefficient-wise and verifies all Bernstein PSD matrices.

Useful targeted checks:

```text
python scripts/higher_level_witness.py --fejer
python scripts/standard_tilted_sos.py
python scripts/quantum_interval.py
python scripts/quantum_face.py
python scripts/randomness_cost.py
```

## Tests and optional numerical regeneration

```text
python -m unittest discover -s tests
python scripts/run_reproduction.py --full
```

The first command includes rejection tests for corrupted identities and positivity
claims. `--full` also reruns historical numerical searches/controls and may be
slow. Solvers CLARABEL/SCS are pinned in requirements-lock.txt. Solver results
can vary by platform; exact certificate verification uses the archived payloads.
`--skip-slow` omits checks and is not a substitute for complete verification.

## Result / artifact / generator

| Result | Data path | Rebuild command |
| --- | --- | --- |
| Fejer family audit | artifacts/prl/fejer/exact_validation.json | python scripts/higher_level_witness.py --fejer --output artifacts/prl/fejer/exact_validation.json |
| Exact interval | artifacts/prl/interval/level3_interval.json | python scripts/run_quantum_interval.py |
| Five optimal-face points | artifacts/prl/quantum_face/ | python scripts/run_quantum_face.py --alpha 1 --level 2 (and the other four paper parameters) |
| Randomness witness | artifacts/prl/randomness/cost.json | python scripts/randomness_cost.py --write |
| Numerical guessing curve | artifacts/prl/randomness/curve.json | python scripts/run_randomness_curve.py |
| Numerical degree survey | artifacts/prl/fejer/numerical_saturation.json | python scripts/survey_conditioned_degree.py |

To rebuild all five point certificates, use `run_quantum_face.py` with
`--alpha 1 --level 2`, `--alpha 5/4 --level 2`, `--alpha 41/32 --level 2`,
`--alpha 13/10 --level 3`, and `--alpha 3/2 --level 3` (one invocation each).
After the default degree survey, `python scripts/survey_conditioned_degree.py --audit`
adds the independently assembled checks at the sensitive numerical points.

## Figures

```text
python scripts/plot_unbounded_cost.py
python scripts/plot_randomness_cost.py
python scripts/make_prl_figure.py
python scripts/make_figures.py
```

The first three use stored data; `make_figures.py` also performs numerical
calculations. Its `fig1.pdf` is an older diagnostic plot, not the current main
Fig. 1; use `plot_unbounded_cost.py` for the current main Fig. 1. Figure outputs go into `figures/`. Matplotlib is included in the
dependency lock, unlike the earlier four-package scientific-core list.

## Limits

Native degree uses reduced PVM words and fixed Alice/Bob roles. The public code
does not assert equivalence to every POVM/localizer filtration. The compiled-game
consequence is restricted to the paper's specified nice-SOS certificate route.
Numerical threshold crossing is distinct from exact closure. None of these
programs constitutes a finite-key randomness protocol or a generic security proof.
