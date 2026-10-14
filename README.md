# The cost of direction in quantum Bell certification

Reproducibility code by Fumin Wang, version 0.3.0 (14 October 2026).

Code: https://github.com/Fumin111994/Unbounded-degree-overhead-for-Alice-conditioned-quantum-Bell-certificates/tree/v0.3.0
Companion dataset: https://doi.org/10.5281/zenodo.23356474
Code license: MIT. Data license: CC BY 4.0.

## Reproduce

Use Python 3.12. Extract the code and data ZIPs into the same directory.
The dataset adds `data/`, `research/` JSON files, and `figures/` without
replacing Python source. All paths below are relative to that combined root.

```text
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_data.py
python verify_revision_results.py
python scripts/verify_relative_precision.py
python scripts/run_reproduction.py
python scripts/make_story_figures.py
```

Run the checksum check first: reproduction regenerates some reports.
The eight current certificate checks include deliberate corruption tests.
The analytic suite contains 28 exact checker groups in six reproduction
stages, with SDP solving disabled in the core checker. Some optional unit
tests and discovery scripts use numerical solvers; they are not proof of
exactness. See `REPRODUCTION.txt` and `VALIDATION_REPORT.json`.

## Evidence map

| Material | Public path |
| --- | --- |
| Coupled witnesses and finite auxiliary certificates | `research/bidirectional_coupling/` |
| Exact standard-three/mixed-eight certificates | `research/mixed_divergence/` |
| Relative-accuracy identities and constants | `scripts/verify_relative_precision.py` |
| Contact geometry and tangent fan | `data/contact_geometry/` |
| Finite separations and rational Grams | `data/finite_separations/` |
| Standard and conditioned analytic examples | `data/analytic/` |
| Level-one families and exactification | `data/level_one_family/`, `data/level_one_exact/` |
| Convention controls | `data/convention_checks/` |
| Conditioned closure certificates | `data/conditioned_closure/` |

The relative-accuracy law is a quantitative corollary of the same coupled
witness and exact upper construction. It is not a separate separation
mechanism. Neither a fixed-noise separation nor unbounded mixed degree
is claimed. The article and Supplementary Information supply the analytic
proofs; finite checks independently verify their algebraic ingredients.
