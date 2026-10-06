# Reproduction for version 0.2.0

The default `python scripts/run_reproduction.py` is the solver-free entry.
Its six stages include the complete exact checker (28 groups), stored control
matrix verification, exact level-one checks, standard-family identities,
anchor coverage, and exact level-two witness checks.

The seven added analytic audit scripts, all included in this release, are:

```text
python scripts/verify_precision_bounds.py
python scripts/verify_fixed_tilt_closure.py
python scripts/verify_matching_upper.py
python scripts/verify_orientation_structure.py
python scripts/verify_orientation_counterexample.py
python scripts/verify_asymmetric_family.py
python scripts/verify_asymmetric_transition.py
```

`verify_asymmetric_transition.py` checks 20 exact audit groups,
including the quartic threshold factorization, Markov normalization and
three complete polynomial pole cancellations. Exact Sturm checks additionally
establish global block positivity at the sharp threshold and for two degree-25
certificates (one interior tilt and one critical tilt). See PROOF_AUDIT.md. The compactness, symmetry and
Markov steps are analytic arguments in the supplement.

`python scripts/run_reproduction.py --full` additionally invokes SDP solvers
for numerical controls and unit tests. Numerical searches and plots have
different evidential status from exact symbolic and rational checks.

The data release inherits the curated 0.1.0 certificate catalog and adds all
current `artifacts/prl` JSON outputs. Historic numerical experiments remain
labeled as such. The current environment uses cvxpy 1.8.2, not 1.9.1;
requirements-lock.txt records the exact dependency versions.

Release validation is run from a separate combined copy of these code/data
files. The archived run passed all six stages and all 28 exact groups, without skips.
Reports are included under artifacts/reproduction/release_0.2.0 in the data ZIP.
