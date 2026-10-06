# Contact geometry and sharp degree costs of quantum Bell certificates

Reproducibility code by Fumin Wang. Version 0.2.0, audited 2026-10-04.
Code version: [v0.2.0](https://github.com/Fumin111994/Unbounded-degree-overhead-for-Alice-conditioned-quantum-Bell-certificates/releases/tag/v0.2.0).
Companion data version 0.2.0: https://doi.org/10.5281/zenodo.23179029
Previous data version: https://doi.org/10.5281/zenodo.22672612

## Main results checked

- Matching exact-degree cost Theta((2-alpha)^(-1/2)) for Alice-tilted CHSH.
- Sharp uniform Bell/randomness precision Theta(k^(-4)).
- Exact degree one for the opposite tilt; a Bob-marginal counterexample.
- Contact-preserving approximation and a scalar geometric sufficient test.
- Asymmetric whole-family cost K*(lambda)=Theta((lambda-1)^(-1/2)).
- Exact whole-family level-two threshold lambda=sqrt(5)/2.

The analytic proofs are in the accompanying manuscript and supplement.
Symbolic audits check their identities and exact constants; finite audits
are not a substitute for the universal arguments. Numerical SDP saturation
at a fixed tolerance is not a claim of exact closure.

## Install and reproduce

Python 3.12.9 was used. In a fresh environment, from this repository root:

```text
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Extract `bell_certificate_data_v0.2.0.zip` into the same directory so that
`artifacts/` and `figures/` are alongside `scripts/` and `src/`. Then run:

```text
python scripts/check_data.py
python scripts/run_reproduction.py
```

The first command checks the pristine dataset manifest. The second runs six
stages, including all 28 groups in `verify_certificates.py`; that core checker
patches `cvxpy.Problem.solve` to raise. Missing certificates are failures.
Some legacy stages regenerate reports: verify hashes BEFORE reproduction.
See `REPRODUCTION.md` for scope and individual commands.

Code license: MIT. Companion data and figures: CC BY 4.0.
