# Unbounded degree overhead for Alice-conditioned quantum Bell certificates

Reproducibility code accompanying the paper by Fumin Wang. Version 0.1.0.

The code compares the standard and Alice-conditioned native PVM hierarchies in
the binary Bell scenario. It checks exact rational/algebraic certificates and
rational-function interval identities, and provides SDP searches and plotting.
The principal analytic result is unbounded conditioned degree overhead despite
standard level-two exactness, with an Omega((2-alpha)^(-1/2)) lower bound.
Numerical saturation levels are tolerance-dependent observations, not exact
upper bounds. See [REPRODUCTION.md](REPRODUCTION.md) for commands and scope.

## Install and verify

Use Python 3.11 or newer (tested with CPython 3.12.9). From the repository root,
create and activate a virtual environment, then run:

```text
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Download the companion Zenodo data archive, `bell_certificate_data_v0.1.0.zip`,
and extract its contents into this repository root. The resulting layout is:

```text
scripts/       src/           tests/
artifacts/     figures/       DATA_MANIFEST.json
```

```text
python scripts/check_data.py
python scripts/run_reproduction.py
python -m unittest discover -s tests
```

The default reproduction entry runs six stages, including 21 exact-certificate
check groups. The core exact checker disables SDP solver calls and fails on
missing or invalid payloads. The unit suite also includes numerical solver
controls. Reproduction writes reports under `artifacts/reproduction/`; some
auxiliary stages rewrite derived reports. Check the archive hashes before running
these stages; re-extract the dataset for a pristine copy when needed.

## Code and data release

This repository contains code and documentation under the [MIT license](LICENSE).
The separately deposited research data and figures use CC BY 4.0. Data and
generated outputs are intentionally ignored by Git. `data_release.json` and
`DATA_MANIFEST.json` identify the exact companion data archive using SHA-256.
The Zenodo DOI and public repository URL are not yet assigned; add them here and
to CITATION.cff after creating the public records. See CITATION.cff for citation
metadata; no unpublished DOI is represented as an existing identifier.

All scientific Python source and tests were copied unchanged from the paper's
working version. `check_data.py` is the additional release-integrity checker.
Exploratory `kkt_*.py`, `run_*` searches and older controls are included for audit;
they are not prerequisites of the analytic theorem.
