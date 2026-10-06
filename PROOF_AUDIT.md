# Pre-release proof audit, 2026-10-04

Scope: the sharp asymmetric-weight transition (main Theorem 2 / SM Theorem
9.9), the scalar geometric sufficient test, and their degree-preserving
SOS dependency. This is a mathematical review with executable algebra
checks, not a machine-checked formal proof or an external referee report.

## Mathematical conclusion

No substantive error was found in the reviewed proofs. In particular:

1. Symmetrization preserves the separator sum and exchanges, rather than
   discards, the two inequalities involving Y. The resulting real P(X) has
   degree at most 2k because the reduction uses XY=-YX and Y^2=1-X^2.
2. The Markov lower bound uses coefficient compactness at a *common fixed
   degree* along subcritical tilts. The derivative contacts at the endpoint
   are inherited from this limit, not inferred from the product-state
   attaining strategy at the critical tilt. Normalizing E by
   2 lambda g (1-c) gives degree at most 2k-1 and derivative 1/2-1/g.
3. The quartic discriminant margin is nonnegative for t>=5/4 and rho<=1/t.
   The determinant condition plus Q(0)>0 establishes both diagonal signs.
   Necessity and sufficiency therefore give K*(lambda)=2 if and only if
   lambda>=sqrt(5)/2. The critical tilt alone is not asserted to have this
   uniform-degree lower bound.
4. The exterior pole is separated from the spectrum: sigma<=d<=8 sigma/3.
   The weighted determinant correction is nonnegative. The one-sided
   polynomial error decreases each Q diagonal by at most one half and its
   two linear determinant losses consume at most the old slack. The
   undivided determinant identity extends continuously to zero diagonal
   endpoints. This justifies the conservative constant 15/(2 sqrt(g))+3;
   no claim of an optimal leading constant is made.
5. In the scalar geometric test, Q0>=gamma D is a relative margin, not an
   absolute positive lower bound. Subtracting 2 tau D gives determinant at
   least eta D/2 and smallest eigenvalue at least eta D/(4M). The kink gap
   and a_min>0 make the Hermite remainder jointly continuous, including
   contacts at the spectral endpoints. This is only a sufficient test.
6. The positivity-to-SOS step uses the closed finite-dimensional Gram cone
   and positive-definite extension on the Cayley path. Proposition 3.4 of
   Bakonyi--Timotin applies to the infinite dihedral group with exactly
   this word-ball domain; the primary source was checked:
   https://arxiv.org/pdf/0808.1273 (Proposition 3.4).

The supplement now spells out the symmetry action, the generalized
spectral bound, and the determinant perturbation without division at zero
diagonals. The main theorem explicitly states the limit lambda down to 1.

## Independent executable checks

`scripts/verify_asymmetric_transition.py` now has 20 audit groups. In
addition to symbolic factorization and rational constant checks, it builds
the actual block diagonals and determinants independently of their claimed
factorizations and uses square-free decomposition plus exact Sturm counts.

- Exact threshold t=5/4, rho=4/5: all four Bob blocks are PSD on [-1,1].
- lambda=11/10, alpha=20/11, a=1: degree-25 pole-cancelled separator;
  contact identities and global positivity are checked exactly.
- lambda=11/10, alpha=3580/2009, a=1991/2009: a genuinely interior
  degree-25 separator; the same global checks pass.
- lambda=101/100, alpha=200/101: degree-76 polynomial; exact pole
  cancellation, contact identities and kernel constants are checked.
  Its positivity is covered by the analytic error-budget proof, not by
  a claimed direct Sturm run of that high-degree polynomial.

The interval sign checker includes positive and negative controls with
repeated roots, endpoint roots, and a narrow negative interval. No grid or
floating-point feasibility threshold is used. These finite audits do not
prove the compactness or Markov arguments and are not presented as such.

Full release reproduction and file-integrity results are stored with the
release, rather than inferred from earlier runs. Historical numerical
controls retain their original evidential status.

## Compilation and publication boundary

The built-in editor compiler was attempted after editing main.tex. It
reported `Unable to find standard directories for platform`, an environment
failure. No separate PDF was compiled in this audit. Previously generated
PDFs are not represented as newly verified outputs. The saved TeX sources
remain authoritative, and the existing editor stays open.

A local release archive is not a public release. Public commit/tag and
version-specific Zenodo DOI must be confirmed before changing the existing
public identifiers or removing the manuscript's version-boundary notice.
