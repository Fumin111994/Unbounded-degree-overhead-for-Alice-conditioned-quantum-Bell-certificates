#!/usr/bin/env python3
"""M6 exactification: exact dual closure certificates at the quantum bound.

For rational tilt alpha below the critical tilt, the one-sided (CFNZ Def. 4.1)
level-2 value equals the quantum value q(alpha) = sqrt(8 + 2 alpha^2).  This
script produces an EXACT, solver-free dual certificate of

    omega_os^2(alpha) <= q(alpha)

i.e. exact data (S_b, Lambda, nu) with the nice-SOS identity

    t - F(m) = sum_b Tr(S_b Phi_b(m))
             + Tr(Lambda (sum_a Phi_{1,a} - sum_a Phi_{0,a}))
             + nu (sum_a m_{0,a,()} - 1)

holding term-by-term at t = q(alpha) (an algebraic number), with every 5x5
S block exactly positive semidefinite.

Method (per alpha):

1. Solve the level-2 SDP with CLARABEL for a double-precision seed (only used
   as a Newton starting point).
2. Build the exact rational identity system Ap u = b(t) in the reduced dual
   variables u = (S entries, y multipliers, nu); the consistency multiplier
   Lambda is compressed to the 7 independent moment positions (symmetric
   filling), removing the multiplier kernel of the identity.
3. Track the analytic center of {Ap u = b(t), S_b positive definite} from
   t = t_seed + 1e-4 down to t = q + delta (delta ~ 1e-60) with a damped
   Newton homotopy in mpmath at high precision (default 250 dps).  The
   damping keeps every S_b inside the positive definite cone.  The limit
   point is the analytic center of the optimal face; its entries are
   algebraic over QQ(q).
4. Identify every entry with algdep/PSLQ (zero, rational, QQ(sqrt d)
   degree 1, then algdep over QQ and over QQ(sqrt d)).
5. Verify EXACTLY with sympy: identity residuals vanish at t = q, and each
   S block passes npa2.exact.is_psd_exact (LDL over the algebraic field).

If identification fails for some entry, the certificate is marked
"unverified" and the high-precision transitional data (entries accurate to
~delta, Newton residuals) is stored instead, per project fallback policy.

Idempotent: artifacts/m6/closure_exact.json is rewritten deterministically
on every run (per-alpha keys replaced, other keys preserved).

Usage:
    python scripts/m6_exactify_closure.py                 # default alphas
    python scripts/m6_exactify_closure.py --alphas 5/4    # subset
    python scripts/m6_exactify_closure.py --dps 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from npa2 import exact  # noqa: E402
from npa2.bell import BellFunctional, Scenario, tilted_chsh  # noqa: E402
from npa2.onesided import build_onesided_problem  # noqa: E402
from npa2.solve import solve_problem  # noqa: E402

NWORDS = 5
BLOCK_WORD_POSITIONS = [(i, j) for i in range(NWORDS) for j in range(i, NWORDS)]
N_S_ENTRIES = 60          # 4 blocks x 15 symmetric positions
N_Y = 7                   # independent consistency multipliers (rank of C)
N_U = N_S_ENTRIES + N_Y + 1   # 68 reduced dual variables
N_KKT = N_U + 29          # + 29 identity-row multipliers rho


# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------


def exact_tilted_functional(alpha: Fraction) -> BellFunctional:
    """Tilted CHSH functional alpha*A_0 + CHSH with exact rational data."""

    signs = {0: Fraction(1), 1: Fraction(-1)}
    joint = {}
    for (x, y), c in {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1}.items():
        for a in (0, 1):
            for b in (0, 1):
                joint[(x, y, a, b)] = Fraction(c) * signs[a] * signs[b]
    alice_local = {(0, a): alpha * signs[a] for a in (0, 1)}
    return BellFunctional(
        name=f"tilted_chsh_exact_alpha_{alpha}",
        scenario=Scenario(2, 2, 2, 2),
        alice_local=alice_local,
        joint=joint,
    )


def quantum_alg(alpha: Fraction) -> sp.Expr:
    return sp.sqrt(8 + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2)


def rational_squarefree(num: int, den: int) -> int:
    """Squarefree d with num/den = (square) * d."""

    product = num * den
    d = 1
    for prime, exponent in sp.factorint(product).items():
        if exponent % 2 == 1:
            d *= prime
    return d


class IdentityData:
    """Exact rational identity system in the reduced dual variables.

    Variables u (length 68): S entries (4 blocks x 15, positions i<=j),
    y (7 independent consistency multipliers), nu.
    Identity: Ap u = b(t) = b0 + t*cvec (29 rows).
    """

    def __init__(self, functional: BellFunctional):
        data = exact.onesided_certificate_unknowns(functional, 2, False)
        self.data = data
        self.block_names = data["block_names"]
        self.sym_keys = list(data["symbols"].keys())
        sym_index = {k: i for i, k in enumerate(self.sym_keys)}
        _, A, b0 = exact.onesided_identity_system(functional, 2, Fraction(0), False)
        _, _, b1 = exact.onesided_identity_system(functional, 2, Fraction(1), False)
        self.A = A
        self.b0 = b0
        self.cvec = b1 - b0

        # Independent consistency rows: elementwise rows of
        # sum_a Phi_{1,a}[i,j] - sum_a Phi_{0,a}[i,j] = 0 in the 28 moments.
        entry_key = data["entry_key"][(0, 0)]
        rows = []
        for (i, j) in BLOCK_WORD_POSITIONS:
            key = entry_key[(i, j)]
            row = [0] * 28
            if key is not None:
                for a in (0, 1):
                    row[sym_index[(1, a, key)]] += 1
                    row[sym_index[(0, a, key)]] -= 1
            rows.append(row)
        _, pivots = sp.Matrix(rows).T.rref()
        self.selected_rows = sorted(pivots)
        self.selected_positions = [BLOCK_WORD_POSITIONS[k] for k in self.selected_rows]
        assert len(self.selected_positions) == N_Y

        def lam_index(i: int, j: int) -> int:
            return i * NWORDS + j

        y_columns = []
        for (i, j) in self.selected_positions:
            column = A[:, 60 + lam_index(i, j)]
            if j != i:
                column = column + A[:, 60 + lam_index(j, i)]
            y_columns.append(column)
        y_matrix = sp.Matrix.hstack(*y_columns)
        # Reduced identity matrix: 29 x 68
        self.Ap = sp.Matrix.hstack(A[:, :60], y_matrix, A[:, 85:86])
        assert self.Ap.rank() == 29

    def unpack_u(self, values):
        """values: sequence of 68 -> (s_blocks, lam, nu) in exact.py layout."""

        s_blocks: dict[tuple[int, int], dict[tuple[int, int], object]] = {}
        offset = 0
        for b in self.block_names:
            entries: dict[tuple[int, int], object] = {}
            for (i, j) in BLOCK_WORD_POSITIONS:
                entries[(i, j)] = values[offset]
                offset += 1
            s_blocks[b] = entries
        y = values[60:67]
        nu = values[67]
        lam_entries: dict[tuple[int, int], object] = {}
        for (i, j), value in zip(self.selected_positions, y):
            lam_entries[(i, j)] = value
            if j != i:
                lam_entries[(j, i)] = value
        return s_blocks, {1: lam_entries}, nu


def entry_names(identity: IdentityData) -> list[str]:
    names = []
    for b in identity.block_names:
        for (i, j) in BLOCK_WORD_POSITIONS:
            names.append(f"S_{b[0]}_{b[1]}:{i}:{j}")
    names += [f"y:{i}:{j}" for (i, j) in identity.selected_positions]
    names.append("nu")
    return names


# ---------------------------------------------------------------------------
# Double-precision seed
# ---------------------------------------------------------------------------


def double_seed(alpha: Fraction, solver: str = "CLARABEL") -> dict:
    functional = tilted_chsh(float(alpha))
    model = build_onesided_problem(functional, 2, enforce_probability_positivity=False)
    summary, arrays = solve_problem(model, solver)
    cons_dual = {
        item.name: np.asarray(item.constraint.dual_value)
        for item in model.constraints
        if item.constraint.dual_value is not None
    }
    return {"summary": summary, "arrays": arrays, "cons_dual": cons_dual}


def seed_u_vector(identity: IdentityData, seed: dict) -> tuple[list[float], float]:
    arrays = seed["arrays"]
    s_values: list[float] = []
    for (x, a) in identity.block_names:
        matrix = arrays[f"dual_psd__Phi_x{x}_a{a}"]
        matrix = 0.5 * (matrix + matrix.T)
        for (i, j) in BLOCK_WORD_POSITIONS:
            s_values.append(float(matrix[i, j]))
    lam = -seed["cons_dual"]["consistency:x1"]
    y_values = [
        float(0.5 * (lam[i, j] + lam[j, i]))
        for (i, j) in identity.selected_positions
    ]
    nu_value = -float(seed["cons_dual"]["normalization"])
    t_value = float(seed["summary"]["value"])
    return s_values + y_values + [nu_value], t_value


# ---------------------------------------------------------------------------
# High-precision analytic-center path
# ---------------------------------------------------------------------------


class AnalyticCenterTracker:
    """Damped Newton tracker for the analytic center of {Ap u = b(t), S_b > 0}.

    Solves the KKT system
        grad B(u) + Ap^T rho = 0,   Ap u = b0 + t*cvec,
    with B(u) = sum_b log det S_b, in 97 unknowns (u, rho).  Damping halves
    the Newton step until every S block stays positive definite (Sylvester
    criterion), so the trajectory is a valid barrier-interior path.
    """

    def __init__(self, identity: IdentityData, dps: int):
        self.identity = identity
        self.dps = dps
        self.old_dps = mp.mp.dps
        mp.mp.dps = dps
        Ap = identity.Ap
        self.Apm = mp.matrix(
            [[mp.mpf(int(Ap[i, j].numerator)) / int(Ap[i, j].denominator)
              for j in range(Ap.cols)] for i in range(Ap.rows)]
        )
        self.b0m = mp.matrix(
            [mp.mpf(int(v.numerator)) / int(v.denominator) for v in identity.b0]
        )
        self.cvm = mp.matrix(
            [mp.mpf(int(v.numerator)) / int(v.denominator) for v in identity.cvec]
        )
        # Right inverse Ap^+ = Ap^T (Ap Ap^T)^{-1} (29 independent rows).
        gram = self.Apm * self.Apm.T
        gram_inv = mp.inverse(gram)
        self.ap_plus = self.Apm.T * gram_inv          # 68 x 29
        self.ap_plus_t = gram_inv * self.Apm          # 29 x 68

    def restore_dps(self):
        mp.mp.dps = self.old_dps

    def project_to_manifold(self, u, t_value):
        """Exact projection of u onto {Ap u = b(t)} via the right inverse."""

        return u + self.ap_plus * ((self.b0m + t_value * self.cvm) - self.Apm * u)

    def _unpack_s(self, u) -> dict:
        blocks = {}
        offset = 0
        for b in self.identity.block_names:
            matrix = mp.matrix(NWORDS, NWORDS)
            for (i, j) in BLOCK_WORD_POSITIONS:
                matrix[i, j] = u[offset]
                matrix[j, i] = u[offset]
                offset += 1
            blocks[b] = matrix
        return blocks

    @staticmethod
    def _is_positive_definite(matrix) -> bool:
        try:
            mp.cholesky(matrix)
            return True
        except (ZeroDivisionError, ValueError):
            return False

    def _all_positive_definite(self, u) -> bool:
        return all(
            self._is_positive_definite(block) for block in self._unpack_s(u).values()
        )

    def _inverses(self, u) -> dict:
        return {b: matrix ** -1 for b, matrix in self._unpack_s(u).items()}

    @staticmethod
    def _gradient_from_inverses(inverses, block_names):
        gradient = mp.matrix(N_U, 1)
        offset = 0
        for b in block_names:
            inverse = inverses[b]
            for (i, j) in BLOCK_WORD_POSITIONS:
                gradient[offset] = inverse[i, j] * (1 if i == j else 2)
                offset += 1
        return gradient

    @staticmethod
    def _hessian_from_inverses(inverses, block_names):
        hessian = mp.matrix(N_U, N_U)
        for block_index, b in enumerate(block_names):
            inverse = inverses[b]
            base = block_index * 15
            for column, (k, l) in enumerate(BLOCK_WORD_POSITIONS):
                basis = mp.matrix(NWORDS, NWORDS)
                basis[k, l] = 1
                basis[l, k] = 1
                derivative = -(inverse * basis * inverse)
                for row, (i, j) in enumerate(BLOCK_WORD_POSITIONS):
                    hessian[base + row, base + column] = (
                        derivative[i, j] * (1 if i == j else 2)
                    )
        return hessian

    def _barrier_gradient(self, u):
        return self._gradient_from_inverses(
            self._inverses(u), self.identity.block_names
        )

    def residual(self, v, t_value):
        u = v[:N_U]
        rho = v[N_U:N_KKT]
        stationarity = self._barrier_gradient(u) + self.Apm.T * rho
        identity_residual = self.Apm * u - (self.b0m + t_value * self.cvm)
        result = mp.matrix(N_KKT, 1)
        for i in range(N_U):
            result[i] = stationarity[i]
        for i in range(29):
            result[N_U + i] = identity_residual[i]
        return result

    def jacobian(self, v):
        u = v[:N_U]
        jacobian = mp.matrix(N_KKT, N_KKT)
        hessian = self._barrier_hessian(u)
        for i in range(N_U):
            for j in range(N_U):
                jacobian[i, j] = hessian[i, j]
        for i in range(N_U):
            for j in range(29):
                jacobian[i, N_U + j] = self.Apm[j, i]
                jacobian[N_U + j, i] = self.Apm[j, i]
        return jacobian

    def _barrier_hessian(self, u):
        return self._hessian_from_inverses(
            self._inverses(u), self.identity.block_names
        )

    def _rho_least_squares(self, u):
        """Best rho for the stationarity residual at u: Ap^T rho = -grad B."""

        return -self.ap_plus_t * self._barrier_gradient(u)

    def newton(self, v, t_value, max_iterations: int = 500):
        """Self-concordant damped Newton for the analytic center at t_value.

        Solves the full KKT system (stationarity + identity residual) with
        step size 1/(1 + lambda), lambda^2 = du^T H du (globally convergent
        for the log-det barrier, Nesterov--Nemirovski); rho is re-synced by
        least squares each step.  Never projects onto the manifold: for
        large t moves the minimal-norm projection leaves the PSD cone.
        """

        tolerance = mp.mpf(10) ** (-(self.dps - 30))
        u = mp.matrix(v[:N_U])
        rho = self._rho_least_squares(u)
        b_t = self.b0m + t_value * self.cvm
        for iteration in range(max_iterations):
            inverses = self._inverses(u)
            gradient = self._gradient_from_inverses(
                inverses, self.identity.block_names
            )
            stationarity = gradient + self.Apm.T * rho
            identity_residual = self.Apm * u - b_t
            norm = max(
                mp.norm(stationarity, p=mp.inf),
                mp.norm(identity_residual, p=mp.inf),
            )
            if norm < tolerance:
                v = mp.matrix(N_KKT, 1)
                for i in range(N_U):
                    v[i] = u[i]
                for i in range(29):
                    v[N_U + i] = rho[i]
                return v, iteration, norm, True
            hessian = self._hessian_from_inverses(
                inverses, self.identity.block_names
            )
            jacobian = mp.matrix(N_KKT, N_KKT)
            for i in range(N_U):
                for j in range(N_U):
                    jacobian[i, j] = hessian[i, j]
            for i in range(N_U):
                for j in range(29):
                    jacobian[i, N_U + j] = self.Apm[j, i]
                    jacobian[N_U + j, i] = self.Apm[j, i]
            rhs = mp.matrix(N_KKT, 1)
            for i in range(N_U):
                rhs[i] = -stationarity[i]
            for i in range(29):
                rhs[N_U + i] = -identity_residual[i]
            step = mp.lu_solve(jacobian, rhs)
            du = step[:N_U]
            hdu = hessian * du
            lam2 = mp.mpf(0)
            for i in range(N_U):
                lam2 += du[i] * hdu[i]
            decrement = mp.sqrt(abs(lam2))
            scale = 1 / (1 + decrement)
            # safety: never leave the positive definite cone
            for _ in range(80):
                trial = u + scale * du
                if self._all_positive_definite(trial[:N_U]):
                    u = trial
                    break
                scale /= 2
            else:
                v_out = mp.matrix(N_KKT, 1)
                for i in range(N_U):
                    v_out[i] = u[i]
                return v_out, iteration, norm, False
            rho = self._rho_least_squares(u)
        v = mp.matrix(N_KKT, 1)
        for i in range(N_U):
            v[i] = u[i]
        for i in range(29):
            v[N_U + i] = rho[i]
        final = max(
            mp.norm(self._barrier_gradient(u) + self.Apm.T * rho, p=mp.inf),
            mp.norm(self.Apm * u - b_t, p=mp.inf),
        )
        return v, max_iterations, final, False


# ---------------------------------------------------------------------------
# Algebraic identification
# ---------------------------------------------------------------------------


def _crootof_from_relation(coeffs: list[int], target) -> sp.Expr:
    """Exact CRootOf matching the numeric target, from an integer relation."""

    variable = sp.Symbol("x")
    poly = sp.Poly([int(c) for c in coeffs], variable)
    roots = sp.Poly(poly.as_expr(), variable).all_roots(radicals=False)
    target_complex = complex(target)

    def distance(root) -> float:
        try:
            return abs(complex(sp.N(root, 40)) - target_complex)
        except (TypeError, ValueError):
            return float("inf")

    return min(roots, key=distance)


def identify_entry(value, d: int, sqrt_d, digits_budget: int,
                   max_degree_qq: int, max_degree_ext: int) -> dict:
    """Identify one mpf value as an exact algebraic number."""

    zero_tol = mp.mpf(10) ** (-(digits_budget - 5))
    if abs(value) < zero_tol:
        return {"method": "zero", "expr": sp.Integer(0), "minpoly": "x"}

    relation = mp.pslq([value, mp.mpf(1)], tol=zero_tol,
                       maxcoeff=10 ** (digits_budget // 3))
    if relation is not None:
        a, b = (int(r) for r in relation)
        return {"method": "rational", "expr": sp.Rational(-b, a),
                "minpoly": f"{a}*x + ({b})"}

    relation = mp.pslq([value, sqrt_d, mp.mpf(1)], tol=zero_tol,
                       maxcoeff=10 ** (digits_budget // 4))
    if relation is not None:
        a, b, c = (int(r) for r in relation)
        expr = sp.nsimplify(0) + (-b * sp.sqrt(d) - c) / a
        return {"method": "sqrt-field", "expr": expr,
                "minpoly": f"{a}*x + ({b})*sqrt({d}) + ({c})"}

    for degree in (2, 3, 4, 6, 8, 12, 16):
        if degree > max_degree_qq:
            break
        powers = [value ** k for k in range(degree, -1, -1)]
        relation = mp.pslq(powers, tol=zero_tol,
                           maxcoeff=10 ** max(2, digits_budget // (degree + 2)))
        if relation is not None:
            coeffs = [int(r) for r in relation]
            expr = _crootof_from_relation(coeffs, value)
            poly = sp.Poly(coeffs, sp.Symbol("x")).as_expr()
            return {"method": f"algdep-qq-deg{degree}", "expr": expr,
                    "minpoly": str(poly)}

    for degree in (2, 3, 4, 6, 8):
        if degree > max_degree_ext:
            break
        powers = []
        for k in range(degree, -1, -1):
            powers.append(value ** k)
            powers.append(sqrt_d * value ** k)
        relation = mp.pslq(powers, tol=zero_tol,
                           maxcoeff=10 ** max(2, digits_budget // (2 * degree + 2)))
        if relation is not None:
            # relation over QQ(sqrt d): sum (c_k + e_k sqrt d) value^k = 0
            coeffs = [int(r) for r in relation]
            variable, root_d = sp.symbols("x rd")
            pairs = [(coeffs[2 * i], coeffs[2 * i + 1])
                     for i in range(degree + 1)]
            poly_ext = sum(
                (c + e * root_d) * variable ** (degree - i)
                for i, (c, e) in enumerate(pairs)
            )
            norm_poly = sp.expand(
                poly_ext * poly_ext.subs(root_d, -root_d)
            ).subs(root_d ** 2, d).subs(root_d, 0)
            norm_coeffs = sp.Poly(norm_poly, variable).all_coeffs()
            expr = _crootof_from_relation([int(c) for c in norm_coeffs], value)
            return {"method": f"algdep-ext-deg{degree}", "expr": expr,
                    "minpoly": str(norm_poly)}

    return {"method": "unidentified", "expr": None, "minpoly": None}


# ---------------------------------------------------------------------------
# Main per-alpha pipeline
# ---------------------------------------------------------------------------


def process_alpha(alpha: Fraction, args) -> dict:
    functional = exact_tilted_functional(alpha)
    q_alg = quantum_alg(alpha)
    q_float = float(q_alg)
    numerator, denominator = sp.fraction(
        sp.Rational(8) + 2 * sp.Rational(alpha.numerator, alpha.denominator) ** 2
    )
    d = rational_squarefree(int(numerator), int(denominator))

    report: dict = {
        "kind": "one-sided level-2 exact dual closure certificate at the quantum bound",
        "alpha": str(alpha),
        "bound": str(q_alg),
        "bound_float": q_float,
        "squarefree_d": d,
        "status": "unverified",
    }

    t_start = time.time()
    identity = IdentityData(functional)
    report["identity_system"] = {
        "rows": int(identity.Ap.rows),
        "reduced_variables": int(identity.Ap.cols),
        "selected_lambda_positions": [list(p) for p in identity.selected_positions],
    }

    seed = double_seed(alpha, args.solver)
    solver_value = float(seed["summary"]["value"])
    report["solver_seed"] = {
        "solver": args.solver,
        "value": solver_value,
        "quantum_float": q_float,
        "value_minus_quantum": solver_value - q_float,
    }
    if abs(solver_value - q_float) > 1e-5:
        report["failures"] = [f"solver value {solver_value} not at quantum {q_float}"]
        return report

    u_seed, t_seed = seed_u_vector(identity, seed)

    tracker = AnalyticCenterTracker(identity, args.dps)
    names = entry_names(identity)
    try:
        q_mp = mp.sqrt(8 + 2 * (mp.mpf(alpha.numerator) / alpha.denominator) ** 2)
        # Start: damped Newton from the raw solver seed (S blocks positive
        # definite) at t1 = t_seed + 1e-6, just above the quantum value.
        # No manifold projection: for large t moves the minimal-norm
        # projection leaves the PSD cone; the full-residual Newton handles
        # the off-manifold start directly.
        t1 = mp.mpf(t_seed) + mp.mpf("1e-6")
        v = mp.matrix(N_KKT, 1)
        for i, value in enumerate(u_seed):
            v[i] = mp.mpf(value)
        v, iterations, residual_norm, ok = tracker.newton(v, t1)
        path_log = [{
            "t_minus_q": mp.nstr(t1 - q_mp, 5), "iterations": iterations,
            "residual": mp.nstr(residual_norm, 3), "converged": bool(ok),
        }]
        if not ok:
            report["failures"] = ["Newton did not converge at the initial analytic center"]
            report["path_log"] = path_log
            return report
        converged = True
        last_good_v = v
        last_good_delta = t1 - q_mp
        prev_v = None
        prev_logdelta = None
        cur_logdelta = mp.log10(t1 - q_mp)
        for exponent in args.delta_schedule:
            delta = mp.mpf(10) ** exponent
            if delta >= t1 - q_mp:
                continue
            # predictor: linear extrapolation of the path in log10(delta)
            v_start = v
            if prev_v is not None:
                ratio = (cur_logdelta - exponent) / (cur_logdelta - prev_logdelta)
                candidate = v + ratio * (v - prev_v)
                if tracker._all_positive_definite(candidate[:N_U]):
                    v_start = candidate
            prev_v = v
            prev_logdelta = cur_logdelta
            v, iterations, residual_norm, ok = tracker.newton(v_start, q_mp + delta)
            path_log.append({
                "t_minus_q": mp.nstr(delta, 3), "iterations": iterations,
                "residual": mp.nstr(residual_norm, 3), "converged": bool(ok),
            })
            if not ok:
                if residual_norm < mp.mpf("1e-100"):
                    # The barrier system stalls on the degenerate optimal
                    # face, but a residual this small leaves >100 correct
                    # digits: accept the iterate for identification.
                    cur_logdelta = mp.mpf(exponent)
                    last_good_v = v
                    last_good_delta = delta
                    converged = True
                else:
                    converged = False
                break
            cur_logdelta = mp.mpf(exponent)
            last_good_v = v
            last_good_delta = delta
        report["path_log"] = path_log
        if not converged:
            report["failures"] = ["homotopy Newton failed", path_log[-1]]
            # keep the best converged high-precision point for the record
            report["high_precision"] = {
                "dps": args.dps,
                "delta_final": mp.nstr(last_good_delta, 3),
                "note": (
                    "homotopy incomplete; entries are the analytic center at "
                    "t = q + delta_final (NOT the optimal face; not usable "
                    "for the t = q certificate)"
                ),
                "entries": {
                    name: mp.nstr(last_good_v[k], 50) for k, name in enumerate(names)
                },
            }
            return report

        hp_values = {name: v[k] for k, name in enumerate(names)}
        final_delta = mp.mpf(10) ** args.delta_schedule[-1]
        report["high_precision"] = {
            "dps": args.dps,
            "delta_final": mp.nstr(final_delta, 3),
            "entries": {name: mp.nstr(value, 50) for name, value in hp_values.items()},
        }

        # budget from the achieved iterate precision (~-log10(residual)),
        # not from the homotopy depth reached before the barrier stalls
        digits_budget = min(args.dps - 60, 100)
        sqrt_d = mp.sqrt(d)
        identification: dict[str, dict] = {}
        for name in names:
            identification[name] = identify_entry(
                hp_values[name], d, sqrt_d, digits_budget,
                args.max_degree_qq, args.max_degree_ext,
            )
    finally:
        tracker.restore_dps()

    report["identification"] = {
        name: {"method": record["method"], "minpoly": record["minpoly"],
               "value": None if record["expr"] is None else str(record["expr"])}
        for name, record in identification.items()
    }
    unidentified = [
        name for name, record in identification.items()
        if record["method"] == "unidentified"
    ]
    if unidentified:
        report["failures"] = [
            f"algdep failed for {len(unidentified)} entries: {unidentified[:8]}"
        ]
        return report

    # Cross-check: identified exact values must reproduce the hp numbers.
    mismatches = []
    mp.mp.dps = 50
    for name, record in identification.items():
        deviation = abs(mp.mpf(sp.N(record["expr"], 50)) - mp.mpf(hp_values[name]))
        if deviation > mp.mpf(10) ** (-(digits_budget - 15)):
            mismatches.append((name, mp.nstr(deviation, 3)))
    mp.mp.dps = 15
    if mismatches:
        report["failures"] = [f"identified values mismatch hp point: {mismatches[:5]}"]
        return report

    ordered_exact = [identification[name]["expr"] for name in names]
    s_blocks, lam, nu = identity.unpack_u(ordered_exact)
    data = identity.data
    residuals = exact._onesided_identity_residuals(
        functional, 2, q_alg, s_blocks, lam, nu, {},
        data["symbols"], data["entry_key"], data["block_names"],
        data["words"], data["algebra"], data["scenario"],
    )
    nonzero_residuals = [str(r) for r in residuals if sp.simplify(r) != 0]
    verification: dict = {"identity_residuals_zero": not nonzero_residuals}
    if nonzero_residuals:
        verification["residuals"] = nonzero_residuals[:5]

    psd_report = {}
    for b in identity.block_names:
        matrix = sp.zeros(NWORDS)
        for (i, j), value in s_blocks[b].items():
            matrix[i, j] = sp.sympify(value)
            matrix[j, i] = matrix[i, j]
        psd_report[str(b)] = {
            "psd": bool(exact.is_psd_exact(matrix)),
            "det": str(sp.simplify(matrix.det())),
            "min_eigenvalue_float": float(
                np.linalg.eigvalsh(np.array(matrix.tolist(), dtype=float))[0]
            ),
        }
    verification["s_blocks"] = psd_report
    verification["all_psd"] = all(item["psd"] for item in psd_report.values())
    report["verification"] = verification

    if verification["identity_residuals_zero"] and verification["all_psd"]:
        report["status"] = "verified"
        report["certificate"] = {
            "t": str(q_alg),
            "s_blocks": {
                str(b): {f"{i},{j}": str(value)
                         for (i, j), value in s_blocks[b].items()}
                for b in identity.block_names
            },
            "lambda": {f"{i},{j}": str(v) for (i, j), v in lam[1].items()},
            "nu": str(nu),
        }
    else:
        report["failures"] = ["exact verification failed"]
    report["runtime_seconds"] = round(time.time() - t_start, 2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", default=["5/4", "41/32"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "m6" / "closure_exact.json")
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--dps", type=int, default=250)
    parser.add_argument("--delta-schedule", nargs="+", type=int,
                        default=[-6, -7, -8, -9, -10, -12, -14, -17, -20,
                                 -25, -30, -35, -40, -45, -50])
    parser.add_argument("--max-degree-qq", type=int, default=16)
    parser.add_argument("--max-degree-ext", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["schema_version"] = 1
    payload["description"] = (
        "M6 exactification closure: exact one-sided level-2 dual certificates "
        "at the quantum bound q(alpha) = sqrt(8 + 2 alpha^2).  Solvers are "
        "used only to seed a high-precision Newton homotopy; verification is "
        "exact (sympy, npa2.exact)."
    )
    results = payload.setdefault("results", {})

    overall_failures = []
    for text in args.alphas:
        alpha = Fraction(text)
        print(f"alpha={alpha}: exactify ...", flush=True)
        report = process_alpha(alpha, args)
        print(f"alpha={alpha}: status={report['status']} "
              f"({report.get('runtime_seconds', '?')}s)", flush=True)
        if report["status"] != "verified":
            print(f"  failures: {report.get('failures')}", flush=True)
        results[str(alpha)] = report
        if report["status"] != "verified":
            overall_failures.append(f"{alpha}: {report.get('failures', 'unverified')}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    payload["status"] = "PASS" if not overall_failures else "PARTIAL"
    payload["failures"] = overall_failures
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "failures": overall_failures},
                     indent=2))
    return 0 if not overall_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
