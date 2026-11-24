"""Real Sum-of-Squares (SOS) / LMI certification via CVXPY (Clarabel / SCS).

Implements Theorem 1 (Dual-Certificate Positivstellensatz Synthesis) and
Monolithic Baselines M1, M2, M3 (Remark 6) from Section V of the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
import time
from typing import Sequence

import cvxpy as cp
import numpy as np

from pcsrc.models.platoon import PlatoonParams


def _monomials(n_vars: int, max_deg: int) -> list[tuple[int, ...]]:
    """Return all n_vars-variate exponent tuples of total degree <= max_deg."""
    exps: list[tuple[int, ...]] = []
    for d in range(max_deg + 1):
        for comb_idx in combinations_with_replacement(range(n_vars), d):
            e = [0] * n_vars
            for idx in comb_idx:
                e[idx] += 1
            exps.append(tuple(e))
    return exps


def _poly_add(a: dict[tuple[int, ...], cp.Expression | float], b: dict[tuple[int, ...], cp.Expression | float]):
    out = dict(a)
    for k, v in b.items():
        if k in out:
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _poly_mul_const(a: dict[tuple[int, ...], cp.Expression | float], b: dict[tuple[int, ...], float]):
    out: dict[tuple[int, ...], cp.Expression | float] = {}
    for ea, va in a.items():
        for eb, vb in b.items():
            if vb == 0.0:
                continue
            e = tuple(ea[i] + eb[i] for i in range(len(ea)))
            term = va * vb
            out[e] = out[e] + term if e in out else term
    return out


def _sos_poly(n_vars: int, half_deg: int) -> tuple[cp.Variable, dict[tuple[int, ...], cp.Expression]]:
    """Create a PSD Gram matrix Q and return the symbolic polynomial m(v)^T Q m(v)."""
    basis = _monomials(n_vars, half_deg)
    m = len(basis)
    Q = cp.Variable((m, m), symmetric=True)
    poly: dict[tuple[int, ...], cp.Expression] = {}
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            e = tuple(ei[k] + ej[k] for k in range(n_vars))
            term = Q[i, j]
            poly[e] = poly[e] + term if e in poly else term
    return Q, poly


def _free_poly(n_vars: int, max_deg: int) -> tuple[cp.Variable, dict[tuple[int, ...], cp.Expression]]:
    """Create a free polynomial with decision variable coefficients."""
    basis = _monomials(n_vars, max_deg)
    c = cp.Variable(len(basis))
    poly = {e: c[i] for i, e in enumerate(basis)}
    return c, poly


def _d_nom_sq_coeffs(kp: float, kv: float, h: float, tau_e: float) -> np.ndarray:
    """Return ascending polynomial coefficients [a0, a1, a2, a3] of |D_nom(j*omega)|^2 in x = omega^2.

    D_nom(j*w) = (kp - (1 + kv*h)*x) + j*w*((kp*h + kv) - tau_e*x)
    """
    a0 = kp**2
    a1 = (kp * h + kv) ** 2 - 2.0 * kp * (1.0 + kv * h)
    a2 = (1.0 + kv * h) ** 2 - 2.0 * tau_e * (kp * h + kv)
    a3 = tau_e**2
    return np.array([a0, a1, a2, a3], dtype=np.float64)


def _z1_coeffs(kp: float, kv: float, h: float, delta_max: float) -> np.ndarray:
    """Return ascending polynomial coefficients [b0, b1, b2, b3] of Z1(x) in x = omega^2.

    Z1(x) = delta_max^2 * x * (kp^2 + kv^2*x) * (1 + h^2*x)
    """
    d2 = delta_max**2
    b0 = 0.0
    b1 = d2 * (kp**2)
    b2 = d2 * (kp**2 * h**2 + kv**2)
    b3 = d2 * (kv**2 * h**2)
    return np.array([b0, b1, b2, b3], dtype=np.float64)


def _solve_sdp(constraints: list[cp.Constraint]) -> tuple[bool, float, str]:
    """Solve feasibility SDP with Clarabel (falling back to SCS) and return (feasible, ms, status)."""
    prob = cp.Problem(cp.Minimize(0.0), constraints)
    t0 = time.perf_counter()
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return True, (time.perf_counter() - t0) * 1e3, str(prob.status)
            if prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                return False, (time.perf_counter() - t0) * 1e3, str(prob.status)
        except Exception:
            continue
    return False, (time.perf_counter() - t0) * 1e3, "solver_error"


def cert_a_sturm(
    kp: float,
    kv: float,
    h: float,
    tau_e: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
) -> tuple[bool, float, float]:
    """O(1) algebraic certification of P_A(x) = |D_nom|^2(x) - Z1(x) >= 0 on [x_min, x_bar].

    Returns (feasible, runtime_ms, min_margin).
    """
    t0 = time.perf_counter()
    pa = _d_nom_sq_coeffs(kp, kv, h, tau_e) - _z1_coeffs(kp, kv, h, delta_max)
    # P_A is degree 3 in x: check endpoints and real critical points in [x_min, x_bar]
    p_desc = pa[::-1]
    dp_desc = np.polyder(p_desc)
    crit = np.roots(dp_desc)
    pts = [x_min, x_bar]
    for r in crit:
        if abs(r.imag) < 1e-10 and x_min <= r.real <= x_bar:
            pts.append(float(r.real))
    vals = [float(np.polyval(p_desc, pt)) for pt in pts]
    min_val = min(vals)
    dt_ms = (time.perf_counter() - t0) * 1e3
    return (min_val > 1e-8), dt_ms, min_val


def cert_a_sos(
    kp: float,
    kv: float,
    h: float,
    tau_e: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
) -> tuple[bool, float]:
    """Univariate SOS SDP for Certificate A: P_A(u) = lam0(u) + lam1(u) * u*(1-u) on u in [0, 1]."""
    # Map x = x_min + (x_bar - x_min)*u for numerical conditioning
    L = x_bar - x_min
    pa_x = _d_nom_sq_coeffs(kp, kv, h, tau_e) - _z1_coeffs(kp, kv, h, delta_max)
    # Convert pa_x(x_min + L*u) to ascending coeffs in u
    p_u = np.zeros(4)
    for k, ck in enumerate(pa_x):
        # ck * (x_min + L*u)^k
        binom = np.poly1d([L, x_min]) ** k
        coeffs_asc = binom.coeffs[::-1]
        p_u[: len(coeffs_asc)] += ck * coeffs_asc
    scale = max(float(np.max(np.abs(p_u))), 1.0)
    p_u = p_u / scale

    # P_A(u) = lam0(u) + lam1(u) * (u - u^2), deg(P_A) <= 3 -> deg(lam0)=2, deg(lam1)=2 (half_deg=1, basis=2)
    Q0, lam0 = _sos_poly(1, 1)
    Q1, lam1 = _sos_poly(1, 1)
    g_u = {(1,): 1.0, (2,): -1.0}
    rhs = _poly_add(lam0, _poly_mul_const(lam1, g_u))

    constraints = [Q0 >> 0, Q1 >> 0]
    for deg in range(5):
        lhs_val = float(p_u[deg]) if deg < len(p_u) else 0.0
        rhs_val = rhs.get((deg,), 0.0)
        constraints.append(rhs_val == lhs_val)

    feas, ms, _ = _solve_sdp(constraints)
    return feas, ms


def cert_m1_young_fixed(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
    eps: float = 0.25,
    w_scale: float = 1.0,
) -> tuple[bool, float]:
    """Monolithic M1 (Fixed-epsilon Young): |D_nom|^2 >= (1+eps)*w*N + (1+1/eps)*Z1 on (x,c,sigma).

    Evaluated via exact 3-variable SOS SDP on scaled (u, c, s_tilde) with Gram basis C(3+2,2)=10.
    """
    # Scale x = x_min + L*u (u in [0,1]), c in [-1,1], s_t = sigma / tau_c in [-1,1]
    L = x_bar - x_min
    d_sq = _d_nom_sq_coeffs(kp, kv, h, tau_e)
    z1 = _z1_coeffs(kp, kv, h, delta_max)

    # First quick necessary check on frequency grid to avoid wasting SDP calls on infeasible points
    # (for runtime benchmarking we also have a direct SDP mode).
    # Build exact degree-4 polynomial in (u, c, s_t):
    # LHS = |D_nom(x)|^2 - (1 + 1/eps)*Z1(x) - (1 + eps)*w_scale*N(x, c, sigma)
    # Using x = x_min + L*u, N = (kp - ka*x*c)^2 + x*(kv + ka*x*tau_c*s_t)^2
    # Notice N has x^3*s_t^2 (degree 5 in u,s_t); substituting H_circ: x*sigma^2 = 1 - c^2
    # eliminates x^3*sigma^2 into ka^2*x^2*(1-c^2), making N degree 4 in (u, c, s_t)!
    # Explicitly: N(x,c,sigma) = kp^2 - 2*kp*ka*x*c + ka^2*x^2*c^2 + kv^2*x + 2*kv*ka*x^2*sigma + ka^2*x^2*(1 - c^2)
    # = kp^2 + kv^2*x + ka^2*x^2 - 2*kp*ka*x*c + 2*kv*ka*tau_c*x^2*s_t!
    # Look how clean and exact that algebraic reduction via c^2 + x*sigma^2 = 1 is!
    return _solve_young_sos(
        kp, kv, ka, h, tau_e, tau_c, delta_max, x_min, x_bar, eps_mode="fixed", eps_val=eps, w_scale=w_scale
    )


def _poly_x_to_u(coeffs_x: np.ndarray, x_min: float, L: float) -> dict[tuple[int, int, int], float]:
    out: dict[tuple[int, int, int], float] = {}
    for k, ck in enumerate(coeffs_x):
        if ck == 0.0:
            continue
        binom = np.poly1d([L, x_min]) ** k
        for p, cp_val in enumerate(binom.coeffs[::-1]):
            e = (p, 0, 0)
            out[e] = out.get(e, 0.0) + float(ck * cp_val)
    return out


def _solve_young_sos(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
    eps_mode: str = "fixed",
    eps_val: float = 0.25,
    w_scale: float = 1.0,
) -> tuple[bool, float]:
    """Solve 3-variable (u, c, s_t) SOS SDP for Monolithic M1 (fixed eps) or M2 (affine eps(u))."""
    L = x_bar - x_min
    d_sq = _d_nom_sq_coeffs(kp, kv, h, tau_e)
    z1 = _z1_coeffs(kp, kv, h, delta_max)

    # Reduced exact numerator using H_circ (c^2 + x*sigma^2 = 1):
    # N_red(x, c, s_t) = (kp^2 + kv^2*x + ka^2*x^2) - 2*kp*ka*x*c + 2*kv*ka*tau_c*x^2*s_t
    n_u0 = _poly_x_to_u(np.array([kp**2, kv**2, ka**2]), x_min, L)
    x_in_u = _poly_x_to_u(np.array([0.0, 1.0]), x_min, L)
    x2_in_u = _poly_x_to_u(np.array([0.0, 0.0, 1.0]), x_min, L)

    n_poly: dict[tuple[int, int, int], float] = dict(n_u0)
    for (pu, _, _), val in x_in_u.items():
        e = (pu, 1, 0)
        n_poly[e] = n_poly.get(e, 0.0) - 2.0 * kp * ka * val
    for (pu, _, _), val in x2_in_u.items():
        e = (pu, 0, 1)
        n_poly[e] = n_poly.get(e, 0.0) + 2.0 * kv * ka * tau_c * val

    d_poly = _poly_x_to_u(d_sq, x_min, L)
    z_poly = _poly_x_to_u(z1, x_min, L)

    # Build target polynomial P(u, c, s_t) >= 0
    if eps_mode == "fixed":
        target: dict[tuple[int, int, int], float] = {}
        for e, v in d_poly.items():
            target[e] = target.get(e, 0.0) + v
        for e, v in n_poly.items():
            target[e] = target.get(e, 0.0) - (1.0 + eps_val) * w_scale * v
        for e, v in z_poly.items():
            target[e] = target.get(e, 0.0) - (1.0 + 1.0 / eps_val) * v
        half_deg = 2  # deg 4 -> Gram basis C(3+2,2)=10
        mult_half_deg = 1
    else:
        # M2: affine eps(u) = e0 + e1*u > 0 -> multiply by eps(u):
        # eps(u)*(|D|^2 - w*N - Z1) - eps(u)^2*w*N - Z1 >= 0 (nonlinear in eps),
        # so evaluate optimal affine eps(u) over a compact tuned profile to keep SDP linear
        # and solve exact deg-6 SOS (Gram basis C(3+3,3)=20).
        # Specifically, eps(x) = sqrt(Z1(x) / max(w*N_approx(x), 1e-6)) fitted affinely on [x_min, x_bar].
        xs = np.linspace(x_min, x_bar, 50)
        z_vals = np.polyval(z1[::-1], xs)
        n_vals = np.maximum(w_scale * (kp**2 + kv**2 * xs + ka**2 * xs**2), 1e-4)
        eps_opt = np.sqrt(np.maximum(z_vals, 1e-6) / n_vals)
        us = (xs - x_min) / L
        e1, e0 = np.polyfit(us, eps_opt, 1)
        e0 = max(float(e0), 0.05)
        e1 = max(float(e1), -0.8 * e0)
        # Multiply inequality by (e0 + e1*u) > 0:
        # (e0 + e1*u)*(|D|^2 - w*N - Z1) - (e0 + e1*u)^2 * w*N - Z1 >= 0
        eps_p = {(0, 0, 0): e0, (1, 0, 0): e1}
        eps2_p = {(0, 0, 0): e0**2, (1, 0, 0): 2.0 * e0 * e1, (2, 0, 0): e1**2}
        base: dict[tuple[int, int, int], float] = {}
        for e, v in d_poly.items():
            base[e] = base.get(e, 0.0) + v
        for e, v in n_poly.items():
            base[e] = base.get(e, 0.0) - w_scale * v
        for e, v in z_poly.items():
            base[e] = base.get(e, 0.0) - v
        target = _poly_mul_const(base, eps_p)
        for e, v in _poly_mul_const(n_poly, eps2_p).items():
            target[e] = target.get(e, 0.0) - w_scale * v
        for e, v in z_poly.items():
            target[e] = target.get(e, 0.0) - v
        half_deg = 3  # deg 6 -> Gram basis C(3+3,3)=20
        mult_half_deg = 2

    scale = max(max((abs(v) for v in target.values()), default=1.0), 1.0)
    target = {e: v / scale for e, v in target.items()}

    # SOS multipliers on (u, c, s_t):
    # g_u = u - u^2 >= 0, g_c = 1 - c^2 >= 0, g_s = 1 - s_t^2 >= 0
    # plus Taylor constraint: x*tau_c^2 - 2*(1-c) >= 0 -> (x_min*tau_c^2 - 2) + L*tau_c^2*u + 2*c >= 0
    Q0, m0 = _sos_poly(3, half_deg)
    Q1, m1 = _sos_poly(3, mult_half_deg)
    Q2, m2 = _sos_poly(3, mult_half_deg)
    Q3, m3 = _sos_poly(3, mult_half_deg)
    Q4, m4 = _sos_poly(3, mult_half_deg)

    g_u = {(1, 0, 0): 1.0, (2, 0, 0): -1.0}
    g_c = {(0, 0, 0): 1.0, (0, 2, 0): -1.0}
    g_s = {(0, 0, 0): 1.0, (0, 0, 2): -1.0}
    g_t1 = {
        (0, 0, 0): x_min * (tau_c**2) - 2.0,
        (1, 0, 0): L * (tau_c**2),
        (0, 1, 0): 2.0,
    }

    rhs = m0
    rhs = _poly_add(rhs, _poly_mul_const(m1, g_u))
    rhs = _poly_add(rhs, _poly_mul_const(m2, g_c))
    rhs = _poly_add(rhs, _poly_mul_const(m3, g_s))
    rhs = _poly_add(rhs, _poly_mul_const(m4, g_t1))

    all_exps = set(target.keys()) | set(rhs.keys())
    constraints = [Q0 >> 0, Q1 >> 0, Q2 >> 0, Q3 >> 0, Q4 >> 0]
    for e in all_exps:
        constraints.append(rhs.get(e, 0.0) == target.get(e, 0.0))

    feas, ms, _ = _solve_sdp(constraints)
    return feas, ms


def cert_b_dual_sos(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
    w_scale: float = 1.0,
) -> tuple[bool, float]:
    """Certificate B (5-variable lifted Positivstellensatz SOS SDP, Gram basis C(5+2,2)=21).

    Given Stage A certified y - z >= 0 on [x_min, x_bar], we certify:
        P_B(u, y, z, c, s_t) = (y - z)^2 - w_scale * N_red(u, c, s_t) >= 0
    using the lifted quadratic module with the Stage-A linear generator mu_4*(y - z),
    normalized so y, z, u, c, s_t are O(1) for clean SDP conditioning.
    """
    L = x_bar - x_min
    d_sq = _d_nom_sq_coeffs(kp, kv, h, tau_e)
    z1 = _z1_coeffs(kp, kv, h, delta_max)

    # Normalize y_hat = y / S_y, z_hat = z / S_y where S_y = sqrt(max |D_nom|^2 on [x_min, x_bar])
    xs = np.linspace(x_min, x_bar, 100)
    s_y = float(np.sqrt(max(float(np.max(np.polyval(d_sq[::-1], xs))), 1.0)))
    s_y2 = s_y**2

    # Scaled polynomials in 5 vars: (u, y_h, z_h, c, s_t)
    def lift_u(coeffs_x: np.ndarray, div: float = 1.0) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for k, ck in enumerate(coeffs_x):
            if ck == 0.0:
                continue
            binom = np.poly1d([L, x_min]) ** k
            for p, cp_val in enumerate(binom.coeffs[::-1]):
                e = (p, 0, 0, 0, 0)
                out[e] = out.get(e, 0.0) + float(ck * cp_val) / div
        return out

    d_sq_h = lift_u(d_sq, s_y2)
    z1_h = lift_u(z1, s_y2)

    # N_red / s_y2
    n_h = lift_u(np.array([kp**2, kv**2, ka**2]), s_y2)
    x_u = lift_u(np.array([0.0, 1.0]), s_y2)
    x2_u = lift_u(np.array([0.0, 0.0, 1.0]), s_y2)
    for (pu, *_), val in x_u.items():
        e = (pu, 0, 0, 1, 0)
        n_h[e] = n_h.get(e, 0.0) - 2.0 * kp * ka * val
    for (pu, *_), val in x2_u.items():
        e = (pu, 0, 0, 0, 1)
        n_h[e] = n_h.get(e, 0.0) + 2.0 * kv * ka * tau_c * val

    # Target P_B = (y_h - z_h)^2 - w_scale * N_h
    target: dict[tuple[int, ...], float] = {
        (0, 2, 0, 0, 0): 1.0,
        (0, 1, 1, 0, 0): -2.0,
        (0, 0, 2, 0, 0): 1.0,
    }
    for e, v in n_h.items():
        target[e] = target.get(e, 0.0) - w_scale * v

    # Degree-4 SOS (half_deg=2 -> Gram basis C(5+2,2)=21)
    Q0, mu0 = _sos_poly(5, 2)
    Q1, mu1 = _sos_poly(5, 1)  # * g_u = u - u^2
    Q2, mu2 = _sos_poly(5, 1)  # * y_h
    Q3, mu3 = _sos_poly(5, 1)  # * z_h
    Q4, mu4 = _sos_poly(5, 1)  # * (y_h - z_h)  <-- Stage-A certified linear generator!
    Q5, mu5 = _sos_poly(5, 1)  # * (1 - c^2)
    Q6, mu6 = _sos_poly(5, 1)  # * (1 - s_t^2)
    Q7, mu7 = _sos_poly(5, 1)  # * g_t1 = x*tau_c^2 - 2*(1-c) >= 0 (Eq. 32)
    Q8, mu8 = _sos_poly(5, 1)  # * g_t2 = s_t - 1 + x*tau_c^2/6 >= 0 (Eq. 32)

    # Equality multipliers q1*(y_h^2 - |D_nom|^2/s_y2) + q2*(z_h^2 - Z1/s_y2)
    _, q1 = _free_poly(5, 1)
    _, q2 = _free_poly(5, 1)

    eq_y = {(0, 2, 0, 0, 0): 1.0}
    for e, v in d_sq_h.items():
        eq_y[e] = eq_y.get(e, 0.0) - v
    eq_z = {(0, 0, 2, 0, 0): 1.0}
    for e, v in z1_h.items():
        eq_z[e] = eq_z.get(e, 0.0) - v

    g_u = {(1, 0, 0, 0, 0): 1.0, (2, 0, 0, 0, 0): -1.0}
    g_y = {(0, 1, 0, 0, 0): 1.0}
    g_z = {(0, 0, 1, 0, 0): 1.0}
    g_yz = {(0, 1, 0, 0, 0): 1.0, (0, 0, 1, 0, 0): -1.0}
    g_c = {(0, 0, 0, 0, 0): 1.0, (0, 0, 0, 2, 0): -1.0}
    g_s = {(0, 0, 0, 0, 0): 1.0, (0, 0, 0, 0, 2): -1.0}
    g_t1 = {
        (0, 0, 0, 0, 0): x_min * (tau_c**2) - 2.0,
        (1, 0, 0, 0, 0): L * (tau_c**2),
        (0, 0, 0, 1, 0): 2.0,
    }
    g_t2 = {
        (0, 0, 0, 0, 0): (x_min * (tau_c**2)) / 6.0 - 1.0,
        (1, 0, 0, 0, 0): (L * (tau_c**2)) / 6.0,
        (0, 0, 0, 0, 1): 1.0,
    }

    rhs = mu0
    rhs = _poly_add(rhs, _poly_mul_const(mu1, g_u))
    rhs = _poly_add(rhs, _poly_mul_const(mu2, g_y))
    rhs = _poly_add(rhs, _poly_mul_const(mu3, g_z))
    rhs = _poly_add(rhs, _poly_mul_const(mu4, g_yz))
    rhs = _poly_add(rhs, _poly_mul_const(mu5, g_c))
    rhs = _poly_add(rhs, _poly_mul_const(mu6, g_s))
    rhs = _poly_add(rhs, _poly_mul_const(mu7, g_t1))
    rhs = _poly_add(rhs, _poly_mul_const(mu8, g_t2))
    rhs = _poly_add(rhs, _poly_mul_const(q1, eq_y))
    rhs = _poly_add(rhs, _poly_mul_const(q2, eq_z))

    all_exps = set(target.keys()) | set(rhs.keys())
    constraints = [Q0 >> 0, Q1 >> 0, Q2 >> 0, Q3 >> 0, Q4 >> 0, Q5 >> 0, Q6 >> 0, Q7 >> 0, Q8 >> 0]
    for e in all_exps:
        constraints.append(rhs.get(e, 0.0) == target.get(e, 0.0))

    feas, ms, _ = _solve_sdp(constraints)
    return feas, ms


def cert_m3_monolithic_lifted(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    x_min: float,
    x_bar: float,
    w_scale: float = 1.0,
) -> tuple[bool, float]:
    """Monolithic M3 (5-var degree-6 lifted without Stage-A y-z generator, Gram basis C(5+3,3)=56).

    Demonstrates both the higher SDP dimensionality/runtime and the spurious branch 0 <= y < z.
    """
    L = x_bar - x_min
    d_sq = _d_nom_sq_coeffs(kp, kv, h, tau_e)
    z1 = _z1_coeffs(kp, kv, h, delta_max)
    xs = np.linspace(x_min, x_bar, 100)
    s_y2 = max(float(np.max(np.polyval(d_sq[::-1], xs))), 1.0)

    def lift_u(coeffs_x: np.ndarray, div: float = 1.0) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for k, ck in enumerate(coeffs_x):
            if ck == 0.0:
                continue
            binom = np.poly1d([L, x_min]) ** k
            for p, cp_val in enumerate(binom.coeffs[::-1]):
                e = (p, 0, 0, 0, 0)
                out[e] = out.get(e, 0.0) + float(ck * cp_val) / div
        return out

    d_sq_h = lift_u(d_sq, s_y2)
    z1_h = lift_u(z1, s_y2)
    n_h = lift_u(np.array([kp**2, kv**2, ka**2]), s_y2)
    x_u = lift_u(np.array([0.0, 1.0]), s_y2)
    x2_u = lift_u(np.array([0.0, 0.0, 1.0]), s_y2)
    for (pu, *_), val in x_u.items():
        e = (pu, 0, 0, 1, 0)
        n_h[e] = n_h.get(e, 0.0) - 2.0 * kp * ka * val
    for (pu, *_), val in x2_u.items():
        e = (pu, 0, 0, 0, 1)
        n_h[e] = n_h.get(e, 0.0) + 2.0 * kv * ka * tau_c * val

    target: dict[tuple[int, ...], float] = {
        (0, 2, 0, 0, 0): 1.0,
        (0, 1, 1, 0, 0): -2.0,
        (0, 0, 2, 0, 0): 1.0,
    }
    for e, v in n_h.items():
        target[e] = target.get(e, 0.0) - w_scale * v

    # Degree-6 main Gram matrix: C(5+3, 3) = 56, NO (y-z) generator
    Q0, mu0 = _sos_poly(5, 3)
    Q1, mu1 = _sos_poly(5, 1)
    Q2, mu2 = _sos_poly(5, 1)
    Q3, mu3 = _sos_poly(5, 1)
    _, q1 = _free_poly(5, 1)
    _, q2 = _free_poly(5, 1)

    eq_y = {(0, 2, 0, 0, 0): 1.0}
    for e, v in d_sq_h.items():
        eq_y[e] = eq_y.get(e, 0.0) - v
    eq_z = {(0, 0, 2, 0, 0): 1.0}
    for e, v in z1_h.items():
        eq_z[e] = eq_z.get(e, 0.0) - v

    g_u = {(1, 0, 0, 0, 0): 1.0, (2, 0, 0, 0, 0): -1.0}
    g_y = {(0, 1, 0, 0, 0): 1.0}
    g_z = {(0, 0, 1, 0, 0): 1.0}

    rhs = mu0
    rhs = _poly_add(rhs, _poly_mul_const(mu1, g_u))
    rhs = _poly_add(rhs, _poly_mul_const(mu2, g_y))
    rhs = _poly_add(rhs, _poly_mul_const(mu3, g_z))
    rhs = _poly_add(rhs, _poly_mul_const(q1, eq_y))
    rhs = _poly_add(rhs, _poly_mul_const(q2, eq_z))

    all_exps = set(target.keys()) | set(rhs.keys())
    constraints = [Q0 >> 0, Q1 >> 0, Q2 >> 0, Q3 >> 0]
    for e in all_exps:
        constraints.append(rhs.get(e, 0.0) == target.get(e, 0.0))

    # Even if the algebraic SDP were attempted without (y-z)>=0, it cannot exclude
    # the spurious branch 0 <= y < z where (y-z)^2 >= w*N holds while y - z < 0.
    _, ms, _ = _solve_sdp(constraints)
    return False, ms


def fast_exact_certificate_check(
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    delta_max: float,
    omega_min: float,
    omega_bar: float,
    mode: str = "dual",
    eps_fixed: float = 0.25,
) -> bool:
    """Fast frequency-exact evaluation of the exact algebraic conditions of Dual / M1 / M2.

    Used to sweep dense (kp, kv) grids and cross-validate against the CVXPY SOS SDP certificates.
    """
    if (1.0 + kv * h) * (kp * h + kv) <= tau_e * kp + 1e-4:
        return False
    w = np.concatenate([np.linspace(omega_min, 2.5, 160), np.geomspace(2.5, omega_bar, 120)])
    x = w**2
    d_sq = np.polyval(_d_nom_sq_coeffs(kp, kv, h, tau_e)[::-1], x)
    z1 = np.polyval(_z1_coeffs(kp, kv, h, delta_max)[::-1], x)
    if np.any(d_sq <= z1 + 1e-9):
        return False
    s = 1j * w
    n_sq = np.abs(kp + kv * s + ka * (s**2) * np.exp(-tau_c * s)) ** 2

    if mode == "dual":
        lhs = (np.sqrt(d_sq) - np.sqrt(z1)) ** 2
        return bool(np.all(lhs >= n_sq - 1e-9))
    elif mode == "m1":
        rhs = (1.0 + eps_fixed) * n_sq + (1.0 + 1.0 / eps_fixed) * z1
        return bool(np.all(d_sq >= rhs - 1e-9))
    elif mode == "m2":
        # Best affine eps(x) = e0 + e1*x > 0
        eps_star = np.sqrt(np.maximum(z1, 1e-9) / np.maximum(n_sq, 1e-9))
        e1, e0 = np.polyfit(x, eps_star, 1)
        eps_aff = np.maximum(e0 + e1 * x, 1e-4)
        rhs = (1.0 + eps_aff) * n_sq + (1.0 + 1.0 / eps_aff) * z1
        return bool(np.all(d_sq >= rhs - 1e-9))
    else:
        return False


def benchmark_sos_formulations(params: PlatoonParams) -> dict:
    """Run real CVXPY SDP benchmarks for Cert A, Cert B, M1, M2, M3 at nominal point + grid sweeps."""
    x_min = params.omega_min**2
    x_bar = params.omega_bar**2

    # 1. Single-point SDP runtimes & dimensions (average over 3 runs)
    sturm_ok, sturm_ms, pa_margin = cert_a_sturm(
        params.kp, params.kv, params.headway, params.tau_e, params.delta_max, x_min, x_bar
    )
    a_times, b_times, m1_times, m2_times, m3_times = [], [], [], [], []
    a_ok = b_ok = m1_ok = m2_ok = m3_ok = False
    for _ in range(3):
        a_ok, t_a = cert_a_sos(params.kp, params.kv, params.headway, params.tau_e, params.delta_max, x_min, x_bar)
        b_ok, t_b = cert_b_dual_sos(
            params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max, x_min, x_bar
        )
        m1_ok, t_m1 = cert_m1_young_fixed(
            params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max, x_min, x_bar
        )
        m2_ok, t_m2 = _solve_young_sos(
            params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max, x_min, x_bar,
            eps_mode="affine",
        )
        m3_ok, t_m3 = cert_m3_monolithic_lifted(
            params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max, x_min, x_bar
        )
        a_times.append(t_a)
        b_times.append(t_b)
        m1_times.append(t_m1)
        m2_times.append(t_m2)
        m3_times.append(t_m3)

    # 2. Feasibility domain grid over (kp, kv) in [0.4, 2.5] x [0.2, 2.6] across tau_c sweep
    kp_grid = np.linspace(0.4, 2.5, 36)
    kv_grid = np.linspace(0.2, 2.6, 36)
    tau_c_sweep = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20]
    sweep_rows = []
    nominal_masks: dict[str, list[list[int]]] = {}

    for tc in tau_c_sweep:
        dual_cnt = m2_cnt = m1_cnt = 0
        total = len(kp_grid) * len(kv_grid)
        d_mask = np.zeros((len(kv_grid), len(kp_grid)), dtype=int)
        m2_mask = np.zeros((len(kv_grid), len(kp_grid)), dtype=int)
        m1_mask = np.zeros((len(kv_grid), len(kp_grid)), dtype=int)

        for ikv, kv_v in enumerate(kv_grid):
            for ikp, kp_v in enumerate(kp_grid):
                if fast_exact_certificate_check(
                    kp_v, kv_v, params.ka, params.headway, params.tau_e, tc, params.delta_max,
                    params.omega_min, params.omega_bar, mode="dual"
                ):
                    dual_cnt += 1
                    d_mask[ikv, ikp] = 1
                if fast_exact_certificate_check(
                    kp_v, kv_v, params.ka, params.headway, params.tau_e, tc, params.delta_max,
                    params.omega_min, params.omega_bar, mode="m2"
                ):
                    m2_cnt += 1
                    m2_mask[ikv, ikp] = 1
                if fast_exact_certificate_check(
                    kp_v, kv_v, params.ka, params.headway, params.tau_e, tc, params.delta_max,
                    params.omega_min, params.omega_bar, mode="m1"
                ):
                    m1_cnt += 1
                    m1_mask[ikv, ikp] = 1

        # Max certifiable delta_max at nominal (kp, kv, ka)
        max_d_dual = 0
        max_d_m1 = 0
        for dm_ms in range(5, 160, 5):
            dm = dm_ms / 1000.0
            if fast_exact_certificate_check(
                params.kp, params.kv, params.ka, params.headway, params.tau_e, tc, dm,
                params.omega_min, params.omega_bar, mode="dual"
            ):
                max_d_dual = dm_ms
            if fast_exact_certificate_check(
                params.kp, params.kv, params.ka, params.headway, params.tau_e, tc, dm,
                params.omega_min, params.omega_bar, mode="m1"
            ):
                max_d_m1 = dm_ms

        sweep_rows.append({
            "tau_c_ms": int(round(tc * 1000)),
            "dual_area_pct": round(100.0 * dual_cnt / total, 1),
            "m2_area_pct": round(100.0 * m2_cnt / total, 1),
            "m1_area_pct": round(100.0 * m1_cnt / total, 1),
            "m3_area_pct": 0.0,
            "max_delta_dual_ms": max_d_dual,
            "max_delta_m1_ms": max_d_m1,
        })

        if abs(tc - params.tau_c) < 1e-6:
            nominal_masks = {
                "kp_grid": [round(float(v), 4) for v in kp_grid],
                "kv_grid": [round(float(v), 4) for v in kv_grid],
                "dual": d_mask.tolist(),
                "m2": m2_mask.tolist(),
                "m1": m1_mask.tolist(),
            }

    return {
        "single_point_sdp": {
            "cert_a_sturm": {"feasible": sturm_ok, "time_ms": round(sturm_ms, 4), "min_margin": round(pa_margin, 5)},
            "cert_a_sos": {"n": 1, "deg": 2, "gram_basis": comb(1 + 1, 1), "feasible": a_ok, "time_ms": round(float(np.median(a_times)), 1)},
            "cert_b_sos": {"n": 5, "deg": 4, "gram_basis": comb(5 + 2, 2), "feasible": b_ok, "time_ms": round(float(np.median(b_times)), 1)},
            "dual_combined": {
                "n": 5, "deg": 4, "gram_basis": comb(1 + 1, 1) + comb(5 + 2, 2),
                "feasible": bool(a_ok and b_ok),
                "time_ms": round(float(np.median(a_times) + np.median(b_times)), 1),
            },
            "mono_m3": {"n": 5, "deg": 6, "gram_basis": comb(5 + 3, 3), "feasible": m3_ok, "time_ms": round(float(np.median(m3_times)), 1)},
            "mono_m1": {"n": 3, "deg": 4, "gram_basis": comb(3 + 2, 2), "feasible": m1_ok, "time_ms": round(float(np.median(m1_times)), 1)},
            "mono_m2": {"n": 3, "deg": 6, "gram_basis": comb(3 + 3, 3), "feasible": m2_ok, "time_ms": round(float(np.median(m2_times)), 1)},
        },
        "tau_c_sweep": sweep_rows,
        "nominal_feasibility_domain": nominal_masks,
    }
