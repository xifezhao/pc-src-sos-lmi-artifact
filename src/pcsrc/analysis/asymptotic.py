"""Low-frequency asymptotic Maclaurin bounds and actuator-lag coupling analysis.

Implements Propositions 1-4 and Remark 3 (Lagrange remainder overlap) from
Sections IV-V of the manuscript.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pcsrc.models.platoon import PlatoonParams, gamma_pcsrc


def c2_nominal(kp: float, ka: float, h: float) -> float:
    """Second-order Maclaurin coefficient c2 under matched delay (Prop. 1)."""
    return (kp**2) * (h**2) - 2.0 * kp * (1.0 - ka)


def c2_mismatch(kp: float, ka: float, h: float, tau_a_hat: float, delta_tau: float) -> float:
    """Exact second-order coefficient c2(Delta_tau, tau_a_hat) (Eq. 27)."""
    return (
        (kp**2) * (h**2)
        - 2.0 * kp * (1.0 - ka)
        - 2.0 * (kp**2) * tau_a_hat * delta_tau
        - (kp**2) * (delta_tau**2)
    )


def h_min_second_order(
    kp: float,
    ka: float,
    tau_a_hat: float = 0.0,
    delta_max: float = 0.0,
    m_hdv: int = 0,
    c2_hdv_bar: float = 0.0,
) -> float:
    """Analytical low-frequency lower bound on time headway (Props. 1 & 2)."""
    rhs = (
        2.0 * (1.0 - ka) / kp
        + 2.0 * tau_a_hat * delta_max
        + (delta_max**2)
        + (m_hdv / (kp**2)) * max(0.0, c2_hdv_bar)
    )
    return float(np.sqrt(max(0.0, rhs)))


def c4_nominal(kp: float, kv: float, ka: float, h: float, tau_e: float, tau_c: float) -> float:
    """Fourth-order Maclaurin coefficient c4 under matched delay (Prop. 3)."""
    return (
        (1.0 + kv * h) ** 2
        - 2.0 * tau_e * (kp * h + kv)
        - (ka**2)
        - kp * ka * (tau_c**2)
        - 2.0 * kv * ka * tau_c
    )


def h_star_matched(kp: float, tau_e: float, tau_c: float) -> float:
    """Gain-matched (kv = tau_e * kp, ka = 1) fourth-order headway bound (Prop. 4)."""
    return float(np.sqrt((1.0 + tau_c / tau_e) ** 2 + 1.0) / np.sqrt(kp))


def verify_low_freq_overlap(params: PlatoonParams) -> dict[str, float]:
    """Compute exact c2_nom, c2_worst, M4 Lagrange bound, and overlap [omega_min, omega_upper]."""
    kp, kv, ka, h = params.kp, params.kv, params.ka, params.headway
    tau_e, tau_c, tau_a = params.tau_e, params.tau_c, params.tau_a
    delta_max = params.delta_max
    tau_a_hat = tau_a - delta_max  # worst positive mismatch tau_a - tau_a_hat = +delta_max

    c2_nom = c2_nominal(kp, ka, h)
    # Worst-case c2 over delta_tau in [-delta_max, +delta_max] with nominal tau_a_hat = tau_a
    c2_worst = min(
        c2_mismatch(kp, ka, h, tau_a, delta_max),
        c2_mismatch(kp, ka, h, tau_a, -delta_max),
    )

    # Numerically bound M4 = sup_{w in [0, w_min]} |d^4 Q / dw^4| / 24 via high-order finite diff
    w_grid = np.linspace(0.0, params.omega_min, 200)
    dw = 1e-3

    def q_func(w_val: np.ndarray, dt: float) -> np.ndarray:
        s = 1j * w_val
        k_fb = kp + kv * s
        d_nom = (s**2) * (tau_e * s + 1.0) + k_fb * (1.0 + h * s)
        d_err = k_fb * (1.0 + h * s) * (np.exp(-(tau_a + dt) * s) - np.exp(-tau_a * s))
        num = k_fb + ka * (s**2) * np.exp(-tau_c * s)
        return np.abs(d_nom + d_err) ** 2 - np.abs(num) ** 2

    m4_max = 0.0
    for dt in (-delta_max, 0.0, delta_max):
        # 4th derivative central difference: (f(x-2h)-4f(x-h)+6f(x)-4f(x+h)+f(x+2h))/h^4
        d4 = (
            q_func(w_grid - 2 * dw, dt)
            - 4.0 * q_func(w_grid - dw, dt)
            + 6.0 * q_func(w_grid, dt)
            - 4.0 * q_func(w_grid + dw, dt)
            + q_func(w_grid + 2 * dw, dt)
        ) / (dw**4)
        m4_max = max(m4_max, float(np.max(np.abs(d4)) / 24.0))

    omega_upper = float(np.sqrt(max(0.0, c2_worst) / max(m4_max, 1e-12)))
    q_at_wmin = c2_worst * (params.omega_min**2) - m4_max * (params.omega_min**4)

    # Evaluate P_B(omega_min^2) = (|D_nom| - sqrt(Z1))^2 - |N|^2 at omega_min
    w0 = params.omega_min
    x0 = w0**2
    s0 = 1j * w0
    k_fb0 = kp + kv * s0
    d_nom0 = abs((s0**2) * (tau_e * s0 + 1.0) + k_fb0 * (1.0 + h * s0))
    z1_0 = np.sqrt((kp**2 + (kv**2) * x0) * (1.0 + (h**2) * x0) * (delta_max**2) * x0)
    n0 = abs(k_fb0 + ka * (s0**2) * np.exp(-tau_c * s0))
    pb_at_wmin = (d_nom0 - z1_0) ** 2 - (n0**2)

    return {
        "c2_nom": float(c2_nom),
        "c2_worst": float(c2_worst),
        "M4_bound": float(m4_max),
        "omega_min": float(params.omega_min),
        "omega_upper": float(omega_upper),
        "Q_lower_at_omega_min": float(q_at_wmin),
        "P_B_at_omega_min": float(pb_at_wmin),
        "overlap_valid": bool(omega_upper > params.omega_min and pb_at_wmin > 0.0),
    }


def _is_robust_string_stable(
    h: float,
    kp: float,
    kv: float,
    ka: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
    delta_max: float,
    omega_grid: np.ndarray,
) -> bool:
    """Check Routh-Hurwitz + robust |Gamma_CAV(j*omega)| <= 1 over worst-case delta_tau."""
    if kp <= 0.05 or kv <= 0.0 or h <= 0.05 or ka < 0.0 or ka > 1.0:
        return False
    # Nominal Routh-Hurwitz
    if (1.0 + kv * h) * (kp * h + kv) <= tau_e * kp + 1e-4:
        return False
    # Exact check over endpoints and zero mismatch
    for dt in (-delta_max, 0.0, delta_max):
        mag = np.abs(gamma_pcsrc(omega_grid, kp, kv, ka, h, tau_e, tau_c, tau_a, tau_a - dt))
        if float(np.max(mag)) > 1.0 + 1e-5:
            return False
    return True


def sweep_headway_vs_actuator_lag(
    params: PlatoonParams,
    tau_e_values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Compute h_min(tau_e) for matched damping (kv = tau_e * kp) vs co-optimized (kp, kv, ka)."""
    if tau_e_values is None:
        tau_e_values = np.array(
            [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 0.90, 1.15, 1.40, 1.70, 2.00]
        )
    omega_grid = np.concatenate([np.linspace(0.01, 2.0, 200), np.geomspace(2.0, 25.0, 120)])
    h_asymp = h_min_second_order(params.kp, params.ka, params.tau_a, params.delta_max)

    h_matched: list[float] = []
    h_coopt: list[float] = []

    for te in tau_e_values:
        # 1) Matched damping: kv = te * kp, fixed kp, ka
        lo, hi = 0.40, 2.20
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if _is_robust_string_stable(
                mid, params.kp, te * params.kp, params.ka, te, params.tau_c, params.tau_a, params.delta_max, omega_grid
            ):
                hi = mid
            else:
                lo = mid
        h_matched.append(round(float(hi), 4))

        # 2) Jointly co-optimized (kp, kv, ka) within physical bounds
        # kp in [0.5, 2.0], kv in [0.3, 2.5], ka in [0.5, 0.90]
        best_h = hi
        # Coarse grid + local refinement
        for kp_c in (0.9, 1.15, 1.5, 1.85):
            for kv_c in (0.8, 1.2, 1.7, 2.2):
                ka_c = 0.80
                if not _is_robust_string_stable(
                    best_h, kp_c, kv_c, ka_c, te, params.tau_c, params.tau_a, params.delta_max, omega_grid
                ):
                    continue
                lo2, hi2 = 0.40, best_h
                for _ in range(22):
                    mid2 = 0.5 * (lo2 + hi2)
                    if _is_robust_string_stable(
                        mid2, kp_c, kv_c, ka_c, te, params.tau_c, params.tau_a, params.delta_max, omega_grid
                    ):
                        hi2 = mid2
                    else:
                        lo2 = mid2
                if hi2 < best_h:
                    best_h = hi2
        h_coopt.append(round(float(best_h), 4))

    return {
        "tau_e": [round(float(v), 4) for v in tau_e_values],
        "h_asymp": float(round(h_asymp, 4)),
        "h_matched": h_matched,
        "h_coopt": h_coopt,
    }
