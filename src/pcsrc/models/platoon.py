"""Unified mixed traffic platoon models: CAV transfer functions and HDV IDM dynamics.

Implements the exact mathematical formulations from Sections II-III of:
"Low-Frequency String Stability Conditions and Lifted Polynomial Certificates
for Mixed Traffic Platoons under Delay Mismatch" (Haiying Dong, Xiaofei Zhao).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlatoonParams:
    """Nominal physical and controller parameters (Table I)."""

    vehicle_length: float = 5.0       # L_i [m]
    standstill_gap: float = 5.0       # s_0 [m]
    headway: float = 1.0              # h [s]
    tau_a: float = 0.20               # true actuation delay [s]
    tau_e: float = 0.35               # actuator time constant [s]
    tau_c: float = 0.18               # V2V communication delay [s]
    a_min: float = -3.0               # min acceleration [m/s^2]
    a_max: float = 3.0                # max acceleration [m/s^2]
    v0: float = 22.0                  # equilibrium cruising speed [m/s]
    v_free: float = 30.0              # IDM desired free-flow speed [m/s]
    kp: float = 1.15                  # proportional spacing gain [s^-2]
    kv: float = 1.18                  # derivative velocity gain [s^-1]
    ka: float = 0.80                  # V2V acceleration feedforward gain [-]
    delta_max: float = 0.08           # max actuation delay mismatch [s]
    omega_min: float = 0.25           # low-frequency certificate boundary [rad/s]
    omega_bar: float = 15.0           # high-frequency truncation horizon [rad/s]


@dataclass(frozen=True)
class HDVUncertaintyBox:
    """Calibrated naturalistic HDV IDM parameter box Theta^HDV (NGSIM/highD)."""

    T_range: tuple[float, float] = (0.70, 1.50)       # desired time headway [s]
    a_range: tuple[float, float] = (0.90, 1.80)       # max acceleration [m/s^2]
    b_range: tuple[float, float] = (1.20, 2.50)       # comfortable decel [m/s^2]
    tau_r_range: tuple[float, float] = (0.30, 0.72)   # neuromuscular lag [s]


def hdv_sensitivities(
    T: float,
    a_max: float,
    b: float,
    v_e: float = 22.0,
    v_free: float = 30.0,
    s0: float = 5.0,
) -> tuple[float, float, float]:
    """Return linearized IDM sensitivities (alpha_1, alpha_2, alpha_3)."""
    # Equilibrium spacing s_e = (s0 + v_e * T) / sqrt(1 - (v_e / v_free)^4)
    # Using the paper's compact linearization s_e = s0 + v_e * T (Eqs. 5-7).
    s_e = s0 + v_e * T
    alpha_1 = np.sqrt(a_max / b) * (v_e / s_e)
    alpha_2 = 2.0 * a_max / s_e
    alpha_3 = 4.0 * a_max * (v_e**3) / (v_free**4) + (2.0 * a_max * T) / s_e
    return float(alpha_1), float(alpha_2), float(alpha_3)


def gamma_hdv(
    omega: np.ndarray,
    T: float,
    a_max: float,
    b: float,
    tau_r: float,
    v_e: float = 22.0,
    v_free: float = 30.0,
    s0: float = 5.0,
) -> np.ndarray:
    """Evaluate linearized HDV transfer function Gamma_HDV(j*omega; theta)."""
    s = 1j * np.asarray(omega, dtype=np.float64)
    a1, a2, a3 = hdv_sensitivities(T, a_max, b, v_e=v_e, v_free=v_free, s0=s0)
    num = a1 * s + a2
    den = tau_r * (s**3) + (s**2) + (a1 + a3) * s + a2
    return num / den


def worst_case_hdv_envelope(
    omega: np.ndarray,
    box: HDVUncertaintyBox = HDVUncertaintyBox(),
    v_e: float = 22.0,
    v_free: float = 30.0,
    s0: float = 5.0,
    n_grid: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute worst-case HDV magnitude H_bar^HDV(omega) over Theta^HDV."""
    w = np.asarray(omega, dtype=np.float64)
    T_vals = np.linspace(box.T_range[0], box.T_range[1], n_grid)
    a_vals = np.linspace(box.a_range[0], box.a_range[1], n_grid)
    b_vals = np.linspace(box.b_range[0], box.b_range[1], n_grid)
    tr_vals = np.linspace(box.tau_r_range[0], box.tau_r_range[1], n_grid)

    h_bar = np.zeros_like(w)
    worst_theta = {"T": 0.0, "a_max": 0.0, "b": 0.0, "tau_r": 0.0, "peak": 0.0, "omega_peak": 0.0}
    for T in T_vals:
        for a_m in a_vals:
            for b_v in b_vals:
                for tr in tr_vals:
                    mag = np.abs(gamma_hdv(w, T, a_m, b_v, tr, v_e=v_e, v_free=v_free, s0=s0))
                    h_bar = np.maximum(h_bar, mag)
                    pk_idx = int(np.argmax(mag))
                    if float(mag[pk_idx]) > worst_theta["peak"]:
                        worst_theta = {
                            "T": float(T),
                            "a_max": float(a_m),
                            "b": float(b_v),
                            "tau_r": float(tr),
                            "peak": float(mag[pk_idx]),
                            "omega_peak": float(w[pk_idx]),
                        }
    return h_bar, worst_theta


def fit_hdv_envelope_poly(
    omega: np.ndarray,
    h_bar: np.ndarray,
    m_hdv: int = 2,
    omega_min: float = 0.25,
    omega_bar: float = 15.0,
) -> np.ndarray:
    """Fit an upper-bounding polynomial w(x) >= (H_bar^HDV(sqrt(x)))^(2m) on [omega_min, omega_bar].

    Returns coefficients [w0, w1, w2] in ascending powers of x = omega^2,
    or a tighter rational-free quadratic shifted upward to guarantee strict upper bound.
    """
    mask = (omega >= omega_min) & (omega <= omega_bar)
    x = omega[mask] ** 2
    y = h_bar[mask] ** (2 * m_hdv)
    # Fit degree-2 polynomial in x over the resonant low-to-mid band and shift to upper-bound
    # Because H_bar^HDV decays sharply at high x, a constant + decaying envelope is conservative;
    # we fit on [omega_min, 3.0] and enforce global upper bound on [omega_min, omega_bar].
    # For clean SOS degree, use a valid constant/quadratic upper bound w(x) = w0 + w1*x + w2*x^2
    # or simple strictly valid upper envelope via LP/Chebyshev shift.
    coeffs_desc = np.polyfit(x, y, 2)
    poly_vals = np.polyval(coeffs_desc, x)
    shift = float(np.max(y - poly_vals)) + 1e-4
    coeffs_desc[-1] += max(0.0, shift)
    # Return ascending [w0, w1, w2]
    return coeffs_desc[::-1]


def gamma_pcsrc(
    omega: np.ndarray,
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
    tau_a_hat: float,
) -> np.ndarray:
    """Closed-loop CAV transfer function Gamma_CAV(j*omega) for PC-SRC (Eq. 15)."""
    s = 1j * np.asarray(omega, dtype=np.float64)
    k_fb = kp + kv * s
    num = (k_fb + ka * (s**2) * np.exp(-tau_c * s)) * np.exp(-tau_a * s)
    d_nom = (s**2) * (tau_e * s + 1.0) + k_fb * (1.0 + h * s)
    delta_err = k_fb * (1.0 + h * s) * (np.exp(-tau_a * s) - np.exp(-tau_a_hat * s))
    return num / (d_nom + delta_err)


def gamma_no_predictor(
    omega: np.ndarray,
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
) -> np.ndarray:
    """Standard CTH + V2V feedforward CACC without actuation-delay predictor."""
    # Equivalent to PC-SRC with tau_a_hat = 0 (no predictor loop: G_0 - G_hat = 0 -> e^{-tau_a s} in den)
    s = 1j * np.asarray(omega, dtype=np.float64)
    k_fb = kp + kv * s
    num = (k_fb + ka * (s**2) * np.exp(-tau_c * s)) * np.exp(-tau_a * s)
    den = (s**2) * (tau_e * s + 1.0) + k_fb * (1.0 + h * s) * np.exp(-tau_a * s)
    return num / den


def gamma_classical_smith(
    omega: np.ndarray,
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
    tau_a_hat: float,
) -> np.ndarray:
    """Classical Smith predictor omitting actuator lag tau_e in internal model (G0_model = 1)."""
    s = 1j * np.asarray(omega, dtype=np.float64)
    k_fb = kp + kv * s
    g_true = np.exp(-tau_a * s) / (tau_e * s + 1.0)
    g_pred_diff = 1.0 - np.exp(-tau_a_hat * s)
    p_s = (1.0 + h * s) / (s**2)
    num = g_true * (k_fb / (s**2) + ka * np.exp(-tau_c * s))
    den = 1.0 + k_fb * p_s * (g_true + g_pred_diff)
    return num / den


def gamma_pade_cacc(
    omega: np.ndarray,
    kp: float,
    kv: float,
    ka: float,
    h: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
    tau_a_hat: float,
) -> np.ndarray:
    """Predictor using [1/1] Pade approximation of actuation delay in internal model."""
    s = 1j * np.asarray(omega, dtype=np.float64)
    k_fb = kp + kv * s
    g_true = np.exp(-tau_a * s) / (tau_e * s + 1.0)
    g0 = 1.0 / (tau_e * s + 1.0)
    pade_delay = (1.0 - 0.5 * tau_a_hat * s) / (1.0 + 0.5 * tau_a_hat * s)
    g_hat_pade = g0 * pade_delay
    p_s = (1.0 + h * s) / (s**2)
    num = g_true * (k_fb / (s**2) + ka * np.exp(-tau_c * s))
    den = 1.0 + k_fb * p_s * (g_true + g0 - g_hat_pade)
    return num / den


def gamma_robust_hinf(
    omega: np.ndarray,
    h: float,
    tau_e: float,
    tau_c: float,
    tau_a: float,
) -> np.ndarray:
    """Heavy-damping Robust H-infinity CACC baseline (Zhou & Zheng 2020 style)."""
    # Conservative high-damping, lower-feedforward gains without delay predictor
    return gamma_no_predictor(
        omega,
        kp=0.55,
        kv=1.45,
        ka=0.60,
        h=h,
        tau_e=tau_e,
        tau_c=tau_c,
        tau_a=tau_a,
    )
