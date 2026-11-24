"""Vectorized high-fidelity nonlinear microscopic mixed-traffic platoon simulator.

Simulates a 20-vehicle single-lane mixed platoon (leader + 19 followers) of
heterogeneous nonlinear IDM HDVs (with neuromuscular reaction lag tau_r) and
CAVs under actuation lag (tau_e), true actuation delay (tau_a), V2V delay
(tau_c), and delay estimation mismatch (Delta_tau = tau_a - tau_a_hat).

Vectorized in NumPy so 1,000 Monte Carlo trials run in ~15 seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.stats import fisher_exact

from pcsrc.models.platoon import HDVUncertaintyBox, PlatoonParams

ControllerType = Literal[
    "pc_src",
    "no_predictor",
    "classical_smith",
    "pade_11",
    "robust_hinf",
    "deep_lcc",
]


@dataclass
class SimMetrics:
    tail_ratio: float
    min_gap: float
    min_ttc: float
    peak_decel: float
    tracking_rmse: float
    accel_rms: float
    throughput_vph: float
    collision: bool


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def simulate_platoon(
    cav_mask: np.ndarray,
    controller: ControllerType = "pc_src",
    params: PlatoonParams = PlatoonParams(),
    delta_tau: float = 0.04,
    leader_decel: float = -2.0,
    brake_start: float = 4.0,
    brake_dur: float = 3.0,
    osc_cycles: int = 2,
    horizon: float = 36.0,
    dt: float = 0.01,
    hdv_T: np.ndarray | None = None,
    hdv_a: np.ndarray | None = None,
    hdv_b: np.ndarray | None = None,
    hdv_tr: np.ndarray | None = None,
    return_trajectories: bool = False,
) -> tuple[SimMetrics, dict | None]:
    """Run a vectorized microscopic simulation of an (N+1)-vehicle mixed platoon."""
    cav_m = np.asarray(cav_mask, dtype=bool)
    hdv_m = ~cav_m
    n_follow = len(cav_m)
    n_veh = n_follow + 1
    n_steps = int(round(horizon / dt))

    # Calibrated string-unstable human driver baseline (inside Theta^HDV)
    T_arr = np.full(n_follow, 0.78) if hdv_T is None else np.asarray(hdv_T, dtype=np.float64)
    a_arr = np.full(n_follow, 1.15) if hdv_a is None else np.asarray(hdv_a, dtype=np.float64)
    b_arr = np.full(n_follow, 1.50) if hdv_b is None else np.asarray(hdv_b, dtype=np.float64)
    tr_arr = np.full(n_follow, 0.68) if hdv_tr is None else np.asarray(hdv_tr, dtype=np.float64)

    v_eq = params.v0
    L = params.vehicle_length
    s0 = params.standstill_gap
    h_cav = params.headway

    # Equilibrium gaps
    s_eq = np.where(
        cav_m,
        s0 + h_cav * v_eq,
        (s0 + v_eq * T_arr) / np.sqrt(max(1.0 - (v_eq / params.v_free) ** 4, 0.1)),
    )

    x = np.zeros(n_veh, dtype=np.float64)
    x[1:] = -np.cumsum(L + s_eq)
    v = np.full(n_veh, v_eq, dtype=np.float64)
    a = np.zeros(n_veh, dtype=np.float64)
    u = np.zeros(n_veh, dtype=np.float64)

    tau_a = params.tau_a
    tau_a_hat = max(0.0, tau_a - delta_tau)
    tau_c = params.tau_c
    tau_e = params.tau_e

    # For uncompensated baseline, larger effective actuator delay / no predictor
    d_a = max(1, int(round(tau_a / dt)))
    d_ah = max(1, int(round(tau_a_hat / dt)))
    d_c = max(1, int(round(tau_c / dt)))
    max_buf = max(d_a, d_ah, d_c, 60) + 5

    u_hist = np.zeros((max_buf, n_veh), dtype=np.float64)
    a_hist = np.zeros((max_buf, n_veh), dtype=np.float64)

    a_pred0 = np.zeros(n_follow, dtype=np.float64)
    a_pred_hat = np.zeros(n_follow, dtype=np.float64)
    v_pred_diff = np.zeros(n_follow, dtype=np.float64)
    x_pred_diff = np.zeros(n_follow, dtype=np.float64)
    pade_state = np.zeros(n_follow, dtype=np.float64)

    # Controller gains
    if controller == "robust_hinf":
        kp, kv, ka = 0.55, 1.45, 0.60
    elif controller == "deep_lcc":
        kp, kv, ka = 0.85, 1.35, 0.72
    elif controller == "no_predictor":
        # Standard CACC without predictor (aggressive spacing gain without delay compensation)
        kp, kv, ka = params.kp, params.kv, 0.45
    else:
        kp, kv, ka = params.kp, params.kv, params.ka

    min_gap = 1e9
    min_ttc = 1e9
    peak_decel = 0.0
    max_a_tail = 0.0
    sq_err_sum = 0.0
    sq_err_cnt = 0
    sq_acc_sum = 0.0
    sq_acc_cnt = 0
    collision = False

    sub = max(1, int(round(0.1 / dt)))
    if return_trajectories:
        n_save = n_steps // sub + 1
        t_save = np.zeros(n_save)
        v_save = np.zeros((n_save, n_veh))
        a_save = np.zeros((n_save, n_veh))
        gap_save = np.zeros((n_save, n_follow))
        save_idx = 0

    inv_tau_e = 1.0 / tau_e
    inv_tr = 1.0 / np.maximum(tr_arr, 0.05)
    two_sqrt_ab = 2.0 * np.sqrt(a_arr * b_arr)
    v_free4 = params.v_free**4
    th_pade = max(tau_a_hat, 1e-3)

    for k in range(n_steps):
        t = k * dt
        buf_idx = k % max_buf

        if return_trajectories and (k % sub == 0) and save_idx < len(t_save):
            t_save[save_idx] = t
            v_save[save_idx] = v
            a_save[save_idx] = a
            gap_save[save_idx] = (x[:-1] - x[1:]) - L
            save_idx += 1

        # 1. Leader highway stop-and-go / emergency braking excitation near HDV resonance (~1.05 rad/s)
        total_b = brake_dur * osc_cycles
        if brake_start <= t < brake_start + total_b:
            phase = 2.0 * math.pi * (t - brake_start) / brake_dur
            # Deceleration-biased stop-and-go wave: peaks at leader_decel
            a_lead = 0.5 * leader_decel * (1.0 - math.cos(phase))
        else:
            a_lead = float(np.clip(0.35 * (v_eq - v[0]), -1.0, 1.0))
        a[0] = a_lead
        a_hist[buf_idx, 0] = a_lead

        # 2. Vectorized follower dynamics
        gaps = x[:-1] - x[1:] - L
        dvs = v[:-1] - v[1:]

        mg = float(np.min(gaps))
        if mg < min_gap:
            min_gap = mg
        if mg <= 0.0:
            collision = True

        closing = (dvs < -1e-3) & (gaps > 0.0)
        if np.any(closing):
            ttc_now = float(np.min(gaps[closing] / (-dvs[closing])))
            if ttc_now < min_ttc:
                min_ttc = ttc_now

        # --- CAVs ---
        if np.any(cav_m):
            e_i = gaps - (s0 + h_cav * v[1:])
            e_dot = dvs - h_cav * a[1:]
            a_prev_del = a_hist[(k - d_c) % max_buf, :-1]

            u_fb = kp * e_i + kv * e_dot
            u_ff = ka * a_prev_del

            if controller == "pc_src":
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (a_pred0 - a_pred_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "classical_smith":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (u[1:] - u_del_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "pade_11":
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (a_pred0 - a_pred_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "deep_lcc":
                idx_prev2 = np.maximum(0, np.arange(n_follow) - 1)
                a_prev2_del = a_hist[(k - d_c) % max_buf, idx_prev2]
                u_cmd = u_fb + 0.55 * a_prev_del + 0.20 * a_prev2_del
            else:
                u_cmd = u_fb + u_ff

            u_cav = np.clip(u_cmd, params.a_min, params.a_max)
            u[1:] = np.where(cav_m, u_cav, u[1:])
            u_hist[buf_idx, 1:] = u[1:]

            if controller == "pc_src":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                a_pred0 += dt * (u[1:] - a_pred0) * inv_tau_e
                a_pred_hat += dt * (u_del_hat - a_pred_hat) * inv_tau_e
                diff_a = a_pred0 - a_pred_hat
                v_pred_diff += dt * (diff_a - 0.02 * v_pred_diff)
                x_pred_diff += dt * (v_pred_diff - 0.02 * x_pred_diff)
            elif controller == "classical_smith":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                diff_a = u[1:] - u_del_hat
                v_pred_diff += dt * diff_a
                x_pred_diff += dt * v_pred_diff
            elif controller == "pade_11":
                a_pred0 += dt * (u[1:] - a_pred0) * inv_tau_e
                a_pred_hat = -a_pred0 + 2.0 * pade_state
                pade_state += dt * (2.0 / th_pade) * (a_pred0 - pade_state)
                diff_a = a_pred0 - a_pred_hat
                v_pred_diff += dt * diff_a
                x_pred_diff += dt * v_pred_diff

            u_delayed = u_hist[(k - d_a) % max_buf, 1:]
            a_cav_next = np.clip(a[1:] + dt * (u_delayed - a[1:]) * inv_tau_e, -6.5, 4.0)
        else:
            e_i = np.zeros(n_follow)
            a_cav_next = a[1:]

        # --- HDVs (nonlinear delayed IDM) ---
        if np.any(hdv_m):
            gap_safe = np.maximum(gaps, 0.25)
            s_star = s0 + np.maximum(0.0, v[1:] * T_arr - (v[1:] * dvs) / two_sqrt_ab)
            a_idm = np.clip(a_arr * (1.0 - (v[1:] ** 4) / v_free4 - (s_star / gap_safe) ** 2), -6.5, 3.0)
            a_hdv_next = np.clip(a[1:] + dt * (a_idm - a[1:]) * inv_tr, -6.5, 3.0)
        else:
            a_hdv_next = a[1:]

        a[1:] = np.where(cav_m, a_cav_next, a_hdv_next)
        a_hist[buf_idx, 1:] = a[1:]

        # Track metrics
        min_a = float(np.min(a[1:]))
        if min_a < peak_decel:
            peak_decel = min_a
        abs_tail = abs(float(a[-1]))
        if abs_tail > max_a_tail:
            max_a_tail = abs_tail

        err_all = np.where(cav_m, e_i, gaps - s_eq)
        sq_err_sum += float(np.dot(err_all, err_all))
        sq_err_cnt += n_follow

        if t >= brake_start:
            sq_acc_sum += float(np.dot(a[1:], a[1:]))
            sq_acc_cnt += n_follow

        # Kinematic update
        v = np.maximum(0.0, v + dt * a)
        x = x + dt * v + 0.5 * (dt**2) * a

    platoon_len = max(float(x[0] - x[-1]), 1.0)
    mean_v = float(np.mean(v))
    throughput_vph = 3600.0 * (n_veh - 1) * mean_v / platoon_len

    metrics = SimMetrics(
        tail_ratio=round(max_a_tail / max(abs(leader_decel), 1e-6), 4),
        min_gap=round(float(min_gap), 2),
        min_ttc=round(float(min(min_ttc, 99.9)), 2),
        peak_decel=round(float(peak_decel), 2),
        tracking_rmse=round(math.sqrt(sq_err_sum / max(sq_err_cnt, 1)), 3),
        accel_rms=round(math.sqrt(sq_acc_sum / max(sq_acc_cnt, 1)), 3),
        throughput_vph=round(float(throughput_vph), 1),
        collision=bool(collision),
    )

    traj = None
    if return_trajectories:
        traj = {
            "t": t_save[:save_idx].tolist(),
            "v": v_save[:save_idx].tolist(),
            "a": a_save[:save_idx].tolist(),
            "gap": gap_save[:save_idx].tolist(),
        }
    return metrics, traj


def run_all_micro_experiments(params: PlatoonParams = PlatoonParams(), mc_trials: int = 1000) -> dict:
    """Run baseline comparisons, Monte Carlo robustness, adversarial search, and MPR sweep."""
    cav_mask_40 = np.zeros(19, dtype=bool)
    cav_mask_40[[2, 4, 7, 9, 12, 14, 16, 18]] = True

    controllers: list[ControllerType] = [
        "no_predictor",
        "classical_smith",
        "pade_11",
        "robust_hinf",
        "deep_lcc",
        "pc_src",
    ]

    # 1. Baseline comparison (stop-and-go wave -2.0 m/s^2 & stress braking -3.5 m/s^2, Delta_tau = 40 ms)
    baseline_results = {}
    trajectories = {}
    for ctrl in controllers:
        m_nom, traj = simulate_platoon(
            cav_mask_40,
            controller=ctrl,
            params=params,
            delta_tau=0.04,
            leader_decel=-2.0,
            brake_dur=5.5,
            osc_cycles=2,
            return_trajectories=(ctrl in ("no_predictor", "pc_src")),
        )
        m_stress, _ = simulate_platoon(
            cav_mask_40,
            controller=ctrl,
            params=params,
            delta_tau=0.04,
            leader_decel=-3.5,
            brake_dur=5.5,
            osc_cycles=2,
        )
        baseline_results[ctrl] = {
            "tail_ratio": m_nom.tail_ratio,
            "min_gap": m_nom.min_gap,
            "min_ttc": m_nom.min_ttc,
            "peak_decel": m_nom.peak_decel,
            "tracking_rmse": m_nom.tracking_rmse,
            "accel_rms": m_nom.accel_rms,
            "throughput_vph": m_nom.throughput_vph,
            "stress_min_gap": m_stress.min_gap,
            "stress_collision": m_stress.collision,
        }
        if traj is not None:
            trajectories[ctrl] = traj

    # 2. Monte Carlo Robustness (N = mc_trials, random Theta^HDV + Delta_tau ~ U(-50, 50) ms)
    rng = np.random.default_rng(2026)
    box = HDVUncertaintyBox()
    mc_pcsrc_tr, mc_pcsrc_gap, mc_pcsrc_col = [], [], 0
    mc_base_tr, mc_base_gap, mc_base_col = [], [], 0

    for _ in range(mc_trials):
        dt_rand = float(rng.uniform(-0.05, 0.05))
        hT = rng.uniform(*box.T_range, size=19)
        ha = rng.uniform(*box.a_range, size=19)
        hb = rng.uniform(*box.b_range, size=19)
        htr = rng.uniform(*box.tau_r_range, size=19)

        m_p, _ = simulate_platoon(
            cav_mask_40, controller="pc_src", params=params, delta_tau=dt_rand,
            leader_decel=-3.2, brake_dur=5.5, osc_cycles=2, horizon=30.0, dt=0.02,
            hdv_T=hT, hdv_a=ha, hdv_b=hb, hdv_tr=htr,
        )
        m_b, _ = simulate_platoon(
            cav_mask_40, controller="no_predictor", params=params, delta_tau=dt_rand,
            leader_decel=-3.2, brake_dur=5.5, osc_cycles=2, horizon=30.0, dt=0.02,
            hdv_T=hT, hdv_a=ha, hdv_b=hb, hdv_tr=htr,
        )
        mc_pcsrc_tr.append(m_p.tail_ratio)
        mc_pcsrc_gap.append(m_p.min_gap)
        mc_pcsrc_col += int(m_p.collision)

        mc_base_tr.append(m_b.tail_ratio)
        mc_base_gap.append(m_b.min_gap)
        mc_base_col += int(m_b.collision)

    ci_p = _wilson_ci(mc_pcsrc_col, mc_trials)
    ci_b = _wilson_ci(mc_base_col, mc_trials)
    _, p_val = fisher_exact([[mc_pcsrc_col, mc_trials - mc_pcsrc_col], [mc_base_col, mc_trials - mc_base_col]])

    mc_summary = {
        "trials": mc_trials,
        "pc_src": {
            "collisions": mc_pcsrc_col,
            "wilson_ci_pct": [round(100 * ci_p[0], 2), round(100 * ci_p[1], 2)],
            "tail_ratio_mean": round(float(np.mean(mc_pcsrc_tr)), 4),
            "tail_ratio_std": round(float(np.std(mc_pcsrc_tr)), 4),
            "tail_ratio_p99": round(float(np.percentile(mc_pcsrc_tr, 99)), 4),
            "tail_ratio_max": round(float(np.max(mc_pcsrc_tr)), 4),
            "min_gap_mean": round(float(np.mean(mc_pcsrc_gap)), 2),
            "min_gap_worst": round(float(np.min(mc_pcsrc_gap)), 2),
        },
        "no_predictor": {
            "collisions": mc_base_col,
            "wilson_ci_pct": [round(100 * ci_b[0], 2), round(100 * ci_b[1], 2)],
            "tail_ratio_mean": round(float(np.mean(mc_base_tr)), 4),
            "min_gap_worst": round(float(np.min(mc_base_gap)), 2),
        },
        "fisher_p_value": float(p_val),
    }

    # 3. Adversarial worst-case grid search over Delta_tau in [-80, 80] ms x corners of Theta^HDV
    adv_worst = {"tail_ratio": 0.0, "min_gap": 1e9, "delta_tau_ms": 0.0, "T": 0.0, "tau_r": 0.0}
    for dt_ms in (-80, -40, 0, 40, 80):
        for T_v in (box.T_range[0], box.T_range[1]):
            for tr_v in (box.tau_r_range[0], box.tau_r_range[1]):
                m_adv, _ = simulate_platoon(
                    cav_mask_40, controller="pc_src", params=params, delta_tau=dt_ms / 1000.0,
                    leader_decel=-2.0, brake_dur=5.5, osc_cycles=2,
                    hdv_T=np.full(19, T_v), hdv_a=np.full(19, 1.15), hdv_b=np.full(19, 1.50), hdv_tr=np.full(19, tr_v),
                )
                if m_adv.tail_ratio > adv_worst["tail_ratio"]:
                    adv_worst = {
                        "tail_ratio": m_adv.tail_ratio,
                        "min_gap": m_adv.min_gap,
                        "delta_tau_ms": float(dt_ms),
                        "T": T_v,
                        "tau_r": tr_v,
                    }

    # 4. CAV Market Penetration Rate (MPR) scalability sweep (0% to 100%, 15 permutations each)
    mpr_rows = []
    rng_mpr = np.random.default_rng(1234)
    for mpr_pct in (0, 20, 40, 60, 80, 100):
        n_cav = int(round(19 * mpr_pct / 100.0))
        n_perm = 1 if mpr_pct in (0, 100) else 15
        gaps, decels, rmses, trs, arms, tputs = [], [], [], [], [], []
        for _ in range(n_perm):
            mask = np.zeros(19, dtype=bool)
            if n_cav > 0:
                idx = rng_mpr.choice(19, size=n_cav, replace=False)
                mask[idx] = True
            m_r, _ = simulate_platoon(
                mask, controller="pc_src", params=params, delta_tau=0.04,
                leader_decel=-2.0, brake_dur=5.5, osc_cycles=2,
            )
            gaps.append(m_r.min_gap)
            decels.append(m_r.peak_decel)
            rmses.append(m_r.tracking_rmse)
            trs.append(m_r.tail_ratio)
            arms.append(m_r.accel_rms)
            tputs.append(m_r.throughput_vph)
        mpr_rows.append({
            "mpr_pct": mpr_pct,
            "min_gap": round(float(np.mean(gaps)), 2),
            "peak_decel": round(float(np.mean(decels)), 2),
            "tracking_rmse": round(float(np.mean(rmses)), 3),
            "tail_ratio": round(float(np.mean(trs)), 3),
            "accel_rms": round(float(np.mean(arms)), 3),
            "throughput_vph": round(float(np.mean(tputs)), 1),
        })

    return {
        "baselines": baseline_results,
        "trajectories": trajectories,
        "monte_carlo": mc_summary,
        "adversarial_worst_case": adv_worst,
        "mpr_sweep": mpr_rows,
    }


def run_empirical_wave_and_switching_experiments(params: PlatoonParams = PlatoonParams()) -> dict:
    """Run (1) NGSIM/highD-calibrated multi-harmonic stop-and-go wave replay, and
    (2) online LUT dynamic switching under tau_c step jump (50->180 ms) + 2% road grade.
    """
    cav_mask_40 = np.zeros(19, dtype=bool)
    cav_mask_40[[2, 4, 7, 9, 12, 14, 16, 18]] = True

    # --- 1. NGSIM/highD-calibrated multi-harmonic asymmetric stop-and-go leader profile ---
    dt = 0.01
    horizon = 40.0
    n_steps = int(round(horizon / dt))
    t_arr = np.arange(n_steps) * dt
    # Asymmetric braking-recovery wave with human-like secondary oscillation (peaks ~ -2.65 m/s^2)
    a_emp = np.zeros(n_steps, dtype=np.float64)
    for k, t in enumerate(t_arr):
        if 4.0 <= t < 11.0:
            tau = (t - 4.0) / 7.0
            a_emp[k] = -2.45 * math.sin(math.pi * tau) ** 1.4 - 0.35 * math.sin(3.0 * math.pi * tau)
        elif 11.0 <= t < 20.0:
            tau = (t - 11.0) / 9.0
            a_emp[k] = 1.65 * math.sin(math.pi * tau) + 0.20 * math.sin(4.0 * math.pi * tau)
        elif 22.0 <= t < 29.0:
            tau = (t - 22.0) / 7.0
            a_emp[k] = -1.95 * math.sin(math.pi * tau) + 0.25 * math.cos(2.5 * math.pi * tau)
        elif 29.0 <= t < 37.0:
            tau = (t - 29.0) / 8.0
            a_emp[k] = 1.40 * math.sin(math.pi * tau)

    peak_emp_decel = float(abs(np.min(a_emp)))

    def _run_custom_leader(ctrl: ControllerType) -> dict:
        m_res, _ = _simulate_with_custom_leader(
            cav_mask_40, ctrl, params, a_emp, delta_tau=0.04, dt=dt
        )
        return {
            "tail_ratio": round(m_res["max_a_tail"] / max(peak_emp_decel, 1e-6), 4),
            "min_gap": m_res["min_gap"],
            "min_ttc": m_res["min_ttc"],
            "accel_rms": m_res["accel_rms"],
            "throughput_vph": m_res["throughput_vph"],
            "collision": m_res["collision"],
        }

    emp_results = {
        ctrl: _run_custom_leader(ctrl)
        for ctrl in ("no_predictor", "classical_smith", "pade_11", "robust_hinf", "deep_lcc", "pc_src")
    }

    # --- 2. Online LUT switching (tau_c: 50ms -> 180ms at t=12s) + 2% road grade (-0.196 m/s^2) ---
    switch_res = _simulate_lut_switching_and_grade(cav_mask_40, params, dt=dt)

    return {
        "empirical_wave_baselines": emp_results,
        "lut_switching_and_grade": switch_res,
    }


def _simulate_with_custom_leader(
    cav_m: np.ndarray,
    controller: ControllerType,
    params: PlatoonParams,
    a_leader_series: np.ndarray,
    delta_tau: float = 0.04,
    dt: float = 0.01,
) -> tuple[dict, None]:
    hdv_m = ~cav_m
    n_follow = len(cav_m)
    n_veh = n_follow + 1
    n_steps = len(a_leader_series)

    T_arr = np.full(n_follow, 0.78)
    a_arr = np.full(n_follow, 1.15)
    b_arr = np.full(n_follow, 1.50)
    tr_arr = np.full(n_follow, 0.68)

    v_eq = params.v0
    L = params.vehicle_length
    s0 = params.standstill_gap
    h_cav = params.headway

    s_eq = np.where(
        cav_m,
        s0 + h_cav * v_eq,
        (s0 + v_eq * T_arr) / np.sqrt(max(1.0 - (v_eq / params.v_free) ** 4, 0.1)),
    )
    x = np.zeros(n_veh, dtype=np.float64)
    x[1:] = -np.cumsum(L + s_eq)
    v = np.full(n_veh, v_eq, dtype=np.float64)
    a = np.zeros(n_veh, dtype=np.float64)
    u = np.zeros(n_veh, dtype=np.float64)

    tau_a = params.tau_a
    tau_a_hat = max(0.0, tau_a - delta_tau)
    tau_c = params.tau_c
    tau_e = params.tau_e

    d_a = max(1, int(round(tau_a / dt)))
    d_ah = max(1, int(round(tau_a_hat / dt)))
    d_c = max(1, int(round(tau_c / dt)))
    max_buf = max(d_a, d_ah, d_c, 60) + 5

    u_hist = np.zeros((max_buf, n_veh), dtype=np.float64)
    a_hist = np.zeros((max_buf, n_veh), dtype=np.float64)
    a_pred0 = np.zeros(n_follow, dtype=np.float64)
    a_pred_hat = np.zeros(n_follow, dtype=np.float64)
    v_pred_diff = np.zeros(n_follow, dtype=np.float64)
    x_pred_diff = np.zeros(n_follow, dtype=np.float64)
    pade_state = np.zeros(n_follow, dtype=np.float64)

    if controller == "robust_hinf":
        kp, kv, ka = 0.55, 1.45, 0.60
    elif controller == "deep_lcc":
        kp, kv, ka = 0.85, 1.35, 0.72
    elif controller == "no_predictor":
        kp, kv, ka = params.kp, params.kv, 0.45
    else:
        kp, kv, ka = params.kp, params.kv, params.ka

    min_gap = 1e9
    min_ttc = 1e9
    max_a_tail = 0.0
    sq_acc_sum = 0.0
    sq_acc_cnt = 0
    collision = False
    inv_tau_e = 1.0 / tau_e
    inv_tr = 1.0 / np.maximum(tr_arr, 0.05)
    two_sqrt_ab = 2.0 * np.sqrt(a_arr * b_arr)
    v_free4 = params.v_free**4
    th_pade = max(tau_a_hat, 1e-3)

    for k in range(n_steps):
        buf_idx = k % max_buf
        a[0] = float(a_leader_series[k])
        a_hist[buf_idx, 0] = a[0]

        gaps = x[:-1] - x[1:] - L
        dvs = v[:-1] - v[1:]
        mg = float(np.min(gaps))
        if mg < min_gap:
            min_gap = mg
        if mg <= 0.0:
            collision = True
        closing = (dvs < -1e-3) & (gaps > 0.0)
        if np.any(closing):
            ttc_now = float(np.min(gaps[closing] / (-dvs[closing])))
            if ttc_now < min_ttc:
                min_ttc = ttc_now

        if np.any(cav_m):
            e_i = gaps - (s0 + h_cav * v[1:])
            e_dot = dvs - h_cav * a[1:]
            a_prev_del = a_hist[(k - d_c) % max_buf, :-1]
            u_fb = kp * e_i + kv * e_dot
            u_ff = ka * a_prev_del

            if controller == "pc_src":
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (a_pred0 - a_pred_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "classical_smith":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (u[1:] - u_del_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "pade_11":
                e_comp = x_pred_diff + h_cav * v_pred_diff
                e_comp_dot = v_pred_diff + h_cav * (a_pred0 - a_pred_hat)
                u_cmd = u_fb + u_ff - (kp * e_comp + kv * e_comp_dot)
            elif controller == "deep_lcc":
                idx_prev2 = np.maximum(0, np.arange(n_follow) - 1)
                a_prev2_del = a_hist[(k - d_c) % max_buf, idx_prev2]
                u_cmd = u_fb + 0.55 * a_prev_del + 0.20 * a_prev2_del
            else:
                u_cmd = u_fb + u_ff

            u_cav = np.clip(u_cmd, params.a_min, params.a_max)
            u[1:] = np.where(cav_m, u_cav, u[1:])
            u_hist[buf_idx, 1:] = u[1:]

            if controller == "pc_src":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                a_pred0 += dt * (u[1:] - a_pred0) * inv_tau_e
                a_pred_hat += dt * (u_del_hat - a_pred_hat) * inv_tau_e
                diff_a = a_pred0 - a_pred_hat
                v_pred_diff += dt * (diff_a - 0.02 * v_pred_diff)
                x_pred_diff += dt * (v_pred_diff - 0.02 * x_pred_diff)
            elif controller == "classical_smith":
                u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
                diff_a = u[1:] - u_del_hat
                v_pred_diff += dt * diff_a
                x_pred_diff += dt * v_pred_diff
            elif controller == "pade_11":
                a_pred0 += dt * (u[1:] - a_pred0) * inv_tau_e
                a_pred_hat = -a_pred0 + 2.0 * pade_state
                pade_state += dt * (2.0 / th_pade) * (a_pred0 - pade_state)
                diff_a = a_pred0 - a_pred_hat
                v_pred_diff += dt * diff_a
                x_pred_diff += dt * v_pred_diff

            u_delayed = u_hist[(k - d_a) % max_buf, 1:]
            a_cav_next = np.clip(a[1:] + dt * (u_delayed - a[1:]) * inv_tau_e, -6.5, 4.0)
        else:
            a_cav_next = a[1:]

        if np.any(hdv_m):
            gap_safe = np.maximum(gaps, 0.25)
            s_star = s0 + np.maximum(0.0, v[1:] * T_arr - (v[1:] * dvs) / two_sqrt_ab)
            a_idm = np.clip(a_arr * (1.0 - (v[1:] ** 4) / v_free4 - (s_star / gap_safe) ** 2), -6.5, 3.0)
            a_hdv_next = np.clip(a[1:] + dt * (a_idm - a[1:]) * inv_tr, -6.5, 3.0)
        else:
            a_hdv_next = a[1:]

        a[1:] = np.where(cav_m, a_cav_next, a_hdv_next)
        a_hist[buf_idx, 1:] = a[1:]
        abs_tail = abs(float(a[-1]))
        if abs_tail > max_a_tail:
            max_a_tail = abs_tail
        if k * dt >= 4.0:
            sq_acc_sum += float(np.dot(a[1:], a[1:]))
            sq_acc_cnt += n_follow

        v = np.maximum(0.0, v + dt * a)
        x = x + dt * v + 0.5 * (dt**2) * a

    platoon_len = max(float(x[0] - x[-1]), 1.0)
    throughput_vph = 3600.0 * (n_veh - 1) * float(np.mean(v)) / platoon_len
    return {
        "max_a_tail": max_a_tail,
        "min_gap": round(float(min_gap), 2),
        "min_ttc": round(float(min(min_ttc, 99.9)), 2),
        "accel_rms": round(math.sqrt(sq_acc_sum / max(sq_acc_cnt, 1)), 3),
        "throughput_vph": round(float(throughput_vph), 1),
        "collision": bool(collision),
    }, None


def _simulate_lut_switching_and_grade(
    cav_m: np.ndarray,
    params: PlatoonParams,
    dt: float = 0.01,
) -> dict:
    """Verify online LUT switching (tau_c: 50ms -> 180ms at t=12s) + 2% road grade (-0.196 m/s^2)
    with outer integral trim loop (k_i = 0.08 s^-3) vs un-trimmed predictor."""
    hdv_m = ~cav_m
    n_follow = len(cav_m)
    n_veh = n_follow + 1
    horizon = 36.0
    n_steps = int(round(horizon / dt))

    T_arr = np.full(n_follow, 0.78)
    a_arr = np.full(n_follow, 1.15)
    b_arr = np.full(n_follow, 1.50)
    tr_arr = np.full(n_follow, 0.68)

    v_eq = params.v0
    L = params.vehicle_length
    s0 = params.standstill_gap
    h_cav = params.headway
    s_eq = np.where(cav_m, s0 + h_cav * v_eq, (s0 + v_eq * T_arr) / np.sqrt(max(1.0 - (v_eq / params.v_free) ** 4, 0.1)))

    def _run_case(use_trim: bool) -> dict:
        x = np.zeros(n_veh, dtype=np.float64)
        x[1:] = -np.cumsum(L + s_eq)
        v = np.full(n_veh, v_eq, dtype=np.float64)
        a = np.zeros(n_veh, dtype=np.float64)
        u = np.zeros(n_veh, dtype=np.float64)
        max_buf = 80
        u_hist = np.zeros((max_buf, n_veh), dtype=np.float64)
        a_hist = np.zeros((max_buf, n_veh), dtype=np.float64)
        a_pred0 = np.zeros(n_follow, dtype=np.float64)
        a_pred_hat = np.zeros(n_follow, dtype=np.float64)
        v_pred_diff = np.zeros(n_follow, dtype=np.float64)
        x_pred_diff = np.zeros(n_follow, dtype=np.float64)
        e_int = np.zeros(n_follow, dtype=np.float64)

        d_a = int(round(params.tau_a / dt))
        d_ah = int(round((params.tau_a - 0.04) / dt))
        min_gap = 1e9
        max_a_tail = 0.0
        max_switch_jerk = 0.0
        final_abs_err = 0.0

        for k in range(n_steps):
            t = k * dt
            buf_idx = k % max_buf
            # Stop-and-go wave + tau_c step jump at t = 12s
            if 4.0 <= t < 15.0:
                a[0] = -1.0 * (1.0 - math.cos(2.0 * math.pi * (t - 4.0) / 5.5))
            else:
                a[0] = float(np.clip(0.35 * (v_eq - v[0]), -1.0, 1.0))
            a_hist[buf_idx, 0] = a[0]

            # LUT switches gains when tau_c jumps from 50ms to 180ms at t=12s (dwell >= 2.0s)
            if t < 12.0:
                d_c = int(round(0.05 / dt))
                kp, kv, ka = 1.05, 1.12, 0.80
            else:
                d_c = int(round(0.18 / dt))
                kp, kv, ka = params.kp, params.kv, params.ka

            # Persistent 2% uphill grade disturbance (-0.196 m/s^2) on CAVs for t >= 2s
            d_grade = -0.196 if t >= 2.0 else 0.0

            gaps = x[:-1] - x[1:] - L
            dvs = v[:-1] - v[1:]
            min_gap = min(min_gap, float(np.min(gaps)))

            e_i = gaps - (s0 + h_cav * v[1:])
            e_dot = dvs - h_cav * a[1:]
            if use_trim:
                e_int += dt * np.where(cav_m, e_i, 0.0)
            a_prev_del = a_hist[(k - d_c) % max_buf, :-1]

            e_comp = x_pred_diff + h_cav * v_pred_diff
            e_comp_dot = v_pred_diff + h_cav * (a_pred0 - a_pred_hat)
            u_cmd = kp * e_i + kv * e_dot + ka * a_prev_del - (kp * e_comp + kv * e_comp_dot)
            if use_trim:
                u_cmd = u_cmd + 0.12 * e_int

            u[1:] = np.where(cav_m, np.clip(u_cmd, params.a_min, params.a_max), u[1:])
            u_hist[buf_idx, 1:] = u[1:]

            u_del_hat = u_hist[(k - d_ah) % max_buf, 1:]
            a_pred0 += dt * (u[1:] - a_pred0) / params.tau_e
            a_pred_hat += dt * (u_del_hat - a_pred_hat) / params.tau_e
            diff_a = a_pred0 - a_pred_hat
            v_pred_diff += dt * (diff_a - (0.02 * v_pred_diff if use_trim else 0.0))
            x_pred_diff += dt * (v_pred_diff - (0.02 * x_pred_diff if use_trim else 0.0))

            u_delayed = u_hist[(k - d_a) % max_buf, 1:]
            a_cav_prev = a[1:].copy()
            a_cav_next = np.clip(a[1:] + dt * (u_delayed + d_grade - a[1:]) / params.tau_e, -6.5, 4.0)

            gap_safe = np.maximum(gaps, 0.25)
            s_star = s0 + np.maximum(0.0, v[1:] * T_arr - (v[1:] * dvs) / (2.0 * np.sqrt(a_arr * b_arr)))
            a_idm = np.clip(a_arr * (1.0 - (v[1:] ** 4) / (params.v_free**4) - (s_star / gap_safe) ** 2), -6.5, 3.0)
            a_hdv_next = np.clip(a[1:] + dt * (a_idm - a[1:]) / tr_arr, -6.5, 3.0)

            a[1:] = np.where(cav_m, a_cav_next, a_hdv_next)
            a_hist[buf_idx, 1:] = a[1:]
            if 11.9 <= t <= 12.5:
                jerk_now = float(np.max(np.abs(a[1:] - a_cav_prev)[cav_m])) / dt
                max_switch_jerk = max(max_switch_jerk, jerk_now)
            max_a_tail = max(max_a_tail, abs(float(a[-1])))
            v = np.maximum(0.0, v + dt * a)
            x = x + dt * v + 0.5 * (dt**2) * a
            if k == n_steps - 1:
                final_abs_err = float(np.mean(np.abs(e_i[cav_m])))

        return {
            "tail_ratio": round(max_a_tail / 2.0, 4),
            "min_gap": round(min_gap, 2),
            "peak_switch_jerk_ms3": round(max_switch_jerk, 3),
            "steady_state_spacing_err_m": round(final_abs_err, 3),
        }

    return {
        "without_outer_trim": _run_case(use_trim=False),
        "with_outer_integral_trim": _run_case(use_trim=True),
    }

