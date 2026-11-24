"""End-to-end Google Colab CLI experiment suite for PC-SRC T-ITS submission.

Runs all analytical, SOS-LMI (CVXPY/Clarabel), frequency-domain Bode,
20-vehicle nonlinear mixed-platoon micro-simulation (baselines, 1000-trial
Monte Carlo, adversarial worst-case, MPR scalability), and O(1) LUT benchmarks.

Generates:
  - colab_verified_results.json (all verified numbers for Tables I-VI)
  - fig_headway_domain.pdf      (Fig. 3: h_min vs tau_e + SOS feasibility domain)
  - fig_bode_mixed.pdf          (Fig. 4: Bode magnitude & mixed-unit attenuation)
  - fig_platoon_validation.pdf  (Fig. 5: 20-veh shockwave damping + MPR sweep)

Usage on Google Colab via `colab` CLI:
  colab run -s pcsrc scripts/run_colab_suite.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src/ is on path when invoked from repo root or Colab /content
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcsrc.analysis.asymptotic import sweep_headway_vs_actuator_lag, verify_low_freq_overlap
from pcsrc.deployment.lut import benchmark_lut
from pcsrc.models.platoon import (
    HDVUncertaintyBox,
    PlatoonParams,
    gamma_classical_smith,
    gamma_no_predictor,
    gamma_pade_cacc,
    gamma_pcsrc,
    gamma_robust_hinf,
    worst_case_hdv_envelope,
)
from pcsrc.simulation.micro_sim import run_all_micro_experiments
from pcsrc.sos.certificates import benchmark_sos_formulations


def _setup_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def run_bode_analysis(params: PlatoonParams) -> dict:
    """Compute frequency-domain Bode responses and mixed-platoon attenuation (m=2 HDVs)."""
    w = np.geomspace(0.01, 100.0, 600)
    h_bar, worst_hdv = worst_case_hdv_envelope(w, HDVUncertaintyBox(), v_e=params.v0, v_free=params.v_free, s0=params.standstill_gap)

    tau_a_hat = params.tau_a - 0.04  # Delta_tau = 40 ms mismatch
    curves = {
        "pc_src": np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "no_predictor": np.abs(gamma_no_predictor(w, params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.tau_a)),
        "classical_smith": np.abs(gamma_classical_smith(w, params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "pade_11": np.abs(gamma_pade_cacc(w, params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "robust_hinf": np.abs(gamma_robust_hinf(w, params.headway, params.tau_e, params.tau_c, params.tau_a)),
    }

    w_peak = worst_hdv["omega_peak"]
    pk_idx = int(np.argmin(np.abs(w - w_peak)))
    h_hdv_pk = float(h_bar[pk_idx])

    table_rows = {}
    for name, mag in curves.items():
        mixed_curve = (h_bar**2) * mag
        hinf_cav = float(np.max(mag))
        at_pk = float(mag[pk_idx])
        mixed_at_pk = float((h_hdv_pk**2) * at_pk)
        hinf_unit = float(np.max(mixed_curve))
        table_rows[name] = {
            "hinf_cav": round(hinf_cav, 4),
            "mag_at_hdv_peak": round(at_pk, 4),
            "mixed_at_hdv_peak": round(mixed_at_pk, 4),
            "hinf_unit": round(hinf_unit, 4),
            "mixed_stable": bool(hinf_unit <= 1.0005),
        }

    return {
        "omega": w.tolist(),
        "h_bar_hdv": h_bar.tolist(),
        "worst_hdv": worst_hdv,
        "curves": {k: v.tolist() for k, v in curves.items()},
        "table": table_rows,
    }


def generate_figures(results: dict, out_dir: Path) -> None:
    """Render 3 publication-quality IEEE double-column vector PDF figures."""
    _setup_ieee_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fig. 3: (a) h_min vs tau_e  &  (b) Certified (kp, kv) domain
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.35))

    hw = results["headway_sweep"]
    te = np.array(hw["tau_e"])
    ax1.axhline(hw["h_asymp"], color="#555555", ls="-.", lw=1.1, label=f"Low-freq bound ({hw['h_asymp']:.3f} s)")
    ax1.plot(te, hw["h_matched"], color="#C62828", ls="--", lw=1.4, marker="s", ms=2.8, label=r"Matched ($k_v=\tau_e k_p$)")
    ax1.plot(te, hw["h_coopt"], color="#1565C0", ls="-", lw=1.6, marker="o", ms=2.8, label=r"Co-optimized $(k_p,k_v,k_a)$")
    ax1.set_xlabel(r"Actuator time constant $\tau_e$ (s)")
    ax1.set_ylabel(r"Min verifiable headway $h_{\min}$ (s)")
    ax1.set_title("(a) Actuator Lag Coupling vs. Headway")
    ax1.set_xlim(0.0, 2.05)
    ax1.set_ylim(0.50, 1.35)
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)

    dom = results["sos_benchmark"]["nominal_feasibility_domain"]
    kp_g = np.array(dom["kp_grid"])
    kv_g = np.array(dom["kv_grid"])
    KP, KV = np.meshgrid(kp_g, kv_g)
    dual_m = np.array(dom["dual"])
    m2_m = np.array(dom["m2"])
    m1_m = np.array(dom["m1"])

    ax2.contourf(KP, KV, dual_m, levels=[0.5, 1.5], colors=["#BBDEFB"], alpha=0.85)
    ax2.contour(KP, KV, dual_m, levels=[0.5], colors=["#1565C0"], linewidths=1.4)
    ax2.contourf(KP, KV, m2_m, levels=[0.5, 1.5], colors=["#FFE0B2"], alpha=0.65)
    ax2.contour(KP, KV, m2_m, levels=[0.5], colors=["#EF6C00"], linewidths=1.1, linestyles="--")
    if np.any(m1_m > 0):
        ax2.contourf(KP, KV, m1_m, levels=[0.5, 1.5], colors=["#FFCDD2"], alpha=0.8)
        ax2.contour(KP, KV, m1_m, levels=[0.5], colors=["#C62828"], linewidths=1.0, linestyles=":")
    ax2.plot(1.15, 1.18, marker="*", ms=7.5, color="#1B5E20", zorder=5, label=r"Nominal $(1.15,1.18)$")
    # Proxy handles for legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#BBDEFB", edgecolor="#1565C0", label="Dual-Cert (Proposed)"),
        Patch(facecolor="#FFE0B2", edgecolor="#EF6C00", ls="--", label="Monolithic M2 (Affine $\\varepsilon$)"),
        Patch(facecolor="#FFCDD2", edgecolor="#C62828", ls=":", label="Monolithic M1 (Fixed $\\varepsilon$)"),
        plt.Line2D([], [], marker="*", color="#1B5E20", ls="None", ms=7, label="Nominal gain"),
    ]
    ax2.set_xlabel(r"Proportional gain $k_p$ ($\mathrm{s}^{-2}$)")
    ax2.set_ylabel(r"Derivative gain $k_v$ ($\mathrm{s}^{-1}$)")
    ax2.set_title(r"(b) Certified Gain Domain ($\tau_c=0.18\,$s, $\delta_{\max}=0.08\,$s)")
    ax2.grid(True)
    ax2.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=6.6)

    plt.tight_layout(w_pad=1.6)
    fig.savefig(out_dir / "fig_headway_domain.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig. 4: Bode magnitude |Gamma_CAV| and mixed unit (H_bar^HDV)^2 |Gamma_CAV|
    # ------------------------------------------------------------------
    bode = results["bode"]
    w = np.array(bode["omega"])
    h_bar = np.array(bode["h_bar_hdv"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.25))
    styles = {
        "pc_src": ("PC-SRC (Proposed)", "#1565C0", "-", 1.7),
        "no_predictor": ("No Predictor (CACC)", "#C62828", "--", 1.2),
        "classical_smith": ("Classical Smith", "#EF6C00", ":", 1.3),
        "pade_11": ("Padé [1/1]", "#6A1B9A", "-.", 1.1),
        "robust_hinf": (r"Robust $\mathcal{H}_\infty$", "#00695C", "-", 1.1),
    }

    for key, (lbl, col, ls, lw) in styles.items():
        mag = np.array(bode["curves"][key])
        ax1.semilogx(w, mag, color=col, ls=ls, lw=lw, label=lbl)
        ax2.semilogx(w, (h_bar**2) * mag, color=col, ls=ls, lw=lw, label=lbl)

    ax1.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
    ax1.set_xlim(0.05, 50.0)
    ax1.set_ylim(0.0, 1.22)
    ax1.set_xlabel(r"Frequency $\omega$ (rad/s)")
    ax1.set_ylabel(r"$|\Gamma_{\mathrm{CAV}}(j\omega)|$")
    ax1.set_title(r"(a) Isolated CAV Frequency Response ($\Delta\tau=40\,$ms)")
    ax1.grid(True, which="both")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.6)

    ax2.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlim(0.05, 10.0)
    ax2.set_ylim(0.0, 1.45)
    ax2.set_xlabel(r"Frequency $\omega$ (rad/s)")
    ax2.set_ylabel(r"$(\bar{H}^{\mathrm{HDV}}(\omega))^2 |\Gamma_{\mathrm{CAV}}(j\omega)|$")
    ax2.set_title(r"(b) Mixed-Unit Attenuation ($m=2$ Preceding HDVs)")
    ax2.grid(True, which="both")

    plt.tight_layout(w_pad=1.6)
    fig.savefig(out_dir / "fig_bode_mixed.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Fig. 5: 20-veh Mixed Platoon Shockwave Damping + MPR Operational Sweep
    # ------------------------------------------------------------------
    micro = results["micro_simulation"]
    t_b = np.array(micro["trajectories"]["no_predictor"]["t"])
    v_b = np.array(micro["trajectories"]["no_predictor"]["v"])
    t_p = np.array(micro["trajectories"]["pc_src"]["t"])
    v_p = np.array(micro["trajectories"]["pc_src"]["v"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.15))
    veh_show = [0, 4, 9, 14, 19]
    cmap = ["#212121", "#EF6C00", "#1565C0", "#2E7D32", "#C62828"]

    for idx_v, col in zip(veh_show, cmap):
        lbl = "Leader 0" if idx_v == 0 else f"Veh {idx_v}"
        ax1.plot(t_b, v_b[:, idx_v], color=col, lw=1.2 if idx_v in (0, 19) else 0.9, label=lbl)
        ax2.plot(t_p, v_p[:, idx_v], color=col, lw=1.2 if idx_v in (0, 19) else 0.9, label=lbl)

    ax1.set_title("(a) No Predictor (40% MPR)")
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel("Speed $v_i$ (m/s)")
    ax1.set_xlim(0, 35)
    ax1.set_ylim(10, 23.5)
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=6.2, ncol=2)

    ax2.set_title("(b) Proposed PC-SRC (40% MPR)")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_xlim(0, 35)
    ax2.set_ylim(10, 23.5)
    ax2.grid(True)

    # Panel (c): MPR Operational Impact (Tail ratio & Throughput)
    mpr = micro["mpr_sweep"]
    mpr_x = [r["mpr_pct"] for r in mpr]
    tr_y = [r["tail_ratio"] for r in mpr]
    tp_y = [r["throughput_vph"] for r in mpr]

    l1 = ax3.plot(mpr_x, tr_y, color="#1565C0", marker="o", ms=3.5, lw=1.5, label="Tail ratio")
    ax3.axhline(1.0, color="#1565C0", ls=":", lw=0.9, alpha=0.7)
    ax3.set_xlabel("CAV Penetration Rate (%)")
    ax3.set_ylabel("Tail Ratio", color="#1565C0")
    ax3.tick_params(axis="y", labelcolor="#1565C0")
    ax3.set_title("(c) Traffic-Flow Scalability")
    ax3.grid(True)

    ax3b = ax3.twinx()
    l2 = ax3b.plot(mpr_x, tp_y, color="#2E7D32", marker="s", ms=3.2, lw=1.4, ls="--", label="Throughput")
    ax3b.set_ylabel("Throughput (veh/h)", color="#2E7D32")
    ax3b.tick_params(axis="y", labelcolor="#2E7D32")
    lines = l1 + l2
    ax3.legend(lines, [l.get_label() for l in lines], loc="center right", frameon=True, framealpha=0.9, fontsize=6.4)

    plt.tight_layout(w_pad=1.1)
    fig.savefig(out_dir / "fig_platoon_validation.pdf")
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    params = PlatoonParams()
    out_dir = Path(os.environ.get("PCSRC_OUT_DIR", str(ROOT / "results" / "colab_run")))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Verifying low-frequency Lagrange remainder overlap...")
    overlap = verify_low_freq_overlap(params)
    print(f"      c2_nom={overlap['c2_nom']:.4f}, c2_worst={overlap['c2_worst']:.4f}, "
          f"M4={overlap['M4_bound']:.4f}, w_upper={overlap['omega_upper']:.4f} rad/s, "
          f"overlap_valid={overlap['overlap_valid']}")

    print("[2/6] Sweeping minimum verifiable headway h_min vs actuator lag tau_e...")
    hw_sweep = sweep_headway_vs_actuator_lag(params)
    print(f"      h_asymp={hw_sweep['h_asymp']:.4f} s, matched[0]={hw_sweep['h_matched'][0]:.4f} s, "
          f"coopt[0]={hw_sweep['h_coopt'][0]:.4f} s")

    print("[3/6] Running real CVXPY SOS-LMI benchmarks (Cert A, Cert B, M1, M2, M3) & tau_c sweeps...")
    sos_res = benchmark_sos_formulations(params)
    sp = sos_res["single_point_sdp"]
    print(f"      Dual-Cert feasible={sp['dual_combined']['feasible']} ({sp['dual_combined']['time_ms']} ms), "
          f"CertA_Sturm={sp['cert_a_sturm']['time_ms']} ms, M3={sp['mono_m3']['time_ms']} ms")

    print("[4/6] Computing frequency-domain Bode & mixed-platoon attenuation (m=2 HDVs)...")
    bode_res = run_bode_analysis(params)
    print(f"      Worst HDV peak={bode_res['worst_hdv']['peak']:.4f} at w={bode_res['worst_hdv']['omega_peak']:.3f} rad/s")
    for k, v in bode_res["table"].items():
        print(f"      {k:16s}: ||G_CAV||={v['hinf_cav']:.4f}, ||G_unit||={v['hinf_unit']:.4f}, stable={v['mixed_stable']}")

    print("[5/6] Running 20-vehicle nonlinear micro-simulation (baselines + 1000 MC + adversarial + MPR)...")
    mc_trials = int(os.environ.get("PCSRC_MC_TRIALS", "1000"))
    micro_res = run_all_micro_experiments(params, mc_trials=mc_trials)
    print(f"      PC-SRC tail_ratio={micro_res['baselines']['pc_src']['tail_ratio']}, "
          f"NoPred tail_ratio={micro_res['baselines']['no_predictor']['tail_ratio']}")
    print(f"      MC ({mc_trials} trials): PC-SRC col={micro_res['monte_carlo']['pc_src']['collisions']}, "
          f"NoPred col={micro_res['monte_carlo']['no_predictor']['collisions']}, "
          f"p={micro_res['monte_carlo']['fisher_p_value']:.4g}")

    print("[6/6] Benchmarking O(1) 3D LUT latency & rendering IEEE PDF figures...")
    lut_res = benchmark_lut(n_queries=20000)
    print(f"      LUT memory={lut_res['memory_bytes']} B ({lut_res['memory_kb']} KB), "
          f"Flat mean={lut_res['flat_index_buffer']['mean_us']} us")

    full_results = {
        "meta": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_elapsed_s": round(time.perf_counter() - t_start, 2),
        },
        "overlap_verification": overlap,
        "headway_sweep": hw_sweep,
        "sos_benchmark": sos_res,
        "bode": bode_res,
        "micro_simulation": micro_res,
        "lut_benchmark": lut_res,
    }

    # Save JSON (without bulky full-res trajectories in summary copy, keep full in results)
    summary_json = dict(full_results)
    summary_json["bode"] = {k: v for k, v in bode_res.items() if k not in ("omega", "h_bar_hdv", "curves")}
    summary_json["micro_simulation"] = {k: v for k, v in micro_res.items() if k != "trajectories"}
    summary_json["sos_benchmark"] = {k: v for k, v in sos_res.items() if k != "nominal_feasibility_domain"}

    (out_dir / "colab_verified_summary.json").write_text(json.dumps(summary_json, indent=2))
    generate_figures(full_results, out_dir)
    print(f"DONE in {full_results['meta']['total_elapsed_s']}s. Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
