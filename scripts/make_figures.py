"""Pure-local, offline IEEE PDF figure generator for the PC-SRC T-ITS artifact.

Generates all three data-driven paper figures (Figs. 3–5) directly from local
Python/NumPy/CVXPY computations without Google Colab or network access:
  - fig_headway_domain.pdf      (Fig. 3: h_min vs tau_e + certified (kp, kv) domain)
  - fig_bode_mixed.pdf          (Fig. 4: isolated CAV Bode + m=2 mixed-unit attenuation)
  - fig_platoon_validation.pdf  (Fig. 5: 20-veh shockwave damping + 0%–100% MPR sweep)

Usage:
    PYTHONPATH=src python3 scripts/make_figures.py --out-dir results/reproduced
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcsrc.analysis.asymptotic import sweep_headway_vs_actuator_lag
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


def _setup_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def compute_bode_curves(params: PlatoonParams) -> dict:
    """Compute frequency-domain Bode curves for all baselines (h=1.00 s & h=1.10 s)."""
    w = np.geomspace(0.01, 100.0, 600)
    h_bar, worst_hdv = worst_case_hdv_envelope(
        w, HDVUncertaintyBox(), v_e=params.v0, v_free=params.v_free, s0=params.standstill_gap
    )
    tau_a_hat = params.tau_a - 0.04
    curves = {
        "pc_src": np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.00, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "pc_src_h110": np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.10, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "no_predictor": np.abs(gamma_no_predictor(w, params.kp, params.kv, params.ka, 1.00, params.tau_e, params.tau_c, params.tau_a)),
        "classical_smith": np.abs(gamma_classical_smith(w, params.kp, params.kv, params.ka, 1.00, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "pade_11": np.abs(gamma_pade_cacc(w, params.kp, params.kv, params.ka, 1.00, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "robust_hinf": np.abs(gamma_robust_hinf(w, 1.00, params.tau_e, params.tau_c, params.tau_a)),
    }
    return {
        "omega": w,
        "h_bar_hdv": h_bar,
        "worst_hdv": worst_hdv,
        "curves": curves,
        "summary": {
            "worst_hdv_peak": round(float(worst_hdv["peak"]), 4),
            "pc_src_h100_hinf_cav": round(float(np.max(curves["pc_src"])), 4),
            "pc_src_h100_hinf_unit": round(float(np.max((h_bar**2) * curves["pc_src"])), 4),
            "pc_src_h110_hinf_unit": round(float(np.max((h_bar**2) * curves["pc_src_h110"])), 4),
        },
    }


def generate_paper_figures(
    headway_sweep: dict,
    sos_res: dict,
    bode_res: dict,
    micro_res: dict,
    out_dir: Path,
) -> list[Path]:
    """Render Figs. 3, 4, and 5 as vector PDFs into ``out_dir``."""
    _setup_ieee_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ------------------------------------------------------------------
    # Fig. 3: (a) h_min vs tau_e  &  (b) Certified (kp, kv) domain
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.35))
    te = np.array(headway_sweep["tau_e"])
    ax1.axhline(
        headway_sweep["h_asymp"],
        color="#555555",
        ls="-.",
        lw=1.1,
        label=f"Low-freq bound ({headway_sweep['h_asymp']:.3f} s)",
    )
    ax1.plot(te, headway_sweep["h_matched"], color="#C62828", ls="--", lw=1.4, marker="s", ms=2.8, label=r"Matched ($k_v=\tau_e k_p$)")
    ax1.plot(te, headway_sweep["h_coopt"], color="#1565C0", ls="-", lw=1.6, marker="o", ms=2.8, label=r"Co-optimized $(k_p,k_v,k_a)$")
    ax1.set_xlabel(r"Actuator time constant $\tau_e$ (s)")
    ax1.set_ylabel(r"Min verifiable headway $h_{\min}$ (s)")
    ax1.set_title("(a) Actuator Lag Coupling vs. Headway")
    ax1.set_xlim(0.0, 2.05)
    ax1.set_ylim(0.50, 1.35)
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)

    dom = sos_res["nominal_feasibility_domain"]
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
    ax2.plot(1.15, 1.18, marker="*", ms=7.5, color="#1B5E20", zorder=5)
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
    p3 = out_dir / "fig_headway_domain.pdf"
    fig.savefig(p3)
    plt.close(fig)
    written.append(p3)

    # ------------------------------------------------------------------
    # Fig. 4: Isolated CAV Bode & m=2 mixed-unit attenuation (h=1.00 & 1.10 s)
    # ------------------------------------------------------------------
    w = np.asarray(bode_res["omega"])
    h_bar = np.asarray(bode_res["h_bar_hdv"])
    curves = bode_res["curves"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.25))
    styles = {
        "pc_src": ("PC-SRC ($h=1.00$ s)", "#1565C0", "-", 1.6),
        "no_predictor": ("No Predictor (CACC)", "#C62828", "--", 1.15),
        "classical_smith": ("Classical Smith", "#EF6C00", ":", 1.25),
        "pade_11": ("Padé [1/1]", "#6A1B9A", "-.", 1.1),
        "robust_hinf": (r"Robust $\mathcal{H}_\infty$", "#00695C", "-", 1.05),
    }
    for key, (lbl, col, ls, lw) in styles.items():
        mag = np.asarray(curves[key])
        ax1.semilogx(w, mag, color=col, ls=ls, lw=lw, label=lbl)
        ax2.semilogx(w, (h_bar**2) * mag, color=col, ls=ls, lw=lw, label=lbl)

    if "pc_src_h110" in curves:
        h110_unit = (h_bar**2) * np.asarray(curves["pc_src_h110"])
        ax2.semilogx(w, h110_unit, color="#2E7D32", ls="-", lw=1.8, label="PC-SRC ($h=1.10$ s)")

    ax1.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
    ax1.set_xlim(0.05, 50.0)
    ax1.set_ylim(0.0, 1.22)
    ax1.set_xlabel(r"Frequency $\omega$ (rad/s)")
    ax1.set_ylabel(r"$|\Gamma_{\mathrm{CAV}}(j\omega)|$")
    ax1.set_title(r"(a) Isolated CAV Response ($\Delta\tau=40\,$ms)")
    ax1.grid(True, which="both")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.4)

    ax2.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlim(0.05, 10.0)
    ax2.set_ylim(0.0, 1.45)
    ax2.set_xlabel(r"Frequency $\omega$ (rad/s)")
    ax2.set_ylabel(r"$(\bar{H}^{\mathrm{HDV}}(\omega))^2 |\Gamma_{\mathrm{CAV}}(j\omega)|$")
    ax2.set_title(r"(b) Mixed-Unit Attenuation ($m=2$ HDVs)")
    ax2.grid(True, which="both")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.2)

    plt.tight_layout(w_pad=1.6)
    p4 = out_dir / "fig_bode_mixed.pdf"
    fig.savefig(p4)
    plt.close(fig)
    written.append(p4)

    # ------------------------------------------------------------------
    # Fig. 5: 20-veh Mixed Platoon Shockwave Damping + MPR Scalability
    # ------------------------------------------------------------------
    t_b = np.asarray(micro_res["trajectories"]["no_predictor"]["t"])
    v_b = np.asarray(micro_res["trajectories"]["no_predictor"]["v"])
    t_p = np.asarray(micro_res["trajectories"]["pc_src"]["t"])
    v_p = np.asarray(micro_res["trajectories"]["pc_src"]["v"])

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

    mpr = micro_res["mpr_sweep"]
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
    p5 = out_dir / "fig_platoon_validation.pdf"
    fig.savefig(p5)
    plt.close(fig)
    written.append(p5)

    return written


def main() -> None:
    from pcsrc.sos.certificates import benchmark_sos_formulations

    parser = argparse.ArgumentParser(description="Generate Figs. 3–5 PDFs locally (offline).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "reproduced")
    args = parser.parse_args()

    t0 = time.perf_counter()
    params = PlatoonParams()
    print("[PC-SRC Figures] Computing analytical headway sweep, SOS domain, Bode, and 20-veh trajectories...")
    hw_sweep = sweep_headway_vs_actuator_lag(params)
    sos_res = benchmark_sos_formulations(params)
    bode_res = compute_bode_curves(params)
    micro_res = run_all_micro_experiments(params, mc_trials=1)
    files = generate_paper_figures(hw_sweep, sos_res, bode_res, micro_res, args.out_dir)
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[PC-SRC Figures] Wrote {len(files)} PDFs in {elapsed}s -> {args.out_dir}")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
