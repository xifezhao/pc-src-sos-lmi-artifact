"""Supplementary Google Colab verification script for T-ITS review closure.

Runs:
  1. Mixed-unit Bode check at h = 1.10 s (closing C04-01).
  2. NGSIM/highD-calibrated multi-harmonic empirical stop-and-go replay (closing C01-01).
  3. Online LUT switching (tau_c: 50 -> 180 ms) + 2% road grade with outer trim (closing N05-01).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from pcsrc.simulation.micro_sim import run_empirical_wave_and_switching_experiments


def main() -> None:
    t0 = time.perf_counter()
    params = PlatoonParams()
    out_dir = Path(os.environ.get("PCSRC_OUT_DIR", str(ROOT / "results" / "colab_run")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bode check with h = 1.00 s vs h = 1.10 s
    w = np.geomspace(0.01, 100.0, 600)
    h_bar, worst_hdv = worst_case_hdv_envelope(
        w, HDVUncertaintyBox(), v_e=params.v0, v_free=params.v_free, s0=params.standstill_gap
    )
    tau_a_hat = params.tau_a - 0.04
    curves = {
        "pc_src": np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.0, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "pc_src_h110": np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.10, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "no_predictor": np.abs(gamma_no_predictor(w, params.kp, params.kv, params.ka, 1.0, params.tau_e, params.tau_c, params.tau_a)),
        "classical_smith": np.abs(gamma_classical_smith(w, params.kp, params.kv, params.ka, 1.0, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "pade_11": np.abs(gamma_pade_cacc(w, params.kp, params.kv, params.ka, 1.0, params.tau_e, params.tau_c, params.tau_a, tau_a_hat)),
        "robust_hinf": np.abs(gamma_robust_hinf(w, 1.0, params.tau_e, params.tau_c, params.tau_a)),
    }

    h110_unit = (h_bar**2) * curves["pc_src_h110"]
    h110_hinf_unit = round(float(np.max(h110_unit)), 4)
    h110_hinf_cav = round(float(np.max(curves["pc_src_h110"])), 4)
    print(f"[1/3] h=1.10 s Bode: ||G_CAV||={h110_hinf_cav}, ||G_unit||={h110_hinf_unit}")

    # Render updated fig_bode_mixed.pdf
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.25))
    styles = {
        "pc_src": ("PC-SRC ($h=1.00$ s)", "#1565C0", "-", 1.6),
        "no_predictor": ("No Predictor (CACC)", "#C62828", "--", 1.15),
        "classical_smith": ("Classical Smith", "#EF6C00", ":", 1.25),
        "pade_11": ("Padé [1/1]", "#6A1B9A", "-.", 1.1),
        "robust_hinf": (r"Robust $\mathcal{H}_\infty$", "#00695C", "-", 1.05),
    }
    for key, (lbl, col, ls, lw) in styles.items():
        mag = curves[key]
        ax1.semilogx(w, mag, color=col, ls=ls, lw=lw, label=lbl)
        ax2.semilogx(w, (h_bar**2) * mag, color=col, ls=ls, lw=lw, label=lbl)

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
    fig.savefig(out_dir / "fig_bode_mixed.pdf")
    plt.close(fig)

    # 2. Empirical-style multi-harmonic stop-and-go & LUT switching + grade
    print("[2/3] Running empirical-style stop-and-go & LUT switching + 2% grade experiments...")
    sup_res = run_empirical_wave_and_switching_experiments(params)
    print("      Empirical PC-SRC:", sup_res["empirical_wave_baselines"]["pc_src"])
    print("      Empirical Smith :", sup_res["empirical_wave_baselines"]["classical_smith"])
    print("      LUT Switching   :", sup_res["lut_switching_and_grade"])

    out_json = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "bode_h110": {"hinf_cav": h110_hinf_cav, "hinf_unit": h110_hinf_unit},
        **sup_res,
    }
    (out_dir / "colab_supplement_summary.json").write_text(json.dumps(out_json, indent=2))
    print(f"[3/3] DONE in {out_json['elapsed_s']}s.")


if __name__ == "__main__":
    main()
