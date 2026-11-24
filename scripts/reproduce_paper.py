"""Unified single-entry reproduction pipeline for IEEE T-ITS PC-SRC manuscript.

Reproduces all Tables (I-V), Figures (3-5), real FHWA NGSIM I-80 trajectory replay,
and online LUT dynamic switching + 2% road grade verification.

Usage:
    # Fast Artifact Evaluation check (< 60 s, 25 Monte Carlo trials)
    PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

    # Full paper reproduction (1,000 Monte Carlo trials)
    PYTHONPATH=src python3 scripts/reproduce_paper.py --full
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

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
from pcsrc.simulation.micro_sim import (
    _simulate_with_custom_leader,
    run_all_micro_experiments,
    run_empirical_wave_and_switching_experiments,
)
from pcsrc.sos.certificates import benchmark_sos_formulations


def replay_bundled_ngsim_csv(csv_path: Path) -> dict:
    """Replay the bundled real FHWA NGSIM I-80 Vehicle_ID=1209 trajectory CSV."""
    times, vels, accs = [], [], []
    veh_id = None
    lane_id = None
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            vels.append(float(row["v_Vel_ms"]))
            accs.append(float(row["v_Acc_ms2"]))
            if veh_id is None:
                veh_id = int(row["Vehicle_ID"])
                lane_id = int(row["Lane_ID"])

    dt = 0.01
    t_10hz = np.array(times, dtype=np.float64)
    a_10hz = np.array(accs, dtype=np.float64)
    t_100hz = np.arange(int(round((t_10hz[-1] + 0.1) / dt))) * dt
    a_100hz = np.interp(t_100hz, t_10hz, a_10hz)
    peak_decel = abs(float(np.min(a_100hz)))

    params = replace(PlatoonParams(), v0=float(vels[0]))
    cav_mask_40 = np.zeros(19, dtype=bool)
    cav_mask_40[[2, 4, 7, 9, 12, 14, 16, 18]] = True

    out = {}
    for ctrl in ("no_predictor", "classical_smith", "pade_11", "robust_hinf", "deep_lcc", "pc_src"):
        res, _ = _simulate_with_custom_leader(cav_mask_40, ctrl, params, a_100hz, delta_tau=0.04, dt=dt)
        out[ctrl] = {
            "tail_ratio": round(res["max_a_tail"] / max(peak_decel, 1e-6), 4),
            "min_gap": res["min_gap"],
            "min_ttc": res["min_ttc"],
            "accel_rms": res["accel_rms"],
            "throughput_vph": res["throughput_vph"],
            "collision": res["collision"],
        }
    return {
        "vehicle_id": veh_id,
        "lane_id": lane_id,
        "num_frames_10hz": len(times),
        "peak_decel_ms2": round(-peak_decel, 3),
        "baselines": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce PC-SRC IEEE T-ITS paper results.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Fast AE verification mode (25 MC trials, <60s).")
    mode.add_argument("--full", action="store_true", help="Full paper reproduction (1,000 MC trials).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "reproduced")
    args = parser.parse_args()

    mc_trials = 1000 if args.full else 25
    lut_queries = 20000 if args.full else 4000
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    params = PlatoonParams()

    print(f"[PC-SRC AE] Mode={'FULL' if args.full else 'FAST'} (mc_trials={mc_trials})")
    overlap = verify_low_freq_overlap(params)
    hw_sweep = sweep_headway_vs_actuator_lag(params)
    sos_res = benchmark_sos_formulations(params)

    # Bode (h=1.00 s and h=1.10 s)
    w = np.geomspace(0.01, 100.0, 400)
    h_bar, worst_hdv = worst_case_hdv_envelope(w, HDVUncertaintyBox(), v_e=params.v0, v_free=params.v_free, s0=params.standstill_gap)
    tau_a_hat = params.tau_a - 0.04
    pc_h100 = np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.00, params.tau_e, params.tau_c, params.tau_a, tau_a_hat))
    pc_h110 = np.abs(gamma_pcsrc(w, params.kp, params.kv, params.ka, 1.10, params.tau_e, params.tau_c, params.tau_a, tau_a_hat))
    bode_summary = {
        "worst_hdv_peak": round(float(worst_hdv["peak"]), 4),
        "pc_src_h100_hinf_cav": round(float(np.max(pc_h100)), 4),
        "pc_src_h100_hinf_unit": round(float(np.max((h_bar**2) * pc_h100)), 4),
        "pc_src_h110_hinf_unit": round(float(np.max((h_bar**2) * pc_h110)), 4),
    }

    micro_res = run_all_micro_experiments(params, mc_trials=mc_trials)
    sup_res = run_empirical_wave_and_switching_experiments(params)
    ngsim_csv = ROOT / "data" / "raw" / "scenario_definitions" / "ngsim_i80_leader_trajectory.csv"
    ngsim_res = replay_bundled_ngsim_csv(ngsim_csv)
    lut_res = benchmark_lut(n_queries=lut_queries, save_dir=args.out_dir)

    # Record environment snapshot alongside reproduced outputs
    snap_lines = [
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"numpy: {np.__version__}",
    ]
    for pkg in ("scipy", "matplotlib", "cvxpy", "clarabel"):
        try:
            mod = __import__(pkg)
            snap_lines.append(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            snap_lines.append(f"{pkg}: not-installed")
    (args.out_dir / "environment_snapshot.txt").write_text("\n".join(snap_lines) + "\n")

    summary = {
        "meta": {
            "mode": "full" if args.full else "fast",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        },
        "overlap_verification": overlap,
        "headway_sweep": hw_sweep,
        "sos_single_point": sos_res["single_point_sdp"],
        "sos_tau_c_sweep": sos_res["tau_c_sweep"],
        "bode_summary": bode_summary,
        "micro_baselines": micro_res["baselines"],
        "monte_carlo": micro_res["monte_carlo"],
        "adversarial_worst_case": micro_res["adversarial_worst_case"],
        "mpr_sweep": micro_res["mpr_sweep"],
        "empirical_wave_baselines": sup_res["empirical_wave_baselines"],
        "lut_switching_and_grade": sup_res["lut_switching_and_grade"],
        "ngsim_i80_replay": ngsim_res,
        "lut_benchmark": lut_res,
    }
    out_file = args.out_dir / "reproduced_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"[PC-SRC AE] Completed in {summary['meta']['elapsed_s']}s -> {out_file}")


if __name__ == "__main__":
    main()
