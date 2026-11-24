"""Google Colab script to extract a real NGSIM I-80 vehicle trajectory CSV
and run 20-vehicle mixed-platoon closed-loop replay across all 6 controllers.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcsrc.models.platoon import PlatoonParams
from pcsrc.simulation.micro_sim import _simulate_with_custom_leader

FT_TO_M = 0.3048
NGSIM_MAT_URL = "https://raw.githubusercontent.com/cemsaz/NGSIM-trajectory-animation/master/trajectories.mat"


def fetch_and_extract_ngsim_csv(csv_out: Path) -> dict:
    mat_path = Path("/tmp/ngsim_i80_trajectories.mat")
    if not mat_path.exists():
        print(f"Downloading real NGSIM I-80 trajectories from {NGSIM_MAT_URL} ...")
        urllib.request.urlretrieve(NGSIM_MAT_URL, mat_path)

    mat = sio.loadmat(str(mat_path))
    data = mat["trajectories"]
    print(f"Loaded NGSIM I-80 matrix shape: {data.shape}")

    # Columns: 0:Vehicle_ID, 1:Frame_ID, 3:Global_Time, 5:Local_Y, 10:v_Class, 11:v_Vel(ft/s), 12:v_Acc(ft/s^2), 13:Lane_ID
    veh_ids = np.unique(data[:, 0])
    best_veh = None
    best_score = -1.0
    best_rows = None

    for vid in veh_ids:
        rows = data[data[:, 0] == vid]
        if len(rows) < 400 or int(rows[0, 10]) != 2:
            continue
        # Sort by Frame_ID
        rows = rows[np.argsort(rows[:, 1])]
        vel_ms = rows[:, 11] * FT_TO_M
        # Smooth 10 Hz velocity slightly with Savitzky-Golay (window=11, poly=2) to differentiate acceleration cleanly
        vel_smooth = savgol_filter(vel_ms, 11, 2)
        acc_ms2 = np.gradient(vel_smooth, 0.1)
        min_a = float(np.min(acc_ms2))
        max_v = float(np.max(vel_smooth))
        min_v = float(np.min(vel_smooth))
        # Look for a realistic stop-and-go wave starting at v_init >= 8.0 m/s
        if -3.3 <= min_a <= -1.9 and (max_v - min_v) >= 5.5 and vel_smooth[0] >= 8.0:
            score = vel_smooth[0] + (max_v - min_v) + abs(min_a)
            if score > best_score:
                best_score = score
                best_veh = int(vid)
                best_rows = (rows[:400], vel_smooth[:400], acc_ms2[:400])

    assert best_rows is not None, "No suitable NGSIM I-80 trajectory found"
    rows, vel_s, acc_s = best_rows
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "Vehicle_ID", "Frame_ID", "Global_Time_ms", "Lane_ID",
            "Local_Y_ft", "v_Vel_fps", "v_Acc_fps2", "v_Vel_ms", "v_Acc_ms2",
        ])
        for k in range(len(rows)):
            writer.writerow([
                round(k * 0.1, 1),
                int(rows[k, 0]),
                int(rows[k, 1]),
                int(rows[k, 3]),
                int(rows[k, 13]),
                round(float(rows[k, 5]), 3),
                round(float(rows[k, 11]), 3),
                round(float(rows[k, 12]), 3),
                round(float(vel_s[k]), 4),
                round(float(acc_s[k]), 4),
            ])

    return {
        "vehicle_id": best_veh,
        "frame_start": int(rows[0, 1]),
        "frame_end": int(rows[-1, 1]),
        "lane_id": int(rows[0, 13]),
        "num_frames_10hz": int(len(rows)),
        "duration_s": round(len(rows) * 0.1, 1),
        "v_initial_ms": round(float(vel_s[0]), 2),
        "v_min_ms": round(float(np.min(vel_s)), 2),
        "v_max_ms": round(float(np.max(vel_s)), 2),
        "peak_decel_ms2": round(float(np.min(acc_s)), 3),
        "peak_accel_ms2": round(float(np.max(acc_s)), 3),
        "acc_10hz": acc_s.tolist(),
        "v0_ms": float(vel_s[0]),
    }


def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path(os.environ.get("PCSRC_OUT_DIR", str(ROOT / "results" / "colab_run")))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ngsim_i80_leader_trajectory.csv"

    meta = fetch_and_extract_ngsim_csv(csv_path)
    print(f"Extracted NGSIM I-80 Vehicle_ID={meta['vehicle_id']} (Frames {meta['frame_start']}..{meta['frame_end']}, "
          f"Lane {meta['lane_id']}), v=[{meta['v_min_ms']}, {meta['v_max_ms']}] m/s, peak_decel={meta['peak_decel_ms2']} m/s^2")

    # Interpolate 10 Hz acceleration profile to 100 Hz (dt = 0.01 s) for 20-vehicle simulator
    dt = 0.01
    t_10hz = np.arange(meta["num_frames_10hz"]) * 0.1
    t_100hz = np.arange(int(round(meta["duration_s"] / dt))) * dt
    a_leader_100hz = np.interp(t_100hz, t_10hz, np.array(meta["acc_10hz"]))
    peak_decel = abs(float(np.min(a_leader_100hz)))

    # Initialize platoon at the true initial speed of the NGSIM leader trajectory
    from dataclasses import replace
    params = replace(PlatoonParams(), v0=float(meta["v0_ms"]))
    cav_mask_40 = np.zeros(19, dtype=bool)
    cav_mask_40[[2, 4, 7, 9, 12, 14, 16, 18]] = True

    controllers = ("no_predictor", "classical_smith", "pade_11", "robust_hinf", "deep_lcc", "pc_src")
    ngsim_results = {}
    for ctrl in controllers:
        res, _ = _simulate_with_custom_leader(
            cav_mask_40, ctrl, params, a_leader_100hz, delta_tau=0.04, dt=dt
        )
        ngsim_results[ctrl] = {
            "tail_ratio": round(res["max_a_tail"] / max(peak_decel, 1e-6), 4),
            "min_gap": res["min_gap"],
            "min_ttc": res["min_ttc"],
            "accel_rms": res["accel_rms"],
            "throughput_vph": res["throughput_vph"],
            "collision": res["collision"],
        }
        print(f"  {ctrl:16s}: {ngsim_results[ctrl]}")

    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "ngsim_metadata": {k: v for k, v in meta.items() if k != "acc_10hz"},
        "ngsim_platoon_baselines": ngsim_results,
    }
    (out_dir / "colab_ngsim_real_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE in {summary['elapsed_s']}s.")


if __name__ == "__main__":
    main()
