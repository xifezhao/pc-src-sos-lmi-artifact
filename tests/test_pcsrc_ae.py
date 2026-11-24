"""Automated Unit & Artifact Evaluation (AE) Test Suite for PC-SRC.

Verifies:
  1. Analytical low-frequency bounds & Lagrange remainder overlap (Section IV, Remark 3).
  2. Dual-Certificate Stage-A Sturm & Stage-B CVXPY/Clarabel SOS feasibility (Theorem 1).
  3. Isolated CAV H-inf <= 1.0 and HDV worst-case envelope (Section VI-C).
  4. 20-vehicle nonlinear mixed platoon safety (zero collision for PC-SRC vs Classical Smith crash).
  5. Bundled real FHWA NGSIM I-80 Vehicle_ID=1209 CSV replay (Section VI-D.1).
  6. Online LUT O(1) 576-byte memory footprint & tau_c switching + 2% grade safety (Section VI-E).
  7. Golden metrics integrity against verified Colab reference JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pcsrc.analysis.asymptotic import (
    c2_mismatch,
    c2_nominal,
    h_min_second_order,
    h_star_matched,
    verify_low_freq_overlap,
)
from pcsrc.deployment.lut import benchmark_lut
from pcsrc.models.platoon import (
    HDVUncertaintyBox,
    PlatoonParams,
    gamma_pcsrc,
    worst_case_hdv_envelope,
)
from pcsrc.simulation.micro_sim import (
    run_empirical_wave_and_switching_experiments,
    simulate_platoon,
)
from reproduce_paper import replay_bundled_ngsim_csv


def test_asymptotic_bounds_and_overlap() -> None:
    params = PlatoonParams()
    assert c2_nominal(params.kp, params.ka, params.headway) == pytest.approx(0.8625, rel=1e-4)
    assert c2_mismatch(params.kp, params.ka, params.headway, params.tau_a, 0.0) == pytest.approx(0.8625, rel=1e-4)
    assert h_min_second_order(params.kp, params.ka) == pytest.approx(np.sqrt(2 * (1 - params.ka) / params.kp), rel=1e-6)
    # Gain-matched h*(tau_e) increases as tau_e decreases (Prop. 4)
    assert h_star_matched(params.kp, 0.05, params.tau_c) > h_star_matched(params.kp, 0.35, params.tau_c)

    overlap = verify_low_freq_overlap(params)
    assert overlap["overlap_valid"] is True
    assert overlap["omega_upper"] > params.omega_min
    assert overlap["Q_lower_at_omega_min"] > 0.0
    assert overlap["P_B_at_omega_min"] > 0.0


def test_bode_isolated_and_hdv_envelope() -> None:
    params = PlatoonParams()
    w = np.geomspace(0.01, 100.0, 300)
    mag = np.abs(
        gamma_pcsrc(w, params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.tau_a, params.tau_a - 0.04)
    )
    assert float(np.max(mag)) <= 1.0005
    h_bar, worst = worst_case_hdv_envelope(w, HDVUncertaintyBox(), v_e=params.v0, v_free=params.v_free, s0=params.standstill_gap, n_grid=3)
    assert worst["peak"] > 1.10
    assert float(np.max(h_bar)) == pytest.approx(worst["peak"], rel=1e-6)


def test_dual_certificate_sturm_and_sos() -> None:
    cvxpy = pytest.importorskip("cvxpy")
    _ = cvxpy
    from pcsrc.sos.certificates import cert_a_sturm, cert_b_dual_sos, fast_exact_certificate_check

    params = PlatoonParams()
    x_min, x_bar = params.omega_min**2, params.omega_bar**2

    sturm_ok, _, margin = cert_a_sturm(params.kp, params.kv, params.headway, params.tau_e, params.delta_max, x_min, x_bar)
    assert sturm_ok is True
    assert margin > 1.0

    b_ok, _ = cert_b_dual_sos(
        params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max, x_min, x_bar
    )
    assert b_ok is True

    assert fast_exact_certificate_check(
        params.kp, params.kv, params.ka, params.headway, params.tau_e, params.tau_c, params.delta_max,
        params.omega_min, params.omega_bar, mode="dual"
    ) is True


def test_micro_platoon_safety_and_stress() -> None:
    params = PlatoonParams()
    cav_mask_40 = np.zeros(19, dtype=bool)
    cav_mask_40[[2, 4, 7, 9, 12, 14, 16, 18]] = True

    m_pcsrc, _ = simulate_platoon(cav_mask_40, controller="pc_src", params=params, delta_tau=0.04, leader_decel=-2.0, brake_dur=5.5, osc_cycles=2)
    m_smith, _ = simulate_platoon(cav_mask_40, controller="classical_smith", params=params, delta_tau=0.04, leader_decel=-2.0, brake_dur=5.5, osc_cycles=2)
    assert m_pcsrc.collision is False
    assert m_pcsrc.tail_ratio < 1.0
    assert m_smith.tail_ratio > 1.0

    m_pcsrc_stress, _ = simulate_platoon(cav_mask_40, controller="pc_src", params=params, delta_tau=0.04, leader_decel=-3.5, brake_dur=5.5, osc_cycles=2)
    m_smith_stress, _ = simulate_platoon(cav_mask_40, controller="classical_smith", params=params, delta_tau=0.04, leader_decel=-3.5, brake_dur=5.5, osc_cycles=2)
    assert m_pcsrc_stress.collision is False
    assert m_pcsrc_stress.min_gap > 1.5
    assert m_smith_stress.collision is True


def test_real_ngsim_i80_csv_replay() -> None:
    csv_path = ROOT / "data" / "raw" / "scenario_definitions" / "ngsim_i80_leader_trajectory.csv"
    assert csv_path.exists()
    res = replay_bundled_ngsim_csv(csv_path)
    assert res["vehicle_id"] == 1209
    assert res["num_frames_10hz"] == 400
    assert res["baselines"]["pc_src"]["collision"] is False
    assert res["baselines"]["pc_src"]["tail_ratio"] < res["baselines"]["no_predictor"]["tail_ratio"]
    assert res["baselines"]["pc_src"]["min_gap"] > res["baselines"]["classical_smith"]["min_gap"]


def test_lut_memory_and_dynamic_switching() -> None:
    lut_res = benchmark_lut(n_queries=1000)
    assert lut_res["grid_points"] == 48
    assert lut_res["certified_cells"] == 48
    assert lut_res["min_dual_cert_margin"] > 0.0
    assert lut_res["memory_bytes"] == 576
    assert lut_res["flat_index_buffer"]["mean_us"] < 50.0

    sup = run_empirical_wave_and_switching_experiments(PlatoonParams())
    sw = sup["lut_switching_and_grade"]
    assert sw["with_outer_integral_trim"]["peak_switch_jerk_ms3"] < 1.0
    assert sw["with_outer_integral_trim"]["steady_state_spacing_err_m"] < sw["without_outer_trim"]["steady_state_spacing_err_m"]


def test_golden_metrics_present_and_consistent() -> None:
    golden = json.loads((ROOT / "data" / "reference" / "golden_metrics.json").read_text())
    assert golden["sos_cert_a_sturm_feasible"] is True
    assert golden["sos_cert_b_feasible"] is True
    assert golden["mc_pcsrc_collisions"] == 0
    assert golden["ngsim_vehicle_id"] == 1209
    assert golden["lut_memory_bytes"] == 576


def test_sos_public_api_exports() -> None:
    cvxpy = pytest.importorskip("cvxpy")
    _ = cvxpy
    import pcsrc.sos as sos_pkg

    assert len(sos_pkg._LAZY_EXPORTS) >= 7
    for name in sos_pkg._LAZY_EXPORTS:
        obj = getattr(sos_pkg, name)
        assert callable(obj), f"pcsrc.sos.{name} should be callable"


def test_nominal_yaml_matches_dataclass() -> None:
    yaml_path = ROOT / "data" / "raw" / "scenario_definitions" / "nominal_platoon_parameters.yaml"
    assert yaml_path.exists()
    raw = yaml_path.read_text()
    try:
        import yaml
        cfg = yaml.safe_load(raw)
    except ImportError:
        cfg = {}
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            k, v = stripped.split(":", 1)
            cfg[k.strip()] = float(v.strip())

    params = PlatoonParams()
    mapping = {
        "vehicle_length_m": params.vehicle_length,
        "standstill_gap_m": params.standstill_gap,
        "headway_s": params.headway,
        "tau_a_s": params.tau_a,
        "tau_e_s": params.tau_e,
        "tau_c_s": params.tau_c,
        "a_min_ms2": params.a_min,
        "a_max_ms2": params.a_max,
        "v0_ms": params.v0,
        "v_free_ms": params.v_free,
        "kp": params.kp,
        "kv": params.kv,
        "ka": params.ka,
        "delta_max_s": params.delta_max,
        "omega_min_rads": params.omega_min,
        "omega_bar_rads": params.omega_bar,
    }
    assert set(cfg.keys()) == set(mapping.keys())
    for k, expected in mapping.items():
        assert float(cfg[k]) == pytest.approx(expected, rel=1e-12), f"Mismatch for {k}"
