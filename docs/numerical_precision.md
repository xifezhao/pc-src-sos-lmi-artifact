# Numerical Precision & Verification Tolerances

All numerical routines in `src/pcsrc/` execute in IEEE-754 double precision (`np.float64`) on CPU. This document records the numerical solver settings, cross-platform BLAS/LAPACK drift expectations, and the exact invariant tolerances enforced by `tests/test_pcsrc_ae.py`.

## Floating-Point & Solver Precision

| Component | Module | Precision / Numerical Scheme | Expected Cross-Host Drift |
|---|---|---|---|
| Low-frequency Maclaurin & remainder bounds | `src/pcsrc/analysis/asymptotic.py` | Analytic closed-form `float64` + scalar grid search | `< 1e-12` (machine epsilon) |
| Stage-A denominator dominance (Sturm) | `src/pcsrc/sos/certificates.py` (`cert_a_sturm`) | Exact real-root counting via Sturm polynomial sequence on $[x_{\min}, \bar{x}]$ | Exact sign / `< 1e-10` on margin |
| Stage-B lifted SOS SDP (Clarabel / SCS) | `src/pcsrc/sos/certificates.py` (`cert_b_dual_sos`) | CVXPY interior-point solver (`cp.CLARABEL`, fallback `cp.SCS`), default primal/dual feasibility tolerances | Feasibility invariant; Gram entries `< 1e-6` |
| Frequency-domain Bode & HDV envelope | `src/pcsrc/models/platoon.py` | Complex `complex128` evaluation over log-spaced $\omega \in [0.01, 100]$ rad/s | `< 1e-10` |
| Nonlinear 20-vehicle micro-simulation | `src/pcsrc/simulation/micro_sim.py` | Fixed-step explicit integration (`dt = 0.01 s`), ring-buffer exact integer delay taps | `< 1e-9` |
| 3D certified LUT lookup | `src/pcsrc/deployment/lut.py` | `4 × 4 × 3` `float64` table (`576 bytes`), exact nearest-grid index | Exact (`0.0` drift); latency is CPU-dependent |

## Enforced Test Invariant Tolerances (`tests/test_pcsrc_ae.py`)

| Test Function | Checked Quantity | Reference / Bound | Tolerance / Rationale |
|---|---|---|---|
| `test_asymptotic_bounds_and_overlap` | $c_2^{\mathrm{nom}}, c_2^{\mathrm{worst}}$ | `0.8625` | `rel=1e-4` (4-decimal manuscript rounding) |
| `test_asymptotic_bounds_and_overlap` | $h_{\min}^{(2)}$ | $\sqrt{2(1-k_a)/k_p}$ | `rel=1e-6` (exact algebraic identity) |
| `test_asymptotic_bounds_and_overlap` | Overlap $\omega_{\mathrm{upper}} > \omega_{\min}$, $Q_{\mathrm{lower}}>0$, $P_B>0$ | Strict positivity | Exact sign check |
| `test_bode_isolated_and_hdv_envelope` | $\max_\omega \|\Gamma_{\mathrm{CAV}}(j\omega)\|$ | `<= 1.0005` | `5e-4` discrete-frequency & solver margin |
| `test_bode_isolated_and_hdv_envelope` | Worst-case HDV peak vs. grid envelope | `rel=1e-6`, peak `> 1.10` | Consistent envelope extraction |
| `test_dual_certificate_sturm_and_sos` | Stage-A Sturm & Stage-B Clarabel SOS | `feasible=True`, `margin > 1.0` | Binary feasibility certificate |
| `test_micro_platoon_safety_and_stress` | PC-SRC vs. Classical Smith (`-2.0` / `-3.5 m/s^2`) | `collision=False`, `tail_ratio < 1.0`, `min_gap > 1.5 m` | Physical safety invariants |
| `test_real_ngsim_i80_csv_replay` | FHWA NGSIM I-80 `Vehicle_ID=1209` (400 frames) | `collision=False`, PC-SRC outperforms baselines | Deterministic CSV replay |
| `test_lut_memory_and_dynamic_switching` | LUT size & switching jerk | `48/48` cells, `576 bytes`, jerk `< 1.0 m/s^3` | Exact memory & comfort bound |
| `test_nominal_yaml_matches_dataclass` | `nominal_platoon_parameters.yaml` vs. `PlatoonParams` | 16/16 fields | `rel=1e-12` (single-source-of-truth guard) |

## Hardware-Dependent vs. Hardware-Independent Outputs

- **Hardware-independent (exact / tight tolerance):** All analytical bounds, SOS feasibility statuses, Bode magnitudes, deterministic trajectories, Monte Carlo collision counts/statistics (under fixed seeds), NGSIM replay metrics, and LUT memory footprint (`576 bytes`).
- **Hardware-dependent (qualitative bound only):** Clarabel wall-clock solve times (e.g., `450.4 ms` reference on Colab x86_64) and LUT query latency (`2.59 µs` reference flat-index mean; `test_lut_memory_and_dynamic_switching` enforces a conservative `< 50.0 µs` upper guard across diverse CPUs).
