# Paper Claims-to-Artifact Mapping (IEEE / ACM 4-Badge Standard)

**Paper:** *Low-Frequency String Stability Conditions and Lifted Polynomial Certificates for Mixed Traffic Platoons under Delay Mismatch*  
**Authors:** Haiying Dong (ORCID: 0000-0002-3909-3246) and Xiaofei Zhao* (ORCID: 0000-0003-0772-8240)  
**Target Venue:** IEEE Transactions on Intelligent Transportation Systems (Regular Paper, 8 pages)

---

## Claim-to-Code & Result Traceability

| Paper Element | Claim / Metric | Implementing Module | Verified Reference Output |
|---|---|---|---|
| **Section IV / Fig. 3(a)** | Low-frequency remainder overlap ($\omega_{\min}=0.25 \le 0.5481$\,rad/s) and co-optimized $h_{\min}(\tau_e)$ eliminating low-$\tau_e$ damping starvation ($0.480$\,s vs $0.603$\,s) | `src/pcsrc/analysis/asymptotic.py` | `results/colab_run/colab_verified_summary.json` (`overlap_verification`, `headway_sweep`), `fig_headway_domain.pdf` |
| **Theorem 1 / Table II / Fig. 3(b)** | Dual-Certificate (Sturm $0.63$\,ms + Stage-B 21-basis SOS $450.4$\,ms = $451.0$\,ms) is $60\times$ faster than coupled M3 ($27{,}116.1$\,ms, infeasible) and certifies $90.7\% \to 88.1\%$ of gain grid across $\tau_c \in [0, 200]$\,ms up to $\delta_{\max}=85\text{--}90$\,ms | `src/pcsrc/sos/certificates.py` (open-source CVXPY + Clarabel) | `results/colab_run/colab_verified_summary.json` (`sos_benchmark`), `fig_headway_domain.pdf` |
| **Table III / Fig. 4** | Isolated CAV $\|\Gamma_{\mathrm{CAV}}\|_{\mathcal{H}_\infty} = 1.0000$, strongest HDV-resonance damping ($0.6039$ at $1.088$\,rad/s), and $h=1.10$\,s near-DC comparison ($1.0302 \to 1.0256$) | `src/pcsrc/models/platoon.py` | `results/colab_run/colab_verified_summary.json` (`bode`), `colab_supplement_summary.json` (`bode_h110`), `fig_bode_mixed.pdf` |
| **Table IV / Fig. 5(a)-(b)** | 20-vehicle 40% MPR mixed platoon: PC-SRC tail ratio $0.673$, safe under $-3.5\,\mathrm{m/s^2}$ stress ($2.34$\,m vs Classical Smith crash $-3.22$\,m); 1,000 Monte Carlo trials $0/1{,}000$ collisions (95% Wilson CI $[0.00\%, 0.38\%]$) | `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/colab_verified_summary.json` (`micro_simulation`), `fig_platoon_validation.pdf` |
| **Section VI-D.1 (NGSIM I-80)** | Direct replay of U.S. FHWA NGSIM I-80 `Vehicle_ID=1209` (Frames 3429–3828, Lane 3, 40.0 s, 10 Hz): PC-SRC tail ratio $0.253$ ($28.8\%$ lower than No Predictor $0.356$), min gap $10.11$\,m, min TTC $9.03$\,s, accel RMS $0.436\,\mathrm{m/s^2}$ | `data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv`, `scripts/reproduce_paper.py` | `results/colab_run/colab_ngsim_real_summary.json` |
| **Table V / Fig. 5(c)** | CAV MPR 0%→100% (15 permutations each): tail ratio $1.685 \to 0.449$, accel RMS $-33.6\%$, instantaneous equivalent platoon throughput $2{,}003.8 \to 2{,}347.9$\,veh/h ($+17.2\%$) | `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/colab_verified_summary.json` (`mpr_sweep`) |
| **Section VI-E (LUT & Switching)** | 48-cell `float32` 3D LUT ($576$\,B): Flat Index $2.59\,\mu\mathrm{s}$ mean ($4.04\,\mu\mathrm{s}$ P99); $\tau_c: 50\to 180$\,ms switching + 2% grade peak jerk $0.413\,\mathrm{m/s^3}$, zero collisions, steady-state error $0.803\to 0.749$\,m | `src/pcsrc/deployment/lut.py`, `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/colab_verified_summary.json` (`lut_benchmark`), `colab_supplement_summary.json` (`lut_switching_and_grade`) |

---

## Single-Command Reproduction & Verification

```bash
# 1. Verify golden file integrity (SHA-256)
make verify-hashes

# 2. Run automated unit & Artifact Evaluation (AE) test suite (< 30 s)
make test

# 3. Fast end-to-end reproduction (25 MC trials, < 60 s)
make reproduce-fast

# 4. Full paper reproduction (1,000 MC trials, ~10 min on CPU / Google Colab)
make reproduce-all
```

No commercial solver licenses (MATLAB / MOSEK) are required; all SOS-LMI SDPs run on open-source **CVXPY + Clarabel**.
