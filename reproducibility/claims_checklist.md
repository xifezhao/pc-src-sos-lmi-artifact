# Paper Claims Mapped to Executable Code & Verified Outputs

| Manuscript Claim | Module / Script | Verified Reference Output |
|---|---|---|
| Low-freq overlap $c_2^{\mathrm{nom}}=0.8625$, $\omega_{\mathrm{upper}}=0.5481$ rad/s | `src/pcsrc/analysis/asymptotic.py` | `results/colab_run/colab_verified_summary.json` (`overlap_verification`) |
| Headway co-optimization ($h_{\min}=0.482$ s at $\tau_e=0.35$ s) | `src/pcsrc/analysis/asymptotic.py` | `results/colab_run/fig_headway_domain.pdf` (Fig. 3a) |
| Dual-Cert SDP (`450.4 ms` Stage-B / `462.6 ms` combined, `60.2x` vs M3, `90.7% -> 88.1%` area) | `src/pcsrc/sos/certificates.py` | `results/colab_run/colab_verified_summary.json` (`sos_benchmark`) |
| Bode & $m=2$ HDV attenuation (`0.6039`, `0.8049`, `h=1.10`: `1.0256`) | `src/pcsrc/models/platoon.py` | `results/colab_run/fig_bode_mixed.pdf` (Fig. 4) |
| 20-vehicle nonlinear platoon (`0.673` tail, `13.48 m` gap, Smith crash `-3.22 m`) | `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/colab_verified_summary.json` (`micro_simulation.baselines`) |
| Real FHWA NGSIM I-80 `ID=1209` replay (`0.253` tail, `10.11 m` gap) | `scripts/reproduce_paper.py` | `results/colab_run/colab_ngsim_real_summary.json` |
| 1,000-trial Monte Carlo (`0/1000` collisions, mean `0.2880 ± 0.0925`) | `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/colab_verified_summary.json` (`monte_carlo`) |
| MPR scalability (`0% -> 100%`, 15 permutations/level, `+17.2%` flow) | `src/pcsrc/simulation/micro_sim.py` | `results/colab_run/fig_platoon_validation.pdf` (Fig. 5) |
| Real Dual-Cert 48-cell (`4x4x3`, `576 B`) LUT latency & 2% grade switching | `src/pcsrc/deployment/lut.py` | `results/colab_run/certified_lut_4x4x3.{npz,csv}`, `colab_supplement_summary.json` |
