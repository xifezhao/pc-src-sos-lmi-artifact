# Known Limitations

1. **Solver Runtime Variance:** Clarabel SDP solve times (`450.4 ms` Stage-B / `462.6 ms` combined for Dual-Cert in `colab_verified_summary.json`) and LUT query latencies (`2.59 µs` flat-index reference) depend on host CPU clock speed and thermal state; functional feasibility and accuracy invariants are hardware-independent.
2. **NGSIM Empirical Derivation:** Velocities and accelerations in `ngsim_i80_leader_trajectory.csv` are Savitzky–Golay smoothed derivatives of raw 10 Hz video-tracked positions (see `data/raw/external/README.md`).
3. **Archival DOI Pending Acceptance:** Permanent Zenodo DOI will be minted and added to `CITATION.cff` upon manuscript acceptance.
