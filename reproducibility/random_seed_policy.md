# Random Seed Policy

All stochastic components use explicit NumPy `default_rng` seeds for deterministic, bit-reproducible sampling:
- **Monte Carlo 1,000-trial robustness (`src/pcsrc/simulation/micro_sim.py`):** `np.random.default_rng(2026)`
- **MPR 15-permutation sweep (`src/pcsrc/simulation/micro_sim.py`):** `np.random.default_rng(1234)`
- **Certified LUT query benchmark (`src/pcsrc/deployment/lut.py`):** `np.random.default_rng(20260816)`

No global unseeded `np.random` calls are used anywhere in `src/pcsrc/` or `scripts/`.
