# Deterministic Execution & Seed Specification

All experiments in this artifact are designed to run deterministically on standard single-node CPUs without GPU nondeterminism or unseeded global random state.

## Deterministic Controls

1. **No Global NumPy RNG Calls:** Every stochastic routine instantiates an explicit `numpy.random.default_rng(seed)` generator locally. No `np.random.seed(...)` or global `np.random.*` calls exist in `src/pcsrc/` or `scripts/`.
2. **Fixed-Step Time Integration:** Nonlinear 20-vehicle mixed-platoon simulations use a fixed time step `dt = 0.01 s` with exact integer delay-buffer indexing (`round(tau / dt)`), eliminating adaptive-step ODE solver variability.
3. **Deterministic Convex Optimization:** Certificate A uses an exact algebraic Sturm sequence (`cert_a_sturm`); Certificate B and monolithic baselines solve convex SDPs via Clarabel's deterministic interior-point algorithm (`cp.CLARABEL`, single-threaded / deterministic factorization).
4. **Offline Empirical Data:** The real FHWA NGSIM I-80 trajectory (`Vehicle_ID=1209`, 400 frames at 10 Hz) is bundled in `data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv` and SHA-256 verified against `data/reference/golden_hashes.sha256`, requiring zero network access during reproduction.

## Seed Registry

| Experiment Component | Source File | Generator & Seed | Purpose |
|---|---|---|---|
| 1,000-trial (or 25-trial `--fast`) Monte Carlo robustness | `src/pcsrc/simulation/micro_sim.py` | `np.random.default_rng(2026)` | Samples heterogeneous HDV IDM parameters from $\Theta^{\mathrm{HDV}}$, random CAV spatial placements, and bounded delay mismatch $\Delta\tau \in [-\delta_{\max}, \delta_{\max}]$ |
| CAV Penetration Rate (MPR 0%–100%) 15-permutation sweep | `src/pcsrc/simulation/micro_sim.py` | `np.random.default_rng(1234)` | Generates deterministic spatial CAV placement permutations per MPR level |
| Certified 3D LUT query latency benchmark | `src/pcsrc/deployment/lut.py` | `np.random.default_rng(20260816)` | Draws uniform query triples $(\tau_c, \delta_{\max}, \tau_e)$ within the certified grid bounds |

## Reproducibility Verification Protocol

Run the following three checks in sequence to verify deterministic reproduction:

```bash
# 1. Verify bit-exact integrity of all 11 reference artifacts
shasum -a 256 -c data/reference/golden_hashes.sha256

# 2. Verify public API exports and all scientific invariants
PYTHONPATH=src pytest -v tests/

# 3. Reproduce all numerical summaries, LUT artifacts, and Figs. 3–5 offline
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast
```

For per-function numerical tolerances and cross-BLAS variance bounds, see [docs/numerical_precision.md](../docs/numerical_precision.md).
