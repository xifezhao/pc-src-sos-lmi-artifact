# PC-SRC SOS-LMI & Mixed Traffic Platoon Artifact

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](pyproject.toml)
[![Artifact Status](https://img.shields.io/badge/Artifact%20Evaluation-Prepared%20(Candidate)-informational.svg)](reproducibility/artifact_badge_checklist.md)

Open-source, self-contained software artifact (**v2.1.0**, prepared for independent Artifact Evaluation) accompanying the manuscript under review at *IEEE Transactions on Intelligent Transportation Systems*:

> **Low-Frequency String Stability Conditions and Lifted Polynomial Certificates for Mixed Traffic Platoons under Delay Mismatch**  
> Haiying Dong (ORCID: `0000-0002-3909-3246`) and Xiaofei Zhao\* (ORCID: `0000-0003-0772-8240`)  
> Repository: <https://github.com/xifezhao/pc-src-sos-lmi-artifact>

> **Note on Artifact Badges:** No ACM/IEEE badges are self-awarded. This repository is structured as a **candidate artifact** prepared for independent Artifact Evaluation (see [reproducibility/artifact_badge_checklist.md](reproducibility/artifact_badge_checklist.md)). A permanent Zenodo archival DOI will be minted upon manuscript acceptance.

## Package Architecture (`src/pcsrc/`)
- `models/platoon.py`: CAV delay-lag transfer functions (`PC-SRC`, `No Predictor`, `Classical Smith`, `Padé [1/1]`, `Robust H-inf`) and calibrated nonlinear/linearized IDM HDV dynamics over $\Theta^{\mathrm{HDV}}$.
- `analysis/asymptotic.py`: Low-frequency Maclaurin expansions (Propositions 1–4), Lagrange remainder overlap verification, and headway-vs-actuator-lag co-optimization.
- `sos/certificates.py`: Open-source CVXPY + Clarabel/SCS SOS-LMI solvers (no commercial solver required) for Certificate A (Sturm sequence + univariate SOS), Certificate B (5-variable lifted Positivstellensatz with Stage-A generator $\mu_4(y-z)$), and Monolithic Baselines M1–M3.
- `simulation/micro_sim.py`: 20-vehicle nonlinear mixed-platoon micro-simulator (6-baseline stop-and-go & stress braking, 1,000-trial Monte Carlo robustness, adversarial worst-case search, CAV penetration rate 0%–100% sweep, real NGSIM I-80 replay, and dynamic LUT switching under 2% road grade).
- `deployment/lut.py`: Offline-to-online $\mathcal{O}(1)$ **real Dual-Certificate-verified** 3D gain Look-Up Table (`4 × 4 × 3 = 48` cells, `576 bytes`), exporting `certified_lut_4x4x3.npz` / `.csv` and benchmarking lookup latency directly against the certified table.

## System Requirements & Runtime
- **Hardware:** Standard CPU (≥ 2 GB RAM, ≥ 250 MB free disk space; no GPU or commercial solver license required).
- **Software:** Python 3.10–3.13 (tested on Python 3.13.18 Linux x86_64 with `requirements-lock.txt`).
- **Expected Runtime:**
  - `pytest -v tests/` (unit & integrity suite): **~15–25 s**
  - `scripts/reproduce_paper.py --fast` (25 MC trials, 4,000 LUT queries): **~45–60 s**
  - `scripts/reproduce_paper.py --full` (1,000 MC trials, 20,000 LUT queries): **~10–12 min** (verified at `636.9 s` on Google Colab CPU)

## Quickstart & One-Command Reproduction

```bash
# 1. Install pinned dependencies and editable package
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e .

# 2. Verify SHA-256 checksums of reference data & outputs
shasum -a 256 -c data/reference/golden_hashes.sha256

# 3. Run automated unit & Artifact Evaluation test suite
PYTHONPATH=src pytest -v tests/

# 4. Run unified reproduction pipeline (--fast: ~50 s; --full: ~11 min)
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast
# PYTHONPATH=src python3 scripts/reproduce_paper.py --full
```

All single-source-of-truth parameters reside in [data/raw/scenario_definitions/nominal_platoon_parameters.yaml](data/raw/scenario_definitions/nominal_platoon_parameters.yaml) (identical to `PlatoonParams` in `src/pcsrc/models/platoon.py`). Reproduced outputs and an automatic `environment_snapshot.txt` are written to `results/reproduced/`.

## Empirical Data Provenance (NGSIM I-80)
The bundled empirical leader trajectory ([data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv](data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv), `Vehicle_ID=1209`, Frames `3429–3828`, Lane 3, 400 frames at 10 Hz) is derived from the U.S. Department of Transportation / FHWA Next Generation Simulation (NGSIM) I-80 dataset (DOI: `10.21949/1504477`, USDOT Data Portal license: CC BY-SA 3.0 / 4.0). Full provenance, mirror URL, preprocessing details (Savitzky–Golay smoothing and numerical differentiation), and SHA-256 verification are documented in [data/raw/external/README.md](data/raw/external/README.md).
