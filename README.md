# PC-SRC SOS-LMI & Mixed Traffic Platoon Artifact

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Reproducibility](https://img.shields.io/badge/IEEE%2FACM-Artifact%20Evaluation-green.svg)](reproducibility/CLAIMS.md)

Reproducible software artifact (**v2.0.0**) for the *IEEE Transactions on Intelligent Transportation Systems* manuscript:

> **Low-Frequency String Stability Conditions and Lifted Polynomial Certificates for Mixed Traffic Platoons under Delay Mismatch**  
> Haiying Dong (ORCID: 0000-0002-3909-3246) and Xiaofei Zhao\* (ORCID: 0000-0003-0772-8240)  
> Repository: <https://github.com/xifezhao/pc-src-sos-lmi-artifact>

## Package Architecture (`src/pcsrc/`)
- `models/platoon.py`: CAV delay-lag transfer functions (`PC-SRC`, `No Predictor`, `Classical Smith`, `Padé [1/1]`, `Robust H-inf`) and calibrated nonlinear/linearized IDM HDV dynamics over $\Theta^{\mathrm{HDV}}$.
- `analysis/asymptotic.py`: Low-frequency Maclaurin expansions (Propositions 1–4), Lagrange remainder overlap verification, and headway-vs-actuator-lag co-optimization.
- `sos/certificates.py`: Open-source CVXPY/Clarabel SOS-LMI solvers for Certificate A (Sturm sequence + univariate SOS), Certificate B (5-variable lifted Positivstellensatz with Stage-A generator $\mu_4(y-z)$), and Monolithic Baselines M1–M3.
- `simulation/micro_sim.py`: 20-vehicle nonlinear mixed-platoon micro-simulator (5-baseline emergency & stress braking, 1,000-trial Monte Carlo robustness, adversarial worst-case search, and CAV penetration rate 0%–100% sweep with safety/throughput/comfort metrics).
- `deployment/lut.py`: Offline-to-online $\mathcal{O}(1)$ 3D certified gain Look-Up Table latency and memory benchmark.

## Running on Google Colab via `colab` CLI

```bash
# Provision a Colab session, upload the package, and run the complete verification suite
colab new -s pcsrc
colab install -s pcsrc cvxpy clarabel scipy matplotlib
colab exec -s pcsrc -f scripts/run_colab_suite.py
colab stop -s pcsrc
```

## Local Reproduction

```bash
pip install -e .
PYTHONPATH=src python3 scripts/run_colab_suite.py
```

Outputs (`colab_verified_summary.json` and vector PDF figures `fig_headway_domain.pdf`, `fig_bode_mixed.pdf`, `fig_platoon_validation.pdf`) are written to `results/colab_run/`.
