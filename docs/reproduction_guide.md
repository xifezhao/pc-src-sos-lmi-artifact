# Step-by-Step Reproduction Guide

1. Run `python scripts/run_paper_experiments_full.py` to execute all SDP solvers, frequency-domain sweeps, microscopic ODE simulations, Monte Carlo / adversarial evaluations, and LUT benchmarks.
2. Run `pytest tests/ -v` to validate all unit, integration, regression, and AE checks.
3. Inspect `results/paper_master_results.json` for the complete consolidated numerical dictionary matching Tables I–VII and Figs. 4–9.
