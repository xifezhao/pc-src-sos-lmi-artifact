# Contributing Guide

1. Install dependencies in a clean virtual environment (`pip install -e .` and `pip install cvxpy clarabel scs scipy pyyaml pytest matplotlib`).
2. Ensure all analytical bounds, SDP Positivstellensatz certificates, and microscopic ODE tests pass via `pytest tests/ -v`.
3. Regenerate full paper tables, figures, and SHA-256 digests via `python scripts/run_paper_experiments_full.py` before submitting pull requests.
