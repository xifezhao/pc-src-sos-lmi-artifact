# Expected Runtime

On a standard 2–8 vCPU cloud VM (e.g., Google Colab CPU runtime):
- Single Dual-Certificate SDP solve (Stage A `2x2` + Stage B `21x21`): ~15–60 ms.
- Full experiment suite (`scripts/run_paper_experiments_full.py`, including 2D/3D SDP grids, 1,000-trial Monte Carlo, Joint Adversarial Grid Search, 15-permutation MPR sweep, and 100,000-query LUT benchmark): ~2–5 minutes.
- Full `pytest tests/ -v` suite: ~15–30 seconds.
