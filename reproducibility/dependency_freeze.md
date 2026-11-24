# Dependency Freeze & Environment Snapshot

- Exact pinned versions are recorded in `requirements-lock.txt` (tested on Python 3.13.18 Linux x86_64 with `cvxpy`, `clarabel`, `numpy`, `scipy`, `matplotlib`, `pytest`).
- GitHub Actions CI installs directly from `requirements-lock.txt`.
- Running `scripts/reproduce_paper.py` automatically writes `results/reproduced/environment_snapshot.txt` capturing Python version, OS/CPU platform, and installed package versions.
