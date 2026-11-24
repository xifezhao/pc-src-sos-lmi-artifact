# Reproducibility Overview

All experiments in the accompanying *IEEE T-ITS* manuscript run on standard CPU hardware using open-source Python packages (`numpy`, `scipy`, `matplotlib`, `cvxpy`, `clarabel`) with **zero commercial solver dependencies**.

## Single Entry Point
```bash
# Fast Artifact Evaluation check (~50 s, 25 MC trials, 4,000 LUT queries)
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

# Full paper reproduction (~10–12 min, 1,000 MC trials, 20,000 LUT queries)
PYTHONPATH=src python3 scripts/reproduce_paper.py --full
```

Outputs are written to `results/reproduced/reproduced_summary.json` together with `results/reproduced/environment_snapshot.txt` and `results/reproduced/certified_lut_4x4x3.{npz,csv}`.
