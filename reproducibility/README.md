# Reproducibility Overview

All experiments in the accompanying *IEEE T-ITS* manuscript run completely offline on standard CPU hardware using open-source Python packages (`numpy`, `scipy`, `matplotlib`, `cvxpy`, `clarabel`) with **zero commercial solver dependencies**.

## Single Entry Point
```bash
# Fast Artifact Evaluation check (~60–90 s, 25 MC trials, 4,000 LUT queries, Figs. 3–5)
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

# Full paper reproduction (~10–12 min, 1,000 MC trials, 20,000 LUT queries, Figs. 3–5)
PYTHONPATH=src python3 scripts/reproduce_paper.py --full

# Standalone offline PDF figure regeneration (Figs. 3–5 only, ~20–40 s)
PYTHONPATH=src python3 scripts/make_figures.py --out-dir results/reproduced
```

Outputs are written to `results/reproduced/reproduced_summary.json` together with `results/reproduced/environment_snapshot.txt`, `results/reproduced/certified_lut_4x4x3.{npz,csv}`, and the three vector PDF figures (`fig_headway_domain.pdf`, `fig_bode_mixed.pdf`, `fig_platoon_validation.pdf`).
