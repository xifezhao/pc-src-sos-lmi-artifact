# Reproducibility Overview

All experiments in the accompanying *IEEE T-ITS* manuscript run completely offline on standard CPU hardware using open-source Python packages (`numpy`, `scipy`, `matplotlib`, `cvxpy`, `clarabel`) with **zero commercial solver dependencies**.

## Single Entry Point
```bash
# Fast Artifact Evaluation check (~60–150 s, 25 MC trials, 4,000 LUT queries, Figs. 3–5)
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast

# Full paper reproduction (~10–12 min, 1,000 MC trials, 20,000 LUT queries, Figs. 3–5)
PYTHONPATH=src python3 scripts/reproduce_paper.py --full

# Standalone offline PDF figure regeneration (Figs. 3–5 only, ~60–120 s)
PYTHONPATH=src python3 scripts/make_figures.py --out-dir results/reproduced
```

Outputs are written to `results/reproduced/` (intentionally gitignored as a clean runtime output directory, while frozen SHA-256-verified reference artifacts reside in `results/colab_run/`), including `reproduced_summary.json`, `environment_snapshot.txt`, `certified_lut_4x4x3.{npz,csv}`, the three vector PDF figures (`fig_headway_domain.pdf`, `fig_bode_mixed.pdf`, `fig_platoon_validation.pdf`), and `reproduced_hashes.sha256`.
