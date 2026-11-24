# Manuscript Figure Mapping

All data-driven PDF figures (Figs. 3–5) can be regenerated offline in one command via `PYTHONPATH=src python3 scripts/reproduce_paper.py --fast` (written to `results/reproduced/`) or standalone via `PYTHONPATH=src python3 scripts/make_figures.py`.

- **Fig. 1 & Fig. 2:** Vector TikZ schematic diagrams rendered directly in LaTeX (mixed-platoon corridor and PC-SRC internal-model delay predictor block diagram).
- **Fig. 3 (`results/colab_run/fig_headway_domain.pdf`):** Minimum verifiable headway $h_{\min}$ vs. $\tau_e$ and certified $(k_p, k_v)$ feasibility domain (`scripts/make_figures.py`, `scripts/reproduce_paper.py`, `scripts/run_colab_suite.py`).
- **Fig. 4 (`results/colab_run/fig_bode_mixed.pdf`):** Isolated CAV Bode response and $m=2$ mixed-unit attenuation at $h=1.00\,$s and $h=1.10\,$s (`scripts/make_figures.py`, `scripts/reproduce_paper.py`, `scripts/run_colab_supplement.py`).
- **Fig. 5 (`results/colab_run/fig_platoon_validation.pdf`):** 20-vehicle nonlinear mixed-platoon wave dissipation and 0%–100% MPR scalability (`scripts/make_figures.py`, `scripts/reproduce_paper.py`, `scripts/run_colab_suite.py`).
