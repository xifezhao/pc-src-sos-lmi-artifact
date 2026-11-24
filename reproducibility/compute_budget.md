# Compute Budget

All benchmarks were executed on a single standard CPU environment (Google Colab Linux x86_64, no GPU, no commercial solver):
- Unit, API & AE test suite (`pytest -v tests/`): **~5–25 s**
- Standalone offline figure regeneration (`scripts/make_figures.py`): **~60–120 s**
- Fast verification (`scripts/reproduce_paper.py --fast`, 25 MC trials + Figs. 3–5): **~60–150 s**
- Full reproduction (`scripts/reproduce_paper.py --full`, 1,000 MC trials + 20,000 LUT queries + Figs. 3–5): **~10–12 min** (`636.9 s` measured)
- Peak RAM: **< 1.5 GB** (recommended ≥ 2 GB); Disk footprint: **< 250 MB**.
