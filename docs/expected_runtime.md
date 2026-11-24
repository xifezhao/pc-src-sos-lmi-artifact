# Compute Budget

All benchmarks were executed on a single standard CPU environment (Google Colab Linux x86_64, no GPU, no commercial solver):
- Unit & AE test suite (`pytest -v tests/`): **~20 s**
- Fast verification (`scripts/reproduce_paper.py --fast`): **~50 s**
- Full reproduction (`scripts/reproduce_paper.py --full`, 1,000 MC trials + 20,000 LUT queries): **~10–12 min** (`636.9 s` measured)
- Peak RAM: **< 1.5 GB** (recommended ≥ 2 GB); Disk footprint: **< 250 MB**.
