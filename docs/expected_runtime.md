# Expected Runtime

Wall-clock timings measured on a single standard CPU (Linux x86_64, Python 3.13, no GPU):

| Command | Scope | Expected Wall-Clock Time |
|---|---|---|
| `shasum -a 256 -c data/reference/golden_hashes.sha256` | 11-file integrity check | < 1 s |
| `PYTHONPATH=src pytest -v tests/` | 9 unit, API, YAML-sync & scientific invariant tests | ~5–25 s |
| `PYTHONPATH=src python3 scripts/make_figures.py` | Offline vector PDF regeneration of Figs. 3–5 | ~60–120 s |
| `PYTHONPATH=src python3 scripts/reproduce_paper.py --fast` | All tables/analyses, 25 MC trials, 4,000 LUT queries, Figs. 3–5 | ~60–150 s |
| `PYTHONPATH=src python3 scripts/reproduce_paper.py --full` | Full paper suite, 1,000 MC trials, 20,000 LUT queries, Figs. 3–5 | ~10–12 min (`636.9 s` measured) |

**Hardware-Dependent Latency Note:** LUT query latency (e.g., `2.59 µs` flat-index reference on Colab x86_64) and Clarabel SDP wall-clock times (`450.4 ms` Stage-B reference) are CPU-dependent and can only be compared numerically on identical hardware; cross-machine verification checks functional feasibility, memory footprint (`576 bytes`), and numerical accuracy invariants.

See [reproducibility/compute_budget.md](../reproducibility/compute_budget.md) for memory and disk budget details.
