# Expected Runtime

Wall-clock timings measured on a single standard CPU (Linux x86_64, Python 3.13, no GPU):

| Command | Scope | Expected Wall-Clock Time |
|---|---|---|
| `shasum -a 256 -c data/reference/golden_hashes.sha256` | 11-file integrity check | < 1 s |
| `PYTHONPATH=src pytest -v tests/` | 7 unit & scientific invariant tests | ~15–25 s |
| `PYTHONPATH=src python3 scripts/reproduce_paper.py --fast` | All tables/analyses, 25 MC trials, 4,000 LUT queries | ~45–60 s |
| `PYTHONPATH=src python3 scripts/reproduce_paper.py --full` | Full paper suite, 1,000 MC trials, 20,000 LUT queries | ~10–12 min (`636.9 s` measured) |

See [reproducibility/compute_budget.md](../reproducibility/compute_budget.md) for memory and disk budget details.
