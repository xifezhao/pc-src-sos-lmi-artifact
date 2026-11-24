# Independent Artifact Evaluation (AE) Verification Script

Reviewers can execute the following commands on Linux/macOS/Colab CPU (≥ 2 GB RAM, ≥ 250 MB disk):

```bash
git clone https://github.com/xifezhao/pc-src-sos-lmi-artifact.git
cd pc-src-sos-lmi-artifact
python3 -m pip install -r requirements-lock.txt
python3 -m pip install --no-deps -e .
shasum -a 256 -c data/reference/golden_hashes.sha256
PYTHONPATH=src python3 -c "import pcsrc, pcsrc.sos; [getattr(pcsrc.sos, n) for n in pcsrc.sos._LAZY_EXPORTS]; print('OK:', len(pcsrc.sos._LAZY_EXPORTS))"
PYTHONPATH=src pytest -v tests/
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast
```

Verify that:
1. All 11 checksums in `data/reference/golden_hashes.sha256` report `OK`.
2. Public API smoke test prints `OK: 7` and all tests in `tests/test_pcsrc_ae.py` pass (`9 passed`).
3. `results/reproduced/` contains `reproduced_summary.json` (matching `data/reference/golden_metrics.json`), `certified_lut_4x4x3.{npz,csv}`, Figs. 3–5 PDFs (`fig_headway_domain.pdf`, `fig_bode_mixed.pdf`, `fig_platoon_validation.pdf`), and `reproduced_hashes.sha256`.
