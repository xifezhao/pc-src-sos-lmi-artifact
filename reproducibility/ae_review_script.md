# Independent Artifact Evaluation (AE) Verification Script

Reviewers can execute the following commands on Linux/macOS/Colab CPU (≥ 2 GB RAM, ≥ 250 MB disk):

```bash
git clone https://github.com/xifezhao/pc-src-sos-lmi-artifact.git
cd pc-src-sos-lmi-artifact
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e .
shasum -a 256 -c data/reference/golden_hashes.sha256
PYTHONPATH=src pytest -v tests/
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast
```

Verify that:
1. All checksums in `data/reference/golden_hashes.sha256` report `OK`.
2. All tests in `tests/test_pcsrc_ae.py` pass (`7 passed`).
3. `results/reproduced/reproduced_summary.json` matches `data/reference/golden_metrics.json`.
