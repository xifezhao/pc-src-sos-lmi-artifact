# One-Command Reproduction

```bash
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e .
shasum -a 256 -c data/reference/golden_hashes.sha256
PYTHONPATH=src pytest -v tests/
PYTHONPATH=src python3 scripts/reproduce_paper.py --fast
```

For full 1,000-trial Monte Carlo and 20,000-query LUT reproduction (~10–12 min on CPU):
```bash
PYTHONPATH=src python3 scripts/reproduce_paper.py --full
```
