# Result Integrity Verification

Verify all bundled reference data and outputs against the SHA-256 manifest:
```bash
shasum -a 256 -c data/reference/golden_hashes.sha256
```
Run the automated test suite verifying all analytical bounds, SOS certificates, simulations, NGSIM replay, and certified LUT benchmarks:
```bash
PYTHONPATH=src pytest -v tests/
```
