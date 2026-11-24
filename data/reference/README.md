# Reference Golden Metrics and Checksums

- `golden_metrics.json`: Verified scalar targets for all key numerical claims (low-frequency overlap, SOS-LMI runtimes/coverage, Bode peaks, 20-vehicle nonlinear platoon safety, real NGSIM I-80 replay, 1,000-trial Monte Carlo, MPR sweep, and certified 48-cell LUT benchmark).
- `golden_hashes.sha256`: SHA-256 digest manifest checked in CI via `shasum -a 256 -c data/reference/golden_hashes.sha256`.
