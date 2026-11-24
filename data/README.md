# Data Directory

This directory contains the single-source-of-truth parameter definitions, bundled empirical NGSIM I-80 trajectory excerpt, and verified golden reference metrics/checksums.

## Layout
- `raw/scenario_definitions/nominal_platoon_parameters.yaml`: Nominal physical, controller (`ka = 0.80`), and HDV uncertainty box (`Theta^HDV`) parameters matching `PlatoonParams` and `HDVUncertaintyBox` in `src/pcsrc/models/platoon.py`.
- `raw/scenario_definitions/ngsim_i80_leader_trajectory.csv`: Extracted 40.0 s (400-frame, 10 Hz) empirical leader trajectory (`Vehicle_ID=1209`, Frames `3429–3828`, Lane 3) derived from USDOT/FHWA NGSIM I-80. SHA-256: `d78f493b635ce64ceceec6a5be08efcb5118296a46491eddb05a2efdf506d915`.
- `raw/external/README.md`: Full NGSIM provenance, official USDOT DOI (`10.21949/1504477`), license terms (CC BY-SA 3.0 / 4.0), mirror URL, and Savitzky–Golay preprocessing notes.
- `reference/golden_metrics.json`: Verified numerical targets reproduced on Google Colab CPU.
- `reference/golden_hashes.sha256`: SHA-256 integrity manifest verified by CI (`shasum -a 256 -c data/reference/golden_hashes.sha256`).
