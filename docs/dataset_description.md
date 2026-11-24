# Bundled Dataset Schema

This repository bundles one empirical trajectory excerpt used for the highway replay benchmark in Section V-D of the manuscript:

- **File:** `data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv`
- **Rows / Sampling:** 400 frames at 10 Hz (`time_s` from `0.0` to `39.9` s), `Vehicle_ID=1209`, `Lane_ID=3`, `Frame_ID=3429..3828`.
- **SHA-256:** `d78f493b635ce64ceceec6a5be08efcb5118296a46491eddb05a2efdf506d915`

## Columns
| Column | Unit | Description |
|---|---|---|
| `time_s` | s | Relative elapsed time (0.1 s step) |
| `Vehicle_ID`, `Frame_ID`, `Global_Time_ms`, `Lane_ID` | — | Original NGSIM I-80 record identifiers |
| `Local_Y_ft`, `v_Vel_fps`, `v_Acc_fps2` | ft, ft/s, ft/s² | Original imperial NGSIM fields |
| `v_Vel_ms`, `v_Acc_ms2` | m/s, m/s² | Derived SI speed and acceleration (Savitzky–Golay smoothed) |

For full USDOT/FHWA provenance, DOI (`10.21949/1504477`), CC BY-SA license terms, and mirror URL, see [data/raw/external/README.md](../data/raw/external/README.md).
