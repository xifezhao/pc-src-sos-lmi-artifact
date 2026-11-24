# External Empirical Data Provenance: USDOT / FHWA NGSIM I-80

## Official Source & Citation
- **Dataset:** U.S. Department of Transportation (USDOT), Federal Highway Administration (FHWA), *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data — Interstate 80 (I-80), Emeryville, CA*.
- **Official DOI:** [10.21949/1504477](https://doi.org/10.21949/1504477)
- **Official Portal:** [USDOT Data Portal — NGSIM Vehicle Trajectories](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
- **License:** Creative Commons Attribution-ShareAlike 3.0 / 4.0 (CC BY-SA) as published on the USDOT Data Portal. Users redistributing derivatives must preserve attribution and share-alike terms.
- **Recommended Citation:**
  > U.S. Department of Transportation Federal Highway Administration (2016). *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data* [Dataset]. Provided by ITS DataHub through Data.transportation.gov. DOI: `10.21949/1504477`.

## Mirror Used by Extraction Script
For automated headless retrieval in `scripts/run_colab_ngsim_real.py`, the raw MATLAB archive `trajectories.mat` was downloaded from the community mirror:
- Mirror URL: [github.com/cemsaz/NGSIM-trajectory-animation/raw/master/trajectories.mat](https://github.com/cemsaz/NGSIM-trajectory-animation/raw/master/trajectories.mat)
- Extracted subset: `Vehicle_ID=1209`, `Lane_ID=3`, `Frame_ID=3429..3828` (400 frames at 10 Hz, 40.0 s duration).
- Bundled CSV path: `data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv`
- Bundled CSV SHA-256: `d78f493b635ce64ceceec6a5be08efcb5118296a46491eddb05a2efdf506d915`

## Preprocessing & Derived Quantities
The raw NGSIM 10 Hz longitudinal positions (`Local_Y_ft`) and reported speeds (`v_Vel_fps`) are converted to SI units (`m`, `m/s`). Because raw NGSIM accelerations contain well-known high-frequency video-tracking noise, `v_Vel_ms` and `v_Acc_ms2` in the bundled CSV are derived via **Savitzky–Golay polynomial smoothing** followed by numerical differentiation of the smoothed speed profile (implemented in `scripts/run_colab_ngsim_real.py`). Both the original imperial columns (`Local_Y_ft`, `v_Vel_fps`, `v_Acc_fps2`) and the derived SI columns (`v_Vel_ms`, `v_Acc_ms2`) are retained in the CSV for full auditability.
