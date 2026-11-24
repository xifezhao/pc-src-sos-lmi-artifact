# External Calibration & Public Empirical Trajectory Data

1. **HDV Uncertainty Box ($\Theta^{\mathrm{HDV}}$):**  
   Calibrated from published naturalistic highway car-following studies on NGSIM and highD trajectories:
   - Desired time headway $T_i \in [0.70, 1.50]\,\mathrm{s}$
   - Maximum acceleration $a_{\max,i} \in [0.90, 1.80]\,\mathrm{m/s^2}$
   - Comfortable deceleration $b_i \in [1.20, 2.50]\,\mathrm{m/s^2}$
   - Neuromuscular reaction lag $\tau_{r,i} \in [0.30, 0.72]\,\mathrm{s}$

2. **Public Empirical NGSIM I-80 Trajectory CSV:**  
   `data/raw/scenario_definitions/ngsim_i80_leader_trajectory.csv` contains a 40.0\,s (400-frame, 10\,Hz) public-domain U.S. FHWA NGSIM I-80 vehicle trajectory (`Vehicle_ID = 1209`, Frames `3429–3828`, Lane `3`, speed `7.11–18.51 m/s`, peak deceleration `-3.174 m/s^2`) extracted via `scripts/run_colab_ngsim_real.py` from the public NGSIM I-80 dataset.
