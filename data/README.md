# Data Directory

- `raw/scenario_definitions/`: Nominal Table I physical parameters, leader braking profiles, and delay perturbation bounds.
- `processed/feasibility_grid/`: SDP-computed 3D and 2D controller gain domain grids and boundary contours.
- `processed/simulations/`: Coupled nonlinear delayed-IDM + PC-SRC platoon trajectories.
- `processed/lut/`: Certified O(1) gain scheduling look-up table (`729`-cell canonical verification grid and `148,716`-cell deployment benchmark).
- `reference/`: Golden SHA-256 digests and regression metrics.
