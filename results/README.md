# Experimental Results & Artifact Archive

This directory contains the collected experimental results, benchmark logs, publication tables, and figure artifacts generated from the reproduction pipeline executed on Google Colab.

## Directory Structure

```text
results/
├── benchmarks/                                # Micro-benchmarks & performance audits
│   ├── certificate_validation/                # Lyapunov dual-certificate validation metrics
│   │   ├── benchmark_certificate_validation.csv
│   │   └── benchmark_certificate_validation.json
│   ├── lut_lookup/                            # O(1) online lookup latency & throughput
│   │   ├── benchmark_lut_lookup.csv
│   │   └── benchmark_lut_lookup.json
│   ├── micro_simulation/                      # Mixed platoon micro-simulation runtime
│   │   ├── benchmark_micro_simulation.csv
│   │   └── benchmark_micro_simulation.json
│   └── sos_grid_runtime/                      # SOS polynomial & LMI grid solve runtime
│       ├── benchmark_sos_grid_runtime.csv
│       └── benchmark_sos_grid_runtime.json
│
├── figures/                                   # Publication PGFPlots & PDF figures
│   ├── fig5_gain_feasible_domain.(pdf|tex)    # Fig. 5: Exact string stability boundary
│   ├── fig6_structural_conservatism.(pdf|tex) # Fig. 6: Non-conservatism vs monolithic LMI
│   ├── fig7_space_time_heatmap.(pdf|tex)      # Fig. 7: Multi-vehicle velocity decay heatmap
│   ├── fig8_time_domain_profiles.(pdf|tex)    # Fig. 8: Time-domain platoon profiles
│   └── fig9_lut_deployment_architecture.(pdf|tex) # Fig. 9: Real-time LUT deployment
│
├── logs/                                      # Full reproduction run execution logs
│   ├── environment_snapshot.txt               # Hardware & software environment metadata
│   ├── full_reproduction.log                  # Full 7-stage Colab pipeline log
│   ├── lut_benchmark.log                      # Look-up table latency benchmark log
│   ├── simulation.log                         # Platoon micro-simulation trajectory log
│   └── solver_grid.log                        # LMI solver status & residual diagnostics
│
├── metrics/                                   # Quantitative evaluation metrics (CSV)
│   ├── acceleration_constraint_metrics.csv    # Actuator bound & clipping counts
│   ├── delay_robustness_metrics.csv           # Robust margins under sensing delays
│   ├── feasibility_summary.csv                # SOS/LMI parametric grid feasibility
│   ├── online_lookup_benchmark.csv            # LUT query latencies & throughput
│   ├── spacing_safety_metrics.csv             # Platoon minimum bumper-to-bumper gap
│   ├── stop_and_go_metrics.csv                # Stop-and-go wave damping performance
│   └── structural_conservatism_metrics.csv    # Dual-certificate vs baseline area ratio
│
└── tables/                                    # Publication LaTeX tables
    ├── table_I_parameters.tex                 # Table I: Platoon physical parameters
    ├── table_II_sos_feasibility.tex           # Table II: SOS/LMI problem dimensions
    ├── table_III_robustness_metrics.tex       # Table III: Delay robustness margins
    ├── table_IV_time_domain_metrics.tex       # Table IV: Emergency braking metrics
    └── table_V_lut_runtime.tex                # Table V: Online LUT latency & memory
```
