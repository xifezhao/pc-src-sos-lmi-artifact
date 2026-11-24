# Reproducibility Guide

## Hardware Requirements
- **CPU:** Multi-core modern processor (e.g., Intel i7 or equivalent)
- **RAM:** Minimum 16GB, recommended 32GB for full-scale simulations.
- **OS:** Linux, macOS, or Windows via WSL2.

## Step-by-Step Reproduction
1. Clone the repository and navigate to this directory.
2. Install the package using pip: `pip install -e .` or using conda: `conda env create -f environment.yml && conda activate pcsrc`
3. Run `make reproduce-fast` for a quick verification.
4. Run `make reproduce-all` to generate all figures and results.

## Expected Outputs and Timing
- `make reproduce-fast`: Completes in < 5 minutes. Outputs summary metrics to the console.
- `make reproduce-all`: May take up to 2 hours depending on hardware. Outputs generated in `results/` and `figures/` directories.

## Troubleshooting
- **Mosek License Issue:** If using Mosek, ensure your license file is placed appropriately (e.g., `~/mosek/mosek.lic`).
- **Memory Errors:** If the full reproduction crashes, try closing other applications or running on a machine with more RAM.
