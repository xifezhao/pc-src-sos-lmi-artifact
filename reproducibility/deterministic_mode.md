# Deterministic Execution

All analytical checks, Sturm root isolations, Clarabel interior-point solves, nonlinear micro-simulations (`dt = 0.01 s`), and seeded NumPy RNG streams run deterministically on CPU. Minor floating-point differences (`< 1e-6`) may occur across BLAS/LAPACK implementations; `tests/test_pcsrc_ae.py` verifies all scientific invariants within documented tolerances.
