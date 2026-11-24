# ==============================================================================
# src/pcsrc/simulation/__init__.py
#
# Simulation utilities package for pcsrc.
#
# Purpose:
#   - Expose simulation-related utilities through a compact public namespace.
#   - Keep package import lightweight with lazy attribute loading.
#   - Support trajectory simulation, sampling, perturbations, Monte Carlo analysis,
#     and closed-loop verification workflows.
#
# Design notes:
#   - This module intentionally avoids importing heavy optional dependencies at
#     package import time.
#   - Submodules are imported lazily by __getattr__.
#   - Missing optional submodules produce clear AttributeError messages rather
#     than failing package import.
# ==============================================================================

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "0.1.0"


# ------------------------------------------------------------------------------
# Lazy public API
# ------------------------------------------------------------------------------

_LAZY_SUBMODULES: dict[str, str] = {
    # Core simulation primitives
    "Trajectory": "pcsrc.simulation.trajectory",
    "TrajectoryPoint": "pcsrc.simulation.trajectory",
    "TrajectorySummary": "pcsrc.simulation.trajectory",
    "simulate_trajectory": "pcsrc.simulation.trajectory",
    "simulate_discrete_time": "pcsrc.simulation.trajectory",
    "simulate_continuous_time": "pcsrc.simulation.trajectory",

    # Integrators
    "IntegratorKind": "pcsrc.simulation.integrators",
    "IntegrationResult": "pcsrc.simulation.integrators",
    "euler_step": "pcsrc.simulation.integrators",
    "rk4_step": "pcsrc.simulation.integrators",
    "integrate_ode": "pcsrc.simulation.integrators",

    # Sampling
    "SampleSet": "pcsrc.simulation.sampling",
    "SamplerConfig": "pcsrc.simulation.sampling",
    "GridSampler": "pcsrc.simulation.sampling",
    "RandomSampler": "pcsrc.simulation.sampling",
    "SobolSampler": "pcsrc.simulation.sampling",
    "sample_box": "pcsrc.simulation.sampling",
    "sample_ball": "pcsrc.simulation.sampling",

    # Perturbations / noise
    "NoiseModel": "pcsrc.simulation.perturbations",
    "BoundedNoise": "pcsrc.simulation.perturbations",
    "GaussianNoise": "pcsrc.simulation.perturbations",
    "AdversarialPerturbation": "pcsrc.simulation.perturbations",
    "apply_perturbation": "pcsrc.simulation.perturbations",

    # Monte Carlo utilities
    "MonteCarloConfig": "pcsrc.simulation.monte_carlo",
    "MonteCarloResult": "pcsrc.simulation.monte_carlo",
    "run_monte_carlo": "pcsrc.simulation.monte_carlo",
    "summarize_monte_carlo": "pcsrc.simulation.monte_carlo",

    # Closed-loop simulation
    "ClosedLoopSystem": "pcsrc.simulation.closed_loop",
    "simulate_closed_loop": "pcsrc.simulation.closed_loop",
    "rollout_policy": "pcsrc.simulation.closed_loop",

    # Diagnostics
    "SimulationStatus": "pcsrc.simulation.diagnostics",
    "SimulationDiagnostic": "pcsrc.simulation.diagnostics",
    "SimulationReport": "pcsrc.simulation.diagnostics",
    "check_simulation_bounds": "pcsrc.simulation.diagnostics",
}


def __getattr__(name: str) -> Any:
    """Lazily import public simulation symbols."""

    if name not in _LAZY_SUBMODULES:
        raise AttributeError(
            f"module 'pcsrc.simulation' has no attribute {name!r}"
        )

    module_name = _LAZY_SUBMODULES[name]

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"attribute {name!r} requires optional module {module_name!r}, "
            "but that module could not be imported"
        ) from exc

    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define public symbol {name!r}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return public symbols for interactive use."""

    return sorted(set(globals()) | set(__all__))


# ------------------------------------------------------------------------------
# Convenience helpers
# ------------------------------------------------------------------------------

def available_simulation_symbols() -> tuple[str, ...]:
    """Return lazily available simulation symbols."""

    return tuple(sorted(_LAZY_SUBMODULES))


def simulation_submodules() -> tuple[str, ...]:
    """Return referenced simulation submodules."""

    return tuple(sorted(set(_LAZY_SUBMODULES.values())))


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------

__all__ = [
    "__version__",
    "AdversarialPerturbation",
    "BoundedNoise",
    "ClosedLoopSystem",
    "GaussianNoise",
    "GridSampler",
    "IntegrationResult",
    "IntegratorKind",
    "MonteCarloConfig",
    "MonteCarloResult",
    "NoiseModel",
    "RandomSampler",
    "SampleSet",
    "SamplerConfig",
    "SimulationDiagnostic",
    "SimulationReport",
    "SimulationStatus",
    "SobolSampler",
    "Trajectory",
    "TrajectoryPoint",
    "TrajectorySummary",
    "apply_perturbation",
    "available_simulation_symbols",
    "check_simulation_bounds",
    "euler_step",
    "integrate_ode",
    "rk4_step",
    "rollout_policy",
    "run_monte_carlo",
    "sample_ball",
    "sample_box",
    "simulate_closed_loop",
    "simulate_continuous_time",
    "simulate_discrete_time",
    "simulate_trajectory",
    "simulation_submodules",
    "summarize_monte_carlo",
]
