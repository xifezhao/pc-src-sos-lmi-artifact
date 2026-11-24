# ==============================================================================
# src/pcsrc/models/__init__.py
#
# Model subpackage initializer for pcsrc.
#
# Purpose:
#   - Expose lightweight model-layer public APIs.
#   - Provide lazy access to vehicle, platoon, delay, uncertainty, and
#     linearization modules.
#   - Avoid importing numerical dependencies at package import time.
#
# Design notes:
#   - Keep this module side-effect free.
#   - Do not import numpy, scipy, pandas, cvxpy, matplotlib, or solver packages
#     here.
#   - Concrete classes/functions are loaded lazily through __getattr__.
# ==============================================================================

from __future__ import annotations

from importlib import import_module
from typing import Final


# ------------------------------------------------------------------------------
# Public lazy submodules
# ------------------------------------------------------------------------------

_LAZY_SUBMODULES: Final[dict[str, str]] = {
    "vehicle": "pcsrc.models.vehicle",
    "platoon": "pcsrc.models.platoon",
    "delay": "pcsrc.models.delay",
    "uncertainty": "pcsrc.models.uncertainty",
    "linearization": "pcsrc.models.linearization",
    "state_space": "pcsrc.models.state_space",
}


# ------------------------------------------------------------------------------
# Public symbols lazily re-exported from implementation modules
# ------------------------------------------------------------------------------

_LAZY_SYMBOLS: Final[dict[str, tuple[str, str]]] = {
    # Vehicle-level models
    "VehicleParams": ("pcsrc.models.vehicle", "VehicleParams"),
    "ActuatorLimits": ("pcsrc.models.vehicle", "ActuatorLimits"),
    "VehicleState": ("pcsrc.models.vehicle", "VehicleState"),
    "LongitudinalVehicleModel": (
        "pcsrc.models.vehicle",
        "LongitudinalVehicleModel",
    ),

    # Platoon-level models
    "PlatoonConfig": ("pcsrc.models.platoon", "PlatoonConfig"),
    "PlatoonState": ("pcsrc.models.platoon", "PlatoonState"),
    "MixedPlatoonModel": ("pcsrc.models.platoon", "MixedPlatoonModel"),

    # Delay and uncertainty descriptions
    "DelayParams": ("pcsrc.models.delay", "DelayParams"),
    "DelayBounds": ("pcsrc.models.delay", "DelayBounds"),
    "UncertaintyBounds": ("pcsrc.models.uncertainty", "UncertaintyBounds"),
    "MismatchBounds": ("pcsrc.models.uncertainty", "MismatchBounds"),

    # State-space and linearization helpers
    "StateSpaceMatrices": ("pcsrc.models.state_space", "StateSpaceMatrices"),
    "LinearizedPlatoonModel": (
        "pcsrc.models.linearization",
        "LinearizedPlatoonModel",
    ),
}


def __getattr__(name: str):
    """Lazily import model submodules and selected public symbols.

    Parameters
    ----------
    name:
        Requested attribute name.

    Returns
    -------
    module | object
        Imported module or symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public lazy API.
    """

    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module

    if name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]
        module = import_module(module_name)
        symbol = getattr(module, symbol_name)
        globals()[name] = symbol
        return symbol

    raise AttributeError(f"module 'pcsrc.models' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sorted public API for this subpackage."""

    return sorted(
        [
            *_LAZY_SUBMODULES.keys(),
            *_LAZY_SYMBOLS.keys(),
        ]
    )


__all__ = [
    *_LAZY_SUBMODULES.keys(),
    *_LAZY_SYMBOLS.keys(),
]
