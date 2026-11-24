# ==============================================================================
# src/pcsrc/control/__init__.py
#
# Control subpackage initialization for pcsrc.
#
# Purpose:
#   - Define the public API for platoon-control modules.
#   - Provide lightweight package metadata and lazy imports.
#   - Avoid importing heavy synthesis/optimization dependencies at package import
#     time.
#
# Submodules:
#   - pcsrc.control.base_controller
#       Abstract controller interfaces, command containers, and runtime context.
#
#   - pcsrc.control.robust_controller
#       SOS/LMI-certified or certificate-backed robust CAV controllers.
#
#   - pcsrc.control.nominal_controller
#       Baseline nominal feedback controllers for simulation and ablation studies.
#
#   - pcsrc.control.saturation
#       Acceleration/rate saturation utilities and safe command projection.
#
#   - pcsrc.control.control_utils
#       Shared helpers for feedback gains, vectorization, validation, and rollout.
#
# Design notes:
#   - Lazy import via ``__getattr__`` keeps ``import pcsrc.control`` cheap.
#   - The symbols listed in ``__all__`` define the intended public API.
#   - Import errors are re-raised with context to make missing optional
#     dependencies easier to diagnose.
# ==============================================================================

from __future__ import annotations

from importlib import import_module
from typing import Any


__version__ = "1.0.0"


# ------------------------------------------------------------------------------
# Lazy public API registry
# ------------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Base controller abstractions
    "BaseController": ("pcsrc.control.base_controller", "BaseController"),
    "ControllerContext": ("pcsrc.control.base_controller", "ControllerContext"),
    "ControllerMetadata": ("pcsrc.control.base_controller", "ControllerMetadata"),
    "ControlCommand": ("pcsrc.control.base_controller", "ControlCommand"),
    "ControlMode": ("pcsrc.control.base_controller", "ControlMode"),
    "ControllerStatus": ("pcsrc.control.base_controller", "ControllerStatus"),

    # Robust / certified controller
    "RobustController": ("pcsrc.control.robust_controller", "RobustController"),
    "RobustControllerConfig": ("pcsrc.control.robust_controller", "RobustControllerConfig"),
    "RobustControllerDiagnostics": (
        "pcsrc.control.robust_controller",
        "RobustControllerDiagnostics",
    ),

    # Nominal baseline controller
    "NominalController": ("pcsrc.control.nominal_controller", "NominalController"),
    "NominalControllerConfig": (
        "pcsrc.control.nominal_controller",
        "NominalControllerConfig",
    ),
    "NominalFeedbackGains": (
        "pcsrc.control.nominal_controller",
        "NominalFeedbackGains",
    ),

    # Saturation and projection utilities
    "SaturationConfig": ("pcsrc.control.saturation", "SaturationConfig"),
    "SaturationResult": ("pcsrc.control.saturation", "SaturationResult"),
    "RateLimiterState": ("pcsrc.control.saturation", "RateLimiterState"),
    "clip_command": ("pcsrc.control.saturation", "clip_command"),
    "apply_rate_limit": ("pcsrc.control.saturation", "apply_rate_limit"),
    "project_command": ("pcsrc.control.saturation", "project_command"),

    # Shared control utilities
    "FeedbackVector": ("pcsrc.control.control_utils", "FeedbackVector"),
    "ControlVectorization": ("pcsrc.control.control_utils", "ControlVectorization"),
    "build_feedback_vector": ("pcsrc.control.control_utils", "build_feedback_vector"),
    "evaluate_linear_feedback": (
        "pcsrc.control.control_utils",
        "evaluate_linear_feedback",
    ),
    "commands_to_mapping": ("pcsrc.control.control_utils", "commands_to_mapping"),
}


def __getattr__(name: str) -> Any:
    """Lazily import public control symbols.

    This hook is invoked when an attribute is not found in the module namespace.
    It allows users to write:

        from pcsrc.control import NominalController

    without importing all control submodules immediately.
    """

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'pcsrc.control' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]

    try:
        module = import_module(module_name)
        value = getattr(module, attr_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {name!r} from {module_name!r}. "
            "Check that optional control dependencies are installed and that "
            "the pcsrc package is on PYTHONPATH."
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes for interactive completion."""

    return sorted(set(globals()) | set(_LAZY_IMPORTS))


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------

__all__ = [
    "__version__",

    # Base controller abstractions
    "BaseController",
    "ControllerContext",
    "ControllerMetadata",
    "ControlCommand",
    "ControlMode",
    "ControllerStatus",

    # Robust / certified controller
    "RobustController",
    "RobustControllerConfig",
    "RobustControllerDiagnostics",

    # Nominal baseline controller
    "NominalController",
    "NominalControllerConfig",
    "NominalFeedbackGains",

    # Saturation utilities
    "SaturationConfig",
    "SaturationResult",
    "RateLimiterState",
    "clip_command",
    "apply_rate_limit",
    "project_command",

    # Shared utilities
    "FeedbackVector",
    "ControlVectorization",
    "build_feedback_vector",
    "evaluate_linear_feedback",
    "commands_to_mapping",
]
