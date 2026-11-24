# ==============================================================================
# src/pcsrc/sos/__init__.py
#
# Sum-of-squares and LMI certification utilities for pcsrc.
#
# Purpose:
#   - Expose a stable public API for SOS/LMI-based controller certification.
#   - Keep heavy optional dependencies isolated from the core simulation stack.
#   - Provide lazy imports so ``import pcsrc.sos`` remains lightweight.
#   - Offer small capability helpers for checking optional solver availability.
#
# Design notes:
#   - SOS/LMI routines often require optional packages such as cvxpy, sympy, and
#     external SDP solvers.
#   - This package-level module should not eagerly import heavy solvers.
#   - Public names are resolved lazily via ``__getattr__``.
#
# Expected package layout:
#
#   pcsrc.sos
#   ├── __init__.py
#   ├── constraints.py
#   ├── certificates.py
#   ├── polynomial.py
#   ├── lmi_problem.py
#   ├── solver.py
#   └── verification.py
#
# The actual project tree may contain a subset of these modules. Missing optional
# modules are handled gracefully at attribute access time.
# ==============================================================================

from __future__ import annotations

from importlib import import_module, util
from typing import Any


# ------------------------------------------------------------------------------
# Package metadata
# ------------------------------------------------------------------------------

__version__ = "0.1.0"


# ------------------------------------------------------------------------------
# Optional dependency discovery
# ------------------------------------------------------------------------------

_OPTIONAL_DEPENDENCIES: tuple[str, ...] = (
    "cvxpy",
    "sympy",
    "numpy",
    "scipy",
)


def has_optional_dependency(name: str) -> bool:
    """Return whether an optional Python dependency is importable.

    Parameters
    ----------
    name:
        Import name of the optional dependency.

    Examples
    --------
    >>> has_optional_dependency("cvxpy")
    True
    """

    return util.find_spec(name) is not None


def available_optional_dependencies() -> dict[str, bool]:
    """Return availability map for common SOS/LMI dependencies."""

    return {
        dependency: has_optional_dependency(dependency)
        for dependency in _OPTIONAL_DEPENDENCIES
    }


def sos_dependencies_available() -> bool:
    """Return whether the common SOS stack is available.

    This checks for packages commonly needed by SOS/LMI workflows. Individual
    modules may require only a subset.
    """

    availability = available_optional_dependencies()

    return bool(
        availability.get("cvxpy", False)
        and availability.get("sympy", False)
        and availability.get("numpy", False)
    )


def require_optional_dependency(name: str) -> None:
    """Raise an informative error if an optional dependency is missing."""

    if has_optional_dependency(name):
        return

    raise ImportError(
        f"Optional dependency {name!r} is required for this SOS/LMI feature. "
        f"Install the project with the appropriate extras, for example: "
        f"`pip install pcsrc[sos]`, or install {name!r} manually."
    )


def require_sos_dependencies() -> None:
    """Raise an informative error if the common SOS stack is unavailable."""

    missing = [
        dependency
        for dependency, available in available_optional_dependencies().items()
        if dependency in {"cvxpy", "sympy", "numpy"} and not available
    ]

    if not missing:
        return

    raise ImportError(
        "SOS/LMI functionality requires optional dependencies that are not "
        f"currently importable: {', '.join(missing)}. "
        "Install the SOS extras, for example: `pip install pcsrc[sos]`."
    )


# ------------------------------------------------------------------------------
# Lazy public API
# ------------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, str] = {
    # constraints.py
    "SOSConstraint": "pcsrc.sos.constraints",
    "SOSConstraintType": "pcsrc.sos.constraints",
    "SOSConstraintSet": "pcsrc.sos.constraints",
    "make_positive_constraint": "pcsrc.sos.constraints",
    "make_negative_constraint": "pcsrc.sos.constraints",
    "make_sos_constraint": "pcsrc.sos.constraints",

    # certificates.py
    "SOSCertificate": "pcsrc.sos.certificates",
    "SOSCertificateStatus": "pcsrc.sos.certificates",
    "CertificateMetadata": "pcsrc.sos.certificates",
    "load_certificate": "pcsrc.sos.certificates",
    "save_certificate": "pcsrc.sos.certificates",

    # polynomial.py
    "Monomial": "pcsrc.sos.polynomial",
    "Polynomial": "pcsrc.sos.polynomial",
    "PolynomialBasis": "pcsrc.sos.polynomial",
    "monomial_powers": "pcsrc.sos.polynomial",
    "make_monomial_basis": "pcsrc.sos.polynomial",

    # lmi_problem.py
    "LMIProblem": "pcsrc.sos.lmi_problem",
    "LMIProblemData": "pcsrc.sos.lmi_problem",
    "LMIResult": "pcsrc.sos.lmi_problem",
    "LMIStatus": "pcsrc.sos.lmi_problem",

    # solver.py
    "SOSSolver": "pcsrc.sos.solver",
    "SOSSolverConfig": "pcsrc.sos.solver",
    "SOSSolverResult": "pcsrc.sos.solver",
    "solve_sos_problem": "pcsrc.sos.solver",
    "installed_solvers": "pcsrc.sos.solver",

    # verification.py
    "VerificationResult": "pcsrc.sos.verification",
    "VerificationStatus": "pcsrc.sos.verification",
    "verify_certificate": "pcsrc.sos.verification",
    "verify_controller_certificate": "pcsrc.sos.verification",

    # residuals.py & paper_positivstellensatz.py
    "compute_paper_certificate_residuals": "pcsrc.sos.residuals",
    "compute_paper_table_metrics_snapshot": "pcsrc.sos.residuals",
    "solve_certificate_A": "pcsrc.sos.paper_positivstellensatz",
    "solve_certificate_B": "pcsrc.sos.paper_positivstellensatz",
    "solve_dual_certificate_full": "pcsrc.sos.paper_positivstellensatz",
    "solve_monolithic_lifted_sdp": "pcsrc.sos.paper_positivstellensatz",
    "solve_monolithic_young_sdp": "pcsrc.sos.paper_positivstellensatz",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public SOS/LMI symbols.

    This keeps importing ``pcsrc.sos`` cheap and prevents optional solver
    dependencies from being imported unless a solver-facing symbol is requested.
    """

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'pcsrc.sos' has no attribute {name!r}")

    module_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)

    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} does not define public symbol {name!r}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted package attributes for interactive discovery."""

    return sorted(
        set(globals())
        | set(_LAZY_EXPORTS)
        | {
            "__version__",
            "available_optional_dependencies",
            "has_optional_dependency",
            "require_optional_dependency",
            "require_sos_dependencies",
            "sos_dependencies_available",
        }
    )


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------

__all__ = [
    "__version__",

    # Dependency helpers
    "available_optional_dependencies",
    "has_optional_dependency",
    "require_optional_dependency",
    "require_sos_dependencies",
    "sos_dependencies_available",

    # constraints.py
    "SOSConstraint",
    "SOSConstraintType",
    "SOSConstraintSet",
    "make_positive_constraint",
    "make_negative_constraint",
    "make_sos_constraint",

    # certificates.py
    "SOSCertificate",
    "SOSCertificateStatus",
    "CertificateMetadata",
    "load_certificate",
    "save_certificate",

    # polynomial.py
    "Monomial",
    "Polynomial",
    "PolynomialBasis",
    "monomial_powers",
    "make_monomial_basis",

    # lmi_problem.py
    "LMIProblem",
    "LMIProblemData",
    "LMIResult",
    "LMIStatus",

    # solver.py
    "SOSSolver",
    "SOSSolverConfig",
    "SOSSolverResult",
    "solve_sos_problem",
    "installed_solvers",

    # verification.py
    "VerificationResult",
    "VerificationStatus",
    "verify_certificate",
    "verify_controller_certificate",
]
