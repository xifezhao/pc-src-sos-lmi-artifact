"""Sum-of-squares (SOS) and LMI certification public API for ``pcsrc``.

Exposes lazy-loaded public certification routines from
``pcsrc.sos.certificates`` so ``import pcsrc.sos`` remains lightweight while
providing direct top-level access to all Dual-Certificate (Theorem 1) and
monolithic baseline (M1–M3) solvers.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Any

from pcsrc.version import __version__

# ------------------------------------------------------------------------------
# Optional dependency discovery
# ------------------------------------------------------------------------------

_OPTIONAL_DEPENDENCIES: tuple[str, ...] = (
    "cvxpy",
    "clarabel",
    "numpy",
    "scipy",
)


def has_optional_dependency(name: str) -> bool:
    """Return whether an optional Python dependency is importable."""
    return util.find_spec(name) is not None


def available_optional_dependencies() -> dict[str, bool]:
    """Return availability map for SOS/LMI solver dependencies."""
    return {
        dependency: has_optional_dependency(dependency)
        for dependency in _OPTIONAL_DEPENDENCIES
    }


def sos_dependencies_available() -> bool:
    """Return whether the core CVXPY + NumPy SOS stack is available."""
    availability = available_optional_dependencies()
    return bool(
        availability.get("cvxpy", False)
        and availability.get("numpy", False)
    )


def require_optional_dependency(name: str) -> None:
    """Raise an informative ImportError if an optional dependency is missing."""
    if has_optional_dependency(name):
        return
    raise ImportError(
        f"Optional dependency {name!r} is required for this SOS/LMI feature. "
        f"Install dependencies via `pip install -r requirements-lock.txt` "
        f"or install {name!r} manually."
    )


def require_sos_dependencies() -> None:
    """Raise an informative ImportError if the core SOS stack is unavailable."""
    missing = [
        dependency
        for dependency, available in available_optional_dependencies().items()
        if dependency in {"cvxpy", "numpy"} and not available
    ]
    if not missing:
        return
    raise ImportError(
        "SOS/LMI functionality requires dependencies that are not currently "
        f"importable: {', '.join(missing)}. "
        "Install dependencies via `pip install -r requirements-lock.txt`."
    )


# ------------------------------------------------------------------------------
# Lazy public API (aligned with pcsrc.sos.certificates)
# ------------------------------------------------------------------------------

_LAZY_EXPORTS: dict[str, str] = {
    "cert_a_sturm": "pcsrc.sos.certificates",
    "cert_a_sos": "pcsrc.sos.certificates",
    "cert_b_dual_sos": "pcsrc.sos.certificates",
    "cert_m1_young_fixed": "pcsrc.sos.certificates",
    "cert_m3_monolithic_lifted": "pcsrc.sos.certificates",
    "fast_exact_certificate_check": "pcsrc.sos.certificates",
    "benchmark_sos_formulations": "pcsrc.sos.certificates",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public SOS/LMI certification symbols."""
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


__all__ = [
    "__version__",
    # Dependency helpers
    "available_optional_dependencies",
    "has_optional_dependency",
    "require_optional_dependency",
    "require_sos_dependencies",
    "sos_dependencies_available",
    # Certification routines (pcsrc.sos.certificates)
    "cert_a_sturm",
    "cert_a_sos",
    "cert_b_dual_sos",
    "cert_m1_young_fixed",
    "cert_m3_monolithic_lifted",
    "fast_exact_certificate_check",
    "benchmark_sos_formulations",
]
