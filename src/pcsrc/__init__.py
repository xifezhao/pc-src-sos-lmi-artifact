# ==============================================================================
# src/pcsrc/__init__.py
#
# Package initializer for pcsrc.
#
# Purpose:
#   - Expose package-level metadata.
#   - Provide lightweight access to common paths and version information.
#   - Avoid importing heavy solver, plotting, or simulation dependencies at import
#     time.
#
# Design notes:
#   - This module must remain side-effect light.
#   - Do not import cvxpy, mosek, matplotlib, scipy, or pandas here.
#   - Subpackages are imported lazily via __getattr__.
# ==============================================================================

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final


_PACKAGE_NAME: Final[str] = "pcsrc"


try:
    __version__: str = version(_PACKAGE_NAME)
except PackageNotFoundError:
    # Editable/local source-tree usage before installation.
    __version__ = "0.0.0+local"


__author__: str = "PC-SRC Authors"
__license__: str = "MIT"
__description__: str = (
    "Certified predictor-corrector string-robust control experiments, "
    "validation utilities, and figure-generation tools."
)


# Public subpackages intentionally exposed at package level.
_LAZY_SUBMODULES: Final[dict[str, str]] = {
    "certification": "pcsrc.certification",
    "controllers": "pcsrc.controllers",
    "data": "pcsrc.data",
    "experiments": "pcsrc.experiments",
    "figures": "pcsrc.figures",
    "models": "pcsrc.models",
    "simulation": "pcsrc.simulation",
    "utils": "pcsrc.utils",
    "validation": "pcsrc.validation",
}


def package_root() -> Path:
    """Return the installed package root directory.

    Returns
    -------
    pathlib.Path
        Path to the ``pcsrc`` Python package directory, typically
        ``<repo>/src/pcsrc`` in editable installs.
    """

    return Path(__file__).resolve().parent


def source_root() -> Path:
    """Return the source root containing the ``pcsrc`` package.

    Returns
    -------
    pathlib.Path
        Path to the source directory, typically ``<repo>/src``.
    """

    return package_root().parent


def repository_root() -> Path:
    """Best-effort discovery of the repository root.

    The function walks upward from the package directory and returns the first
    directory containing one of the project sentinel files/directories:
    ``pyproject.toml``, ``.git``, or ``configs``.

    Returns
    -------
    pathlib.Path
        Best-effort repository root. If no sentinel is found, the source root is
        returned.
    """

    current = package_root()

    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").exists()
            or (candidate / ".git").exists()
            or (candidate / "configs").exists()
        ):
            return candidate

    return source_root()


def config_root() -> Path:
    """Return the repository-level configuration directory."""

    return repository_root() / "configs"


def data_root() -> Path:
    """Return the repository-level data directory."""

    return repository_root() / "data"


def results_root() -> Path:
    """Return the repository-level results directory."""

    return repository_root() / "results"


def certificates_root() -> Path:
    """Return the repository-level certificates directory."""

    return repository_root() / "certificates"


def get_version() -> str:
    """Return the installed package version string."""

    return __version__


def __getattr__(name: str):
    """Lazily import selected public subpackages.

    Parameters
    ----------
    name:
        Attribute name requested from the ``pcsrc`` package.

    Returns
    -------
    module
        Imported submodule corresponding to ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not a declared lazy submodule.
    """

    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module {_PACKAGE_NAME!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return a sorted list of public package attributes."""

    return sorted(
        [
            "__author__",
            "__description__",
            "__license__",
            "__version__",
            "certificates_root",
            "config_root",
            "data_root",
            "get_version",
            "package_root",
            "repository_root",
            "results_root",
            "source_root",
            *_LAZY_SUBMODULES.keys(),
        ]
    )


__all__ = [
    "__author__",
    "__description__",
    "__license__",
    "__version__",
    "certificates_root",
    "config_root",
    "data_root",
    "get_version",
    "package_root",
    "repository_root",
    "results_root",
    "source_root",
    *_LAZY_SUBMODULES.keys(),
]
