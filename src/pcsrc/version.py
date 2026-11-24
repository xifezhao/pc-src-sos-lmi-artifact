# ==============================================================================
# src/pcsrc/version.py
#
# Version and build metadata helpers for pcsrc.
#
# Purpose:
#   - Provide a single lightweight source of package version information.
#   - Expose optional build/release metadata used by reproducibility reports.
#   - Avoid importing heavy project dependencies.
#
# Design notes:
#   - This module is safe to import from CLI, figure scripts, validation tools,
#     and package initialization.
#   - Do not import cvxpy, numpy, pandas, scipy, matplotlib, or solver packages.
# ==============================================================================

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final


PACKAGE_NAME: Final[str] = "pcsrc"


try:
    __version__: str = version(PACKAGE_NAME)
except PackageNotFoundError:
    # Fallback for editable/source-tree execution before package installation.
    __version__ = "0.0.0+local"


__version_info__: tuple[int, int, int] | None = None
__release__: str = __version__
__build__: str = "local"
__commit__: str | None = None
__build_time_utc__: str | None = None


@dataclass(frozen=True)
class VersionInfo:
    """Structured package and runtime version metadata.

    Attributes
    ----------
    package:
        Python distribution/package name.
    version:
        Installed package version.
    release:
        Release identifier. Defaults to the package version.
    build:
        Build label such as ``local``, ``wheel``, ``sdist``, or CI build ID.
    commit:
        Git commit hash if available.
    build_time_utc:
        Build timestamp in UTC if available.
    python:
        Python interpreter version.
    platform:
        Platform string from :func:`platform.platform`.
    implementation:
        Python implementation, e.g. ``CPython``.
    executable:
        Path to the Python executable.
    """

    package: str
    version: str
    release: str
    build: str
    commit: str | None
    build_time_utc: str | None
    python: str
    platform: str
    implementation: str
    executable: str


def get_version() -> str:
    """Return the package version string."""

    return __version__


def get_release() -> str:
    """Return the release identifier."""

    return __release__


def get_build() -> str:
    """Return the build label."""

    return __build__


def get_commit() -> str | None:
    """Return the package build commit hash if available."""

    return __commit__


def get_build_time_utc() -> str | None:
    """Return the package build timestamp in UTC if available."""

    return __build_time_utc__


def parse_version_tuple(version_string: str | None = None) -> tuple[int, int, int] | None:
    """Parse a semantic version string into ``(major, minor, patch)``.

    The parser intentionally stays lightweight and only extracts the first three
    dot-separated numeric components. Local/build suffixes such as
    ``+local`` or pre-release tags such as ``1.2.3rc1`` are ignored when they
    cannot be interpreted safely.

    Parameters
    ----------
    version_string:
        Version string to parse. If omitted, the package ``__version__`` is used.

    Returns
    -------
    tuple[int, int, int] | None
        Parsed ``(major, minor, patch)`` tuple, or ``None`` if parsing fails.
    """

    raw = version_string or __version__
    core = raw.split("+", maxsplit=1)[0].split("-", maxsplit=1)[0]
    parts = core.split(".")

    if len(parts) < 3:
        return None

    parsed: list[int] = []
    for part in parts[:3]:
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break

        if digits == "":
            return None

        parsed.append(int(digits))

    return tuple(parsed)  # type: ignore[return-value]


def runtime_summary() -> dict[str, str]:
    """Return a compact runtime summary suitable for metadata YAML/JSON."""

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def version_info() -> VersionInfo:
    """Return structured package and runtime metadata."""

    return VersionInfo(
        package=PACKAGE_NAME,
        version=__version__,
        release=__release__,
        build=__build__,
        commit=__commit__,
        build_time_utc=__build_time_utc__,
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        implementation=platform.python_implementation(),
        executable=sys.executable,
    )


def version_info_dict() -> dict[str, Any]:
    """Return :func:`version_info` as a plain dictionary."""

    return asdict(version_info())


def git_commit_from_repo(start: str | Path | None = None) -> str | None:
    """Best-effort read of the current Git commit hash.

    This helper avoids invoking ``git`` as a subprocess. It reads ``.git/HEAD``
    and referenced files directly when the source tree is available.

    Parameters
    ----------
    start:
        Directory from which to search upward. If omitted, the directory
        containing this file is used.

    Returns
    -------
    str | None
        Git commit hash if discoverable, otherwise ``None``.
    """

    start_path = Path(start).resolve() if start is not None else Path(__file__).resolve()

    for candidate in (start_path, *start_path.parents):
        git_dir = candidate / ".git"

        if not git_dir.exists():
            continue

        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None

        head = head_file.read_text(encoding="utf-8").strip()

        if head.startswith("ref:"):
            ref_path = head.split(" ", maxsplit=1)[1]
            ref_file = git_dir / ref_path
            if ref_file.exists():
                return ref_file.read_text(encoding="utf-8").strip()

            packed_refs = git_dir / "packed-refs"
            if packed_refs.exists():
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    commit, ref = line.split(" ", maxsplit=1)
                    if ref == ref_path:
                        return commit

            return None

        if len(head) >= 7:
            return head

    return None


def full_version_text() -> str:
    """Return a human-readable one-line version summary."""

    info = version_info()
    commit = info.commit or git_commit_from_repo() or "unknown"
    return (
        f"{info.package} {info.version} "
        f"(release={info.release}, build={info.build}, commit={commit})"
    )


__version_info__ = parse_version_tuple(__version__)


__all__ = [
    "PACKAGE_NAME",
    "VersionInfo",
    "__build__",
    "__build_time_utc__",
    "__commit__",
    "__release__",
    "__version__",
    "__version_info__",
    "full_version_text",
    "get_build",
    "get_build_time_utc",
    "get_commit",
    "get_release",
    "get_version",
    "git_commit_from_repo",
    "parse_version_tuple",
    "runtime_summary",
    "version_info",
    "version_info_dict",
]
