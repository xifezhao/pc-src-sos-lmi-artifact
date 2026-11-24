# ==============================================================================
# src/pcsrc/validation/validate_figures.py
#
# Figure artifact validation utilities.
#
# Purpose:
#   - Validate generated figure files and related artifacts.
#   - Check file existence, size, extension, image readability, and optional
#     metadata sidecars.
#   - Validate expected figures from a manifest or default project conventions.
#   - Provide CLI entry point for CI / reproducibility workflows.
#
# Design notes:
#   - Uses only Python standard library at import time.
#   - Pillow is optional; if installed, image dimensions are checked more deeply.
#   - Matplotlib is not required.
#   - Designed to be conservative: missing optional sidecars can be warnings
#     rather than hard failures.
#
# Typical usage:
#
#   from pcsrc.validation.validate_figures import validate_figure_directory
#
#   report = validate_figure_directory("results/figures")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.validate_figures \
#       --figure-dir results/figures \
#       --manifest results/figures/manifest.json \
#       --json-output results/figures/validation_report.json
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import json
import os


Number = int | float
JSONDict = dict[str, Any]


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

def _validate_name(name: str, field_name: str = "name") -> str:
    """Validate a non-empty symbolic name."""

    parsed = str(name).strip()

    if not parsed:
        raise ValueError(f"{field_name} must be non-empty")

    return parsed


def _as_nonnegative_int(value: Any, name: str = "value") -> int:
    """Convert value to non-negative integer."""

    converted = int(value)

    if converted < 0:
        raise ValueError(f"{name} must be non-negative, got {converted}")

    return converted


def _as_positive_int(value: Any, name: str = "value") -> int:
    """Convert value to positive integer."""

    converted = int(value)

    if converted <= 0:
        raise ValueError(f"{name} must be positive, got {converted}")

    return converted


def _as_nonnegative_float(value: Any, name: str = "value") -> float:
    """Convert value to non-negative finite float."""

    converted = float(value)

    if not isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")

    return converted


def _json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, bool | int | float | str):
        if isinstance(value, float) and not isfinite(value):
            return None
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]

    return str(value)


def _load_json(path: str | Path) -> JSONDict:
    """Load JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> Path:
    """Write JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )

    return output_path


def _human_size(num_bytes: int) -> str:
    """Return human-readable file size."""

    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB")

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{num_bytes} B"


# ------------------------------------------------------------------------------
# Enums and dataclasses
# ------------------------------------------------------------------------------

class FigureValidationStatus(str, Enum):
    """Validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class FigureArtifactKind(str, Enum):
    """Supported figure artifact kind."""

    IMAGE = "image"
    PDF = "pdf"
    SVG = "svg"
    TEX = "tex"
    JSON = "json"
    DATA = "data"
    UNKNOWN = "unknown"


class FigureCheckSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_VECTOR_EXTENSIONS = {".pdf", ".svg"}
_TEX_EXTENSIONS = {".tex"}
_JSON_EXTENSIONS = {".json"}
_DATA_EXTENSIONS = {".csv", ".tsv", ".npz", ".npy", ".pkl", ".pickle"}


@dataclass(frozen=True, slots=True)
class FigureValidationConfig:
    """Configuration for figure validation."""

    name: str = "figure_validation"

    figure_dir: str | None = None
    recursive: bool = True

    allowed_extensions: tuple[str, ...] = (
        ".png",
        ".pdf",
        ".svg",
        ".jpg",
        ".jpeg",
        ".tex",
        ".json",
    )

    min_file_size_bytes: int = 128
    max_file_size_bytes: int = 100 * 1024 * 1024

    min_image_width: int = 300
    min_image_height: int = 200

    require_nonempty: bool = True
    require_sidecar_json: bool = False
    require_source_tex: bool = False

    expected_figures: tuple[str, ...] = ()
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = tuple(
            item.lower() if str(item).startswith(".") else "." + str(item).lower()
            for item in self.allowed_extensions
        )

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "figure_dir", None if self.figure_dir is None else str(self.figure_dir))
        object.__setattr__(self, "recursive", bool(self.recursive))
        object.__setattr__(self, "allowed_extensions", allowed)
        object.__setattr__(
            self,
            "min_file_size_bytes",
            _as_nonnegative_int(self.min_file_size_bytes, "min_file_size_bytes"),
        )
        object.__setattr__(
            self,
            "max_file_size_bytes",
            _as_positive_int(self.max_file_size_bytes, "max_file_size_bytes"),
        )
        object.__setattr__(self, "min_image_width", _as_positive_int(self.min_image_width, "min_image_width"))
        object.__setattr__(self, "min_image_height", _as_positive_int(self.min_image_height, "min_image_height"))
        object.__setattr__(self, "require_nonempty", bool(self.require_nonempty))
        object.__setattr__(self, "require_sidecar_json", bool(self.require_sidecar_json))
        object.__setattr__(self, "require_source_tex", bool(self.require_source_tex))
        object.__setattr__(self, "expected_figures", tuple(str(item) for item in self.expected_figures))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FigureValidationConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "figure_validation")),
            figure_dir=data.get("figure_dir", None),
            recursive=bool(data.get("recursive", True)),
            allowed_extensions=tuple(data.get("allowed_extensions", (
                ".png",
                ".pdf",
                ".svg",
                ".jpg",
                ".jpeg",
                ".tex",
                ".json",
            ))),
            min_file_size_bytes=int(data.get("min_file_size_bytes", 128)),
            max_file_size_bytes=int(data.get("max_file_size_bytes", 100 * 1024 * 1024)),
            min_image_width=int(data.get("min_image_width", 300)),
            min_image_height=int(data.get("min_image_height", 200)),
            require_nonempty=bool(data.get("require_nonempty", True)),
            require_sidecar_json=bool(data.get("require_sidecar_json", False)),
            require_source_tex=bool(data.get("require_source_tex", False)),
            expected_figures=tuple(data.get("expected_figures", ())),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "figure_dir": self.figure_dir,
            "recursive": self.recursive,
            "allowed_extensions": list(self.allowed_extensions),
            "min_file_size_bytes": self.min_file_size_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "min_image_width": self.min_image_width,
            "min_image_height": self.min_image_height,
            "require_nonempty": self.require_nonempty,
            "require_sidecar_json": self.require_sidecar_json,
            "require_source_tex": self.require_source_tex,
            "expected_figures": list(self.expected_figures),
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FigureCheck:
    """One validation check result."""

    name: str
    status: FigureValidationStatus
    severity: FigureCheckSeverity = FigureCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, FigureValidationStatus)
            else FigureValidationStatus(self.status)
        )
        severity = (
            self.severity
            if isinstance(self.severity, FigureCheckSeverity)
            else FigureCheckSeverity(self.severity)
        )

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(self, "details", dict(_json_safe(self.details)))

    @property
    def passed(self) -> bool:
        """Whether the check is passing."""

        return self.status is FigureValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check is failing."""

        return self.status is FigureValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check is warning."""

        return self.status is FigureValidationStatus.WARNING

    def to_dict(self) -> JSONDict:
        """Serialize check."""

        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """Figure artifact metadata."""

    path: str
    kind: FigureArtifactKind
    size_bytes: int
    extension: str
    width: int | None = None
    height: int | None = None
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, FigureArtifactKind) else FigureArtifactKind(self.kind)

        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "extension", str(self.extension).lower())
        object.__setattr__(self, "width", None if self.width is None else _as_positive_int(self.width, "width"))
        object.__setattr__(self, "height", None if self.height is None else _as_positive_int(self.height, "height"))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    def to_dict(self) -> JSONDict:
        """Serialize artifact."""

        return {
            "path": self.path,
            "kind": self.kind.value,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "extension": self.extension,
            "width": self.width,
            "height": self.height,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FigureValidationReport:
    """Complete figure validation report."""

    name: str = "figure_validation_report"
    status: FigureValidationStatus = FigureValidationStatus.PASS
    artifacts: tuple[FigureArtifact, ...] = ()
    checks: tuple[FigureCheck, ...] = ()
    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, FigureValidationStatus)
            else FigureValidationStatus(self.status)
        )

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "config", dict(_json_safe(self.config)))
        object.__setattr__(self, "metrics", dict(_json_safe(self.metrics)))

    @property
    def passed(self) -> bool:
        """Whether report passed."""

        return self.status is FigureValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is FigureValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[FigureCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[FigureCheck, ...]:
        """Return failing checks."""

        return tuple(check for check in self.checks if check.failed)

    def to_dict(self) -> JSONDict:
        """Serialize report."""

        return {
            "name": self.name,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self) -> str:
        """Render report as human-readable text."""

        lines = [
            f"Figure validation report: {self.status.value.upper()}",
            f"Artifacts: {len(self.artifacts)}",
            f"Checks: {len(self.checks)}",
            f"Failures: {len(self.failures)}",
            f"Warnings: {len(self.warnings)}",
            "",
        ]

        if self.failures:
            lines.append("Failures:")
            for check in self.failures:
                location = f" [{check.path}]" if check.path else ""
                lines.append(f"  - {check.name}{location}: {check.message}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for check in self.warnings:
                location = f" [{check.path}]" if check.path else ""
                lines.append(f"  - {check.name}{location}: {check.message}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------------
# Artifact discovery and metadata
# ------------------------------------------------------------------------------

def infer_artifact_kind(path: str | Path) -> FigureArtifactKind:
    """Infer artifact kind from file extension."""

    suffix = Path(path).suffix.lower()

    if suffix in _IMAGE_EXTENSIONS:
        return FigureArtifactKind.IMAGE

    if suffix == ".pdf":
        return FigureArtifactKind.PDF

    if suffix == ".svg":
        return FigureArtifactKind.SVG

    if suffix in _TEX_EXTENSIONS:
        return FigureArtifactKind.TEX

    if suffix in _JSON_EXTENSIONS:
        return FigureArtifactKind.JSON

    if suffix in _DATA_EXTENSIONS:
        return FigureArtifactKind.DATA

    return FigureArtifactKind.UNKNOWN


def _try_read_image_size(path: Path) -> tuple[int | None, int | None, str | None]:
    """Read image dimensions using Pillow if available."""

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None, None, "Pillow not installed; image dimension check skipped"

    try:
        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height), None
    except Exception as exc:
        return None, None, f"Unable to read image dimensions: {exc}"


def collect_artifact(path: str | Path) -> FigureArtifact:
    """Collect metadata for one artifact path."""

    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    kind = infer_artifact_kind(p)
    width: int | None = None
    height: int | None = None
    metadata: JSONDict = {}

    if kind is FigureArtifactKind.IMAGE and p.exists():
        width, height, error = _try_read_image_size(p)
        if error:
            metadata["image_size_warning"] = error

    return FigureArtifact(
        path=str(p),
        kind=kind,
        size_bytes=size,
        extension=p.suffix.lower(),
        width=width,
        height=height,
        metadata=metadata,
    )


def discover_figure_files(
    figure_dir: str | Path,
    *,
    recursive: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Discover figure artifact files under a directory."""

    root = Path(figure_dir)

    if not root.exists():
        return ()

    if not root.is_dir():
        return (root,)

    allowed = None
    if allowed_extensions is not None:
        allowed = {
            item.lower() if str(item).startswith(".") else "." + str(item).lower()
            for item in allowed_extensions
        }

    pattern = "**/*" if recursive else "*"
    paths: list[Path] = []

    for path in root.glob(pattern):
        if not path.is_file:
            continue

        if not path.is_file():
            continue

        if allowed is not None and path.suffix.lower() not in allowed:
            continue

        paths.append(path)

    return tuple(sorted(paths))


# ------------------------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------------------------

def validate_artifact(
    path: str | Path,
    config: FigureValidationConfig | Mapping[str, Any] | None = None,
) -> tuple[FigureArtifact | None, tuple[FigureCheck, ...]]:
    """Validate one figure artifact."""

    cfg = (
        FigureValidationConfig()
        if config is None
        else config
        if isinstance(config, FigureValidationConfig)
        else FigureValidationConfig.from_dict(config)
    )

    p = Path(path)
    checks: list[FigureCheck] = []

    if not p.exists():
        checks.append(
            FigureCheck(
                name="exists",
                status=FigureValidationStatus.FAIL,
                severity=FigureCheckSeverity.ERROR,
                message="File does not exist",
                path=str(p),
            )
        )
        return None, tuple(checks)

    if not p.is_file():
        checks.append(
            FigureCheck(
                name="is_file",
                status=FigureValidationStatus.FAIL,
                severity=FigureCheckSeverity.ERROR,
                message="Path is not a file",
                path=str(p),
            )
        )
        return None, tuple(checks)

    artifact = collect_artifact(p)

    if artifact.extension not in cfg.allowed_extensions:
        checks.append(
            FigureCheck(
                name="allowed_extension",
                status=FigureValidationStatus.WARNING,
                severity=FigureCheckSeverity.WARNING,
                message=f"Extension {artifact.extension!r} is not in allowed extensions",
                path=str(p),
                details={"allowed_extensions": list(cfg.allowed_extensions)},
            )
        )
    else:
        checks.append(
            FigureCheck(
                name="allowed_extension",
                status=FigureValidationStatus.PASS,
                message="Extension is allowed",
                path=str(p),
            )
        )

    if cfg.require_nonempty and artifact.size_bytes <= 0:
        checks.append(
            FigureCheck(
                name="nonempty",
                status=FigureValidationStatus.FAIL,
                severity=FigureCheckSeverity.ERROR,
                message="File is empty",
                path=str(p),
            )
        )
    elif artifact.size_bytes < cfg.min_file_size_bytes:
        checks.append(
            FigureCheck(
                name="min_file_size",
                status=FigureValidationStatus.WARNING,
                severity=FigureCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} is below minimum "
                    f"{_human_size(cfg.min_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    elif artifact.size_bytes > cfg.max_file_size_bytes:
        checks.append(
            FigureCheck(
                name="max_file_size",
                status=FigureValidationStatus.WARNING,
                severity=FigureCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} exceeds maximum "
                    f"{_human_size(cfg.max_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    else:
        checks.append(
            FigureCheck(
                name="file_size",
                status=FigureValidationStatus.PASS,
                message=f"File size is {_human_size(artifact.size_bytes)}",
                path=str(p),
            )
        )

    if artifact.kind is FigureArtifactKind.IMAGE:
        if artifact.width is None or artifact.height is None:
            checks.append(
                FigureCheck(
                    name="image_readable",
                    status=FigureValidationStatus.WARNING,
                    severity=FigureCheckSeverity.WARNING,
                    message=artifact.metadata.get(
                        "image_size_warning",
                        "Image dimensions unavailable",
                    ),
                    path=str(p),
                )
            )
        elif artifact.width < cfg.min_image_width or artifact.height < cfg.min_image_height:
            checks.append(
                FigureCheck(
                    name="image_dimensions",
                    status=FigureValidationStatus.WARNING,
                    severity=FigureCheckSeverity.WARNING,
                    message=(
                        f"Image dimensions {artifact.width}x{artifact.height} are below "
                        f"{cfg.min_image_width}x{cfg.min_image_height}"
                    ),
                    path=str(p),
                    details={
                        "width": artifact.width,
                        "height": artifact.height,
                        "min_width": cfg.min_image_width,
                        "min_height": cfg.min_image_height,
                    },
                )
            )
        else:
            checks.append(
                FigureCheck(
                    name="image_dimensions",
                    status=FigureValidationStatus.PASS,
                    message=f"Image dimensions are {artifact.width}x{artifact.height}",
                    path=str(p),
                )
            )

    if cfg.require_sidecar_json and artifact.kind in {
        FigureArtifactKind.IMAGE,
        FigureArtifactKind.PDF,
        FigureArtifactKind.SVG,
    }:
        sidecar = p.with_suffix(".json")
        checks.append(
            FigureCheck(
                name="sidecar_json",
                status=FigureValidationStatus.PASS if sidecar.exists() else FigureValidationStatus.WARNING,
                severity=FigureCheckSeverity.INFO if sidecar.exists() else FigureCheckSeverity.WARNING,
                message="Sidecar JSON exists" if sidecar.exists() else "Sidecar JSON is missing",
                path=str(sidecar),
            )
        )

    if cfg.require_source_tex and artifact.kind in {
        FigureArtifactKind.IMAGE,
        FigureArtifactKind.PDF,
        FigureArtifactKind.SVG,
    }:
        tex = p.with_suffix(".tex")
        checks.append(
            FigureCheck(
                name="source_tex",
                status=FigureValidationStatus.PASS if tex.exists() else FigureValidationStatus.WARNING,
                severity=FigureCheckSeverity.INFO if tex.exists() else FigureCheckSeverity.WARNING,
                message="Source TeX exists" if tex.exists() else "Source TeX is missing",
                path=str(tex),
            )
        )

    return artifact, tuple(checks)


def _validate_expected_figures(
    figure_dir: Path,
    expected_figures: Sequence[str],
) -> tuple[FigureCheck, ...]:
    """Validate expected figure files."""

    checks: list[FigureCheck] = []

    for item in expected_figures:
        path = Path(item)
        if not path.is_absolute():
            path = figure_dir / path

        checks.append(
            FigureCheck(
                name="expected_figure",
                status=FigureValidationStatus.PASS if path.exists() else FigureValidationStatus.FAIL,
                severity=FigureCheckSeverity.INFO if path.exists() else FigureCheckSeverity.ERROR,
                message="Expected figure exists" if path.exists() else "Expected figure is missing",
                path=str(path),
                details={"expected": item},
            )
        )

    return tuple(checks)


def _compute_report_status(
    checks: Sequence[FigureCheck],
    *,
    fail_on_warning: bool = False,
) -> FigureValidationStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return FigureValidationStatus.FAIL

    if any(check.warning for check in checks):
        return FigureValidationStatus.FAIL if fail_on_warning else FigureValidationStatus.WARNING

    return FigureValidationStatus.PASS


def validate_figure_directory(
    figure_dir: str | Path,
    config: FigureValidationConfig | Mapping[str, Any] | None = None,
) -> FigureValidationReport:
    """Validate all figure artifacts in a directory."""

    cfg = (
        FigureValidationConfig(figure_dir=str(figure_dir))
        if config is None
        else config
        if isinstance(config, FigureValidationConfig)
        else FigureValidationConfig.from_dict(config)
    )

    root = Path(figure_dir)
    checks: list[FigureCheck] = []
    artifacts: list[FigureArtifact] = []

    if not root.exists():
        checks.append(
            FigureCheck(
                name="figure_dir_exists",
                status=FigureValidationStatus.FAIL,
                severity=FigureCheckSeverity.ERROR,
                message="Figure directory does not exist",
                path=str(root),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)
        return FigureValidationReport(
            status=status,
            artifacts=(),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_artifacts": 0, "num_checks": len(checks)},
        )

    paths = discover_figure_files(
        root,
        recursive=cfg.recursive,
        allowed_extensions=cfg.allowed_extensions,
    )

    checks.append(
        FigureCheck(
            name="figure_discovery",
            status=FigureValidationStatus.PASS if paths else FigureValidationStatus.WARNING,
            severity=FigureCheckSeverity.INFO if paths else FigureCheckSeverity.WARNING,
            message=f"Discovered {len(paths)} artifact(s)",
            path=str(root),
        )
    )

    for path in paths:
        artifact, artifact_checks = validate_artifact(path, cfg)
        if artifact is not None:
            artifacts.append(artifact)
        checks.extend(artifact_checks)

    checks.extend(_validate_expected_figures(root, cfg.expected_figures))

    total_size = sum(artifact.size_bytes for artifact in artifacts)
    kind_counts: dict[str, int] = {}
    for artifact in artifacts:
        kind_counts[artifact.kind.value] = kind_counts.get(artifact.kind.value, 0) + 1

    status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

    metrics = {
        "num_artifacts": len(artifacts),
        "num_checks": len(checks),
        "num_failures": sum(1 for check in checks if check.failed),
        "num_warnings": sum(1 for check in checks if check.warning),
        "total_size_bytes": total_size,
        "total_size_human": _human_size(total_size),
        "kind_counts": kind_counts,
    }

    return FigureValidationReport(
        status=status,
        artifacts=tuple(artifacts),
        checks=tuple(checks),
        config=cfg.to_dict(),
        metrics=metrics,
    )


def validate_figures_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    figure_dir: str | Path | None = None,
) -> FigureValidationReport:
    """Validate figures using a manifest mapping or JSON file.

    Supported manifest fields:
    - figure_dir
    - expected_figures
    - allowed_extensions
    - min_file_size_bytes
    - max_file_size_bytes
    - min_image_width
    - min_image_height
    - require_sidecar_json
    - require_source_tex
    - fail_on_warning
    """

    data = _load_json(manifest) if isinstance(manifest, str | Path) else dict(manifest)

    if figure_dir is not None:
        data["figure_dir"] = str(figure_dir)

    root = data.get("figure_dir", None)
    if root is None:
        raise ValueError("manifest must define figure_dir or figure_dir must be provided")

    cfg = FigureValidationConfig.from_dict(data)

    return validate_figure_directory(root, cfg)


def validate_default_project_figures(
    figure_dir: str | Path = "results/figures",
) -> FigureValidationReport:
    """Validate the default expected project figure set."""

    expected = (
        "fig5_gain_feasible_domain.png",
        "fig6_structural_conservatism.png",
        "fig7_space_time_heatmap.png",
        "fig8_time_domain_profiles.png",
        "fig9_lut_deployment.png",
    )

    cfg = FigureValidationConfig(
        figure_dir=str(figure_dir),
        expected_figures=expected,
        recursive=True,
        require_nonempty=True,
        require_sidecar_json=False,
        require_source_tex=False,
    )

    return validate_figure_directory(figure_dir, cfg)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_csv(text: str | None) -> tuple[str, ...]:
    """Parse comma-separated string values."""

    if not text:
        return ()

    return tuple(item.strip() for item in text.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.validate_figures",
        description="Validate generated figure artifacts.",
    )

    parser.add_argument("--figure-dir", default="results/figures")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument(
        "--allowed-extensions",
        default=".png,.pdf,.svg,.jpg,.jpeg,.tex,.json",
    )
    parser.add_argument("--expected-figures", default=None)

    parser.add_argument("--min-file-size-bytes", type=int, default=128)
    parser.add_argument("--max-file-size-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--min-image-width", type=int, default=300)
    parser.add_argument("--min-image-height", type=int, default=200)

    parser.add_argument("--require-sidecar-json", action="store_true")
    parser.add_argument("--require-source-tex", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--default-project", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.default_project:
        report = validate_default_project_figures(args.figure_dir)

    elif args.manifest:
        report = validate_figures_from_manifest(args.manifest, figure_dir=args.figure_dir)

    else:
        cfg = FigureValidationConfig(
            figure_dir=args.figure_dir,
            recursive=not args.non_recursive,
            allowed_extensions=_parse_csv(args.allowed_extensions),
            expected_figures=_parse_csv(args.expected_figures),
            min_file_size_bytes=args.min_file_size_bytes,
            max_file_size_bytes=args.max_file_size_bytes,
            min_image_width=args.min_image_width,
            min_image_height=args.min_image_height,
            require_sidecar_json=args.require_sidecar_json,
            require_source_tex=args.require_source_tex,
            fail_on_warning=args.fail_on_warning,
        )
        report = validate_figure_directory(args.figure_dir, cfg)

    print(report.to_text())

    if args.json_output:
        _write_json(args.json_output, report.to_dict())

    if args.text_output:
        output_path = Path(args.text_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_text(), encoding="utf-8")

    return 0 if report.passed else 1


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------

__all__ = [
    "FigureArtifact",
    "FigureArtifactKind",
    "FigureCheck",
    "FigureCheckSeverity",
    "FigureValidationConfig",
    "FigureValidationReport",
    "FigureValidationStatus",
    "build_arg_parser",
    "collect_artifact",
    "discover_figure_files",
    "infer_artifact_kind",
    "main",
    "validate_artifact",
    "validate_default_project_figures",
    "validate_figure_directory",
    "validate_figures_from_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
