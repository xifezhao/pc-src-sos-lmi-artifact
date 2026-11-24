# ==============================================================================
# src/pcsrc/validation/validate_certificates.py
#
# Certificate artifact validation utilities.
#
# Purpose:
#   - Validate generated safety / feasibility / robustness certificate artifacts.
#   - Check existence, file size, extension, schema fields, numeric finiteness,
#     constraint residuals, and tolerance margins.
#   - Support JSON, JSONL, CSV, TXT, and optionally NPZ/NPY certificate files.
#   - Provide CLI entry point for CI / reproducibility workflows.
#
# Design notes:
#   - Pure Python standard library at import time.
#   - NumPy is optional and imported lazily only for .npz/.npy files.
#   - The validator is schema-tolerant: it recognizes several common field names
#     for status, residuals, margins, violations, and metadata.
#
# Typical usage:
#
#   from pcsrc.validation.validate_certificates import validate_certificate_directory
#
#   report = validate_certificate_directory("results/certificates")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.validate_certificates \
#       --certificate-dir results/certificates \
#       --manifest results/certificates/manifest.json \
#       --json-output results/certificates/validation_report.json
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import csv
import json


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

    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())

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


def _normalize_extension(value: str) -> str:
    """Normalize file extension."""

    text = str(value).strip().lower()

    if not text:
        raise ValueError("extension must be non-empty")

    return text if text.startswith(".") else "." + text


def _parse_csv_text(text: str | None) -> tuple[str, ...]:
    """Parse comma-separated CLI values."""

    if not text:
        return ()

    return tuple(item.strip() for item in text.split(",") if item.strip())


def _is_finite_number(value: Any) -> bool:
    """Return whether value is a finite number or finite numeric text."""

    if value is None:
        return False

    if isinstance(value, bool):
        return False

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False

    return isfinite(parsed)


def _flatten_numbers(value: Any) -> tuple[float, ...]:
    """Extract finite and non-finite numeric-like values recursively.

    Non-numeric values are ignored. Numeric values are converted to float even if
    they are NaN/inf, so callers can detect non-finite entries.
    """

    numbers: list[float] = []

    if value is None or isinstance(value, bool):
        return ()

    if isinstance(value, int | float):
        numbers.append(float(value))
        return tuple(numbers)

    if isinstance(value, str):
        try:
            numbers.append(float(value.strip()))
        except ValueError:
            pass
        return tuple(numbers)

    if isinstance(value, Mapping):
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
        return tuple(numbers)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return tuple(numbers)

    if hasattr(value, "tolist") and callable(value.tolist):
        return _flatten_numbers(value.tolist())

    return ()


def _get_nested(mapping: Mapping[str, Any], key_path: str) -> Any:
    """Get nested mapping value via dotted key path."""

    current: Any = mapping

    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]

    return current


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return first present field among flat or dotted keys."""

    for key in keys:
        if "." in key:
            value = _get_nested(mapping, key)
            if value is not None:
                return value
        elif key in mapping:
            return mapping[key]

    return None


def _status_is_success(value: Any) -> bool:
    """Interpret certificate status field."""

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {
        "pass",
        "passed",
        "success",
        "successful",
        "valid",
        "verified",
        "certified",
        "feasible",
        "sat",
        "true",
        "1",
        "ok",
    }


# ------------------------------------------------------------------------------
# Enums and dataclasses
# ------------------------------------------------------------------------------

class CertificateValidationStatus(str, Enum):
    """Validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class CertificateCheckSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CertificateArtifactKind(str, Enum):
    """Supported certificate artifact kinds."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    NPZ = "npz"
    NPY = "npy"
    TEXT = "text"
    UNKNOWN = "unknown"


_CERTIFICATE_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".npz",
    ".npy",
    ".txt",
}


_STATUS_KEYS = (
    "status",
    "result",
    "verified",
    "certified",
    "feasible",
    "success",
    "certificate.status",
    "validation.status",
)

_MARGIN_KEYS = (
    "margin",
    "safety_margin",
    "min_margin",
    "minimum_margin",
    "certificate.margin",
    "certificate.min_margin",
    "summary.margin",
    "summary.min_margin",
)

_RESIDUAL_KEYS = (
    "residual",
    "max_residual",
    "constraint_residual",
    "max_constraint_residual",
    "violation",
    "max_violation",
    "constraint_violation",
    "max_constraint_violation",
    "certificate.residual",
    "certificate.max_residual",
    "summary.max_residual",
    "summary.max_violation",
)

_VIOLATION_COUNT_KEYS = (
    "num_violations",
    "violations",
    "n_violations",
    "constraint_violations",
    "summary.num_violations",
)


@dataclass(frozen=True, slots=True)
class CertificateValidationConfig:
    """Configuration for certificate validation."""

    name: str = "certificate_validation"

    certificate_dir: str | None = None
    recursive: bool = True

    allowed_extensions: tuple[str, ...] = (
        ".json",
        ".jsonl",
        ".csv",
        ".npz",
        ".npy",
        ".txt",
    )

    min_file_size_bytes: int = 1
    max_file_size_bytes: int = 100 * 1024 * 1024

    residual_tolerance: float = 1e-7
    margin_tolerance: float = 0.0

    require_nonempty: bool = True
    require_status_success: bool = True
    require_finite_numeric_values: bool = True
    require_schema_fields: tuple[str, ...] = ()

    expected_certificates: tuple[str, ...] = ()
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "certificate_dir", None if self.certificate_dir is None else str(self.certificate_dir))
        object.__setattr__(self, "recursive", bool(self.recursive))
        object.__setattr__(
            self,
            "allowed_extensions",
            tuple(_normalize_extension(item) for item in self.allowed_extensions),
        )
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
        object.__setattr__(
            self,
            "residual_tolerance",
            _as_nonnegative_float(self.residual_tolerance, "residual_tolerance"),
        )
        object.__setattr__(
            self,
            "margin_tolerance",
            _as_nonnegative_float(self.margin_tolerance, "margin_tolerance"),
        )
        object.__setattr__(self, "require_nonempty", bool(self.require_nonempty))
        object.__setattr__(self, "require_status_success", bool(self.require_status_success))
        object.__setattr__(self, "require_finite_numeric_values", bool(self.require_finite_numeric_values))
        object.__setattr__(self, "require_schema_fields", tuple(str(item) for item in self.require_schema_fields))
        object.__setattr__(self, "expected_certificates", tuple(str(item) for item in self.expected_certificates))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CertificateValidationConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "certificate_validation")),
            certificate_dir=data.get("certificate_dir", None),
            recursive=bool(data.get("recursive", True)),
            allowed_extensions=tuple(data.get("allowed_extensions", (
                ".json",
                ".jsonl",
                ".csv",
                ".npz",
                ".npy",
                ".txt",
            ))),
            min_file_size_bytes=int(data.get("min_file_size_bytes", 1)),
            max_file_size_bytes=int(data.get("max_file_size_bytes", 100 * 1024 * 1024)),
            residual_tolerance=float(data.get("residual_tolerance", 1e-7)),
            margin_tolerance=float(data.get("margin_tolerance", 0.0)),
            require_nonempty=bool(data.get("require_nonempty", True)),
            require_status_success=bool(data.get("require_status_success", True)),
            require_finite_numeric_values=bool(data.get("require_finite_numeric_values", True)),
            require_schema_fields=tuple(data.get("require_schema_fields", ())),
            expected_certificates=tuple(data.get("expected_certificates", ())),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "certificate_dir": self.certificate_dir,
            "recursive": self.recursive,
            "allowed_extensions": list(self.allowed_extensions),
            "min_file_size_bytes": self.min_file_size_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "residual_tolerance": self.residual_tolerance,
            "margin_tolerance": self.margin_tolerance,
            "require_nonempty": self.require_nonempty,
            "require_status_success": self.require_status_success,
            "require_finite_numeric_values": self.require_finite_numeric_values,
            "require_schema_fields": list(self.require_schema_fields),
            "expected_certificates": list(self.expected_certificates),
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CertificateCheck:
    """One validation check result."""

    name: str
    status: CertificateValidationStatus
    severity: CertificateCheckSeverity = CertificateCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, CertificateValidationStatus)
            else CertificateValidationStatus(self.status)
        )
        severity = (
            self.severity
            if isinstance(self.severity, CertificateCheckSeverity)
            else CertificateCheckSeverity(self.severity)
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

        return self.status is CertificateValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check is failing."""

        return self.status is CertificateValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check is warning."""

        return self.status is CertificateValidationStatus.WARNING

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
class CertificateArtifact:
    """Certificate artifact metadata."""

    path: str
    kind: CertificateArtifactKind
    size_bytes: int
    extension: str
    records: int | None = None
    fields: tuple[str, ...] = ()
    max_residual: float | None = None
    min_margin: float | None = None
    num_violations: int | None = None
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, CertificateArtifactKind) else CertificateArtifactKind(self.kind)

        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "extension", str(self.extension).lower())
        object.__setattr__(self, "records", None if self.records is None else _as_nonnegative_int(self.records, "records"))
        object.__setattr__(self, "fields", tuple(str(item) for item in self.fields))
        object.__setattr__(self, "max_residual", None if self.max_residual is None else float(self.max_residual))
        object.__setattr__(self, "min_margin", None if self.min_margin is None else float(self.min_margin))
        object.__setattr__(
            self,
            "num_violations",
            None if self.num_violations is None else _as_nonnegative_int(self.num_violations, "num_violations"),
        )
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    def to_dict(self) -> JSONDict:
        """Serialize artifact."""

        return {
            "path": self.path,
            "kind": self.kind.value,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "extension": self.extension,
            "records": self.records,
            "fields": list(self.fields),
            "max_residual": self.max_residual,
            "min_margin": self.min_margin,
            "num_violations": self.num_violations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CertificateValidationReport:
    """Complete certificate validation report."""

    name: str = "certificate_validation_report"
    status: CertificateValidationStatus = CertificateValidationStatus.PASS
    artifacts: tuple[CertificateArtifact, ...] = ()
    checks: tuple[CertificateCheck, ...] = ()
    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, CertificateValidationStatus)
            else CertificateValidationStatus(self.status)
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

        return self.status is CertificateValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is CertificateValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[CertificateCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[CertificateCheck, ...]:
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
            f"Certificate validation report: {self.status.value.upper()}",
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
# Artifact discovery and reading
# ------------------------------------------------------------------------------

def infer_certificate_kind(path: str | Path) -> CertificateArtifactKind:
    """Infer certificate artifact kind from extension."""

    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        return CertificateArtifactKind.JSON

    if suffix == ".jsonl":
        return CertificateArtifactKind.JSONL

    if suffix == ".csv":
        return CertificateArtifactKind.CSV

    if suffix == ".npz":
        return CertificateArtifactKind.NPZ

    if suffix == ".npy":
        return CertificateArtifactKind.NPY

    if suffix == ".txt":
        return CertificateArtifactKind.TEXT

    return CertificateArtifactKind.UNKNOWN


def discover_certificate_files(
    certificate_dir: str | Path,
    *,
    recursive: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Discover certificate artifact files under a directory."""

    root = Path(certificate_dir)

    if not root.exists():
        return ()

    if not root.is_dir():
        return (root,)

    allowed = None
    if allowed_extensions is not None:
        allowed = {_normalize_extension(item) for item in allowed_extensions}

    pattern = "**/*" if recursive else "*"
    paths: list[Path] = []

    for path in root.glob(pattern):
        if not path.is_file():
            continue

        if allowed is not None and path.suffix.lower() not in allowed:
            continue

        paths.append(path)

    return tuple(sorted(paths))


def _read_json_certificate(path: Path) -> tuple[Any, JSONDict]:
    """Read JSON certificate."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {"json_type": type(data).__name__}


def _read_jsonl_certificate(path: Path) -> tuple[list[Any], JSONDict]:
    """Read JSONL certificate records."""

    records: list[Any] = []
    errors: list[str] = []

    for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_idx}: {exc}")

    metadata: JSONDict = {
        "json_type": "list",
        "records": len(records),
    }

    if errors:
        metadata["jsonl_errors"] = errors[:20]
        metadata["num_jsonl_errors"] = len(errors)

    return records, metadata


def _read_csv_certificate(path: Path) -> tuple[list[JSONDict], JSONDict]:
    """Read CSV certificate rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [dict(row) for row in reader]

    return records, {
        "json_type": "list",
        "records": len(records),
        "fieldnames": list(reader.fieldnames or ()),
    }


def _read_numpy_certificate(path: Path) -> tuple[Any, JSONDict]:
    """Read NPZ/NPY certificate using NumPy lazily."""

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise ImportError(
            "NumPy is required to validate .npz/.npy certificate files. "
            "Install it with `pip install numpy`."
        ) from exc

    loaded = np.load(path, allow_pickle=True)

    if path.suffix.lower() == ".npz":
        payload = {key: loaded[key].tolist() for key in loaded.files}
        return payload, {
            "json_type": "dict",
            "npz_keys": list(loaded.files),
        }

    if hasattr(loaded, "tolist"):
        return loaded.tolist(), {
            "json_type": type(loaded).__name__,
            "shape": list(getattr(loaded, "shape", ())),
        }

    return loaded, {"json_type": type(loaded).__name__}


def _read_text_certificate(path: Path) -> tuple[JSONDict, JSONDict]:
    """Read a plain text certificate."""

    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    status = None
    if any("valid" in line.lower() or "verified" in line.lower() for line in lines):
        status = "valid"
    elif any("invalid" in line.lower() or "failed" in line.lower() for line in lines):
        status = "failed"

    payload = {
        "status": status,
        "text": text,
    }

    return payload, {
        "json_type": "dict",
        "nonempty_lines": len(lines),
    }


def read_certificate_payload(path: str | Path) -> tuple[Any, JSONDict]:
    """Read certificate payload from supported artifact file."""

    p = Path(path)
    kind = infer_certificate_kind(p)

    if kind is CertificateArtifactKind.JSON:
        return _read_json_certificate(p)

    if kind is CertificateArtifactKind.JSONL:
        return _read_jsonl_certificate(p)

    if kind is CertificateArtifactKind.CSV:
        return _read_csv_certificate(p)

    if kind in {CertificateArtifactKind.NPZ, CertificateArtifactKind.NPY}:
        return _read_numpy_certificate(p)

    if kind is CertificateArtifactKind.TEXT:
        return _read_text_certificate(p)

    raise ValueError(f"unsupported certificate extension: {p.suffix}")


def _records_from_payload(payload: Any) -> list[Any]:
    """Normalize certificate payload into record list."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, Mapping):
        for key in ("records", "certificates", "data", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        return [payload]

    return [payload]


def _field_names_from_payload(payload: Any) -> tuple[str, ...]:
    """Collect top-level field names from certificate payload."""

    fields: list[str] = []
    seen: set[str] = set()

    for record in _records_from_payload(payload):
        if isinstance(record, Mapping):
            for key in record.keys():
                text = str(key)
                if text not in seen:
                    seen.add(text)
                    fields.append(text)

    if isinstance(payload, Mapping):
        for key in payload.keys():
            text = str(key)
            if text not in seen:
                seen.add(text)
                fields.append(text)

    return tuple(fields)


def _extract_residual_values(payload: Any) -> tuple[float, ...]:
    """Extract residual / violation values from payload."""

    values: list[float] = []

    for record in _records_from_payload(payload):
        if not isinstance(record, Mapping):
            continue

        raw = _first_present(record, _RESIDUAL_KEYS)
        if raw is not None:
            values.extend(_flatten_numbers(raw))

    if isinstance(payload, Mapping):
        raw = _first_present(payload, _RESIDUAL_KEYS)
        if raw is not None:
            values.extend(_flatten_numbers(raw))

    return tuple(float(value) for value in values if isfinite(value))


def _extract_margin_values(payload: Any) -> tuple[float, ...]:
    """Extract margin values from payload."""

    values: list[float] = []

    for record in _records_from_payload(payload):
        if not isinstance(record, Mapping):
            continue

        raw = _first_present(record, _MARGIN_KEYS)
        if raw is not None:
            values.extend(_flatten_numbers(raw))

    if isinstance(payload, Mapping):
        raw = _first_present(payload, _MARGIN_KEYS)
        if raw is not None:
            values.extend(_flatten_numbers(raw))

    return tuple(float(value) for value in values if isfinite(value))


def _extract_violation_count(payload: Any) -> int | None:
    """Extract violation count if available."""

    candidates: list[int] = []

    for record in _records_from_payload(payload):
        if not isinstance(record, Mapping):
            continue

        raw = _first_present(record, _VIOLATION_COUNT_KEYS)
        if raw is None:
            continue

        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
            candidates.append(len(raw))
        elif _is_finite_number(raw):
            candidates.append(max(0, int(float(raw))))

    if isinstance(payload, Mapping):
        raw = _first_present(payload, _VIOLATION_COUNT_KEYS)
        if raw is not None:
            if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
                candidates.append(len(raw))
            elif _is_finite_number(raw):
                candidates.append(max(0, int(float(raw))))

    if not candidates:
        return None

    return max(candidates)


def _extract_status_values(payload: Any) -> tuple[Any, ...]:
    """Extract certificate status values."""

    values: list[Any] = []

    for record in _records_from_payload(payload):
        if not isinstance(record, Mapping):
            continue

        raw = _first_present(record, _STATUS_KEYS)
        if raw is not None:
            values.append(raw)

    if isinstance(payload, Mapping):
        raw = _first_present(payload, _STATUS_KEYS)
        if raw is not None:
            values.append(raw)

    return tuple(values)


def collect_certificate_artifact(path: str | Path) -> CertificateArtifact:
    """Collect metadata for one certificate artifact."""

    p = Path(path)
    kind = infer_certificate_kind(p)
    size = p.stat().st_size if p.exists() else 0
    records: int | None = None
    fields: tuple[str, ...] = ()
    max_residual: float | None = None
    min_margin: float | None = None
    num_violations: int | None = None
    metadata: JSONDict = {}

    if p.exists() and p.is_file() and size > 0:
        try:
            payload, read_metadata = read_certificate_payload(p)
            metadata.update(read_metadata)
            records = len(_records_from_payload(payload))
            fields = _field_names_from_payload(payload)

            residuals = _extract_residual_values(payload)
            margins = _extract_margin_values(payload)

            if residuals:
                max_residual = max(abs(value) for value in residuals)

            if margins:
                min_margin = min(margins)

            num_violations = _extract_violation_count(payload)

            numeric_values = _flatten_numbers(payload)
            metadata["num_numeric_values"] = len(numeric_values)
            metadata["num_nonfinite_numeric_values"] = sum(1 for value in numeric_values if not isfinite(value))
            metadata["status_values"] = list(_extract_status_values(payload))

        except Exception as exc:
            metadata = {
                "read_error": str(exc),
                "read_error_type": type(exc).__name__,
            }

    return CertificateArtifact(
        path=str(p),
        kind=kind,
        size_bytes=size,
        extension=p.suffix.lower(),
        records=records,
        fields=fields,
        max_residual=max_residual,
        min_margin=min_margin,
        num_violations=num_violations,
        metadata=metadata,
    )


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_certificate_artifact(
    path: str | Path,
    config: CertificateValidationConfig | Mapping[str, Any] | None = None,
) -> tuple[CertificateArtifact | None, tuple[CertificateCheck, ...]]:
    """Validate one certificate artifact."""

    cfg = (
        CertificateValidationConfig()
        if config is None
        else config
        if isinstance(config, CertificateValidationConfig)
        else CertificateValidationConfig.from_dict(config)
    )

    p = Path(path)
    checks: list[CertificateCheck] = []

    if not p.exists():
        checks.append(
            CertificateCheck(
                name="exists",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message="File does not exist",
                path=str(p),
            )
        )
        return None, tuple(checks)

    if not p.is_file():
        checks.append(
            CertificateCheck(
                name="is_file",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message="Path is not a file",
                path=str(p),
            )
        )
        return None, tuple(checks)

    artifact = collect_certificate_artifact(p)

    if artifact.extension not in cfg.allowed_extensions:
        checks.append(
            CertificateCheck(
                name="allowed_extension",
                status=CertificateValidationStatus.WARNING,
                severity=CertificateCheckSeverity.WARNING,
                message=f"Extension {artifact.extension!r} is not in allowed extensions",
                path=str(p),
                details={"allowed_extensions": list(cfg.allowed_extensions)},
            )
        )
    else:
        checks.append(
            CertificateCheck(
                name="allowed_extension",
                status=CertificateValidationStatus.PASS,
                message="Extension is allowed",
                path=str(p),
            )
        )

    if cfg.require_nonempty and artifact.size_bytes <= 0:
        checks.append(
            CertificateCheck(
                name="nonempty",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message="File is empty",
                path=str(p),
            )
        )
    elif artifact.size_bytes < cfg.min_file_size_bytes:
        checks.append(
            CertificateCheck(
                name="min_file_size",
                status=CertificateValidationStatus.WARNING,
                severity=CertificateCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} is below minimum "
                    f"{_human_size(cfg.min_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    elif artifact.size_bytes > cfg.max_file_size_bytes:
        checks.append(
            CertificateCheck(
                name="max_file_size",
                status=CertificateValidationStatus.WARNING,
                severity=CertificateCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} exceeds maximum "
                    f"{_human_size(cfg.max_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    else:
        checks.append(
            CertificateCheck(
                name="file_size",
                status=CertificateValidationStatus.PASS,
                message=f"File size is {_human_size(artifact.size_bytes)}",
                path=str(p),
            )
        )

    if "read_error" in artifact.metadata:
        checks.append(
            CertificateCheck(
                name="readable",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message=f"Unable to read certificate: {artifact.metadata['read_error']}",
                path=str(p),
                details=artifact.metadata,
            )
        )
        return artifact, tuple(checks)

    checks.append(
        CertificateCheck(
            name="readable",
            status=CertificateValidationStatus.PASS,
            message="Certificate metadata parsed",
            path=str(p),
        )
    )

    if "num_jsonl_errors" in artifact.metadata:
        checks.append(
            CertificateCheck(
                name="jsonl_parse_errors",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message=f"JSONL contains {artifact.metadata['num_jsonl_errors']} parse error(s)",
                path=str(p),
                details=artifact.metadata,
            )
        )

    if cfg.require_schema_fields:
        available = set(artifact.fields)
        missing = [field for field in cfg.require_schema_fields if field not in available]

        checks.append(
            CertificateCheck(
                name="required_schema_fields",
                status=CertificateValidationStatus.PASS if not missing else CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.INFO if not missing else CertificateCheckSeverity.ERROR,
                message="All required schema fields exist" if not missing else f"Missing schema fields: {missing}",
                path=str(p),
                details={
                    "required_schema_fields": list(cfg.require_schema_fields),
                    "available_fields": list(artifact.fields),
                    "missing_fields": missing,
                },
            )
        )

    if cfg.require_status_success:
        status_values = tuple(artifact.metadata.get("status_values", ()))

        if not status_values:
            checks.append(
                CertificateCheck(
                    name="status_success",
                    status=CertificateValidationStatus.WARNING,
                    severity=CertificateCheckSeverity.WARNING,
                    message="No explicit certificate status field found",
                    path=str(p),
                )
            )
        else:
            bad_statuses = [value for value in status_values if not _status_is_success(value)]
            checks.append(
                CertificateCheck(
                    name="status_success",
                    status=CertificateValidationStatus.PASS if not bad_statuses else CertificateValidationStatus.FAIL,
                    severity=CertificateCheckSeverity.INFO if not bad_statuses else CertificateCheckSeverity.ERROR,
                    message=(
                        "Certificate status indicates success"
                        if not bad_statuses
                        else f"Certificate status contains non-success values: {bad_statuses}"
                    ),
                    path=str(p),
                    details={
                        "status_values": list(status_values),
                        "bad_statuses": bad_statuses,
                    },
                )
            )

    if cfg.require_finite_numeric_values:
        nonfinite_count = int(artifact.metadata.get("num_nonfinite_numeric_values", 0))
        checks.append(
            CertificateCheck(
                name="finite_numeric_values",
                status=CertificateValidationStatus.PASS if nonfinite_count == 0 else CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.INFO if nonfinite_count == 0 else CertificateCheckSeverity.ERROR,
                message=(
                    "All numeric values are finite"
                    if nonfinite_count == 0
                    else f"Found {nonfinite_count} non-finite numeric value(s)"
                ),
                path=str(p),
                details={
                    "num_numeric_values": artifact.metadata.get("num_numeric_values", 0),
                    "num_nonfinite_numeric_values": nonfinite_count,
                },
            )
        )

    if artifact.max_residual is not None:
        checks.append(
            CertificateCheck(
                name="residual_tolerance",
                status=(
                    CertificateValidationStatus.PASS
                    if artifact.max_residual <= cfg.residual_tolerance
                    else CertificateValidationStatus.FAIL
                ),
                severity=(
                    CertificateCheckSeverity.INFO
                    if artifact.max_residual <= cfg.residual_tolerance
                    else CertificateCheckSeverity.ERROR
                ),
                message=(
                    f"Max residual {artifact.max_residual:.6g} is within tolerance {cfg.residual_tolerance:.6g}"
                    if artifact.max_residual <= cfg.residual_tolerance
                    else f"Max residual {artifact.max_residual:.6g} exceeds tolerance {cfg.residual_tolerance:.6g}"
                ),
                path=str(p),
                details={
                    "max_residual": artifact.max_residual,
                    "residual_tolerance": cfg.residual_tolerance,
                },
            )
        )

    if artifact.min_margin is not None:
        checks.append(
            CertificateCheck(
                name="margin_tolerance",
                status=(
                    CertificateValidationStatus.PASS
                    if artifact.min_margin >= -cfg.margin_tolerance
                    else CertificateValidationStatus.FAIL
                ),
                severity=(
                    CertificateCheckSeverity.INFO
                    if artifact.min_margin >= -cfg.margin_tolerance
                    else CertificateCheckSeverity.ERROR
                ),
                message=(
                    f"Min margin {artifact.min_margin:.6g} is acceptable"
                    if artifact.min_margin >= -cfg.margin_tolerance
                    else f"Min margin {artifact.min_margin:.6g} is below tolerance {-cfg.margin_tolerance:.6g}"
                ),
                path=str(p),
                details={
                    "min_margin": artifact.min_margin,
                    "margin_tolerance": cfg.margin_tolerance,
                },
            )
        )

    if artifact.num_violations is not None:
        checks.append(
            CertificateCheck(
                name="violation_count",
                status=CertificateValidationStatus.PASS if artifact.num_violations == 0 else CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.INFO if artifact.num_violations == 0 else CertificateCheckSeverity.ERROR,
                message=(
                    "No constraint violations reported"
                    if artifact.num_violations == 0
                    else f"Certificate reports {artifact.num_violations} violation(s)"
                ),
                path=str(p),
                details={"num_violations": artifact.num_violations},
            )
        )

    return artifact, tuple(checks)


def _validate_expected_certificates(
    certificate_dir: Path,
    expected_certificates: Sequence[str],
) -> tuple[CertificateCheck, ...]:
    """Validate expected certificate files."""

    checks: list[CertificateCheck] = []

    for item in expected_certificates:
        path = Path(item)
        if not path.is_absolute():
            path = certificate_dir / path

        checks.append(
            CertificateCheck(
                name="expected_certificate",
                status=CertificateValidationStatus.PASS if path.exists() else CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.INFO if path.exists() else CertificateCheckSeverity.ERROR,
                message="Expected certificate exists" if path.exists() else "Expected certificate is missing",
                path=str(path),
                details={"expected": item},
            )
        )

    return tuple(checks)


def _compute_report_status(
    checks: Sequence[CertificateCheck],
    *,
    fail_on_warning: bool = False,
) -> CertificateValidationStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return CertificateValidationStatus.FAIL

    if any(check.warning for check in checks):
        return CertificateValidationStatus.FAIL if fail_on_warning else CertificateValidationStatus.WARNING

    return CertificateValidationStatus.PASS


def validate_certificate_directory(
    certificate_dir: str | Path,
    config: CertificateValidationConfig | Mapping[str, Any] | None = None,
) -> CertificateValidationReport:
    """Validate all certificate artifacts in a directory."""

    cfg = (
        CertificateValidationConfig(certificate_dir=str(certificate_dir))
        if config is None
        else config
        if isinstance(config, CertificateValidationConfig)
        else CertificateValidationConfig.from_dict(config)
    )

    root = Path(certificate_dir)
    checks: list[CertificateCheck] = []
    artifacts: list[CertificateArtifact] = []

    if not root.exists():
        checks.append(
            CertificateCheck(
                name="certificate_dir_exists",
                status=CertificateValidationStatus.FAIL,
                severity=CertificateCheckSeverity.ERROR,
                message="Certificate directory does not exist",
                path=str(root),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)
        return CertificateValidationReport(
            status=status,
            artifacts=(),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_artifacts": 0, "num_checks": len(checks)},
        )

    paths = discover_certificate_files(
        root,
        recursive=cfg.recursive,
        allowed_extensions=cfg.allowed_extensions,
    )

    checks.append(
        CertificateCheck(
            name="certificate_discovery",
            status=CertificateValidationStatus.PASS if paths else CertificateValidationStatus.WARNING,
            severity=CertificateCheckSeverity.INFO if paths else CertificateCheckSeverity.WARNING,
            message=f"Discovered {len(paths)} certificate artifact(s)",
            path=str(root),
        )
    )

    for path in paths:
        artifact, artifact_checks = validate_certificate_artifact(path, cfg)
        if artifact is not None:
            artifacts.append(artifact)
        checks.extend(artifact_checks)

    checks.extend(_validate_expected_certificates(root, cfg.expected_certificates))

    total_size = sum(artifact.size_bytes for artifact in artifacts)
    kind_counts: dict[str, int] = {}
    residuals = [artifact.max_residual for artifact in artifacts if artifact.max_residual is not None]
    margins = [artifact.min_margin for artifact in artifacts if artifact.min_margin is not None]
    violation_counts = [artifact.num_violations for artifact in artifacts if artifact.num_violations is not None]

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
        "max_residual": max(residuals) if residuals else None,
        "min_margin": min(margins) if margins else None,
        "total_violations": sum(violation_counts) if violation_counts else None,
    }

    return CertificateValidationReport(
        status=status,
        artifacts=tuple(artifacts),
        checks=tuple(checks),
        config=cfg.to_dict(),
        metrics=metrics,
    )


def validate_certificates_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    certificate_dir: str | Path | None = None,
) -> CertificateValidationReport:
    """Validate certificates using a manifest mapping or JSON file."""

    data = _load_json(manifest) if isinstance(manifest, str | Path) else dict(manifest)

    if certificate_dir is not None:
        data["certificate_dir"] = str(certificate_dir)

    root = data.get("certificate_dir", None)
    if root is None:
        raise ValueError("manifest must define certificate_dir or certificate_dir must be provided")

    cfg = CertificateValidationConfig.from_dict(data)

    return validate_certificate_directory(root, cfg)


def validate_default_project_certificates(
    certificate_dir: str | Path = "results/certificates",
) -> CertificateValidationReport:
    """Validate the default expected project certificate set."""

    expected = (
        "safety_certificate.json",
        "feasibility_certificate.json",
        "robustness_certificate.json",
    )

    cfg = CertificateValidationConfig(
        certificate_dir=str(certificate_dir),
        expected_certificates=expected,
        recursive=True,
        require_nonempty=True,
        require_status_success=True,
        require_finite_numeric_values=True,
        residual_tolerance=1e-7,
        margin_tolerance=0.0,
    )

    return validate_certificate_directory(certificate_dir, cfg)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.validate_certificates",
        description="Validate generated certificate artifacts.",
    )

    parser.add_argument("--certificate-dir", default="results/certificates")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument(
        "--allowed-extensions",
        default=".json,.jsonl,.csv,.npz,.npy,.txt",
    )
    parser.add_argument("--expected-certificates", default=None)

    parser.add_argument("--min-file-size-bytes", type=int, default=1)
    parser.add_argument("--max-file-size-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--margin-tolerance", type=float, default=0.0)

    parser.add_argument("--no-require-status-success", action="store_true")
    parser.add_argument("--no-require-finite-numeric-values", action="store_true")
    parser.add_argument("--required-schema-fields", default=None)

    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--default-project", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.default_project:
        report = validate_default_project_certificates(args.certificate_dir)

    elif args.manifest:
        report = validate_certificates_from_manifest(args.manifest, certificate_dir=args.certificate_dir)

    else:
        cfg = CertificateValidationConfig(
            certificate_dir=args.certificate_dir,
            recursive=not args.non_recursive,
            allowed_extensions=_parse_csv_text(args.allowed_extensions),
            expected_certificates=_parse_csv_text(args.expected_certificates),
            min_file_size_bytes=args.min_file_size_bytes,
            max_file_size_bytes=args.max_file_size_bytes,
            residual_tolerance=args.residual_tolerance,
            margin_tolerance=args.margin_tolerance,
            require_status_success=not args.no_require_status_success,
            require_finite_numeric_values=not args.no_require_finite_numeric_values,
            require_schema_fields=_parse_csv_text(args.required_schema_fields),
            fail_on_warning=args.fail_on_warning,
        )
        report = validate_certificate_directory(args.certificate_dir, cfg)

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
    "CertificateArtifact",
    "CertificateArtifactKind",
    "CertificateCheck",
    "CertificateCheckSeverity",
    "CertificateValidationConfig",
    "CertificateValidationReport",
    "CertificateValidationStatus",
    "JSONDict",
    "Number",
    "build_arg_parser",
    "collect_certificate_artifact",
    "discover_certificate_files",
    "infer_certificate_kind",
    "main",
    "read_certificate_payload",
    "validate_certificate_artifact",
    "validate_certificate_directory",
    "validate_certificates_from_manifest",
    "validate_default_project_certificates",
]


if __name__ == "__main__":
    raise SystemExit(main())
