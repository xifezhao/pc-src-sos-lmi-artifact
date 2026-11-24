# ==============================================================================
# src/pcsrc/validation/validate_lut.py
#
# LUT artifact validation utilities.
#
# Purpose:
#   - Validate generated look-up table artifacts for deployment and reproduction.
#   - Check existence, file size, extension, schema fields, grid dimensions,
#     numeric finiteness, monotonic axes, value bounds, and expected artifacts.
#   - Support JSON, JSONL, CSV, TSV, TXT, and optionally NPZ/NPY files.
#   - Provide CLI entry point for CI / reproducibility workflows.
#
# Design notes:
#   - Pure Python standard library at import time.
#   - NumPy is optional and imported lazily only for .npz/.npy files.
#   - The validator is intentionally schema-tolerant and recognizes common field
#     names such as grid, axes, values, table, lut, gain, action, policy, etc.
#
# Typical usage:
#
#   from pcsrc.validation.validate_lut import validate_lut_directory
#
#   report = validate_lut_directory("results/lut")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.validate_lut \
#       --lut-dir results/lut \
#       --manifest results/lut/manifest.json \
#       --json-output results/lut/validation_report.json
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence
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


def _as_finite_float(value: Any, name: str = "value") -> float:
    """Convert value to finite float."""

    converted = float(value)

    if not isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")

    return converted


def _as_nonnegative_float(value: Any, name: str = "value") -> float:
    """Convert value to non-negative finite float."""

    converted = _as_finite_float(value, name)

    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")

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


def _parse_float_pair(text: str | None) -> tuple[float, float] | None:
    """Parse lower,upper pair from CLI."""

    if not text:
        return None

    parts = [item.strip() for item in text.split(",") if item.strip()]

    if len(parts) != 2:
        raise ValueError("bounds must be formatted as lower,upper")

    lower = _as_finite_float(parts[0], "lower")
    upper = _as_finite_float(parts[1], "upper")

    if lower > upper:
        raise ValueError("lower bound must not exceed upper bound")

    return lower, upper


def _is_numeric_like(value: Any) -> bool:
    """Return whether value is numeric-like."""

    if value is None or isinstance(value, bool):
        return False

    try:
        float(value)
    except (TypeError, ValueError):
        return False

    return True


def _flatten_numbers(value: Any) -> tuple[float, ...]:
    """Extract numeric-like values recursively, preserving NaN/inf."""

    numbers: list[float] = []

    if value is None or isinstance(value, bool):
        return ()

    if isinstance(value, int | float):
        return (float(value),)

    if isinstance(value, str):
        try:
            return (float(value.strip()),)
        except ValueError:
            return ()

    if hasattr(value, "tolist") and callable(value.tolist):
        return _flatten_numbers(value.tolist())

    if isinstance(value, Mapping):
        for item in value.values():
            numbers.extend(_flatten_numbers(item))
        return tuple(numbers)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            numbers.extend(_flatten_numbers(item))
        return tuple(numbers)

    return ()


def _shape_of_nested_sequence(value: Any) -> tuple[int, ...]:
    """Infer rectangular nested sequence shape where possible."""

    if hasattr(value, "shape"):
        try:
            return tuple(int(item) for item in value.shape)
        except Exception:
            pass

    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()

    length = len(value)

    if length == 0:
        return (0,)

    child_shapes = [_shape_of_nested_sequence(item) for item in value]

    first = child_shapes[0]
    if all(shape == first for shape in child_shapes):
        return (length,) + first

    return (length,)


def _is_strictly_increasing(values: Sequence[Any]) -> bool:
    """Return whether sequence is strictly increasing."""

    numeric = [float(value) for value in values if _is_numeric_like(value)]

    if len(numeric) != len(values):
        return False

    return all(b > a for a, b in zip(numeric, numeric[1:]))


def _get_nested(mapping: Mapping[str, Any], key_path: str) -> Any:
    """Get nested value by dotted path."""

    current: Any = mapping

    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]

    return current


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return first present value among flat or dotted keys."""

    for key in keys:
        if "." in key:
            value = _get_nested(mapping, key)
            if value is not None:
                return value
        elif key in mapping:
            return mapping[key]

    return None


# ------------------------------------------------------------------------------
# Enums and constants
# ------------------------------------------------------------------------------

class LUTValidationStatus(str, Enum):
    """Validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class LUTCheckSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LUTArtifactKind(str, Enum):
    """Supported LUT artifact kinds."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    NPZ = "npz"
    NPY = "npy"
    TEXT = "text"
    UNKNOWN = "unknown"


_LUT_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".npz",
    ".npy",
    ".txt",
}

_AXIS_KEYS = (
    "axes",
    "grid",
    "grids",
    "axis",
    "state_grid",
    "input_grid",
    "metadata.axes",
    "metadata.grid",
)

_VALUE_KEYS = (
    "values",
    "table",
    "lut",
    "lookup_table",
    "gain",
    "gains",
    "policy",
    "action",
    "actions",
    "controller",
    "data",
)

_VERSION_KEYS = (
    "version",
    "schema_version",
    "lut_version",
    "metadata.version",
    "metadata.schema_version",
)


# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LUTValidationConfig:
    """Configuration for LUT validation."""

    name: str = "lut_validation"

    lut_dir: str | None = None
    recursive: bool = True

    allowed_extensions: tuple[str, ...] = (
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".npz",
        ".npy",
        ".txt",
    )

    min_file_size_bytes: int = 1
    max_file_size_bytes: int = 200 * 1024 * 1024

    min_records: int = 1
    min_numeric_values: int = 1
    max_dimensions: int = 8

    require_nonempty: bool = True
    require_finite_numeric_values: bool = True
    require_monotonic_axes: bool = True
    require_schema_fields: tuple[str, ...] = ()

    value_bounds: tuple[float, float] | None = None
    expected_luts: tuple[str, ...] = ()
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        value_bounds = None
        if self.value_bounds is not None:
            if len(self.value_bounds) != 2:
                raise ValueError("value_bounds must contain exactly two values")
            lower = _as_finite_float(self.value_bounds[0], "value_bounds[0]")
            upper = _as_finite_float(self.value_bounds[1], "value_bounds[1]")
            if lower > upper:
                raise ValueError("value_bounds lower must not exceed upper")
            value_bounds = (lower, upper)

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "lut_dir", None if self.lut_dir is None else str(self.lut_dir))
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
        object.__setattr__(self, "min_records", _as_nonnegative_int(self.min_records, "min_records"))
        object.__setattr__(
            self,
            "min_numeric_values",
            _as_nonnegative_int(self.min_numeric_values, "min_numeric_values"),
        )
        object.__setattr__(self, "max_dimensions", _as_positive_int(self.max_dimensions, "max_dimensions"))
        object.__setattr__(self, "require_nonempty", bool(self.require_nonempty))
        object.__setattr__(self, "require_finite_numeric_values", bool(self.require_finite_numeric_values))
        object.__setattr__(self, "require_monotonic_axes", bool(self.require_monotonic_axes))
        object.__setattr__(self, "require_schema_fields", tuple(str(item) for item in self.require_schema_fields))
        object.__setattr__(self, "value_bounds", value_bounds)
        object.__setattr__(self, "expected_luts", tuple(str(item) for item in self.expected_luts))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LUTValidationConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "lut_validation")),
            lut_dir=data.get("lut_dir", None),
            recursive=bool(data.get("recursive", True)),
            allowed_extensions=tuple(data.get("allowed_extensions", (
                ".json",
                ".jsonl",
                ".csv",
                ".tsv",
                ".npz",
                ".npy",
                ".txt",
            ))),
            min_file_size_bytes=int(data.get("min_file_size_bytes", 1)),
            max_file_size_bytes=int(data.get("max_file_size_bytes", 200 * 1024 * 1024)),
            min_records=int(data.get("min_records", 1)),
            min_numeric_values=int(data.get("min_numeric_values", 1)),
            max_dimensions=int(data.get("max_dimensions", 8)),
            require_nonempty=bool(data.get("require_nonempty", True)),
            require_finite_numeric_values=bool(data.get("require_finite_numeric_values", True)),
            require_monotonic_axes=bool(data.get("require_monotonic_axes", True)),
            require_schema_fields=tuple(data.get("require_schema_fields", ())),
            value_bounds=None if data.get("value_bounds", None) is None else tuple(data["value_bounds"]),
            expected_luts=tuple(data.get("expected_luts", ())),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "lut_dir": self.lut_dir,
            "recursive": self.recursive,
            "allowed_extensions": list(self.allowed_extensions),
            "min_file_size_bytes": self.min_file_size_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "min_records": self.min_records,
            "min_numeric_values": self.min_numeric_values,
            "max_dimensions": self.max_dimensions,
            "require_nonempty": self.require_nonempty,
            "require_finite_numeric_values": self.require_finite_numeric_values,
            "require_monotonic_axes": self.require_monotonic_axes,
            "require_schema_fields": list(self.require_schema_fields),
            "value_bounds": None if self.value_bounds is None else list(self.value_bounds),
            "expected_luts": list(self.expected_luts),
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LUTCheck:
    """One validation check result."""

    name: str
    status: LUTValidationStatus
    severity: LUTCheckSeverity = LUTCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, LUTValidationStatus) else LUTValidationStatus(self.status)
        severity = self.severity if isinstance(self.severity, LUTCheckSeverity) else LUTCheckSeverity(self.severity)

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(self, "details", dict(_json_safe(self.details)))

    @property
    def passed(self) -> bool:
        """Whether the check is passing."""

        return self.status is LUTValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check is failing."""

        return self.status is LUTValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check is warning."""

        return self.status is LUTValidationStatus.WARNING

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
class LUTArtifact:
    """LUT artifact metadata."""

    path: str
    kind: LUTArtifactKind
    size_bytes: int
    extension: str

    records: int | None = None
    fields: tuple[str, ...] = ()
    value_shape: tuple[int, ...] = ()
    axis_shapes: tuple[tuple[int, ...], ...] = ()

    num_numeric_values: int = 0
    num_nonfinite_numeric_values: int = 0
    min_value: float | None = None
    max_value: float | None = None

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, LUTArtifactKind) else LUTArtifactKind(self.kind)

        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "extension", str(self.extension).lower())
        object.__setattr__(self, "records", None if self.records is None else _as_nonnegative_int(self.records, "records"))
        object.__setattr__(self, "fields", tuple(str(item) for item in self.fields))
        object.__setattr__(self, "value_shape", tuple(int(item) for item in self.value_shape))
        object.__setattr__(
            self,
            "axis_shapes",
            tuple(tuple(int(item) for item in shape) for shape in self.axis_shapes),
        )
        object.__setattr__(
            self,
            "num_numeric_values",
            _as_nonnegative_int(self.num_numeric_values, "num_numeric_values"),
        )
        object.__setattr__(
            self,
            "num_nonfinite_numeric_values",
            _as_nonnegative_int(self.num_nonfinite_numeric_values, "num_nonfinite_numeric_values"),
        )
        object.__setattr__(self, "min_value", None if self.min_value is None else float(self.min_value))
        object.__setattr__(self, "max_value", None if self.max_value is None else float(self.max_value))
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
            "value_shape": list(self.value_shape),
            "axis_shapes": [list(shape) for shape in self.axis_shapes],
            "num_numeric_values": self.num_numeric_values,
            "num_nonfinite_numeric_values": self.num_nonfinite_numeric_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LUTValidationReport:
    """Complete LUT validation report."""

    name: str = "lut_validation_report"
    status: LUTValidationStatus = LUTValidationStatus.PASS
    artifacts: tuple[LUTArtifact, ...] = ()
    checks: tuple[LUTCheck, ...] = ()
    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, LUTValidationStatus) else LUTValidationStatus(self.status)

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "config", dict(_json_safe(self.config)))
        object.__setattr__(self, "metrics", dict(_json_safe(self.metrics)))

    @property
    def passed(self) -> bool:
        """Whether report passed."""

        return self.status is LUTValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is LUTValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[LUTCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[LUTCheck, ...]:
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
            f"LUT validation report: {self.status.value.upper()}",
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

def infer_lut_kind(path: str | Path) -> LUTArtifactKind:
    """Infer LUT artifact kind from extension."""

    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        return LUTArtifactKind.JSON

    if suffix == ".jsonl":
        return LUTArtifactKind.JSONL

    if suffix == ".csv":
        return LUTArtifactKind.CSV

    if suffix == ".tsv":
        return LUTArtifactKind.TSV

    if suffix == ".npz":
        return LUTArtifactKind.NPZ

    if suffix == ".npy":
        return LUTArtifactKind.NPY

    if suffix == ".txt":
        return LUTArtifactKind.TEXT

    return LUTArtifactKind.UNKNOWN


def discover_lut_files(
    lut_dir: str | Path,
    *,
    recursive: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Discover LUT artifact files under a directory."""

    root = Path(lut_dir)

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


def _read_json_lut(path: Path) -> tuple[Any, JSONDict]:
    """Read JSON LUT payload."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {"payload_type": type(data).__name__}


def _read_jsonl_lut(path: Path) -> tuple[list[Any], JSONDict]:
    """Read JSONL LUT records."""

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
        "payload_type": "list",
        "records": len(records),
    }

    if errors:
        metadata["jsonl_errors"] = errors[:20]
        metadata["num_jsonl_errors"] = len(errors)

    return records, metadata


def _read_delimited_lut(path: Path, delimiter: str) -> tuple[list[JSONDict], JSONDict]:
    """Read CSV/TSV LUT as list of records."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        records = [dict(row) for row in reader]

    return records, {
        "payload_type": "list",
        "records": len(records),
        "fieldnames": list(reader.fieldnames or ()),
    }


def _read_numpy_lut(path: Path) -> tuple[Any, JSONDict]:
    """Read NPZ/NPY LUT using NumPy lazily."""

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise ImportError(
            "NumPy is required to validate .npz/.npy LUT files. "
            "Install it with `pip install numpy`."
        ) from exc

    loaded = np.load(path, allow_pickle=True)

    if path.suffix.lower() == ".npz":
        payload = {key: loaded[key].tolist() for key in loaded.files}
        return payload, {
            "payload_type": "dict",
            "npz_keys": list(loaded.files),
        }

    return loaded.tolist() if hasattr(loaded, "tolist") else loaded, {
        "payload_type": type(loaded).__name__,
        "shape": list(getattr(loaded, "shape", ())),
    }


def _read_text_lut(path: Path) -> tuple[JSONDict, JSONDict]:
    """Read plain text LUT artifact."""

    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    numeric_values: list[float] = []
    for line in lines:
        for token in line.replace(",", " ").split():
            if _is_numeric_like(token):
                numeric_values.append(float(token))

    payload = {
        "text": text,
        "values": numeric_values,
    }

    return payload, {
        "payload_type": "dict",
        "nonempty_lines": len(lines),
    }


def read_lut_payload(path: str | Path) -> tuple[Any, JSONDict]:
    """Read LUT payload from supported artifact file."""

    p = Path(path)
    kind = infer_lut_kind(p)

    if kind is LUTArtifactKind.JSON:
        return _read_json_lut(p)

    if kind is LUTArtifactKind.JSONL:
        return _read_jsonl_lut(p)

    if kind is LUTArtifactKind.CSV:
        return _read_delimited_lut(p, ",")

    if kind is LUTArtifactKind.TSV:
        return _read_delimited_lut(p, "\t")

    if kind in {LUTArtifactKind.NPZ, LUTArtifactKind.NPY}:
        return _read_numpy_lut(p)

    if kind is LUTArtifactKind.TEXT:
        return _read_text_lut(p)

    raise ValueError(f"unsupported LUT extension: {p.suffix}")


# ------------------------------------------------------------------------------
# Payload analysis
# ------------------------------------------------------------------------------

def _records_from_payload(payload: Any) -> list[Any]:
    """Normalize payload into records."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, Mapping):
        for key in ("records", "rows", "entries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        return [payload]

    return [payload]


def _field_names_from_payload(payload: Any) -> tuple[str, ...]:
    """Collect top-level field names."""

    fields: list[str] = []
    seen: set[str] = set()

    if isinstance(payload, Mapping):
        for key in payload.keys():
            text = str(key)
            if text not in seen:
                seen.add(text)
                fields.append(text)

    for record in _records_from_payload(payload):
        if isinstance(record, Mapping):
            for key in record.keys():
                text = str(key)
                if text not in seen:
                    seen.add(text)
                    fields.append(text)

    return tuple(fields)


def _extract_value_payload(payload: Any) -> Any:
    """Extract LUT value/table payload."""

    if isinstance(payload, Mapping):
        value = _first_present(payload, _VALUE_KEYS)
        if value is not None:
            return value

    return payload


def _extract_axis_payloads(payload: Any) -> tuple[Any, ...]:
    """Extract axis/grid payloads."""

    if not isinstance(payload, Mapping):
        return ()

    raw = _first_present(payload, _AXIS_KEYS)

    if raw is None:
        return ()

    if isinstance(raw, Mapping):
        return tuple(raw.values())

    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
        if raw and all(isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray) for item in raw):
            return tuple(raw)
        return (raw,)

    return ()


def _axis_monotonicity(axis_payloads: Sequence[Any]) -> tuple[bool, list[int]]:
    """Check monotonicity for axis arrays."""

    bad_indices: list[int] = []

    for idx, axis in enumerate(axis_payloads):
        if isinstance(axis, Mapping):
            values = axis.get("values", axis.get("grid", axis.get("axis", None)))
            if values is None:
                bad_indices.append(idx)
                continue
            axis = values

        if not isinstance(axis, Sequence) or isinstance(axis, str | bytes | bytearray):
            bad_indices.append(idx)
            continue

        if not _is_strictly_increasing(axis):
            bad_indices.append(idx)

    return len(bad_indices) == 0, bad_indices


def collect_lut_artifact(path: str | Path) -> LUTArtifact:
    """Collect metadata for one LUT artifact."""

    p = Path(path)
    kind = infer_lut_kind(p)
    size = p.stat().st_size if p.exists() else 0

    records: int | None = None
    fields: tuple[str, ...] = ()
    value_shape: tuple[int, ...] = ()
    axis_shapes: tuple[tuple[int, ...], ...] = ()
    num_numeric_values = 0
    num_nonfinite_numeric_values = 0
    min_value: float | None = None
    max_value: float | None = None
    metadata: JSONDict = {}

    if p.exists() and p.is_file() and size > 0:
        try:
            payload, read_metadata = read_lut_payload(p)
            metadata.update(read_metadata)

            records = len(_records_from_payload(payload))
            fields = _field_names_from_payload(payload)

            value_payload = _extract_value_payload(payload)
            axis_payloads = _extract_axis_payloads(payload)

            value_shape = _shape_of_nested_sequence(value_payload)
            axis_shapes = tuple(_shape_of_nested_sequence(axis) for axis in axis_payloads)

            all_numbers = _flatten_numbers(payload)
            value_numbers = _flatten_numbers(value_payload)
            finite_value_numbers = [value for value in value_numbers if isfinite(value)]

            num_numeric_values = len(all_numbers)
            num_nonfinite_numeric_values = sum(1 for value in all_numbers if not isfinite(value))

            if finite_value_numbers:
                min_value = min(finite_value_numbers)
                max_value = max(finite_value_numbers)

            axes_ok, bad_axes = _axis_monotonicity(axis_payloads)

            metadata["num_value_numeric_values"] = len(value_numbers)
            metadata["num_axis_payloads"] = len(axis_payloads)
            metadata["axes_monotonic"] = axes_ok
            metadata["bad_axis_indices"] = bad_axes
            metadata["version"] = _first_present(payload, _VERSION_KEYS) if isinstance(payload, Mapping) else None

        except Exception as exc:
            metadata = {
                "read_error": str(exc),
                "read_error_type": type(exc).__name__,
            }

    return LUTArtifact(
        path=str(p),
        kind=kind,
        size_bytes=size,
        extension=p.suffix.lower(),
        records=records,
        fields=fields,
        value_shape=value_shape,
        axis_shapes=axis_shapes,
        num_numeric_values=num_numeric_values,
        num_nonfinite_numeric_values=num_nonfinite_numeric_values,
        min_value=min_value,
        max_value=max_value,
        metadata=metadata,
    )


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_lut_artifact(
    path: str | Path,
    config: LUTValidationConfig | Mapping[str, Any] | None = None,
) -> tuple[LUTArtifact | None, tuple[LUTCheck, ...]]:
    """Validate one LUT artifact."""

    cfg = (
        LUTValidationConfig()
        if config is None
        else config
        if isinstance(config, LUTValidationConfig)
        else LUTValidationConfig.from_dict(config)
    )

    p = Path(path)
    checks: list[LUTCheck] = []

    if not p.exists():
        checks.append(
            LUTCheck(
                name="exists",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message="File does not exist",
                path=str(p),
            )
        )
        return None, tuple(checks)

    if not p.is_file():
        checks.append(
            LUTCheck(
                name="is_file",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message="Path is not a file",
                path=str(p),
            )
        )
        return None, tuple(checks)

    artifact = collect_lut_artifact(p)

    if artifact.extension not in cfg.allowed_extensions:
        checks.append(
            LUTCheck(
                name="allowed_extension",
                status=LUTValidationStatus.WARNING,
                severity=LUTCheckSeverity.WARNING,
                message=f"Extension {artifact.extension!r} is not in allowed extensions",
                path=str(p),
                details={"allowed_extensions": list(cfg.allowed_extensions)},
            )
        )
    else:
        checks.append(
            LUTCheck(
                name="allowed_extension",
                status=LUTValidationStatus.PASS,
                message="Extension is allowed",
                path=str(p),
            )
        )

    if cfg.require_nonempty and artifact.size_bytes <= 0:
        checks.append(
            LUTCheck(
                name="nonempty",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message="File is empty",
                path=str(p),
            )
        )
    elif artifact.size_bytes < cfg.min_file_size_bytes:
        checks.append(
            LUTCheck(
                name="min_file_size",
                status=LUTValidationStatus.WARNING,
                severity=LUTCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} is below minimum "
                    f"{_human_size(cfg.min_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    elif artifact.size_bytes > cfg.max_file_size_bytes:
        checks.append(
            LUTCheck(
                name="max_file_size",
                status=LUTValidationStatus.WARNING,
                severity=LUTCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} exceeds maximum "
                    f"{_human_size(cfg.max_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    else:
        checks.append(
            LUTCheck(
                name="file_size",
                status=LUTValidationStatus.PASS,
                message=f"File size is {_human_size(artifact.size_bytes)}",
                path=str(p),
            )
        )

    if "read_error" in artifact.metadata:
        checks.append(
            LUTCheck(
                name="readable",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message=f"Unable to read LUT: {artifact.metadata['read_error']}",
                path=str(p),
                details=artifact.metadata,
            )
        )
        return artifact, tuple(checks)

    checks.append(
        LUTCheck(
            name="readable",
            status=LUTValidationStatus.PASS,
            message="LUT metadata parsed",
            path=str(p),
        )
    )

    if "num_jsonl_errors" in artifact.metadata:
        checks.append(
            LUTCheck(
                name="jsonl_parse_errors",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message=f"JSONL contains {artifact.metadata['num_jsonl_errors']} parse error(s)",
                path=str(p),
                details=artifact.metadata,
            )
        )

    if artifact.records is not None:
        checks.append(
            LUTCheck(
                name="record_count",
                status=LUTValidationStatus.PASS if artifact.records >= cfg.min_records else LUTValidationStatus.WARNING,
                severity=LUTCheckSeverity.INFO if artifact.records >= cfg.min_records else LUTCheckSeverity.WARNING,
                message=(
                    f"Record count is {artifact.records}"
                    if artifact.records >= cfg.min_records
                    else f"Record count {artifact.records} is below minimum {cfg.min_records}"
                ),
                path=str(p),
            )
        )

    checks.append(
        LUTCheck(
            name="numeric_value_count",
            status=(
                LUTValidationStatus.PASS
                if artifact.num_numeric_values >= cfg.min_numeric_values
                else LUTValidationStatus.WARNING
            ),
            severity=(
                LUTCheckSeverity.INFO
                if artifact.num_numeric_values >= cfg.min_numeric_values
                else LUTCheckSeverity.WARNING
            ),
            message=(
                f"Numeric value count is {artifact.num_numeric_values}"
                if artifact.num_numeric_values >= cfg.min_numeric_values
                else (
                    f"Numeric value count {artifact.num_numeric_values} is below "
                    f"minimum {cfg.min_numeric_values}"
                )
            ),
            path=str(p),
        )
    )

    if cfg.require_finite_numeric_values:
        checks.append(
            LUTCheck(
                name="finite_numeric_values",
                status=(
                    LUTValidationStatus.PASS
                    if artifact.num_nonfinite_numeric_values == 0
                    else LUTValidationStatus.FAIL
                ),
                severity=(
                    LUTCheckSeverity.INFO
                    if artifact.num_nonfinite_numeric_values == 0
                    else LUTCheckSeverity.ERROR
                ),
                message=(
                    "All numeric values are finite"
                    if artifact.num_nonfinite_numeric_values == 0
                    else f"Found {artifact.num_nonfinite_numeric_values} non-finite numeric value(s)"
                ),
                path=str(p),
                details={
                    "num_numeric_values": artifact.num_numeric_values,
                    "num_nonfinite_numeric_values": artifact.num_nonfinite_numeric_values,
                },
            )
        )

    if artifact.value_shape:
        checks.append(
            LUTCheck(
                name="value_dimensions",
                status=(
                    LUTValidationStatus.PASS
                    if len(artifact.value_shape) <= cfg.max_dimensions
                    else LUTValidationStatus.WARNING
                ),
                severity=(
                    LUTCheckSeverity.INFO
                    if len(artifact.value_shape) <= cfg.max_dimensions
                    else LUTCheckSeverity.WARNING
                ),
                message=(
                    f"Value shape is {artifact.value_shape}"
                    if len(artifact.value_shape) <= cfg.max_dimensions
                    else (
                        f"Value dimension {len(artifact.value_shape)} exceeds "
                        f"maximum {cfg.max_dimensions}"
                    )
                ),
                path=str(p),
                details={"value_shape": list(artifact.value_shape)},
            )
        )

    if cfg.require_schema_fields:
        available = set(artifact.fields)
        missing = [field for field in cfg.require_schema_fields if field not in available]

        checks.append(
            LUTCheck(
                name="required_schema_fields",
                status=LUTValidationStatus.PASS if not missing else LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.INFO if not missing else LUTCheckSeverity.ERROR,
                message="All required schema fields exist" if not missing else f"Missing schema fields: {missing}",
                path=str(p),
                details={
                    "required_schema_fields": list(cfg.require_schema_fields),
                    "available_fields": list(artifact.fields),
                    "missing_fields": missing,
                },
            )
        )

    if cfg.require_monotonic_axes and int(artifact.metadata.get("num_axis_payloads", 0)) > 0:
        axes_ok = bool(artifact.metadata.get("axes_monotonic", False))
        checks.append(
            LUTCheck(
                name="monotonic_axes",
                status=LUTValidationStatus.PASS if axes_ok else LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.INFO if axes_ok else LUTCheckSeverity.ERROR,
                message="All axes are strictly increasing" if axes_ok else "One or more axes are not strictly increasing",
                path=str(p),
                details={
                    "axis_shapes": [list(shape) for shape in artifact.axis_shapes],
                    "bad_axis_indices": artifact.metadata.get("bad_axis_indices", []),
                },
            )
        )

    if cfg.value_bounds is not None and artifact.min_value is not None and artifact.max_value is not None:
        lower, upper = cfg.value_bounds
        in_bounds = artifact.min_value >= lower and artifact.max_value <= upper

        checks.append(
            LUTCheck(
                name="value_bounds",
                status=LUTValidationStatus.PASS if in_bounds else LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.INFO if in_bounds else LUTCheckSeverity.ERROR,
                message=(
                    f"Values are within bounds [{lower}, {upper}]"
                    if in_bounds
                    else (
                        f"Value range [{artifact.min_value}, {artifact.max_value}] "
                        f"exceeds bounds [{lower}, {upper}]"
                    )
                ),
                path=str(p),
                details={
                    "lower": lower,
                    "upper": upper,
                    "min_value": artifact.min_value,
                    "max_value": artifact.max_value,
                },
            )
        )

    return artifact, tuple(checks)


def _validate_expected_luts(
    lut_dir: Path,
    expected_luts: Sequence[str],
) -> tuple[LUTCheck, ...]:
    """Validate expected LUT files."""

    checks: list[LUTCheck] = []

    for item in expected_luts:
        path = Path(item)
        if not path.is_absolute():
            path = lut_dir / path

        checks.append(
            LUTCheck(
                name="expected_lut",
                status=LUTValidationStatus.PASS if path.exists() else LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.INFO if path.exists() else LUTCheckSeverity.ERROR,
                message="Expected LUT exists" if path.exists() else "Expected LUT is missing",
                path=str(path),
                details={"expected": item},
            )
        )

    return tuple(checks)


def _compute_report_status(
    checks: Sequence[LUTCheck],
    *,
    fail_on_warning: bool = False,
) -> LUTValidationStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return LUTValidationStatus.FAIL

    if any(check.warning for check in checks):
        return LUTValidationStatus.FAIL if fail_on_warning else LUTValidationStatus.WARNING

    return LUTValidationStatus.PASS


def validate_lut_directory(
    lut_dir: str | Path,
    config: LUTValidationConfig | Mapping[str, Any] | None = None,
) -> LUTValidationReport:
    """Validate all LUT artifacts in a directory."""

    cfg = (
        LUTValidationConfig(lut_dir=str(lut_dir))
        if config is None
        else config
        if isinstance(config, LUTValidationConfig)
        else LUTValidationConfig.from_dict(config)
    )

    root = Path(lut_dir)
    checks: list[LUTCheck] = []
    artifacts: list[LUTArtifact] = []

    if not root.exists():
        checks.append(
            LUTCheck(
                name="lut_dir_exists",
                status=LUTValidationStatus.FAIL,
                severity=LUTCheckSeverity.ERROR,
                message="LUT directory does not exist",
                path=str(root),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

        return LUTValidationReport(
            status=status,
            artifacts=(),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_artifacts": 0, "num_checks": len(checks)},
        )

    paths = discover_lut_files(
        root,
        recursive=cfg.recursive,
        allowed_extensions=cfg.allowed_extensions,
    )

    checks.append(
        LUTCheck(
            name="lut_discovery",
            status=LUTValidationStatus.PASS if paths else LUTValidationStatus.WARNING,
            severity=LUTCheckSeverity.INFO if paths else LUTCheckSeverity.WARNING,
            message=f"Discovered {len(paths)} LUT artifact(s)",
            path=str(root),
        )
    )

    for path in paths:
        artifact, artifact_checks = validate_lut_artifact(path, cfg)
        if artifact is not None:
            artifacts.append(artifact)
        checks.extend(artifact_checks)

    checks.extend(_validate_expected_luts(root, cfg.expected_luts))

    total_size = sum(artifact.size_bytes for artifact in artifacts)
    kind_counts: dict[str, int] = {}
    numeric_counts = [artifact.num_numeric_values for artifact in artifacts]
    min_values = [artifact.min_value for artifact in artifacts if artifact.min_value is not None]
    max_values = [artifact.max_value for artifact in artifacts if artifact.max_value is not None]

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
        "numeric_values_total": sum(numeric_counts),
        "global_min_value": min(min_values) if min_values else None,
        "global_max_value": max(max_values) if max_values else None,
    }

    return LUTValidationReport(
        status=status,
        artifacts=tuple(artifacts),
        checks=tuple(checks),
        config=cfg.to_dict(),
        metrics=metrics,
    )


def validate_lut_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    lut_dir: str | Path | None = None,
) -> LUTValidationReport:
    """Validate LUT artifacts using a manifest mapping or JSON file."""

    data = _load_json(manifest) if isinstance(manifest, str | Path) else dict(manifest)

    if lut_dir is not None:
        data["lut_dir"] = str(lut_dir)

    root = data.get("lut_dir", None)
    if root is None:
        raise ValueError("manifest must define lut_dir or lut_dir must be provided")

    cfg = LUTValidationConfig.from_dict(data)

    return validate_lut_directory(root, cfg)


def validate_default_project_lut(
    lut_dir: str | Path = "results/lut",
) -> LUTValidationReport:
    """Validate the default expected project LUT set."""

    expected = (
        "controller_lut.json",
        "gain_lut.json",
        "deployment_grid.json",
    )

    cfg = LUTValidationConfig(
        lut_dir=str(lut_dir),
        expected_luts=expected,
        recursive=True,
        require_nonempty=True,
        require_finite_numeric_values=True,
        require_monotonic_axes=True,
        min_numeric_values=1,
    )

    return validate_lut_directory(lut_dir, cfg)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.validate_lut",
        description="Validate generated LUT deployment artifacts.",
    )

    parser.add_argument("--lut-dir", default="results/lut")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument(
        "--allowed-extensions",
        default=".json,.jsonl,.csv,.tsv,.npz,.npy,.txt",
    )
    parser.add_argument("--expected-luts", default=None)

    parser.add_argument("--min-file-size-bytes", type=int, default=1)
    parser.add_argument("--max-file-size-bytes", type=int, default=200 * 1024 * 1024)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-numeric-values", type=int, default=1)
    parser.add_argument("--max-dimensions", type=int, default=8)

    parser.add_argument("--required-schema-fields", default=None)
    parser.add_argument("--value-bounds", default=None, help="Value bounds formatted as lower,upper.")

    parser.add_argument("--no-require-finite-numeric-values", action="store_true")
    parser.add_argument("--no-require-monotonic-axes", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--default-project", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.default_project:
        report = validate_default_project_lut(args.lut_dir)

    elif args.manifest:
        report = validate_lut_from_manifest(args.manifest, lut_dir=args.lut_dir)

    else:
        cfg = LUTValidationConfig(
            lut_dir=args.lut_dir,
            recursive=not args.non_recursive,
            allowed_extensions=_parse_csv_text(args.allowed_extensions),
            expected_luts=_parse_csv_text(args.expected_luts),
            min_file_size_bytes=args.min_file_size_bytes,
            max_file_size_bytes=args.max_file_size_bytes,
            min_records=args.min_records,
            min_numeric_values=args.min_numeric_values,
            max_dimensions=args.max_dimensions,
            require_schema_fields=_parse_csv_text(args.required_schema_fields),
            value_bounds=_parse_float_pair(args.value_bounds),
            require_finite_numeric_values=not args.no_require_finite_numeric_values,
            require_monotonic_axes=not args.no_require_monotonic_axes,
            fail_on_warning=args.fail_on_warning,
        )

        report = validate_lut_directory(args.lut_dir, cfg)

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
    "JSONDict",
    "LUTArtifact",
    "LUTArtifactKind",
    "LUTCheck",
    "LUTCheckSeverity",
    "LUTValidationConfig",
    "LUTValidationReport",
    "LUTValidationStatus",
    "Number",
    "build_arg_parser",
    "collect_lut_artifact",
    "discover_lut_files",
    "infer_lut_kind",
    "main",
    "read_lut_payload",
    "validate_default_project_lut",
    "validate_lut_artifact",
    "validate_lut_directory",
    "validate_lut_from_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
