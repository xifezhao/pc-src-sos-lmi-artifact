# ==============================================================================
# src/pcsrc/validation/validate_simulation_metrics.py
#
# Simulation metric validation utilities.
#
# Purpose:
#   - Validate generated simulation metric artifacts.
#   - Check existence, size, schema fields, required metric names, numeric
#     finiteness, ranges, thresholds, and expected metric files.
#   - Support JSON, JSONL, CSV, TSV, and TXT metric artifacts.
#   - Provide CLI entry point for CI / reproducibility workflows.
#
# Design notes:
#   - Pure Python standard library.
#   - Schema-tolerant metric extraction:
#       * {"metrics": {"rmse": 0.1, "success_rate": 0.98}}
#       * {"records": [{"metric": "rmse", "value": 0.1}]}
#       * CSV with columns metric,value or wide metric columns.
#   - Threshold rules are represented as:
#       {
#           "rmse": {"max": 0.2},
#           "success_rate": {"min": 0.95},
#           "violation_rate": {"max": 0.01}
#       }
#
# Typical usage:
#
#   from pcsrc.validation.validate_simulation_metrics import (
#       validate_simulation_metric_directory,
#   )
#
#   report = validate_simulation_metric_directory("results/metrics")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.validate_simulation_metrics \
#       --metric-dir results/metrics \
#       --manifest results/metrics/manifest.json \
#       --json-output results/metrics/validation_report.json
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


def _is_numeric_like(value: Any) -> bool:
    """Return whether value can be parsed as finite or non-finite float."""

    if value is None or isinstance(value, bool):
        return False

    try:
        float(value)
    except (TypeError, ValueError):
        return False

    return True


def _to_float_or_none(value: Any) -> float | None:
    """Convert numeric-like value to float, else None."""

    if not _is_numeric_like(value):
        return None

    return float(value)


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


def _parse_threshold_rules(text: str | None) -> JSONDict:
    """Parse CLI threshold rules.

    Format:
        metric:min=0,max=1;rmse:max=0.2;success_rate:min=0.95
    """

    if not text:
        return {}

    rules: JSONDict = {}

    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece:
            continue

        if ":" not in piece:
            raise ValueError("threshold rule must be formatted as metric:key=value,key=value")

        metric_name, rule_text = piece.split(":", 1)
        metric_name = metric_name.strip()

        if not metric_name:
            raise ValueError("threshold metric name must be non-empty")

        rule: JSONDict = {}
        for item in rule_text.split(","):
            item = item.strip()
            if not item:
                continue

            if "=" not in item:
                raise ValueError("threshold rule item must be key=value")

            key, value = item.split("=", 1)
            rule[key.strip()] = _as_finite_float(value.strip(), key.strip())

        rules[metric_name] = rule

    return rules


# ------------------------------------------------------------------------------
# Enums and constants
# ------------------------------------------------------------------------------

class SimulationMetricValidationStatus(str, Enum):
    """Validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class SimulationMetricCheckSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SimulationMetricArtifactKind(str, Enum):
    """Supported metric artifact kinds."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    TEXT = "text"
    UNKNOWN = "unknown"


_METRIC_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
}

_METRIC_CONTAINER_KEYS = (
    "metrics",
    "summary",
    "results",
    "statistics",
    "stats",
    "validation.metrics",
    "simulation.metrics",
)

_RECORD_CONTAINER_KEYS = (
    "records",
    "rows",
    "data",
    "measurements",
    "runs",
)

_METRIC_NAME_KEYS = (
    "metric",
    "name",
    "key",
    "field",
)

_METRIC_VALUE_KEYS = (
    "value",
    "mean",
    "score",
    "estimate",
    "result",
)


# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimulationMetricValidationConfig:
    """Configuration for simulation metric validation."""

    name: str = "simulation_metric_validation"

    metric_dir: str | None = None
    recursive: bool = True

    allowed_extensions: tuple[str, ...] = (
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".txt",
    )

    min_file_size_bytes: int = 1
    max_file_size_bytes: int = 100 * 1024 * 1024

    min_metrics: int = 1
    min_records: int = 1

    required_metrics: tuple[str, ...] = ()
    threshold_rules: JSONDict = field(default_factory=dict)

    require_nonempty: bool = True
    require_finite_numeric_values: bool = True
    require_metric_values: bool = True

    expected_metric_files: tuple[str, ...] = ()
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_rules: JSONDict = {}

        for metric_name, rule in self.threshold_rules.items():
            if not isinstance(rule, Mapping):
                raise ValueError("threshold rule must be a mapping")

            parsed_rule: JSONDict = {}
            for key, value in rule.items():
                key_text = str(key)
                if key_text not in {"min", "max", "eq", "abs_max"}:
                    raise ValueError(f"unsupported threshold rule key: {key_text}")
                parsed_rule[key_text] = _as_finite_float(value, f"threshold_rules[{metric_name}].{key_text}")

            normalized_rules[str(metric_name)] = parsed_rule

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "metric_dir", None if self.metric_dir is None else str(self.metric_dir))
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
        object.__setattr__(self, "min_metrics", _as_nonnegative_int(self.min_metrics, "min_metrics"))
        object.__setattr__(self, "min_records", _as_nonnegative_int(self.min_records, "min_records"))
        object.__setattr__(self, "required_metrics", tuple(str(item) for item in self.required_metrics))
        object.__setattr__(self, "threshold_rules", normalized_rules)
        object.__setattr__(self, "require_nonempty", bool(self.require_nonempty))
        object.__setattr__(self, "require_finite_numeric_values", bool(self.require_finite_numeric_values))
        object.__setattr__(self, "require_metric_values", bool(self.require_metric_values))
        object.__setattr__(self, "expected_metric_files", tuple(str(item) for item in self.expected_metric_files))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationMetricValidationConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "simulation_metric_validation")),
            metric_dir=data.get("metric_dir", data.get("metrics_dir", None)),
            recursive=bool(data.get("recursive", True)),
            allowed_extensions=tuple(data.get("allowed_extensions", (
                ".json",
                ".jsonl",
                ".csv",
                ".tsv",
                ".txt",
            ))),
            min_file_size_bytes=int(data.get("min_file_size_bytes", 1)),
            max_file_size_bytes=int(data.get("max_file_size_bytes", 100 * 1024 * 1024)),
            min_metrics=int(data.get("min_metrics", 1)),
            min_records=int(data.get("min_records", 1)),
            required_metrics=tuple(data.get("required_metrics", ())),
            threshold_rules=dict(data.get("threshold_rules", data.get("thresholds", {})) or {}),
            require_nonempty=bool(data.get("require_nonempty", True)),
            require_finite_numeric_values=bool(data.get("require_finite_numeric_values", True)),
            require_metric_values=bool(data.get("require_metric_values", True)),
            expected_metric_files=tuple(data.get("expected_metric_files", data.get("expected_metrics", ()))),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "metric_dir": self.metric_dir,
            "recursive": self.recursive,
            "allowed_extensions": list(self.allowed_extensions),
            "min_file_size_bytes": self.min_file_size_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "min_metrics": self.min_metrics,
            "min_records": self.min_records,
            "required_metrics": list(self.required_metrics),
            "threshold_rules": dict(self.threshold_rules),
            "require_nonempty": self.require_nonempty,
            "require_finite_numeric_values": self.require_finite_numeric_values,
            "require_metric_values": self.require_metric_values,
            "expected_metric_files": list(self.expected_metric_files),
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SimulationMetricCheck:
    """One validation check result."""

    name: str
    status: SimulationMetricValidationStatus
    severity: SimulationMetricCheckSeverity = SimulationMetricCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, SimulationMetricValidationStatus)
            else SimulationMetricValidationStatus(self.status)
        )
        severity = (
            self.severity
            if isinstance(self.severity, SimulationMetricCheckSeverity)
            else SimulationMetricCheckSeverity(self.severity)
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

        return self.status is SimulationMetricValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check is failing."""

        return self.status is SimulationMetricValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check is warning."""

        return self.status is SimulationMetricValidationStatus.WARNING

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
class SimulationMetricArtifact:
    """Simulation metric artifact metadata."""

    path: str
    kind: SimulationMetricArtifactKind
    size_bytes: int
    extension: str

    records: int | None = None
    metric_names: tuple[str, ...] = ()
    metric_values: JSONDict = field(default_factory=dict)

    num_numeric_values: int = 0
    num_nonfinite_numeric_values: int = 0
    min_numeric_value: float | None = None
    max_numeric_value: float | None = None

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = (
            self.kind
            if isinstance(self.kind, SimulationMetricArtifactKind)
            else SimulationMetricArtifactKind(self.kind)
        )

        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "extension", str(self.extension).lower())
        object.__setattr__(self, "records", None if self.records is None else _as_nonnegative_int(self.records, "records"))
        object.__setattr__(self, "metric_names", tuple(str(item) for item in self.metric_names))
        object.__setattr__(self, "metric_values", dict(_json_safe(self.metric_values)))
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
        object.__setattr__(
            self,
            "min_numeric_value",
            None if self.min_numeric_value is None else float(self.min_numeric_value),
        )
        object.__setattr__(
            self,
            "max_numeric_value",
            None if self.max_numeric_value is None else float(self.max_numeric_value),
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
            "metric_names": list(self.metric_names),
            "metric_values": dict(self.metric_values),
            "num_numeric_values": self.num_numeric_values,
            "num_nonfinite_numeric_values": self.num_nonfinite_numeric_values,
            "min_numeric_value": self.min_numeric_value,
            "max_numeric_value": self.max_numeric_value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SimulationMetricValidationReport:
    """Complete simulation metric validation report."""

    name: str = "simulation_metric_validation_report"
    status: SimulationMetricValidationStatus = SimulationMetricValidationStatus.PASS
    artifacts: tuple[SimulationMetricArtifact, ...] = ()
    checks: tuple[SimulationMetricCheck, ...] = ()
    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, SimulationMetricValidationStatus)
            else SimulationMetricValidationStatus(self.status)
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

        return self.status is SimulationMetricValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is SimulationMetricValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[SimulationMetricCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[SimulationMetricCheck, ...]:
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
            f"Simulation metric validation report: {self.status.value.upper()}",
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

def infer_simulation_metric_kind(path: str | Path) -> SimulationMetricArtifactKind:
    """Infer simulation metric artifact kind from extension."""

    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        return SimulationMetricArtifactKind.JSON

    if suffix == ".jsonl":
        return SimulationMetricArtifactKind.JSONL

    if suffix == ".csv":
        return SimulationMetricArtifactKind.CSV

    if suffix == ".tsv":
        return SimulationMetricArtifactKind.TSV

    if suffix == ".txt":
        return SimulationMetricArtifactKind.TEXT

    return SimulationMetricArtifactKind.UNKNOWN


def discover_simulation_metric_files(
    metric_dir: str | Path,
    *,
    recursive: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Discover simulation metric files under a directory."""

    root = Path(metric_dir)

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


def _read_json_metrics(path: Path) -> tuple[Any, JSONDict]:
    """Read JSON metrics."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {"payload_type": type(data).__name__}


def _read_jsonl_metrics(path: Path) -> tuple[list[Any], JSONDict]:
    """Read JSONL metric records."""

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


def _read_delimited_metrics(path: Path, delimiter: str) -> tuple[list[JSONDict], JSONDict]:
    """Read CSV/TSV metrics."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        records = [dict(row) for row in reader]

    return records, {
        "payload_type": "list",
        "records": len(records),
        "fieldnames": list(reader.fieldnames or ()),
    }


def _read_text_metrics(path: Path) -> tuple[JSONDict, JSONDict]:
    """Read plain text metrics.

    Supports simple lines like:
        rmse: 0.1
        success_rate = 0.98
    """

    text = path.read_text(encoding="utf-8")
    metrics: JSONDict = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        separator = ":" if ":" in stripped else "=" if "=" in stripped else None
        if separator is None:
            continue

        key, value = stripped.split(separator, 1)
        key = key.strip()
        value = value.strip()

        parsed = _to_float_or_none(value)
        metrics[key] = parsed if parsed is not None else value

    return {
        "metrics": metrics,
        "text": text,
    }, {
        "payload_type": "dict",
        "parsed_text_metrics": len(metrics),
    }


def read_simulation_metric_payload(path: str | Path) -> tuple[Any, JSONDict]:
    """Read simulation metric payload from supported file."""

    p = Path(path)
    kind = infer_simulation_metric_kind(p)

    if kind is SimulationMetricArtifactKind.JSON:
        return _read_json_metrics(p)

    if kind is SimulationMetricArtifactKind.JSONL:
        return _read_jsonl_metrics(p)

    if kind is SimulationMetricArtifactKind.CSV:
        return _read_delimited_metrics(p, ",")

    if kind is SimulationMetricArtifactKind.TSV:
        return _read_delimited_metrics(p, "\t")

    if kind is SimulationMetricArtifactKind.TEXT:
        return _read_text_metrics(p)

    raise ValueError(f"unsupported metric extension: {p.suffix}")


# ------------------------------------------------------------------------------
# Metric extraction
# ------------------------------------------------------------------------------

def _records_from_payload(payload: Any) -> list[Any]:
    """Normalize payload into record list."""

    if isinstance(payload, list):
        return payload

    if isinstance(payload, Mapping):
        for key in _RECORD_CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        return [payload]

    return [payload]


def _extract_metric_mapping_from_mapping(mapping: Mapping[str, Any]) -> JSONDict:
    """Extract metric mapping from one mapping object."""

    metrics: JSONDict = {}

    container = _first_present(mapping, _METRIC_CONTAINER_KEYS)
    if isinstance(container, Mapping):
        for key, value in container.items():
            parsed = _to_float_or_none(value)
            if parsed is not None:
                metrics[str(key)] = parsed

    metric_name = _first_present(mapping, _METRIC_NAME_KEYS)
    metric_value = _first_present(mapping, _METRIC_VALUE_KEYS)

    if metric_name is not None and metric_value is not None:
        parsed = _to_float_or_none(metric_value)
        if parsed is not None:
            metrics[str(metric_name)] = parsed

    # Wide-format fallback: numeric top-level fields are treated as metrics.
    for key, value in mapping.items():
        if key in set(_RECORD_CONTAINER_KEYS) | set(_METRIC_CONTAINER_KEYS):
            continue

        if isinstance(value, bool):
            continue

        parsed = _to_float_or_none(value)
        if parsed is not None:
            metrics.setdefault(str(key), parsed)

    return metrics


def extract_metric_values(payload: Any) -> JSONDict:
    """Extract metric values from flexible metric payload."""

    metrics: JSONDict = {}

    if isinstance(payload, Mapping):
        metrics.update(_extract_metric_mapping_from_mapping(payload))

    for record in _records_from_payload(payload):
        if isinstance(record, Mapping):
            for key, value in _extract_metric_mapping_from_mapping(record).items():
                metrics[key] = value

    return metrics


def collect_simulation_metric_artifact(path: str | Path) -> SimulationMetricArtifact:
    """Collect metadata for one metric artifact."""

    p = Path(path)
    kind = infer_simulation_metric_kind(p)
    size = p.stat().st_size if p.exists() else 0

    records: int | None = None
    metric_values: JSONDict = {}
    metric_names: tuple[str, ...] = ()
    num_numeric_values = 0
    num_nonfinite_numeric_values = 0
    min_numeric_value: float | None = None
    max_numeric_value: float | None = None
    metadata: JSONDict = {}

    if p.exists() and p.is_file() and size > 0:
        try:
            payload, read_metadata = read_simulation_metric_payload(p)
            metadata.update(read_metadata)

            records = len(_records_from_payload(payload))
            metric_values = extract_metric_values(payload)
            metric_names = tuple(sorted(metric_values))

            all_numbers = _flatten_numbers(payload)
            finite_numbers = [value for value in all_numbers if isfinite(value)]

            num_numeric_values = len(all_numbers)
            num_nonfinite_numeric_values = sum(1 for value in all_numbers if not isfinite(value))

            if finite_numbers:
                min_numeric_value = min(finite_numbers)
                max_numeric_value = max(finite_numbers)

            metadata["num_extracted_metrics"] = len(metric_values)

        except Exception as exc:
            metadata = {
                "read_error": str(exc),
                "read_error_type": type(exc).__name__,
            }

    return SimulationMetricArtifact(
        path=str(p),
        kind=kind,
        size_bytes=size,
        extension=p.suffix.lower(),
        records=records,
        metric_names=metric_names,
        metric_values=metric_values,
        num_numeric_values=num_numeric_values,
        num_nonfinite_numeric_values=num_nonfinite_numeric_values,
        min_numeric_value=min_numeric_value,
        max_numeric_value=max_numeric_value,
        metadata=metadata,
    )


# ------------------------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------------------------

def _check_threshold_rule(metric_name: str, value: float, rule: Mapping[str, Any]) -> tuple[bool, str]:
    """Check a single metric threshold rule."""

    failures: list[str] = []

    if "min" in rule and value < float(rule["min"]):
        failures.append(f"value {value:.6g} < min {float(rule['min']):.6g}")

    if "max" in rule and value > float(rule["max"]):
        failures.append(f"value {value:.6g} > max {float(rule['max']):.6g}")

    if "eq" in rule and value != float(rule["eq"]):
        failures.append(f"value {value:.6g} != eq {float(rule['eq']):.6g}")

    if "abs_max" in rule and abs(value) > float(rule["abs_max"]):
        failures.append(f"abs(value) {abs(value):.6g} > abs_max {float(rule['abs_max']):.6g}")

    if failures:
        return False, f"{metric_name}: " + "; ".join(failures)

    return True, f"{metric_name}: value {value:.6g} satisfies threshold rule"


def validate_simulation_metric_artifact(
    path: str | Path,
    config: SimulationMetricValidationConfig | Mapping[str, Any] | None = None,
) -> tuple[SimulationMetricArtifact | None, tuple[SimulationMetricCheck, ...]]:
    """Validate one simulation metric artifact."""

    cfg = (
        SimulationMetricValidationConfig()
        if config is None
        else config
        if isinstance(config, SimulationMetricValidationConfig)
        else SimulationMetricValidationConfig.from_dict(config)
    )

    p = Path(path)
    checks: list[SimulationMetricCheck] = []

    if not p.exists():
        checks.append(
            SimulationMetricCheck(
                name="exists",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message="File does not exist",
                path=str(p),
            )
        )
        return None, tuple(checks)

    if not p.is_file():
        checks.append(
            SimulationMetricCheck(
                name="is_file",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message="Path is not a file",
                path=str(p),
            )
        )
        return None, tuple(checks)

    artifact = collect_simulation_metric_artifact(p)

    if artifact.extension not in cfg.allowed_extensions:
        checks.append(
            SimulationMetricCheck(
                name="allowed_extension",
                status=SimulationMetricValidationStatus.WARNING,
                severity=SimulationMetricCheckSeverity.WARNING,
                message=f"Extension {artifact.extension!r} is not in allowed extensions",
                path=str(p),
                details={"allowed_extensions": list(cfg.allowed_extensions)},
            )
        )
    else:
        checks.append(
            SimulationMetricCheck(
                name="allowed_extension",
                status=SimulationMetricValidationStatus.PASS,
                message="Extension is allowed",
                path=str(p),
            )
        )

    if cfg.require_nonempty and artifact.size_bytes <= 0:
        checks.append(
            SimulationMetricCheck(
                name="nonempty",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message="File is empty",
                path=str(p),
            )
        )
    elif artifact.size_bytes < cfg.min_file_size_bytes:
        checks.append(
            SimulationMetricCheck(
                name="min_file_size",
                status=SimulationMetricValidationStatus.WARNING,
                severity=SimulationMetricCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} is below minimum "
                    f"{_human_size(cfg.min_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    elif artifact.size_bytes > cfg.max_file_size_bytes:
        checks.append(
            SimulationMetricCheck(
                name="max_file_size",
                status=SimulationMetricValidationStatus.WARNING,
                severity=SimulationMetricCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} exceeds maximum "
                    f"{_human_size(cfg.max_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    else:
        checks.append(
            SimulationMetricCheck(
                name="file_size",
                status=SimulationMetricValidationStatus.PASS,
                message=f"File size is {_human_size(artifact.size_bytes)}",
                path=str(p),
            )
        )

    if "read_error" in artifact.metadata:
        checks.append(
            SimulationMetricCheck(
                name="readable",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message=f"Unable to read metric artifact: {artifact.metadata['read_error']}",
                path=str(p),
                details=artifact.metadata,
            )
        )
        return artifact, tuple(checks)

    checks.append(
        SimulationMetricCheck(
            name="readable",
            status=SimulationMetricValidationStatus.PASS,
            message="Metric metadata parsed",
            path=str(p),
        )
    )

    if "num_jsonl_errors" in artifact.metadata:
        checks.append(
            SimulationMetricCheck(
                name="jsonl_parse_errors",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message=f"JSONL contains {artifact.metadata['num_jsonl_errors']} parse error(s)",
                path=str(p),
                details=artifact.metadata,
            )
        )

    if artifact.records is not None:
        checks.append(
            SimulationMetricCheck(
                name="record_count",
                status=(
                    SimulationMetricValidationStatus.PASS
                    if artifact.records >= cfg.min_records
                    else SimulationMetricValidationStatus.WARNING
                ),
                severity=(
                    SimulationMetricCheckSeverity.INFO
                    if artifact.records >= cfg.min_records
                    else SimulationMetricCheckSeverity.WARNING
                ),
                message=(
                    f"Record count is {artifact.records}"
                    if artifact.records >= cfg.min_records
                    else f"Record count {artifact.records} is below minimum {cfg.min_records}"
                ),
                path=str(p),
            )
        )

    if cfg.require_metric_values:
        checks.append(
            SimulationMetricCheck(
                name="metric_value_count",
                status=(
                    SimulationMetricValidationStatus.PASS
                    if len(artifact.metric_names) >= cfg.min_metrics
                    else SimulationMetricValidationStatus.WARNING
                ),
                severity=(
                    SimulationMetricCheckSeverity.INFO
                    if len(artifact.metric_names) >= cfg.min_metrics
                    else SimulationMetricCheckSeverity.WARNING
                ),
                message=(
                    f"Extracted {len(artifact.metric_names)} metric value(s)"
                    if len(artifact.metric_names) >= cfg.min_metrics
                    else (
                        f"Extracted {len(artifact.metric_names)} metric value(s), "
                        f"below minimum {cfg.min_metrics}"
                    )
                ),
                path=str(p),
                details={"metric_names": list(artifact.metric_names)},
            )
        )

    if cfg.require_finite_numeric_values:
        checks.append(
            SimulationMetricCheck(
                name="finite_numeric_values",
                status=(
                    SimulationMetricValidationStatus.PASS
                    if artifact.num_nonfinite_numeric_values == 0
                    else SimulationMetricValidationStatus.FAIL
                ),
                severity=(
                    SimulationMetricCheckSeverity.INFO
                    if artifact.num_nonfinite_numeric_values == 0
                    else SimulationMetricCheckSeverity.ERROR
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

    if cfg.required_metrics:
        available = set(artifact.metric_names)
        missing = [name for name in cfg.required_metrics if name not in available]

        checks.append(
            SimulationMetricCheck(
                name="required_metrics",
                status=SimulationMetricValidationStatus.PASS if not missing else SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.INFO if not missing else SimulationMetricCheckSeverity.ERROR,
                message="All required metrics exist" if not missing else f"Missing required metrics: {missing}",
                path=str(p),
                details={
                    "required_metrics": list(cfg.required_metrics),
                    "available_metrics": list(artifact.metric_names),
                    "missing_metrics": missing,
                },
            )
        )

    for metric_name, rule in cfg.threshold_rules.items():
        if metric_name not in artifact.metric_values:
            checks.append(
                SimulationMetricCheck(
                    name="threshold_metric_exists",
                    status=SimulationMetricValidationStatus.FAIL,
                    severity=SimulationMetricCheckSeverity.ERROR,
                    message=f"Metric {metric_name!r} required by threshold rule is missing",
                    path=str(p),
                    details={"metric": metric_name, "rule": dict(rule)},
                )
            )
            continue

        value = float(artifact.metric_values[metric_name])
        ok, message = _check_threshold_rule(metric_name, value, rule)

        checks.append(
            SimulationMetricCheck(
                name="threshold_rule",
                status=SimulationMetricValidationStatus.PASS if ok else SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.INFO if ok else SimulationMetricCheckSeverity.ERROR,
                message=message,
                path=str(p),
                details={
                    "metric": metric_name,
                    "value": value,
                    "rule": dict(rule),
                },
            )
        )

    return artifact, tuple(checks)


def _validate_expected_metric_files(
    metric_dir: Path,
    expected_metric_files: Sequence[str],
) -> tuple[SimulationMetricCheck, ...]:
    """Validate expected metric files."""

    checks: list[SimulationMetricCheck] = []

    for item in expected_metric_files:
        path = Path(item)
        if not path.is_absolute():
            path = metric_dir / path

        checks.append(
            SimulationMetricCheck(
                name="expected_metric_file",
                status=SimulationMetricValidationStatus.PASS if path.exists() else SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.INFO if path.exists() else SimulationMetricCheckSeverity.ERROR,
                message="Expected metric file exists" if path.exists() else "Expected metric file is missing",
                path=str(path),
                details={"expected": item},
            )
        )

    return tuple(checks)


def _compute_report_status(
    checks: Sequence[SimulationMetricCheck],
    *,
    fail_on_warning: bool = False,
) -> SimulationMetricValidationStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return SimulationMetricValidationStatus.FAIL

    if any(check.warning for check in checks):
        return SimulationMetricValidationStatus.FAIL if fail_on_warning else SimulationMetricValidationStatus.WARNING

    return SimulationMetricValidationStatus.PASS


def validate_simulation_metric_directory(
    metric_dir: str | Path,
    config: SimulationMetricValidationConfig | Mapping[str, Any] | None = None,
) -> SimulationMetricValidationReport:
    """Validate all simulation metric artifacts in a directory."""

    cfg = (
        SimulationMetricValidationConfig(metric_dir=str(metric_dir))
        if config is None
        else config
        if isinstance(config, SimulationMetricValidationConfig)
        else SimulationMetricValidationConfig.from_dict(config)
    )

    root = Path(metric_dir)
    checks: list[SimulationMetricCheck] = []
    artifacts: list[SimulationMetricArtifact] = []

    if not root.exists():
        checks.append(
            SimulationMetricCheck(
                name="metric_dir_exists",
                status=SimulationMetricValidationStatus.FAIL,
                severity=SimulationMetricCheckSeverity.ERROR,
                message="Metric directory does not exist",
                path=str(root),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

        return SimulationMetricValidationReport(
            status=status,
            artifacts=(),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_artifacts": 0, "num_checks": len(checks)},
        )

    paths = discover_simulation_metric_files(
        root,
        recursive=cfg.recursive,
        allowed_extensions=cfg.allowed_extensions,
    )

    checks.append(
        SimulationMetricCheck(
            name="metric_discovery",
            status=SimulationMetricValidationStatus.PASS if paths else SimulationMetricValidationStatus.WARNING,
            severity=SimulationMetricCheckSeverity.INFO if paths else SimulationMetricCheckSeverity.WARNING,
            message=f"Discovered {len(paths)} metric artifact(s)",
            path=str(root),
        )
    )

    for path in paths:
        artifact, artifact_checks = validate_simulation_metric_artifact(path, cfg)
        if artifact is not None:
            artifacts.append(artifact)
        checks.extend(artifact_checks)

    checks.extend(_validate_expected_metric_files(root, cfg.expected_metric_files))

    total_size = sum(artifact.size_bytes for artifact in artifacts)
    kind_counts: dict[str, int] = {}
    all_metric_names: set[str] = set()
    all_metric_values: dict[str, list[float]] = {}

    for artifact in artifacts:
        kind_counts[artifact.kind.value] = kind_counts.get(artifact.kind.value, 0) + 1
        all_metric_names.update(artifact.metric_names)

        for name, value in artifact.metric_values.items():
            parsed = _to_float_or_none(value)
            if parsed is not None and isfinite(parsed):
                all_metric_values.setdefault(name, []).append(parsed)

    metric_summary: JSONDict = {}
    for name, values in sorted(all_metric_values.items()):
        metric_summary[name] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

    metrics = {
        "num_artifacts": len(artifacts),
        "num_checks": len(checks),
        "num_failures": sum(1 for check in checks if check.failed),
        "num_warnings": sum(1 for check in checks if check.warning),
        "total_size_bytes": total_size,
        "total_size_human": _human_size(total_size),
        "kind_counts": kind_counts,
        "num_unique_metrics": len(all_metric_names),
        "unique_metrics": sorted(all_metric_names),
        "metric_summary": metric_summary,
    }

    return SimulationMetricValidationReport(
        status=status,
        artifacts=tuple(artifacts),
        checks=tuple(checks),
        config=cfg.to_dict(),
        metrics=metrics,
    )


def validate_simulation_metrics_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    metric_dir: str | Path | None = None,
) -> SimulationMetricValidationReport:
    """Validate simulation metrics using a manifest mapping or JSON file."""

    data = _load_json(manifest) if isinstance(manifest, str | Path) else dict(manifest)

    if metric_dir is not None:
        data["metric_dir"] = str(metric_dir)

    root = data.get("metric_dir", data.get("metrics_dir", None))
    if root is None:
        raise ValueError("manifest must define metric_dir or metric_dir must be provided")

    cfg = SimulationMetricValidationConfig.from_dict(data)

    return validate_simulation_metric_directory(root, cfg)


def validate_default_project_simulation_metrics(
    metric_dir: str | Path = "results/metrics",
) -> SimulationMetricValidationReport:
    """Validate the default expected project simulation metrics."""

    expected = (
        "simulation_summary.json",
        "safety_metrics.json",
        "runtime_metrics.json",
    )

    required = (
        "success_rate",
        "violation_rate",
        "mean_runtime",
    )

    thresholds = {
        "success_rate": {"min": 0.0, "max": 1.0},
        "violation_rate": {"min": 0.0, "max": 1.0},
        "mean_runtime": {"min": 0.0},
    }

    cfg = SimulationMetricValidationConfig(
        metric_dir=str(metric_dir),
        expected_metric_files=expected,
        required_metrics=required,
        threshold_rules=thresholds,
        recursive=True,
        require_nonempty=True,
        require_finite_numeric_values=True,
        require_metric_values=True,
    )

    return validate_simulation_metric_directory(metric_dir, cfg)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.validate_simulation_metrics",
        description="Validate generated simulation metric artifacts.",
    )

    parser.add_argument("--metric-dir", default="results/metrics")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument(
        "--allowed-extensions",
        default=".json,.jsonl,.csv,.tsv,.txt",
    )
    parser.add_argument("--expected-metric-files", default=None)

    parser.add_argument("--min-file-size-bytes", type=int, default=1)
    parser.add_argument("--max-file-size-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--min-metrics", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)

    parser.add_argument("--required-metrics", default=None)
    parser.add_argument(
        "--threshold-rules",
        default=None,
        help="Rules like 'rmse:max=0.2;success_rate:min=0.95;violation_rate:max=0.01'.",
    )

    parser.add_argument("--no-require-finite-numeric-values", action="store_true")
    parser.add_argument("--no-require-metric-values", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--default-project", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.default_project:
        report = validate_default_project_simulation_metrics(args.metric_dir)

    elif args.manifest:
        report = validate_simulation_metrics_from_manifest(args.manifest, metric_dir=args.metric_dir)

    else:
        cfg = SimulationMetricValidationConfig(
            metric_dir=args.metric_dir,
            recursive=not args.non_recursive,
            allowed_extensions=_parse_csv_text(args.allowed_extensions),
            expected_metric_files=_parse_csv_text(args.expected_metric_files),
            min_file_size_bytes=args.min_file_size_bytes,
            max_file_size_bytes=args.max_file_size_bytes,
            min_metrics=args.min_metrics,
            min_records=args.min_records,
            required_metrics=_parse_csv_text(args.required_metrics),
            threshold_rules=_parse_threshold_rules(args.threshold_rules),
            require_finite_numeric_values=not args.no_require_finite_numeric_values,
            require_metric_values=not args.no_require_metric_values,
            fail_on_warning=args.fail_on_warning,
        )

        report = validate_simulation_metric_directory(args.metric_dir, cfg)

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
    "Number",
    "SimulationMetricArtifact",
    "SimulationMetricArtifactKind",
    "SimulationMetricCheck",
    "SimulationMetricCheckSeverity",
    "SimulationMetricValidationConfig",
    "SimulationMetricValidationReport",
    "SimulationMetricValidationStatus",
    "build_arg_parser",
    "collect_simulation_metric_artifact",
    "discover_simulation_metric_files",
    "extract_metric_values",
    "infer_simulation_metric_kind",
    "main",
    "read_simulation_metric_payload",
    "validate_default_project_simulation_metrics",
    "validate_simulation_metric_artifact",
    "validate_simulation_metric_directory",
    "validate_simulation_metrics_from_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
