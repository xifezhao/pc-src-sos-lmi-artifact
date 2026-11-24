# ==============================================================================
# src/pcsrc/validation/validate_tables.py
#
# Table artifact validation utilities.
#
# Purpose:
#   - Validate generated table files and related metadata.
#   - Check existence, file size, extension, row/column counts, required columns,
#     numeric field validity, and optional sidecar JSON files.
#   - Support CSV, TSV, JSON, JSONL, and LaTeX table artifacts.
#   - Provide CLI entry point for CI / reproducibility workflows.
#
# Design notes:
#   - Pure Python standard library.
#   - Conservative validation: structural problems fail, optional sidecars warn.
#   - Pairs naturally with validate_figures.py for publication artifact checks.
#
# Typical usage:
#
#   from pcsrc.validation.validate_tables import validate_table_directory
#
#   report = validate_table_directory("results/tables")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.validate_tables \
#       --table-dir results/tables \
#       --manifest results/tables/manifest.json \
#       --json-output results/tables/validation_report.json
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


def _is_finite_number_text(value: Any) -> bool:
    """Return whether value can be parsed as a finite number."""

    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    try:
        parsed = float(text)
    except ValueError:
        return False

    return isfinite(parsed)


def _looks_like_numeric_column(values: Sequence[Any], *, threshold: float = 0.75) -> bool:
    """Heuristically decide whether a column should be numeric."""

    nonempty = [value for value in values if str(value).strip()]

    if not nonempty:
        return False

    numeric = sum(1 for value in nonempty if _is_finite_number_text(value))

    return numeric / len(nonempty) >= threshold


# ------------------------------------------------------------------------------
# Enums and dataclasses
# ------------------------------------------------------------------------------

class TableValidationStatus(str, Enum):
    """Validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class TableCheckSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TableArtifactKind(str, Enum):
    """Supported table artifact kinds."""

    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"
    LATEX = "latex"
    TEXT = "text"
    UNKNOWN = "unknown"


_TABLE_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".tex", ".txt"}


@dataclass(frozen=True, slots=True)
class TableValidationConfig:
    """Configuration for table validation."""

    name: str = "table_validation"

    table_dir: str | None = None
    recursive: bool = True

    allowed_extensions: tuple[str, ...] = (
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".tex",
    )

    min_file_size_bytes: int = 1
    max_file_size_bytes: int = 50 * 1024 * 1024

    min_rows: int = 1
    max_rows: int = 1_000_000
    min_columns: int = 1
    max_columns: int = 10_000

    required_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()

    require_nonempty: bool = True
    require_sidecar_json: bool = False

    expected_tables: tuple[str, ...] = ()
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "table_dir", None if self.table_dir is None else str(self.table_dir))
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
        object.__setattr__(self, "min_rows", _as_nonnegative_int(self.min_rows, "min_rows"))
        object.__setattr__(self, "max_rows", _as_positive_int(self.max_rows, "max_rows"))
        object.__setattr__(self, "min_columns", _as_nonnegative_int(self.min_columns, "min_columns"))
        object.__setattr__(self, "max_columns", _as_positive_int(self.max_columns, "max_columns"))
        object.__setattr__(self, "required_columns", tuple(str(item) for item in self.required_columns))
        object.__setattr__(self, "numeric_columns", tuple(str(item) for item in self.numeric_columns))
        object.__setattr__(self, "require_nonempty", bool(self.require_nonempty))
        object.__setattr__(self, "require_sidecar_json", bool(self.require_sidecar_json))
        object.__setattr__(self, "expected_tables", tuple(str(item) for item in self.expected_tables))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableValidationConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "table_validation")),
            table_dir=data.get("table_dir", None),
            recursive=bool(data.get("recursive", True)),
            allowed_extensions=tuple(data.get("allowed_extensions", (
                ".csv",
                ".tsv",
                ".json",
                ".jsonl",
                ".tex",
            ))),
            min_file_size_bytes=int(data.get("min_file_size_bytes", 1)),
            max_file_size_bytes=int(data.get("max_file_size_bytes", 50 * 1024 * 1024)),
            min_rows=int(data.get("min_rows", 1)),
            max_rows=int(data.get("max_rows", 1_000_000)),
            min_columns=int(data.get("min_columns", 1)),
            max_columns=int(data.get("max_columns", 10_000)),
            required_columns=tuple(data.get("required_columns", ())),
            numeric_columns=tuple(data.get("numeric_columns", ())),
            require_nonempty=bool(data.get("require_nonempty", True)),
            require_sidecar_json=bool(data.get("require_sidecar_json", False)),
            expected_tables=tuple(data.get("expected_tables", ())),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "table_dir": self.table_dir,
            "recursive": self.recursive,
            "allowed_extensions": list(self.allowed_extensions),
            "min_file_size_bytes": self.min_file_size_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_columns": self.min_columns,
            "max_columns": self.max_columns,
            "required_columns": list(self.required_columns),
            "numeric_columns": list(self.numeric_columns),
            "require_nonempty": self.require_nonempty,
            "require_sidecar_json": self.require_sidecar_json,
            "expected_tables": list(self.expected_tables),
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TableCheck:
    """One validation check result."""

    name: str
    status: TableValidationStatus
    severity: TableCheckSeverity = TableCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, TableValidationStatus)
            else TableValidationStatus(self.status)
        )
        severity = (
            self.severity
            if isinstance(self.severity, TableCheckSeverity)
            else TableCheckSeverity(self.severity)
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

        return self.status is TableValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check is failing."""

        return self.status is TableValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check is warning."""

        return self.status is TableValidationStatus.WARNING

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
class TableArtifact:
    """Table artifact metadata."""

    path: str
    kind: TableArtifactKind
    size_bytes: int
    extension: str
    rows: int | None = None
    columns: int | None = None
    column_names: tuple[str, ...] = ()
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, TableArtifactKind) else TableArtifactKind(self.kind)

        object.__setattr__(self, "path", str(self.path))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "extension", str(self.extension).lower())
        object.__setattr__(self, "rows", None if self.rows is None else _as_nonnegative_int(self.rows, "rows"))
        object.__setattr__(self, "columns", None if self.columns is None else _as_nonnegative_int(self.columns, "columns"))
        object.__setattr__(self, "column_names", tuple(str(item) for item in self.column_names))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    def to_dict(self) -> JSONDict:
        """Serialize artifact."""

        return {
            "path": self.path,
            "kind": self.kind.value,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "extension": self.extension,
            "rows": self.rows,
            "columns": self.columns,
            "column_names": list(self.column_names),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TableValidationReport:
    """Complete table validation report."""

    name: str = "table_validation_report"
    status: TableValidationStatus = TableValidationStatus.PASS
    artifacts: tuple[TableArtifact, ...] = ()
    checks: tuple[TableCheck, ...] = ()
    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, TableValidationStatus)
            else TableValidationStatus(self.status)
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

        return self.status is TableValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is TableValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[TableCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[TableCheck, ...]:
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
            f"Table validation report: {self.status.value.upper()}",
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
# Artifact discovery and table reading
# ------------------------------------------------------------------------------

def infer_table_kind(path: str | Path) -> TableArtifactKind:
    """Infer table artifact kind from extension."""

    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return TableArtifactKind.CSV

    if suffix == ".tsv":
        return TableArtifactKind.TSV

    if suffix == ".json":
        return TableArtifactKind.JSON

    if suffix == ".jsonl":
        return TableArtifactKind.JSONL

    if suffix == ".tex":
        return TableArtifactKind.LATEX

    if suffix == ".txt":
        return TableArtifactKind.TEXT

    return TableArtifactKind.UNKNOWN


def discover_table_files(
    table_dir: str | Path,
    *,
    recursive: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Discover table artifact files under a directory."""

    root = Path(table_dir)

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


def _read_delimited_table(path: Path, delimiter: str) -> tuple[int, int, tuple[str, ...], JSONDict]:
    """Read CSV/TSV metadata."""

    metadata: JSONDict = {}
    rows = 0
    columns = 0
    column_names: tuple[str, ...] = ()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        all_rows = list(reader)

    if not all_rows:
        return 0, 0, (), metadata

    column_names = tuple(str(item) for item in all_rows[0])
    columns = len(column_names)
    rows = max(0, len(all_rows) - 1)

    if rows:
        body = all_rows[1:]
        widths = [len(row) for row in body]
        metadata["row_widths"] = {
            "min": min(widths),
            "max": max(widths),
        }

        column_values: dict[str, list[Any]] = {
            name: []
            for name in column_names
        }

        for row in body:
            for idx, name in enumerate(column_names):
                column_values[name].append(row[idx] if idx < len(row) else "")

        numeric_like = [
            name
            for name, values in column_values.items()
            if _looks_like_numeric_column(values)
        ]
        metadata["numeric_like_columns"] = numeric_like

    return rows, columns, column_names, metadata


def _read_json_table(path: Path) -> tuple[int, int, tuple[str, ...], JSONDict]:
    """Read JSON table metadata."""

    data = json.loads(path.read_text(encoding="utf-8"))
    metadata: JSONDict = {"json_type": type(data).__name__}

    records: list[Any]

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            records = data["rows"]
        elif isinstance(data.get("records"), list):
            records = data["records"]
        elif isinstance(data.get("data"), list):
            records = data["data"]
        else:
            records = [data]
    else:
        records = []

    rows = len(records)
    column_names: tuple[str, ...] = ()

    if records and all(isinstance(item, Mapping) for item in records):
        keys: list[str] = []
        seen: set[str] = set()

        for record in records:
            for key in record.keys():
                key_text = str(key)
                if key_text not in seen:
                    seen.add(key_text)
                    keys.append(key_text)

        column_names = tuple(keys)
        columns = len(column_names)
    elif records and all(isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray) for item in records):
        columns = max(len(item) for item in records)
        column_names = tuple(f"col_{idx}" for idx in range(columns))
    else:
        columns = 0

    return rows, columns, column_names, metadata


def _read_jsonl_table(path: Path) -> tuple[int, int, tuple[str, ...], JSONDict]:
    """Read JSONL table metadata."""

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

    metadata: JSONDict = {}
    if errors:
        metadata["jsonl_errors"] = errors[:20]
        metadata["num_jsonl_errors"] = len(errors)

    rows = len(records)

    if records and all(isinstance(item, Mapping) for item in records):
        keys: list[str] = []
        seen: set[str] = set()

        for record in records:
            for key in record.keys():
                key_text = str(key)
                if key_text not in seen:
                    seen.add(key_text)
                    keys.append(key_text)

        column_names = tuple(keys)
        columns = len(column_names)
    else:
        column_names = ()
        columns = 0

    return rows, columns, column_names, metadata


def _read_latex_table(path: Path) -> tuple[int, int, tuple[str, ...], JSONDict]:
    """Read approximate LaTeX table metadata."""

    text = path.read_text(encoding="utf-8")
    metadata: JSONDict = {
        "contains_tabular": "\\begin{tabular" in text,
        "contains_booktabs": any(token in text for token in ("\\toprule", "\\midrule", "\\bottomrule")),
    }

    candidate_rows = [
        line
        for line in text.splitlines()
        if "&" in line and "\\\\" in line
    ]

    rows = len(candidate_rows)

    columns = 0
    if candidate_rows:
        columns = max(line.count("&") + 1 for line in candidate_rows)

    column_names = tuple(f"col_{idx}" for idx in range(columns))

    return rows, columns, column_names, metadata


def collect_table_artifact(path: str | Path) -> TableArtifact:
    """Collect metadata for one table artifact."""

    p = Path(path)
    kind = infer_table_kind(p)
    size = p.stat().st_size if p.exists() else 0
    rows: int | None = None
    columns: int | None = None
    column_names: tuple[str, ...] = ()
    metadata: JSONDict = {}

    if p.exists() and p.is_file() and size > 0:
        try:
            if kind is TableArtifactKind.CSV:
                rows, columns, column_names, metadata = _read_delimited_table(p, ",")
            elif kind is TableArtifactKind.TSV:
                rows, columns, column_names, metadata = _read_delimited_table(p, "\t")
            elif kind is TableArtifactKind.JSON:
                rows, columns, column_names, metadata = _read_json_table(p)
            elif kind is TableArtifactKind.JSONL:
                rows, columns, column_names, metadata = _read_jsonl_table(p)
            elif kind is TableArtifactKind.LATEX:
                rows, columns, column_names, metadata = _read_latex_table(p)
            elif kind is TableArtifactKind.TEXT:
                lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
                rows = len(lines)
                columns = None
                metadata = {"nonempty_lines": len(lines)}
        except Exception as exc:
            metadata = {
                "read_error": str(exc),
                "read_error_type": type(exc).__name__,
            }

    return TableArtifact(
        path=str(p),
        kind=kind,
        size_bytes=size,
        extension=p.suffix.lower(),
        rows=rows,
        columns=columns,
        column_names=column_names,
        metadata=metadata,
    )


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_table_artifact(
    path: str | Path,
    config: TableValidationConfig | Mapping[str, Any] | None = None,
) -> tuple[TableArtifact | None, tuple[TableCheck, ...]]:
    """Validate one table artifact."""

    cfg = (
        TableValidationConfig()
        if config is None
        else config
        if isinstance(config, TableValidationConfig)
        else TableValidationConfig.from_dict(config)
    )

    p = Path(path)
    checks: list[TableCheck] = []

    if not p.exists():
        checks.append(
            TableCheck(
                name="exists",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message="File does not exist",
                path=str(p),
            )
        )
        return None, tuple(checks)

    if not p.is_file():
        checks.append(
            TableCheck(
                name="is_file",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message="Path is not a file",
                path=str(p),
            )
        )
        return None, tuple(checks)

    artifact = collect_table_artifact(p)

    if artifact.extension not in cfg.allowed_extensions:
        checks.append(
            TableCheck(
                name="allowed_extension",
                status=TableValidationStatus.WARNING,
                severity=TableCheckSeverity.WARNING,
                message=f"Extension {artifact.extension!r} is not in allowed extensions",
                path=str(p),
                details={"allowed_extensions": list(cfg.allowed_extensions)},
            )
        )
    else:
        checks.append(
            TableCheck(
                name="allowed_extension",
                status=TableValidationStatus.PASS,
                message="Extension is allowed",
                path=str(p),
            )
        )

    if cfg.require_nonempty and artifact.size_bytes <= 0:
        checks.append(
            TableCheck(
                name="nonempty",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message="File is empty",
                path=str(p),
            )
        )
    elif artifact.size_bytes < cfg.min_file_size_bytes:
        checks.append(
            TableCheck(
                name="min_file_size",
                status=TableValidationStatus.WARNING,
                severity=TableCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} is below minimum "
                    f"{_human_size(cfg.min_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    elif artifact.size_bytes > cfg.max_file_size_bytes:
        checks.append(
            TableCheck(
                name="max_file_size",
                status=TableValidationStatus.WARNING,
                severity=TableCheckSeverity.WARNING,
                message=(
                    f"File size {_human_size(artifact.size_bytes)} exceeds maximum "
                    f"{_human_size(cfg.max_file_size_bytes)}"
                ),
                path=str(p),
            )
        )
    else:
        checks.append(
            TableCheck(
                name="file_size",
                status=TableValidationStatus.PASS,
                message=f"File size is {_human_size(artifact.size_bytes)}",
                path=str(p),
            )
        )

    if "read_error" in artifact.metadata:
        checks.append(
            TableCheck(
                name="readable",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message=f"Unable to read table: {artifact.metadata['read_error']}",
                path=str(p),
                details=artifact.metadata,
            )
        )
    else:
        checks.append(
            TableCheck(
                name="readable",
                status=TableValidationStatus.PASS,
                message="Table metadata parsed",
                path=str(p),
            )
        )

    if "num_jsonl_errors" in artifact.metadata:
        checks.append(
            TableCheck(
                name="jsonl_parse_errors",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message=f"JSONL contains {artifact.metadata['num_jsonl_errors']} parse error(s)",
                path=str(p),
                details=artifact.metadata,
            )
        )

    if artifact.rows is not None:
        if artifact.rows < cfg.min_rows:
            checks.append(
                TableCheck(
                    name="min_rows",
                    status=TableValidationStatus.WARNING,
                    severity=TableCheckSeverity.WARNING,
                    message=f"Row count {artifact.rows} is below minimum {cfg.min_rows}",
                    path=str(p),
                )
            )
        elif artifact.rows > cfg.max_rows:
            checks.append(
                TableCheck(
                    name="max_rows",
                    status=TableValidationStatus.WARNING,
                    severity=TableCheckSeverity.WARNING,
                    message=f"Row count {artifact.rows} exceeds maximum {cfg.max_rows}",
                    path=str(p),
                )
            )
        else:
            checks.append(
                TableCheck(
                    name="row_count",
                    status=TableValidationStatus.PASS,
                    message=f"Row count is {artifact.rows}",
                    path=str(p),
                )
            )

    if artifact.columns is not None:
        if artifact.columns < cfg.min_columns:
            checks.append(
                TableCheck(
                    name="min_columns",
                    status=TableValidationStatus.WARNING,
                    severity=TableCheckSeverity.WARNING,
                    message=f"Column count {artifact.columns} is below minimum {cfg.min_columns}",
                    path=str(p),
                )
            )
        elif artifact.columns > cfg.max_columns:
            checks.append(
                TableCheck(
                    name="max_columns",
                    status=TableValidationStatus.WARNING,
                    severity=TableCheckSeverity.WARNING,
                    message=f"Column count {artifact.columns} exceeds maximum {cfg.max_columns}",
                    path=str(p),
                )
            )
        else:
            checks.append(
                TableCheck(
                    name="column_count",
                    status=TableValidationStatus.PASS,
                    message=f"Column count is {artifact.columns}",
                    path=str(p),
                )
            )

    if cfg.required_columns and artifact.column_names:
        available = set(artifact.column_names)
        missing = [name for name in cfg.required_columns if name not in available]

        checks.append(
            TableCheck(
                name="required_columns",
                status=TableValidationStatus.PASS if not missing else TableValidationStatus.FAIL,
                severity=TableCheckSeverity.INFO if not missing else TableCheckSeverity.ERROR,
                message="All required columns exist" if not missing else f"Missing required columns: {missing}",
                path=str(p),
                details={
                    "required_columns": list(cfg.required_columns),
                    "available_columns": list(artifact.column_names),
                    "missing_columns": missing,
                },
            )
        )

    if cfg.numeric_columns and artifact.kind in {TableArtifactKind.CSV, TableArtifactKind.TSV}:
        numeric_checks = _validate_numeric_columns_for_delimited_file(p, artifact, cfg.numeric_columns)

        checks.extend(numeric_checks)

    if cfg.require_sidecar_json and artifact.kind is not TableArtifactKind.JSON:
        sidecar = p.with_suffix(".json")
        checks.append(
            TableCheck(
                name="sidecar_json",
                status=TableValidationStatus.PASS if sidecar.exists() else TableValidationStatus.WARNING,
                severity=TableCheckSeverity.INFO if sidecar.exists() else TableCheckSeverity.WARNING,
                message="Sidecar JSON exists" if sidecar.exists() else "Sidecar JSON is missing",
                path=str(sidecar),
            )
        )

    return artifact, tuple(checks)


def _validate_numeric_columns_for_delimited_file(
    path: Path,
    artifact: TableArtifact,
    numeric_columns: Sequence[str],
) -> tuple[TableCheck, ...]:
    """Validate finite numeric values for selected CSV/TSV columns."""

    delimiter = "," if artifact.kind is TableArtifactKind.CSV else "\t"
    checks: list[TableCheck] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        records = list(reader)

    available = set(reader.fieldnames or ())

    for column in numeric_columns:
        if column not in available:
            checks.append(
                TableCheck(
                    name="numeric_column_exists",
                    status=TableValidationStatus.FAIL,
                    severity=TableCheckSeverity.ERROR,
                    message=f"Numeric column {column!r} is missing",
                    path=str(path),
                    details={"column": column},
                )
            )
            continue

        bad_rows: list[int] = []
        for row_idx, row in enumerate(records, start=1):
            if not _is_finite_number_text(row.get(column, "")):
                bad_rows.append(row_idx)

        checks.append(
            TableCheck(
                name="numeric_column_finite",
                status=TableValidationStatus.PASS if not bad_rows else TableValidationStatus.FAIL,
                severity=TableCheckSeverity.INFO if not bad_rows else TableCheckSeverity.ERROR,
                message=(
                    f"Column {column!r} contains finite numeric values"
                    if not bad_rows
                    else f"Column {column!r} has {len(bad_rows)} non-finite/non-numeric value(s)"
                ),
                path=str(path),
                details={
                    "column": column,
                    "bad_rows": bad_rows[:50],
                    "num_bad_rows": len(bad_rows),
                },
            )
        )

    return tuple(checks)


def _validate_expected_tables(
    table_dir: Path,
    expected_tables: Sequence[str],
) -> tuple[TableCheck, ...]:
    """Validate expected table files."""

    checks: list[TableCheck] = []

    for item in expected_tables:
        path = Path(item)
        if not path.is_absolute():
            path = table_dir / path

        checks.append(
            TableCheck(
                name="expected_table",
                status=TableValidationStatus.PASS if path.exists() else TableValidationStatus.FAIL,
                severity=TableCheckSeverity.INFO if path.exists() else TableCheckSeverity.ERROR,
                message="Expected table exists" if path.exists() else "Expected table is missing",
                path=str(path),
                details={"expected": item},
            )
        )

    return tuple(checks)


def _compute_report_status(
    checks: Sequence[TableCheck],
    *,
    fail_on_warning: bool = False,
) -> TableValidationStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return TableValidationStatus.FAIL

    if any(check.warning for check in checks):
        return TableValidationStatus.FAIL if fail_on_warning else TableValidationStatus.WARNING

    return TableValidationStatus.PASS


def validate_table_directory(
    table_dir: str | Path,
    config: TableValidationConfig | Mapping[str, Any] | None = None,
) -> TableValidationReport:
    """Validate all table artifacts in a directory."""

    cfg = (
        TableValidationConfig(table_dir=str(table_dir))
        if config is None
        else config
        if isinstance(config, TableValidationConfig)
        else TableValidationConfig.from_dict(config)
    )

    root = Path(table_dir)
    checks: list[TableCheck] = []
    artifacts: list[TableArtifact] = []

    if not root.exists():
        checks.append(
            TableCheck(
                name="table_dir_exists",
                status=TableValidationStatus.FAIL,
                severity=TableCheckSeverity.ERROR,
                message="Table directory does not exist",
                path=str(root),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)
        return TableValidationReport(
            status=status,
            artifacts=(),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_artifacts": 0, "num_checks": len(checks)},
        )

    paths = discover_table_files(
        root,
        recursive=cfg.recursive,
        allowed_extensions=cfg.allowed_extensions,
    )

    checks.append(
        TableCheck(
            name="table_discovery",
            status=TableValidationStatus.PASS if paths else TableValidationStatus.WARNING,
            severity=TableCheckSeverity.INFO if paths else TableCheckSeverity.WARNING,
            message=f"Discovered {len(paths)} artifact(s)",
            path=str(root),
        )
    )

    for path in paths:
        artifact, artifact_checks = validate_table_artifact(path, cfg)
        if artifact is not None:
            artifacts.append(artifact)
        checks.extend(artifact_checks)

    checks.extend(_validate_expected_tables(root, cfg.expected_tables))

    total_size = sum(artifact.size_bytes for artifact in artifacts)
    kind_counts: dict[str, int] = {}
    row_counts = [artifact.rows for artifact in artifacts if artifact.rows is not None]
    column_counts = [artifact.columns for artifact in artifacts if artifact.columns is not None]

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
        "row_count_min": min(row_counts) if row_counts else None,
        "row_count_max": max(row_counts) if row_counts else None,
        "column_count_min": min(column_counts) if column_counts else None,
        "column_count_max": max(column_counts) if column_counts else None,
    }

    return TableValidationReport(
        status=status,
        artifacts=tuple(artifacts),
        checks=tuple(checks),
        config=cfg.to_dict(),
        metrics=metrics,
    )


def validate_tables_from_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    table_dir: str | Path | None = None,
) -> TableValidationReport:
    """Validate tables using a manifest mapping or JSON file."""

    data = _load_json(manifest) if isinstance(manifest, str | Path) else dict(manifest)

    if table_dir is not None:
        data["table_dir"] = str(table_dir)

    root = data.get("table_dir", None)
    if root is None:
        raise ValueError("manifest must define table_dir or table_dir must be provided")

    cfg = TableValidationConfig.from_dict(data)

    return validate_table_directory(root, cfg)


def validate_default_project_tables(
    table_dir: str | Path = "results/tables",
) -> TableValidationReport:
    """Validate the default expected project table set."""

    expected = (
        "tab1_parameters.csv",
        "tab2_performance_summary.csv",
        "tab3_ablation_results.csv",
        "tab4_runtime_breakdown.csv",
    )

    cfg = TableValidationConfig(
        table_dir=str(table_dir),
        expected_tables=expected,
        recursive=True,
        require_nonempty=True,
        require_sidecar_json=False,
    )

    return validate_table_directory(table_dir, cfg)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.validate_tables",
        description="Validate generated table artifacts.",
    )

    parser.add_argument("--table-dir", default="results/tables")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument(
        "--allowed-extensions",
        default=".csv,.tsv,.json,.jsonl,.tex",
    )
    parser.add_argument("--expected-tables", default=None)

    parser.add_argument("--min-file-size-bytes", type=int, default=1)
    parser.add_argument("--max-file-size-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--min-columns", type=int, default=1)
    parser.add_argument("--max-columns", type=int, default=10_000)

    parser.add_argument("--required-columns", default=None)
    parser.add_argument("--numeric-columns", default=None)

    parser.add_argument("--require-sidecar-json", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--default-project", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.default_project:
        report = validate_default_project_tables(args.table_dir)

    elif args.manifest:
        report = validate_tables_from_manifest(args.manifest, table_dir=args.table_dir)

    else:
        cfg = TableValidationConfig(
            table_dir=args.table_dir,
            recursive=not args.non_recursive,
            allowed_extensions=_parse_csv_text(args.allowed_extensions),
            expected_tables=_parse_csv_text(args.expected_tables),
            min_file_size_bytes=args.min_file_size_bytes,
            max_file_size_bytes=args.max_file_size_bytes,
            min_rows=args.min_rows,
            max_rows=args.max_rows,
            min_columns=args.min_columns,
            max_columns=args.max_columns,
            required_columns=_parse_csv_text(args.required_columns),
            numeric_columns=_parse_csv_text(args.numeric_columns),
            require_sidecar_json=args.require_sidecar_json,
            fail_on_warning=args.fail_on_warning,
        )
        report = validate_table_directory(args.table_dir, cfg)

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
    "TableArtifact",
    "TableArtifactKind",
    "TableCheck",
    "TableCheckSeverity",
    "TableValidationConfig",
    "TableValidationReport",
    "TableValidationStatus",
    "build_arg_parser",
    "collect_table_artifact",
    "discover_table_files",
    "infer_table_kind",
    "main",
    "validate_default_project_tables",
    "validate_table_artifact",
    "validate_table_directory",
    "validate_tables_from_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
