# ==============================================================================
# src/pcsrc/validation/numerical_regression.py
#
# Numerical regression testing utilities.
#
# Purpose:
#   - Compare current numerical artifacts against baseline artifacts.
#   - Detect unexpected numerical drift in simulation, validation, plotting, or
#     table-generation pipelines.
#   - Support JSON, JSONL, CSV, TSV, and simple TXT metric files.
#   - Provide per-key and global absolute/relative tolerance checks.
#
# Design notes:
#   - Pure Python standard library.
#   - Schema-tolerant numeric extraction from nested mappings/sequences.
#   - Deterministic reports suitable for CI.
#   - Optional baseline creation/update helper.
#
# Typical usage:
#
#   from pcsrc.validation.numerical_regression import compare_artifacts
#
#   report = compare_artifacts(
#       baseline="results/baseline/metrics.json",
#       current="results/metrics/metrics.json",
#       abs_tol=1e-8,
#       rel_tol=1e-6,
#   )
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.numerical_regression \
#       --baseline results/baseline/metrics.json \
#       --current results/metrics/metrics.json \
#       --json-output results/regression_report.json
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import csv
import json
import shutil


Number = int | float
JSONDict = dict[str, Any]
NumericMap = dict[str, float]


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

def _validate_name(name: str, field_name: str = "name") -> str:
    """Validate a non-empty symbolic name."""

    parsed = str(name).strip()

    if not parsed:
        raise ValueError(f"{field_name} must be non-empty")

    return parsed


def _as_nonnegative_float(value: Any, name: str = "value") -> float:
    """Convert value to non-negative finite float."""

    converted = float(value)

    if not isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")

    return converted


def _as_nonnegative_int(value: Any, name: str = "value") -> int:
    """Convert value to non-negative integer."""

    converted = int(value)

    if converted < 0:
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


def _load_json(path: str | Path) -> Any:
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


def _normalize_extension(value: str) -> str:
    """Normalize file extension."""

    text = str(value).strip().lower()

    if not text:
        raise ValueError("extension must be non-empty")

    return text if text.startswith(".") else "." + text


def _parse_csv_text(text: str | None) -> tuple[str, ...]:
    """Parse comma-separated values."""

    if not text:
        return ()

    return tuple(item.strip() for item in text.split(",") if item.strip())


def _is_numeric_like(value: Any) -> bool:
    """Return whether value can be converted to float."""

    if value is None or isinstance(value, bool):
        return False

    try:
        float(value)
    except (TypeError, ValueError):
        return False

    return True


def _to_float(value: Any) -> float:
    """Convert numeric-like value to float."""

    return float(value)


def _path_join(prefix: str, key: str) -> str:
    """Join dotted numeric paths."""

    if not prefix:
        return str(key)

    return f"{prefix}.{key}"


def _safe_rel_error(current: float, baseline: float) -> float:
    """Compute relative error with stable denominator."""

    denom = max(abs(baseline), 1e-300)

    return abs(current - baseline) / denom


def _parse_tolerance_rules(text: str | None) -> JSONDict:
    """Parse per-key tolerance rules from CLI.

    Format:
        rmse:abs=1e-8,rel=1e-6;success_rate:abs=1e-12
    """

    if not text:
        return {}

    rules: JSONDict = {}

    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece:
            continue

        if ":" not in piece:
            raise ValueError("tolerance rule must be formatted as key:abs=...,rel=...")

        key, rule_text = piece.split(":", 1)
        key = key.strip()

        if not key:
            raise ValueError("tolerance rule key must be non-empty")

        rule: JSONDict = {}
        for item in rule_text.split(","):
            item = item.strip()
            if not item:
                continue

            if "=" not in item:
                raise ValueError("tolerance rule item must be key=value")

            rule_key, raw_value = item.split("=", 1)
            rule_key = rule_key.strip()

            if rule_key not in {"abs", "rel", "abs_tol", "rel_tol"}:
                raise ValueError(f"unsupported tolerance key: {rule_key}")

            normalized_key = "abs_tol" if rule_key in {"abs", "abs_tol"} else "rel_tol"
            rule[normalized_key] = _as_nonnegative_float(raw_value, normalized_key)

        rules[key] = rule

    return rules


# ------------------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------------------

class NumericalRegressionStatus(str, Enum):
    """Regression status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class NumericalRegressionSeverity(str, Enum):
    """Check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class NumericalArtifactKind(str, Enum):
    """Supported artifact kind."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    TEXT = "text"
    UNKNOWN = "unknown"


_NUMERIC_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
}


# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NumericalRegressionConfig:
    """Configuration for numerical regression checks."""

    name: str = "numerical_regression"

    abs_tol: float = 1e-8
    rel_tol: float = 1e-6

    missing_key_is_failure: bool = True
    extra_key_is_failure: bool = False
    require_finite: bool = True

    tolerance_rules: JSONDict = field(default_factory=dict)
    ignored_keys: tuple[str, ...] = ()

    min_common_values: int = 1
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed_rules: JSONDict = {}

        for key, rule in self.tolerance_rules.items():
            if not isinstance(rule, Mapping):
                raise ValueError("tolerance rule must be a mapping")

            parsed_rule: JSONDict = {}

            if "abs" in rule:
                parsed_rule["abs_tol"] = _as_nonnegative_float(rule["abs"], f"tolerance_rules[{key}].abs")
            if "rel" in rule:
                parsed_rule["rel_tol"] = _as_nonnegative_float(rule["rel"], f"tolerance_rules[{key}].rel")
            if "abs_tol" in rule:
                parsed_rule["abs_tol"] = _as_nonnegative_float(rule["abs_tol"], f"tolerance_rules[{key}].abs_tol")
            if "rel_tol" in rule:
                parsed_rule["rel_tol"] = _as_nonnegative_float(rule["rel_tol"], f"tolerance_rules[{key}].rel_tol")

            parsed_rules[str(key)] = parsed_rule

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "abs_tol", _as_nonnegative_float(self.abs_tol, "abs_tol"))
        object.__setattr__(self, "rel_tol", _as_nonnegative_float(self.rel_tol, "rel_tol"))
        object.__setattr__(self, "missing_key_is_failure", bool(self.missing_key_is_failure))
        object.__setattr__(self, "extra_key_is_failure", bool(self.extra_key_is_failure))
        object.__setattr__(self, "require_finite", bool(self.require_finite))
        object.__setattr__(self, "tolerance_rules", parsed_rules)
        object.__setattr__(self, "ignored_keys", tuple(str(item) for item in self.ignored_keys))
        object.__setattr__(self, "min_common_values", _as_nonnegative_int(self.min_common_values, "min_common_values"))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NumericalRegressionConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "numerical_regression")),
            abs_tol=float(data.get("abs_tol", data.get("absolute_tolerance", 1e-8))),
            rel_tol=float(data.get("rel_tol", data.get("relative_tolerance", 1e-6))),
            missing_key_is_failure=bool(data.get("missing_key_is_failure", True)),
            extra_key_is_failure=bool(data.get("extra_key_is_failure", False)),
            require_finite=bool(data.get("require_finite", True)),
            tolerance_rules=dict(data.get("tolerance_rules", {})),
            ignored_keys=tuple(data.get("ignored_keys", ())),
            min_common_values=int(data.get("min_common_values", 1)),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "missing_key_is_failure": self.missing_key_is_failure,
            "extra_key_is_failure": self.extra_key_is_failure,
            "require_finite": self.require_finite,
            "tolerance_rules": dict(self.tolerance_rules),
            "ignored_keys": list(self.ignored_keys),
            "min_common_values": self.min_common_values,
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NumericalDifference:
    """One numeric value comparison."""

    key: str
    baseline: float
    current: float
    abs_error: float
    rel_error: float
    abs_tol: float
    rel_tol: float
    passed: bool

    def to_dict(self) -> JSONDict:
        """Serialize difference."""

        return {
            "key": self.key,
            "baseline": self.baseline,
            "current": self.current,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class NumericalRegressionCheck:
    """One regression check."""

    name: str
    status: NumericalRegressionStatus
    severity: NumericalRegressionSeverity = NumericalRegressionSeverity.INFO
    message: str = ""
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, NumericalRegressionStatus)
            else NumericalRegressionStatus(self.status)
        )
        severity = (
            self.severity
            if isinstance(self.severity, NumericalRegressionSeverity)
            else NumericalRegressionSeverity(self.severity)
        )

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "details", dict(_json_safe(self.details)))

    @property
    def passed(self) -> bool:
        """Whether the check passed."""

        return self.status is NumericalRegressionStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check failed."""

        return self.status is NumericalRegressionStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check warned."""

        return self.status is NumericalRegressionStatus.WARNING

    def to_dict(self) -> JSONDict:
        """Serialize check."""

        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class NumericalRegressionReport:
    """Complete numerical regression report."""

    name: str = "numerical_regression_report"
    status: NumericalRegressionStatus = NumericalRegressionStatus.PASS

    baseline_path: str | None = None
    current_path: str | None = None

    checks: tuple[NumericalRegressionCheck, ...] = ()
    differences: tuple[NumericalDifference, ...] = ()

    missing_keys: tuple[str, ...] = ()
    extra_keys: tuple[str, ...] = ()

    config: JSONDict = field(default_factory=dict)
    metrics: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, NumericalRegressionStatus)
            else NumericalRegressionStatus(self.status)
        )

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "baseline_path", None if self.baseline_path is None else str(self.baseline_path))
        object.__setattr__(self, "current_path", None if self.current_path is None else str(self.current_path))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "differences", tuple(self.differences))
        object.__setattr__(self, "missing_keys", tuple(str(item) for item in self.missing_keys))
        object.__setattr__(self, "extra_keys", tuple(str(item) for item in self.extra_keys))
        object.__setattr__(self, "config", dict(_json_safe(self.config)))
        object.__setattr__(self, "metrics", dict(_json_safe(self.metrics)))

    @property
    def passed(self) -> bool:
        """Whether report passed."""

        return self.status is NumericalRegressionStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is NumericalRegressionStatus.FAIL

    @property
    def failures(self) -> tuple[NumericalRegressionCheck, ...]:
        """Return failed checks."""

        return tuple(check for check in self.checks if check.failed)

    @property
    def warnings(self) -> tuple[NumericalRegressionCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    def to_dict(self) -> JSONDict:
        """Serialize report."""

        return {
            "name": self.name,
            "status": self.status.value,
            "baseline_path": self.baseline_path,
            "current_path": self.current_path,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
            "missing_keys": list(self.missing_keys),
            "extra_keys": list(self.extra_keys),
            "differences": [item.to_dict() for item in self.differences],
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self, *, max_items: int = 20) -> str:
        """Render report as human-readable text."""

        failed_diffs = [item for item in self.differences if not item.passed]

        lines = [
            f"Numerical regression report: {self.status.value.upper()}",
            f"Baseline: {self.baseline_path}",
            f"Current: {self.current_path}",
            f"Compared values: {self.metrics.get('num_common_values', 0)}",
            f"Failed values: {len(failed_diffs)}",
            f"Missing keys: {len(self.missing_keys)}",
            f"Extra keys: {len(self.extra_keys)}",
            "",
        ]

        if failed_diffs:
            lines.append("Failed differences:")
            for diff in failed_diffs[:max_items]:
                lines.append(
                    "  - "
                    f"{diff.key}: baseline={diff.baseline:.12g}, "
                    f"current={diff.current:.12g}, "
                    f"abs_err={diff.abs_error:.3g}, "
                    f"rel_err={diff.rel_error:.3g}, "
                    f"abs_tol={diff.abs_tol:.3g}, "
                    f"rel_tol={diff.rel_tol:.3g}"
                )

            if len(failed_diffs) > max_items:
                lines.append(f"  ... {len(failed_diffs) - max_items} more")
            lines.append("")

        if self.missing_keys:
            lines.append("Missing keys:")
            for key in self.missing_keys[:max_items]:
                lines.append(f"  - {key}")
            if len(self.missing_keys) > max_items:
                lines.append(f"  ... {len(self.missing_keys) - max_items} more")
            lines.append("")

        if self.extra_keys:
            lines.append("Extra keys:")
            for key in self.extra_keys[:max_items]:
                lines.append(f"  - {key}")
            if len(self.extra_keys) > max_items:
                lines.append(f"  ... {len(self.extra_keys) - max_items} more")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------------
# Artifact reading and numeric extraction
# ------------------------------------------------------------------------------

def infer_numerical_artifact_kind(path: str | Path) -> NumericalArtifactKind:
    """Infer artifact kind by extension."""

    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        return NumericalArtifactKind.JSON

    if suffix == ".jsonl":
        return NumericalArtifactKind.JSONL

    if suffix == ".csv":
        return NumericalArtifactKind.CSV

    if suffix == ".tsv":
        return NumericalArtifactKind.TSV

    if suffix == ".txt":
        return NumericalArtifactKind.TEXT

    return NumericalArtifactKind.UNKNOWN


def read_numerical_artifact(path: str | Path) -> Any:
    """Read numerical artifact from supported file format."""

    p = Path(path)
    kind = infer_numerical_artifact_kind(p)

    if kind is NumericalArtifactKind.JSON:
        return json.loads(p.read_text(encoding="utf-8"))

    if kind is NumericalArtifactKind.JSONL:
        records: list[Any] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records

    if kind in {NumericalArtifactKind.CSV, NumericalArtifactKind.TSV}:
        delimiter = "," if kind is NumericalArtifactKind.CSV else "\t"
        with p.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for row in reader]

    if kind is NumericalArtifactKind.TEXT:
        payload: JSONDict = {}
        lines = p.read_text(encoding="utf-8").splitlines()

        for idx, line in enumerate(lines):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            separator = ":" if ":" in stripped else "=" if "=" in stripped else None

            if separator is None:
                numbers = [
                    float(token)
                    for token in stripped.replace(",", " ").split()
                    if _is_numeric_like(token)
                ]
                for number_idx, number in enumerate(numbers):
                    payload[f"line_{idx}.value_{number_idx}"] = number
                continue

            key, value = stripped.split(separator, 1)
            key = key.strip()
            value = value.strip()

            if _is_numeric_like(value):
                payload[key] = float(value)

        return payload

    raise ValueError(f"unsupported numerical artifact extension: {p.suffix}")


def extract_numeric_map(
    payload: Any,
    *,
    prefix: str = "",
    include_nonfinite: bool = True,
) -> NumericMap:
    """Extract numeric values from nested payload as dotted-path mapping."""

    values: NumericMap = {}

    if payload is None or isinstance(payload, bool):
        return values

    if isinstance(payload, int | float):
        number = float(payload)
        if include_nonfinite or isfinite(number):
            values[prefix or "value"] = number
        return values

    if isinstance(payload, str):
        if _is_numeric_like(payload):
            number = float(payload)
            if include_nonfinite or isfinite(number):
                values[prefix or "value"] = number
        return values

    if hasattr(payload, "tolist") and callable(payload.tolist):
        return extract_numeric_map(payload.tolist(), prefix=prefix, include_nonfinite=include_nonfinite)

    if isinstance(payload, Mapping):
        for key, item in payload.items():
            values.update(
                extract_numeric_map(
                    item,
                    prefix=_path_join(prefix, str(key)),
                    include_nonfinite=include_nonfinite,
                )
            )
        return values

    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        for idx, item in enumerate(payload):
            values.update(
                extract_numeric_map(
                    item,
                    prefix=_path_join(prefix, str(idx)),
                    include_nonfinite=include_nonfinite,
                )
            )
        return values

    return values


def load_numeric_map(path: str | Path, *, include_nonfinite: bool = True) -> NumericMap:
    """Load artifact and extract numeric map."""

    return extract_numeric_map(
        read_numerical_artifact(path),
        include_nonfinite=include_nonfinite,
    )


# ------------------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------------------

def _is_ignored(key: str, ignored_keys: Sequence[str]) -> bool:
    """Return whether key should be ignored.

    Exact match or prefix match with trailing wildcard are supported:
        "metadata.seed"
        "timing.*"
    """

    for pattern in ignored_keys:
        if pattern.endswith("*"):
            if key.startswith(pattern[:-1]):
                return True
        elif key == pattern:
            return True

    return False


def _tolerance_for_key(key: str, config: NumericalRegressionConfig) -> tuple[float, float]:
    """Resolve tolerance for one key."""

    abs_tol = config.abs_tol
    rel_tol = config.rel_tol

    for pattern, rule in config.tolerance_rules.items():
        matched = key == pattern or (pattern.endswith("*") and key.startswith(pattern[:-1]))

        if matched:
            abs_tol = float(rule.get("abs_tol", abs_tol))
            rel_tol = float(rule.get("rel_tol", rel_tol))

    return abs_tol, rel_tol


def compare_numeric_maps(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    config: NumericalRegressionConfig | Mapping[str, Any] | None = None,
) -> NumericalRegressionReport:
    """Compare two numeric maps."""

    cfg = (
        NumericalRegressionConfig()
        if config is None
        else config
        if isinstance(config, NumericalRegressionConfig)
        else NumericalRegressionConfig.from_dict(config)
    )

    baseline_filtered = {
        key: float(value)
        for key, value in baseline.items()
        if not _is_ignored(key, cfg.ignored_keys)
    }
    current_filtered = {
        key: float(value)
        for key, value in current.items()
        if not _is_ignored(key, cfg.ignored_keys)
    }

    baseline_keys = set(baseline_filtered)
    current_keys = set(current_filtered)

    common_keys = sorted(baseline_keys & current_keys)
    missing_keys = tuple(sorted(baseline_keys - current_keys))
    extra_keys = tuple(sorted(current_keys - baseline_keys))

    checks: list[NumericalRegressionCheck] = []
    differences: list[NumericalDifference] = []

    if len(common_keys) < cfg.min_common_values:
        checks.append(
            NumericalRegressionCheck(
                name="min_common_values",
                status=NumericalRegressionStatus.FAIL,
                severity=NumericalRegressionSeverity.ERROR,
                message=(
                    f"Only {len(common_keys)} common numeric value(s), "
                    f"below minimum {cfg.min_common_values}"
                ),
                details={"num_common_values": len(common_keys), "min_common_values": cfg.min_common_values},
            )
        )
    else:
        checks.append(
            NumericalRegressionCheck(
                name="min_common_values",
                status=NumericalRegressionStatus.PASS,
                message=f"Found {len(common_keys)} common numeric value(s)",
                details={"num_common_values": len(common_keys)},
            )
        )

    if missing_keys:
        checks.append(
            NumericalRegressionCheck(
                name="missing_keys",
                status=NumericalRegressionStatus.FAIL if cfg.missing_key_is_failure else NumericalRegressionStatus.WARNING,
                severity=(
                    NumericalRegressionSeverity.ERROR
                    if cfg.missing_key_is_failure
                    else NumericalRegressionSeverity.WARNING
                ),
                message=f"{len(missing_keys)} baseline key(s) missing from current output",
                details={"missing_keys": list(missing_keys)},
            )
        )
    else:
        checks.append(
            NumericalRegressionCheck(
                name="missing_keys",
                status=NumericalRegressionStatus.PASS,
                message="No missing keys",
            )
        )

    if extra_keys:
        checks.append(
            NumericalRegressionCheck(
                name="extra_keys",
                status=NumericalRegressionStatus.FAIL if cfg.extra_key_is_failure else NumericalRegressionStatus.WARNING,
                severity=(
                    NumericalRegressionSeverity.ERROR
                    if cfg.extra_key_is_failure
                    else NumericalRegressionSeverity.WARNING
                ),
                message=f"{len(extra_keys)} extra key(s) found in current output",
                details={"extra_keys": list(extra_keys)},
            )
        )
    else:
        checks.append(
            NumericalRegressionCheck(
                name="extra_keys",
                status=NumericalRegressionStatus.PASS,
                message="No extra keys",
            )
        )

    nonfinite_keys: list[str] = []

    for key in common_keys:
        b = baseline_filtered[key]
        c = current_filtered[key]

        if cfg.require_finite and (not isfinite(b) or not isfinite(c)):
            nonfinite_keys.append(key)

        abs_error = abs(c - b) if isfinite(c) and isfinite(b) else float("inf")
        rel_error = _safe_rel_error(c, b) if isfinite(c) and isfinite(b) else float("inf")
        abs_tol, rel_tol = _tolerance_for_key(key, cfg)

        passed = (
            isfinite(abs_error)
            and isfinite(rel_error)
            and (abs_error <= abs_tol or rel_error <= rel_tol)
        )

        differences.append(
            NumericalDifference(
                key=key,
                baseline=b,
                current=c,
                abs_error=abs_error,
                rel_error=rel_error,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                passed=passed,
            )
        )

    if cfg.require_finite:
        checks.append(
            NumericalRegressionCheck(
                name="finite_values",
                status=NumericalRegressionStatus.PASS if not nonfinite_keys else NumericalRegressionStatus.FAIL,
                severity=NumericalRegressionSeverity.INFO if not nonfinite_keys else NumericalRegressionSeverity.ERROR,
                message=(
                    "All compared values are finite"
                    if not nonfinite_keys
                    else f"{len(nonfinite_keys)} compared key(s) contain non-finite values"
                ),
                details={"nonfinite_keys": nonfinite_keys},
            )
        )

    failed_differences = [item for item in differences if not item.passed]

    checks.append(
        NumericalRegressionCheck(
            name="tolerance",
            status=NumericalRegressionStatus.PASS if not failed_differences else NumericalRegressionStatus.FAIL,
            severity=NumericalRegressionSeverity.INFO if not failed_differences else NumericalRegressionSeverity.ERROR,
            message=(
                "All common values satisfy tolerance"
                if not failed_differences
                else f"{len(failed_differences)} value(s) exceed tolerance"
            ),
            details={"num_failed_differences": len(failed_differences)},
        )
    )

    abs_errors = [item.abs_error for item in differences if isfinite(item.abs_error)]
    rel_errors = [item.rel_error for item in differences if isfinite(item.rel_error)]

    metrics = {
        "num_baseline_values": len(baseline_filtered),
        "num_current_values": len(current_filtered),
        "num_common_values": len(common_keys),
        "num_missing_keys": len(missing_keys),
        "num_extra_keys": len(extra_keys),
        "num_failed_differences": len(failed_differences),
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "max_rel_error": max(rel_errors) if rel_errors else None,
        "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "rmse": sqrt(sum(value * value for value in abs_errors) / len(abs_errors)) if abs_errors else None,
    }

    status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

    return NumericalRegressionReport(
        status=status,
        checks=tuple(checks),
        differences=tuple(differences),
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        config=cfg.to_dict(),
        metrics=metrics,
    )


def _compute_report_status(
    checks: Sequence[NumericalRegressionCheck],
    *,
    fail_on_warning: bool = False,
) -> NumericalRegressionStatus:
    """Compute aggregate report status."""

    if any(check.failed for check in checks):
        return NumericalRegressionStatus.FAIL

    if any(check.warning for check in checks):
        return NumericalRegressionStatus.FAIL if fail_on_warning else NumericalRegressionStatus.WARNING

    return NumericalRegressionStatus.PASS


def compare_artifacts(
    baseline: str | Path,
    current: str | Path,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    config: NumericalRegressionConfig | Mapping[str, Any] | None = None,
    ignored_keys: Sequence[str] = (),
    tolerance_rules: Mapping[str, Any] | None = None,
) -> NumericalRegressionReport:
    """Compare two artifact files."""

    if config is None:
        cfg = NumericalRegressionConfig(
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            ignored_keys=tuple(ignored_keys),
            tolerance_rules=dict(tolerance_rules or {}),
        )
    elif isinstance(config, NumericalRegressionConfig):
        cfg = config
    else:
        cfg = NumericalRegressionConfig.from_dict(config)

    baseline_path = Path(baseline)
    current_path = Path(current)

    checks: list[NumericalRegressionCheck] = []

    if not baseline_path.exists():
        checks.append(
            NumericalRegressionCheck(
                name="baseline_exists",
                status=NumericalRegressionStatus.FAIL,
                severity=NumericalRegressionSeverity.ERROR,
                message="Baseline artifact does not exist",
                details={"baseline": str(baseline_path)},
            )
        )

    if not current_path.exists():
        checks.append(
            NumericalRegressionCheck(
                name="current_exists",
                status=NumericalRegressionStatus.FAIL,
                severity=NumericalRegressionSeverity.ERROR,
                message="Current artifact does not exist",
                details={"current": str(current_path)},
            )
        )

    if checks:
        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)
        return NumericalRegressionReport(
            status=status,
            baseline_path=str(baseline_path),
            current_path=str(current_path),
            checks=tuple(checks),
            config=cfg.to_dict(),
            metrics={"num_common_values": 0},
        )

    baseline_map = load_numeric_map(baseline_path, include_nonfinite=True)
    current_map = load_numeric_map(current_path, include_nonfinite=True)

    report = compare_numeric_maps(baseline_map, current_map, cfg)

    return NumericalRegressionReport(
        status=report.status,
        baseline_path=str(baseline_path),
        current_path=str(current_path),
        checks=report.checks,
        differences=report.differences,
        missing_keys=report.missing_keys,
        extra_keys=report.extra_keys,
        config=report.config,
        metrics=report.metrics,
    )


def compare_artifact_directories(
    baseline_dir: str | Path,
    current_dir: str | Path,
    *,
    allowed_extensions: Sequence[str] = tuple(_NUMERIC_EXTENSIONS),
    config: NumericalRegressionConfig | Mapping[str, Any] | None = None,
) -> NumericalRegressionReport:
    """Compare matching numerical artifacts in two directories.

    Files are matched by relative path.
    """

    cfg = (
        NumericalRegressionConfig()
        if config is None
        else config
        if isinstance(config, NumericalRegressionConfig)
        else NumericalRegressionConfig.from_dict(config)
    )

    base_root = Path(baseline_dir)
    curr_root = Path(current_dir)

    allowed = {_normalize_extension(item) for item in allowed_extensions}

    checks: list[NumericalRegressionCheck] = []
    all_diffs: list[NumericalDifference] = []
    missing_files: list[str] = []
    extra_files: list[str] = []

    if not base_root.exists():
        return NumericalRegressionReport(
            status=NumericalRegressionStatus.FAIL,
            baseline_path=str(base_root),
            current_path=str(curr_root),
            checks=(
                NumericalRegressionCheck(
                    name="baseline_dir_exists",
                    status=NumericalRegressionStatus.FAIL,
                    severity=NumericalRegressionSeverity.ERROR,
                    message="Baseline directory does not exist",
                ),
            ),
            config=cfg.to_dict(),
        )

    if not curr_root.exists():
        return NumericalRegressionReport(
            status=NumericalRegressionStatus.FAIL,
            baseline_path=str(base_root),
            current_path=str(curr_root),
            checks=(
                NumericalRegressionCheck(
                    name="current_dir_exists",
                    status=NumericalRegressionStatus.FAIL,
                    severity=NumericalRegressionSeverity.ERROR,
                    message="Current directory does not exist",
                ),
            ),
            config=cfg.to_dict(),
        )

    baseline_files = {
        path.relative_to(base_root).as_posix(): path
        for path in base_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    }
    current_files = {
        path.relative_to(curr_root).as_posix(): path
        for path in curr_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    }

    for rel in sorted(set(baseline_files) - set(current_files)):
        missing_files.append(rel)

    for rel in sorted(set(current_files) - set(baseline_files)):
        extra_files.append(rel)

    for rel in sorted(set(baseline_files) & set(current_files)):
        sub_report = compare_artifacts(
            baseline_files[rel],
            current_files[rel],
            config=cfg,
        )

        for diff in sub_report.differences:
            all_diffs.append(
                NumericalDifference(
                    key=f"{rel}:{diff.key}",
                    baseline=diff.baseline,
                    current=diff.current,
                    abs_error=diff.abs_error,
                    rel_error=diff.rel_error,
                    abs_tol=diff.abs_tol,
                    rel_tol=diff.rel_tol,
                    passed=diff.passed,
                )
            )

    if missing_files:
        checks.append(
            NumericalRegressionCheck(
                name="missing_files",
                status=NumericalRegressionStatus.FAIL if cfg.missing_key_is_failure else NumericalRegressionStatus.WARNING,
                severity=(
                    NumericalRegressionSeverity.ERROR
                    if cfg.missing_key_is_failure
                    else NumericalRegressionSeverity.WARNING
                ),
                message=f"{len(missing_files)} baseline file(s) missing from current directory",
                details={"missing_files": missing_files},
            )
        )
    else:
        checks.append(
            NumericalRegressionCheck(
                name="missing_files",
                status=NumericalRegressionStatus.PASS,
                message="No missing files",
            )
        )

    if extra_files:
        checks.append(
            NumericalRegressionCheck(
                name="extra_files",
                status=NumericalRegressionStatus.FAIL if cfg.extra_key_is_failure else NumericalRegressionStatus.WARNING,
                severity=(
                    NumericalRegressionSeverity.ERROR
                    if cfg.extra_key_is_failure
                    else NumericalRegressionSeverity.WARNING
                ),
                message=f"{len(extra_files)} extra file(s) found in current directory",
                details={"extra_files": extra_files},
            )
        )
    else:
        checks.append(
            NumericalRegressionCheck(
                name="extra_files",
                status=NumericalRegressionStatus.PASS,
                message="No extra files",
            )
        )

    failed_diffs = [item for item in all_diffs if not item.passed]
    abs_errors = [item.abs_error for item in all_diffs if isfinite(item.abs_error)]
    rel_errors = [item.rel_error for item in all_diffs if isfinite(item.rel_error)]

    checks.append(
        NumericalRegressionCheck(
            name="directory_tolerance",
            status=NumericalRegressionStatus.PASS if not failed_diffs else NumericalRegressionStatus.FAIL,
            severity=NumericalRegressionSeverity.INFO if not failed_diffs else NumericalRegressionSeverity.ERROR,
            message=(
                "All matched directory values satisfy tolerance"
                if not failed_diffs
                else f"{len(failed_diffs)} value(s) exceed tolerance"
            ),
        )
    )

    status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

    return NumericalRegressionReport(
        status=status,
        baseline_path=str(base_root),
        current_path=str(curr_root),
        checks=tuple(checks),
        differences=tuple(all_diffs),
        missing_keys=tuple(missing_files),
        extra_keys=tuple(extra_files),
        config=cfg.to_dict(),
        metrics={
            "num_baseline_files": len(baseline_files),
            "num_current_files": len(current_files),
            "num_common_files": len(set(baseline_files) & set(current_files)),
            "num_common_values": len(all_diffs),
            "num_failed_differences": len(failed_diffs),
            "max_abs_error": max(abs_errors) if abs_errors else None,
            "max_rel_error": max(rel_errors) if rel_errors else None,
            "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
            "rmse": sqrt(sum(value * value for value in abs_errors) / len(abs_errors)) if abs_errors else None,
        },
    )


def update_baseline(
    current: str | Path,
    baseline: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Copy current artifact or directory to baseline location."""

    current_path = Path(current)
    baseline_path = Path(baseline)

    if not current_path.exists():
        raise FileNotFoundError(f"current path does not exist: {current_path}")

    if baseline_path.exists() and not overwrite:
        raise FileExistsError(f"baseline path already exists: {baseline_path}")

    if current_path.is_dir():
        if baseline_path.exists():
            shutil.rmtree(baseline_path)
        shutil.copytree(current_path, baseline_path)
    else:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, baseline_path)

    return baseline_path


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.numerical_regression",
        description="Compare numerical artifacts against baselines.",
    )

    parser.add_argument("--baseline", required=True, help="Baseline artifact or directory.")
    parser.add_argument("--current", required=True, help="Current artifact or directory.")

    parser.add_argument("--directory", action="store_true", help="Compare directories recursively.")
    parser.add_argument(
        "--allowed-extensions",
        default=".json,.jsonl,.csv,.tsv,.txt",
        help="Comma-separated extensions for directory comparison.",
    )

    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--tolerance-rules", default=None)
    parser.add_argument("--ignored-keys", default=None)

    parser.add_argument("--missing-key-warning", action="store_true")
    parser.add_argument("--extra-key-failure", action="store_true")
    parser.add_argument("--no-require-finite", action="store_true")
    parser.add_argument("--min-common-values", type=int, default=1)
    parser.add_argument("--fail-on-warning", action="store_true")

    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text-output", default=None)

    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.update_baseline:
        path = update_baseline(
            current=args.current,
            baseline=args.baseline,
            overwrite=not args.no_overwrite,
        )
        print(json.dumps({"updated_baseline": str(path)}, ensure_ascii=False, indent=2))
        return 0

    cfg = NumericalRegressionConfig(
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        tolerance_rules=_parse_tolerance_rules(args.tolerance_rules),
        ignored_keys=_parse_csv_text(args.ignored_keys),
        missing_key_is_failure=not args.missing_key_warning,
        extra_key_is_failure=args.extra_key_failure,
        require_finite=not args.no_require_finite,
        min_common_values=args.min_common_values,
        fail_on_warning=args.fail_on_warning,
    )

    if args.directory:
        report = compare_artifact_directories(
            baseline_dir=args.baseline,
            current_dir=args.current,
            allowed_extensions=_parse_csv_text(args.allowed_extensions),
            config=cfg,
        )
    else:
        report = compare_artifacts(
            baseline=args.baseline,
            current=args.current,
            config=cfg,
        )

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
    "NumericMap",
    "NumericalArtifactKind",
    "NumericalDifference",
    "NumericalRegressionCheck",
    "NumericalRegressionConfig",
    "NumericalRegressionReport",
    "NumericalRegressionSeverity",
    "NumericalRegressionStatus",
    "build_arg_parser",
    "compare_artifact_directories",
    "compare_artifacts",
    "compare_numeric_maps",
    "extract_numeric_map",
    "infer_numerical_artifact_kind",
    "load_numeric_map",
    "main",
    "read_numerical_artifact",
    "update_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
