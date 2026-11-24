# ==============================================================================
# src/pcsrc/validation/hash_artifacts.py
#
# Artifact hashing and integrity validation utilities.
#
# Purpose:
#   - Compute deterministic hashes for experiment artifacts.
#   - Generate JSON manifests for reproducibility and CI.
#   - Verify current artifacts against saved hash manifests.
#   - Support files and recursive directory hashing.
#
# Design notes:
#   - Pure Python standard library.
#   - Uses streaming reads for large files.
#   - Manifest paths are stored relative to a selected root where possible.
#   - Default algorithm is SHA-256.
#
# Typical usage:
#
#   from pcsrc.validation.hash_artifacts import create_hash_manifest, verify_hash_manifest
#
#   manifest = create_hash_manifest("results", output="results/hash_manifest.json")
#   report = verify_hash_manifest("results/hash_manifest.json")
#   print(report.to_text())
#
# CLI:
#
#   python -m pcsrc.validation.hash_artifacts \
#       --root results \
#       --output results/hash_manifest.json
#
#   python -m pcsrc.validation.hash_artifacts \
#       --verify results/hash_manifest.json
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import algorithms_available, new as new_hash
from math import isfinite
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import argparse
import fnmatch
import json
import os
import platform
import sys
import time


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


def _as_positive_int(value: Any, name: str = "value") -> int:
    """Convert value to positive integer."""

    converted = int(value)

    if converted <= 0:
        raise ValueError(f"{name} must be positive, got {converted}")

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


def _load_json(path: str | Path) -> JSONDict:
    """Load JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> Path:
    """Write JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=indent, sort_keys=True),
        encoding="utf-8",
    )

    return output_path


def _human_size(num_bytes: int) -> str:
    """Return human-readable file size."""

    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{num_bytes} B"


def _normalize_algorithm(algorithm: str) -> str:
    """Normalize and validate hash algorithm name."""

    parsed = str(algorithm).strip().lower().replace("-", "")

    aliases = {
        "sha256": "sha256",
        "sha512": "sha512",
        "sha1": "sha1",
        "md5": "md5",
        "blake2b": "blake2b",
        "blake2s": "blake2s",
    }

    parsed = aliases.get(parsed, parsed)

    if parsed not in algorithms_available:
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}")

    return parsed


def _safe_relative_path(path: Path, root: Path) -> str:
    """Return POSIX relative path if possible, otherwise absolute path."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _path_matches_any(path_text: str, patterns: Sequence[str]) -> bool:
    """Return whether path text matches any glob pattern."""

    normalized = path_text.replace("\\", "/")

    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _parse_csv_text(text: str | None) -> tuple[str, ...]:
    """Parse comma-separated CLI values."""

    if not text:
        return ()

    return tuple(item.strip() for item in text.split(",") if item.strip())


def _default_exclude_patterns() -> tuple[str, ...]:
    """Return default ignored path patterns."""

    return (
        ".git/*",
        "**/.git/*",
        "__pycache__/*",
        "**/__pycache__/*",
        "*.pyc",
        "*.pyo",
        ".DS_Store",
        "**/.DS_Store",
        "*.tmp",
        "*.temp",
        "*.log",
    )


# ------------------------------------------------------------------------------
# Enums and dataclasses
# ------------------------------------------------------------------------------

class HashValidationStatus(str, Enum):
    """Hash validation status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class HashCheckSeverity(str, Enum):
    """Hash check severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ArtifactHashKind(str, Enum):
    """Artifact kind."""

    FILE = "file"
    DIRECTORY = "directory"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HashConfig:
    """Configuration for hashing artifacts."""

    name: str = "artifact_hashing"

    algorithm: str = "sha256"
    chunk_size: int = 1024 * 1024
    recursive: bool = True

    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = field(default_factory=_default_exclude_patterns)

    follow_symlinks: bool = False
    include_metadata: bool = True
    fail_on_warning: bool = False

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "algorithm", _normalize_algorithm(self.algorithm))
        object.__setattr__(self, "chunk_size", _as_positive_int(self.chunk_size, "chunk_size"))
        object.__setattr__(self, "recursive", bool(self.recursive))
        object.__setattr__(self, "include_patterns", tuple(str(item) for item in self.include_patterns))
        object.__setattr__(self, "exclude_patterns", tuple(str(item) for item in self.exclude_patterns))
        object.__setattr__(self, "follow_symlinks", bool(self.follow_symlinks))
        object.__setattr__(self, "include_metadata", bool(self.include_metadata))
        object.__setattr__(self, "fail_on_warning", bool(self.fail_on_warning))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HashConfig":
        """Create config from mapping."""

        return cls(
            name=str(data.get("name", "artifact_hashing")),
            algorithm=str(data.get("algorithm", "sha256")),
            chunk_size=int(data.get("chunk_size", 1024 * 1024)),
            recursive=bool(data.get("recursive", True)),
            include_patterns=tuple(data.get("include_patterns", ())),
            exclude_patterns=tuple(data.get("exclude_patterns", _default_exclude_patterns())),
            follow_symlinks=bool(data.get("follow_symlinks", False)),
            include_metadata=bool(data.get("include_metadata", True)),
            fail_on_warning=bool(data.get("fail_on_warning", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize config."""

        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "chunk_size": self.chunk_size,
            "recursive": self.recursive,
            "include_patterns": list(self.include_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "follow_symlinks": self.follow_symlinks,
            "include_metadata": self.include_metadata,
            "fail_on_warning": self.fail_on_warning,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ArtifactHash:
    """Hash entry for one artifact."""

    path: str
    digest: str
    algorithm: str = "sha256"
    kind: ArtifactHashKind = ArtifactHashKind.FILE

    size_bytes: int = 0
    mtime_ns: int | None = None
    mode: int | None = None

    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ArtifactHashKind) else ArtifactHashKind(self.kind)

        object.__setattr__(self, "path", str(self.path).replace("\\", "/"))
        object.__setattr__(self, "digest", str(self.digest))
        object.__setattr__(self, "algorithm", _normalize_algorithm(self.algorithm))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size_bytes", _as_nonnegative_int(self.size_bytes, "size_bytes"))
        object.__setattr__(self, "mtime_ns", None if self.mtime_ns is None else int(self.mtime_ns))
        object.__setattr__(self, "mode", None if self.mode is None else int(self.mode))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactHash":
        """Create artifact hash from mapping."""

        return cls(
            path=str(data["path"]),
            digest=str(data["digest"]),
            algorithm=str(data.get("algorithm", "sha256")),
            kind=ArtifactHashKind(data.get("kind", ArtifactHashKind.FILE.value)),
            size_bytes=int(data.get("size_bytes", 0)),
            mtime_ns=None if data.get("mtime_ns", None) is None else int(data["mtime_ns"]),
            mode=None if data.get("mode", None) is None else int(data["mode"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize artifact hash."""

        return {
            "path": self.path,
            "digest": self.digest,
            "algorithm": self.algorithm,
            "kind": self.kind.value,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "mtime_ns": self.mtime_ns,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HashManifest:
    """Hash manifest document."""

    root: str
    entries: tuple[ArtifactHash, ...]

    name: str = "hash_manifest"
    created_at_unix: float = field(default_factory=time.time)
    config: JSONDict = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        parsed_entries = tuple(
            entry if isinstance(entry, ArtifactHash) else ArtifactHash.from_dict(entry)
            for entry in self.entries
        )

        object.__setattr__(self, "root", str(self.root))
        object.__setattr__(self, "entries", parsed_entries)
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "created_at_unix", float(self.created_at_unix))
        object.__setattr__(self, "config", dict(_json_safe(self.config)))
        object.__setattr__(self, "metadata", dict(_json_safe(self.metadata)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HashManifest":
        """Create manifest from mapping."""

        return cls(
            root=str(data.get("root", ".")),
            entries=tuple(ArtifactHash.from_dict(item) for item in data.get("entries", ())),
            name=str(data.get("name", "hash_manifest")),
            created_at_unix=float(data.get("created_at_unix", time.time())),
            config=dict(data.get("config", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        """Serialize manifest."""

        return {
            "name": self.name,
            "root": self.root,
            "created_at_unix": self.created_at_unix,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def by_path(self) -> dict[str, ArtifactHash]:
        """Return entry map by path."""

        return {entry.path: entry for entry in self.entries}


@dataclass(frozen=True, slots=True)
class HashCheck:
    """One hash verification check."""

    name: str
    status: HashValidationStatus
    severity: HashCheckSeverity = HashCheckSeverity.INFO
    message: str = ""
    path: str | None = None
    details: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, HashValidationStatus) else HashValidationStatus(self.status)
        severity = self.severity if isinstance(self.severity, HashCheckSeverity) else HashCheckSeverity(self.severity)

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "path", None if self.path is None else str(self.path))
        object.__setattr__(self, "details", dict(_json_safe(self.details)))

    @property
    def passed(self) -> bool:
        """Whether the check passed."""

        return self.status is HashValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether the check failed."""

        return self.status is HashValidationStatus.FAIL

    @property
    def warning(self) -> bool:
        """Whether the check emitted a warning."""

        return self.status is HashValidationStatus.WARNING

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
class HashVerificationReport:
    """Complete hash verification report."""

    name: str = "hash_verification_report"
    status: HashValidationStatus = HashValidationStatus.PASS

    manifest_path: str | None = None
    root: str | None = None

    checks: tuple[HashCheck, ...] = ()
    expected_entries: tuple[ArtifactHash, ...] = ()
    current_entries: tuple[ArtifactHash, ...] = ()

    missing_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    extra_paths: tuple[str, ...] = ()

    metrics: JSONDict = field(default_factory=dict)
    config: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, HashValidationStatus) else HashValidationStatus(self.status)

        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "manifest_path", None if self.manifest_path is None else str(self.manifest_path))
        object.__setattr__(self, "root", None if self.root is None else str(self.root))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "expected_entries", tuple(self.expected_entries))
        object.__setattr__(self, "current_entries", tuple(self.current_entries))
        object.__setattr__(self, "missing_paths", tuple(str(item) for item in self.missing_paths))
        object.__setattr__(self, "changed_paths", tuple(str(item) for item in self.changed_paths))
        object.__setattr__(self, "extra_paths", tuple(str(item) for item in self.extra_paths))
        object.__setattr__(self, "metrics", dict(_json_safe(self.metrics)))
        object.__setattr__(self, "config", dict(_json_safe(self.config)))

    @property
    def passed(self) -> bool:
        """Whether report passed."""

        return self.status is HashValidationStatus.PASS

    @property
    def failed(self) -> bool:
        """Whether report failed."""

        return self.status is HashValidationStatus.FAIL

    @property
    def warnings(self) -> tuple[HashCheck, ...]:
        """Return warning checks."""

        return tuple(check for check in self.checks if check.warning)

    @property
    def failures(self) -> tuple[HashCheck, ...]:
        """Return failed checks."""

        return tuple(check for check in self.checks if check.failed)

    def to_dict(self) -> JSONDict:
        """Serialize report."""

        return {
            "name": self.name,
            "status": self.status.value,
            "manifest_path": self.manifest_path,
            "root": self.root,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
            "missing_paths": list(self.missing_paths),
            "changed_paths": list(self.changed_paths),
            "extra_paths": list(self.extra_paths),
            "expected_entries": [entry.to_dict() for entry in self.expected_entries],
            "current_entries": [entry.to_dict() for entry in self.current_entries],
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_text(self, *, max_items: int = 30) -> str:
        """Render report as human-readable text."""

        lines = [
            f"Hash verification report: {self.status.value.upper()}",
            f"Manifest: {self.manifest_path}",
            f"Root: {self.root}",
            f"Expected entries: {len(self.expected_entries)}",
            f"Current entries: {len(self.current_entries)}",
            f"Missing: {len(self.missing_paths)}",
            f"Changed: {len(self.changed_paths)}",
            f"Extra: {len(self.extra_paths)}",
            "",
        ]

        if self.missing_paths:
            lines.append("Missing paths:")
            for item in self.missing_paths[:max_items]:
                lines.append(f"  - {item}")
            if len(self.missing_paths) > max_items:
                lines.append(f"  ... {len(self.missing_paths) - max_items} more")
            lines.append("")

        if self.changed_paths:
            lines.append("Changed paths:")
            for item in self.changed_paths[:max_items]:
                lines.append(f"  - {item}")
            if len(self.changed_paths) > max_items:
                lines.append(f"  ... {len(self.changed_paths) - max_items} more")
            lines.append("")

        if self.extra_paths:
            lines.append("Extra paths:")
            for item in self.extra_paths[:max_items]:
                lines.append(f"  - {item}")
            if len(self.extra_paths) > max_items:
                lines.append(f"  ... {len(self.extra_paths) - max_items} more")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------------------------
# Hash computation
# ------------------------------------------------------------------------------

def hash_stream(
    stream: BinaryIO,
    *,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash a binary stream."""

    parsed_algorithm = _normalize_algorithm(algorithm)
    parsed_chunk_size = _as_positive_int(chunk_size, "chunk_size")

    hasher = new_hash(parsed_algorithm)

    while True:
        chunk = stream.read(parsed_chunk_size)
        if not chunk:
            break
        hasher.update(chunk)

    return hasher.hexdigest()


def hash_file(
    path: str | Path,
    *,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash one file."""

    p = Path(path)

    with p.open("rb") as handle:
        return hash_stream(handle, algorithm=algorithm, chunk_size=chunk_size)


def collect_artifact_hash(
    path: str | Path,
    *,
    root: str | Path | None = None,
    config: HashConfig | Mapping[str, Any] | None = None,
) -> ArtifactHash:
    """Collect hash entry for one file."""

    cfg = (
        HashConfig()
        if config is None
        else config
        if isinstance(config, HashConfig)
        else HashConfig.from_dict(config)
    )

    p = Path(path)
    root_path = Path(root) if root is not None else p.parent

    if not p.exists():
        return ArtifactHash(
            path=_safe_relative_path(p, root_path),
            digest="",
            algorithm=cfg.algorithm,
            kind=ArtifactHashKind.MISSING,
        )

    if p.is_dir():
        return ArtifactHash(
            path=_safe_relative_path(p, root_path),
            digest="",
            algorithm=cfg.algorithm,
            kind=ArtifactHashKind.DIRECTORY,
        )

    stat = p.stat()
    digest = hash_file(p, algorithm=cfg.algorithm, chunk_size=cfg.chunk_size)

    metadata: JSONDict = {}
    if cfg.include_metadata:
        metadata = {
            "absolute_path": str(p.resolve()),
            "suffix": p.suffix.lower(),
        }

    return ArtifactHash(
        path=_safe_relative_path(p, root_path),
        digest=digest,
        algorithm=cfg.algorithm,
        kind=ArtifactHashKind.FILE,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        mode=stat.st_mode,
        metadata=metadata,
    )


def discover_artifacts(
    root: str | Path,
    *,
    recursive: bool = True,
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
    follow_symlinks: bool = False,
) -> tuple[Path, ...]:
    """Discover files under root according to include/exclude patterns."""

    root_path = Path(root)

    if not root_path.exists():
        return ()

    if root_path.is_file():
        return (root_path,)

    pattern = "**/*" if recursive else "*"
    paths: list[Path] = []

    for path in root_path.glob(pattern):
        if path.is_symlink() and not follow_symlinks:
            continue

        if not path.is_file():
            continue

        rel = _safe_relative_path(path, root_path)

        if include_patterns and not _path_matches_any(rel, include_patterns):
            continue

        if exclude_patterns and _path_matches_any(rel, exclude_patterns):
            continue

        paths.append(path)

    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def compute_directory_hash(
    root: str | Path,
    *,
    algorithm: str = "sha256",
    recursive: bool = True,
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
    follow_symlinks: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute deterministic aggregate hash for a directory.

    The aggregate hash is computed from relative paths and file digests.
    """

    cfg = HashConfig(
        algorithm=algorithm,
        recursive=recursive,
        include_patterns=tuple(include_patterns),
        exclude_patterns=tuple(exclude_patterns),
        follow_symlinks=follow_symlinks,
        chunk_size=chunk_size,
    )

    root_path = Path(root)
    entries = create_hash_entries(root_path, config=cfg)

    hasher = new_hash(cfg.algorithm)

    for entry in entries:
        line = f"{entry.path}\0{entry.digest}\0{entry.size_bytes}\n"
        hasher.update(line.encode("utf-8"))

    return hasher.hexdigest()


def create_hash_entries(
    root: str | Path,
    *,
    config: HashConfig | Mapping[str, Any] | None = None,
) -> tuple[ArtifactHash, ...]:
    """Create hash entries for a file or directory."""

    cfg = (
        HashConfig()
        if config is None
        else config
        if isinstance(config, HashConfig)
        else HashConfig.from_dict(config)
    )

    root_path = Path(root)
    paths = discover_artifacts(
        root_path,
        recursive=cfg.recursive,
        include_patterns=cfg.include_patterns,
        exclude_patterns=cfg.exclude_patterns,
        follow_symlinks=cfg.follow_symlinks,
    )

    return tuple(
        collect_artifact_hash(path, root=root_path if root_path.is_dir() else root_path.parent, config=cfg)
        for path in paths
    )


def create_hash_manifest(
    root: str | Path,
    *,
    output: str | Path | None = None,
    config: HashConfig | Mapping[str, Any] | None = None,
    name: str = "hash_manifest",
    metadata: Mapping[str, Any] | None = None,
) -> HashManifest:
    """Create hash manifest for a root path, optionally writing JSON output."""

    cfg = (
        HashConfig()
        if config is None
        else config
        if isinstance(config, HashConfig)
        else HashConfig.from_dict(config)
    )

    root_path = Path(root)
    entries = create_hash_entries(root_path, config=cfg)

    manifest_metadata: JSONDict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    manifest_metadata.update(dict(metadata or {}))

    manifest = HashManifest(
        root=str(root_path),
        entries=entries,
        name=name,
        config=cfg.to_dict(),
        metadata=manifest_metadata,
    )

    if output is not None:
        _write_json(output, manifest.to_dict())

    return manifest


def load_hash_manifest(path: str | Path) -> HashManifest:
    """Load hash manifest from JSON file."""

    return HashManifest.from_dict(_load_json(path))


# ------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------

def _compute_report_status(
    checks: Sequence[HashCheck],
    *,
    fail_on_warning: bool = False,
) -> HashValidationStatus:
    """Compute aggregate verification status."""

    if any(check.failed for check in checks):
        return HashValidationStatus.FAIL

    if any(check.warning for check in checks):
        return HashValidationStatus.FAIL if fail_on_warning else HashValidationStatus.WARNING

    return HashValidationStatus.PASS


def verify_hash_manifest(
    manifest: str | Path | HashManifest | Mapping[str, Any],
    *,
    root: str | Path | None = None,
    include_extra: bool = True,
    config: HashConfig | Mapping[str, Any] | None = None,
) -> HashVerificationReport:
    """Verify current artifacts against a saved manifest."""

    manifest_path: str | None = None

    if isinstance(manifest, HashManifest):
        loaded_manifest = manifest
    elif isinstance(manifest, Mapping):
        loaded_manifest = HashManifest.from_dict(manifest)
    else:
        manifest_path = str(manifest)
        loaded_manifest = load_hash_manifest(manifest)

    cfg = (
        HashConfig.from_dict(loaded_manifest.config)
        if config is None and loaded_manifest.config
        else HashConfig()
        if config is None
        else config
        if isinstance(config, HashConfig)
        else HashConfig.from_dict(config)
    )

    root_path = Path(root) if root is not None else Path(loaded_manifest.root)

    checks: list[HashCheck] = []

    if not root_path.exists():
        checks.append(
            HashCheck(
                name="root_exists",
                status=HashValidationStatus.FAIL,
                severity=HashCheckSeverity.ERROR,
                message="Root path does not exist",
                path=str(root_path),
            )
        )

        status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

        return HashVerificationReport(
            status=status,
            manifest_path=manifest_path,
            root=str(root_path),
            checks=tuple(checks),
            expected_entries=loaded_manifest.entries,
            config=cfg.to_dict(),
        )

    expected = loaded_manifest.by_path()

    current_entries = create_hash_entries(root_path, config=cfg)
    current = {entry.path: entry for entry in current_entries}

    missing_paths = tuple(sorted(set(expected) - set(current)))
    extra_paths = tuple(sorted(set(current) - set(expected))) if include_extra else ()

    changed_paths: list[str] = []
    changed_details: list[JSONDict] = []

    for path_text in sorted(set(expected) & set(current)):
        expected_entry = expected[path_text]
        current_entry = current[path_text]

        if expected_entry.digest != current_entry.digest:
            changed_paths.append(path_text)
            changed_details.append(
                {
                    "path": path_text,
                    "expected_digest": expected_entry.digest,
                    "current_digest": current_entry.digest,
                    "expected_size_bytes": expected_entry.size_bytes,
                    "current_size_bytes": current_entry.size_bytes,
                }
            )

    checks.append(
        HashCheck(
            name="missing_paths",
            status=HashValidationStatus.PASS if not missing_paths else HashValidationStatus.FAIL,
            severity=HashCheckSeverity.INFO if not missing_paths else HashCheckSeverity.ERROR,
            message="No missing paths" if not missing_paths else f"{len(missing_paths)} path(s) are missing",
            details={"missing_paths": list(missing_paths)},
        )
    )

    checks.append(
        HashCheck(
            name="changed_paths",
            status=HashValidationStatus.PASS if not changed_paths else HashValidationStatus.FAIL,
            severity=HashCheckSeverity.INFO if not changed_paths else HashCheckSeverity.ERROR,
            message="No changed paths" if not changed_paths else f"{len(changed_paths)} path(s) changed",
            details={"changed": changed_details},
        )
    )

    if include_extra:
        checks.append(
            HashCheck(
                name="extra_paths",
                status=HashValidationStatus.PASS if not extra_paths else HashValidationStatus.WARNING,
                severity=HashCheckSeverity.INFO if not extra_paths else HashCheckSeverity.WARNING,
                message="No extra paths" if not extra_paths else f"{len(extra_paths)} extra path(s) found",
                details={"extra_paths": list(extra_paths)},
            )
        )

    total_size = sum(entry.size_bytes for entry in current_entries)
    status = _compute_report_status(checks, fail_on_warning=cfg.fail_on_warning)

    return HashVerificationReport(
        status=status,
        manifest_path=manifest_path,
        root=str(root_path),
        checks=tuple(checks),
        expected_entries=loaded_manifest.entries,
        current_entries=current_entries,
        missing_paths=missing_paths,
        changed_paths=tuple(changed_paths),
        extra_paths=extra_paths,
        metrics={
            "num_expected_entries": len(loaded_manifest.entries),
            "num_current_entries": len(current_entries),
            "num_missing_paths": len(missing_paths),
            "num_changed_paths": len(changed_paths),
            "num_extra_paths": len(extra_paths),
            "total_current_size_bytes": total_size,
            "total_current_size_human": _human_size(total_size),
        },
        config=cfg.to_dict(),
    )


def hash_artifact(path: str | Path, *, algorithm: str = "sha256") -> str:
    """Compatibility helper returning a file or directory hash."""

    p = Path(path)

    if p.is_dir():
        return compute_directory_hash(p, algorithm=algorithm)

    return hash_file(p, algorithm=algorithm)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        prog="pcsrc.validation.hash_artifacts",
        description="Hash experiment artifacts and verify hash manifests.",
    )

    parser.add_argument("--root", default="results", help="Artifact root for manifest creation.")
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")

    parser.add_argument("--verify", default=None, help="Verify an existing manifest JSON.")
    parser.add_argument("--verify-root", default=None, help="Override root path during verification.")
    parser.add_argument("--json-output", default=None, help="Write verification report JSON.")
    parser.add_argument("--text-output", default=None, help="Write verification report text.")

    parser.add_argument("--algorithm", default="sha256")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--non-recursive", action="store_true")
    parser.add_argument("--include-patterns", default=None)
    parser.add_argument("--exclude-patterns", default=None)
    parser.add_argument("--no-default-excludes", action="store_true")
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--no-extra-check", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")

    parser.add_argument("--print", action="store_true", help="Print manifest JSON when creating manifest.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    exclude_patterns = _parse_csv_text(args.exclude_patterns)

    if not args.no_default_excludes:
        exclude_patterns = tuple(_default_exclude_patterns()) + exclude_patterns

    cfg = HashConfig(
        algorithm=args.algorithm,
        chunk_size=args.chunk_size,
        recursive=not args.non_recursive,
        include_patterns=_parse_csv_text(args.include_patterns),
        exclude_patterns=exclude_patterns,
        follow_symlinks=args.follow_symlinks,
        include_metadata=not args.no_metadata,
        fail_on_warning=args.fail_on_warning,
    )

    if args.verify:
        report = verify_hash_manifest(
            args.verify,
            root=args.verify_root,
            include_extra=not args.no_extra_check,
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

    manifest = create_hash_manifest(
        args.root,
        output=args.output,
        config=cfg,
    )

    if args.print or args.output is None:
        print(json.dumps(_json_safe(manifest.to_dict()), ensure_ascii=False, indent=2, sort_keys=True))

    else:
        print(
            json.dumps(
                {
                    "manifest": str(args.output),
                    "root": str(args.root),
                    "entries": len(manifest.entries),
                    "algorithm": cfg.algorithm,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------

__all__ = [
    "ArtifactHash",
    "ArtifactHashKind",
    "HashCheck",
    "HashCheckSeverity",
    "HashConfig",
    "HashManifest",
    "HashValidationStatus",
    "HashVerificationReport",
    "build_arg_parser",
    "collect_artifact_hash",
    "compute_directory_hash",
    "create_hash_entries",
    "create_hash_manifest",
    "discover_artifacts",
    "hash_artifact",
    "hash_file",
    "hash_stream",
    "load_hash_manifest",
    "main",
    "verify_hash_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
