# ==============================================================================
# src/pcsrc/control/saturation.py
#
# Saturation, rate limiting, and command projection utilities for pcsrc.
#
# Purpose:
#   - Clip CAV acceleration commands to physical/control limits.
#   - Apply optional command-rate limits between simulation steps.
#   - Provide diagnostics describing whether a command was saturated or
#     rate-limited.
#   - Support scalar commands, indexed command mappings, and simple batch
#     command processing.
#
# Convention:
#   command acceleration u is expressed in m/s^2.
#   command rate du/dt is expressed in m/s^3.
#
# Design notes:
#   - This module uses only the Python standard library.
#   - It intentionally avoids NumPy so it remains lightweight.
#   - It is compatible with pcsrc.models.vehicle.ActuatorLimits when available,
#     but does not require users to pass that type.
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Any, ClassVar, Mapping, MutableMapping, Sequence


Number = int | float


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

def _as_float(value: Number, name: str) -> float:
    """Convert ``value`` to a finite float."""

    converted = float(value)

    if not isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")

    return converted


def _nonnegative(value: Number, name: str) -> float:
    """Convert ``value`` to finite float and require non-negativity."""

    converted = _as_float(value, name)

    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")

    return converted


def _positive(value: Number, name: str) -> float:
    """Convert ``value`` to finite float and require positivity."""

    converted = _as_float(value, name)

    if converted <= 0.0:
        raise ValueError(f"{name} must be positive, got {converted}")

    return converted


def _clip(value: Number, lower: Number, upper: Number) -> float:
    """Clip scalar value to ``[lower, upper]``."""

    x = _as_float(value, "value")
    lo = _as_float(lower, "lower")
    hi = _as_float(upper, "upper")

    if lo > hi:
        raise ValueError(f"lower must be <= upper, got {lo} > {hi}")

    return min(max(x, lo), hi)


def _changed(left: Number, right: Number, tol: Number = 1e-12) -> bool:
    """Return whether two scalar values differ by more than tolerance."""

    tolerance = _nonnegative(tol, "tol")
    return abs(float(left) - float(right)) > tolerance


# ------------------------------------------------------------------------------
# Saturation configuration and result containers
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SaturationConfig:
    """Configuration for command saturation and rate limiting.

    Attributes
    ----------
    min_command:
        Minimum allowed acceleration command in m/s².
    max_command:
        Maximum allowed acceleration command in m/s².
    max_rate:
        Optional symmetric command-rate limit in m/s³. If ``None``, no rate
        limiting is applied.
    min_rate:
        Optional lower command-rate limit in m/s³. If omitted while ``max_rate``
        is provided, ``-max_rate`` is used.
    enable_saturation:
        Whether amplitude saturation is enabled.
    enable_rate_limit:
        Whether rate limiting is enabled.
    """

    min_command: float = -3.0
    max_command: float = 3.0
    max_rate: float | None = None
    min_rate: float | None = None
    enable_saturation: bool = True
    enable_rate_limit: bool = True

    DEFAULT_MIN_COMMAND: ClassVar[float] = -3.0
    DEFAULT_MAX_COMMAND: ClassVar[float] = 3.0

    def __post_init__(self) -> None:
        min_command = _as_float(self.min_command, "min_command")
        max_command = _as_float(self.max_command, "max_command")

        if min_command > max_command:
            raise ValueError(
                f"min_command must be <= max_command, got "
                f"{min_command} > {max_command}"
            )

        if self.max_rate is None:
            max_rate = None
        else:
            max_rate = _positive(self.max_rate, "max_rate")

        if self.min_rate is None:
            min_rate = None if max_rate is None else -max_rate
        else:
            min_rate = _as_float(self.min_rate, "min_rate")

        if max_rate is not None and min_rate is not None and min_rate > max_rate:
            raise ValueError(
                f"min_rate must be <= max_rate, got {min_rate} > {max_rate}"
            )

        object.__setattr__(self, "min_command", min_command)
        object.__setattr__(self, "max_command", max_command)
        object.__setattr__(self, "max_rate", max_rate)
        object.__setattr__(self, "min_rate", min_rate)
        object.__setattr__(self, "enable_saturation", bool(self.enable_saturation))
        object.__setattr__(self, "enable_rate_limit", bool(self.enable_rate_limit))

    @property
    def has_rate_limit(self) -> bool:
        """Return whether finite rate limits are configured."""

        return self.min_rate is not None and self.max_rate is not None

    def clip(self, command: Number) -> float:
        """Clip command to amplitude bounds if saturation is enabled."""

        value = _as_float(command, "command")

        if not self.enable_saturation:
            return value

        return _clip(value, self.min_command, self.max_command)

    def rate_clip(
        self,
        command: Number,
        previous_command: Number,
        dt: Number,
    ) -> float:
        """Clip command according to configured rate limits.

        Parameters
        ----------
        command:
            Candidate command after amplitude saturation.
        previous_command:
            Previous command value.
        dt:
            Positive sample time in seconds.
        """

        value = _as_float(command, "command")
        previous = _as_float(previous_command, "previous_command")
        step = _positive(dt, "dt")

        if not self.enable_rate_limit or not self.has_rate_limit:
            return value

        assert self.min_rate is not None
        assert self.max_rate is not None

        lower = previous + self.min_rate * step
        upper = previous + self.max_rate * step

        return _clip(value, lower, upper)

    def contains(self, command: Number, tol: Number = 0.0) -> bool:
        """Return whether command lies within amplitude bounds."""

        tolerance = _nonnegative(tol, "tol")
        value = _as_float(command, "command")

        return (
            self.min_command - tolerance
            <= value
            <= self.max_command + tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""

        return asdict(self)

    def replace(self, **changes: Any) -> "SaturationConfig":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SaturationConfig":
        """Create configuration from mapping."""

        return cls(
            min_command=float(data.get("min_command", cls.DEFAULT_MIN_COMMAND)),
            max_command=float(data.get("max_command", cls.DEFAULT_MAX_COMMAND)),
            max_rate=(
                None
                if data.get("max_rate", None) is None
                else float(data["max_rate"])
            ),
            min_rate=(
                None
                if data.get("min_rate", None) is None
                else float(data["min_rate"])
            ),
            enable_saturation=bool(data.get("enable_saturation", True)),
            enable_rate_limit=bool(data.get("enable_rate_limit", True)),
        )

    @classmethod
    def from_actuator_limits(
        cls,
        limits: Any,
        max_rate: Number | None = None,
    ) -> "SaturationConfig":
        """Create config from an object exposing actuator-limit fields.

        Compatible objects may expose either:

        - ``min_acceleration`` / ``max_acceleration``;
        - ``minimum`` / ``maximum``.
        """

        if hasattr(limits, "min_acceleration") and hasattr(limits, "max_acceleration"):
            min_command = float(limits.min_acceleration)
            max_command = float(limits.max_acceleration)
        elif hasattr(limits, "minimum") and hasattr(limits, "maximum"):
            min_command = float(limits.minimum)
            max_command = float(limits.maximum)
        elif isinstance(limits, Mapping):
            min_command = float(
                limits.get("min_acceleration", limits.get("minimum", cls.DEFAULT_MIN_COMMAND))
            )
            max_command = float(
                limits.get("max_acceleration", limits.get("maximum", cls.DEFAULT_MAX_COMMAND))
            )
        else:
            raise TypeError(
                "limits must expose min_acceleration/max_acceleration, "
                "minimum/maximum, or be a mapping"
            )

        return cls(
            min_command=min_command,
            max_command=max_command,
            max_rate=None if max_rate is None else float(max_rate),
        )


@dataclass(frozen=True, slots=True)
class SaturationResult:
    """Result of applying command saturation and optional rate limiting."""

    command: float
    raw_command: float
    saturated_command: float
    previous_command: float | None = None
    dt: float | None = None
    saturated: bool = False
    rate_limited: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _as_float(self.command, "command"))
        object.__setattr__(
            self,
            "raw_command",
            _as_float(self.raw_command, "raw_command"),
        )
        object.__setattr__(
            self,
            "saturated_command",
            _as_float(self.saturated_command, "saturated_command"),
        )

        if self.previous_command is not None:
            object.__setattr__(
                self,
                "previous_command",
                _as_float(self.previous_command, "previous_command"),
            )

        if self.dt is not None:
            object.__setattr__(self, "dt", _positive(self.dt, "dt"))

        if self.lower_bound is not None:
            object.__setattr__(
                self,
                "lower_bound",
                _as_float(self.lower_bound, "lower_bound"),
            )

        if self.upper_bound is not None:
            object.__setattr__(
                self,
                "upper_bound",
                _as_float(self.upper_bound, "upper_bound"),
            )

        object.__setattr__(self, "saturated", bool(self.saturated))
        object.__setattr__(self, "rate_limited", bool(self.rate_limited))

    @property
    def changed(self) -> bool:
        """Return whether the final command differs from the raw command."""

        return self.saturated or self.rate_limited

    @property
    def delta_from_previous(self) -> float | None:
        """Return command change relative to previous command if available."""

        if self.previous_command is None:
            return None

        return self.command - self.previous_command

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""

        return {
            "command": self.command,
            "raw_command": self.raw_command,
            "saturated_command": self.saturated_command,
            "previous_command": self.previous_command,
            "dt": self.dt,
            "saturated": self.saturated,
            "rate_limited": self.rate_limited,
            "changed": self.changed,
            "delta_from_previous": self.delta_from_previous,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(slots=True)
class RateLimiterState:
    """Runtime memory for command-rate limiting.

    Attributes
    ----------
    previous_commands:
        Mapping from vehicle/control index to previously applied command.
    """

    previous_commands: MutableMapping[int, float] = field(default_factory=dict)

    def get(self, index: int, default: Number = 0.0) -> float:
        """Return previous command for index."""

        return float(self.previous_commands.get(int(index), float(default)))

    def set(self, index: int, command: Number) -> None:
        """Store previous command for index."""

        self.previous_commands[int(index)] = _as_float(command, "command")

    def update(self, commands: Mapping[int, Number]) -> None:
        """Update memory from an index-command mapping."""

        for index, command in commands.items():
            self.set(int(index), command)

    def reset(self) -> None:
        """Clear all stored previous commands."""

        self.previous_commands.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize rate limiter state."""

        return {
            "previous_commands": {
                int(index): float(command)
                for index, command in self.previous_commands.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RateLimiterState":
        """Create rate limiter state from mapping."""

        state = cls()

        for index, command in data.get("previous_commands", {}).items():
            state.set(int(index), float(command))

        return state


# ------------------------------------------------------------------------------
# Core scalar operations
# ------------------------------------------------------------------------------

def clip_command(
    command: Number,
    min_command: Number = SaturationConfig.DEFAULT_MIN_COMMAND,
    max_command: Number = SaturationConfig.DEFAULT_MAX_COMMAND,
) -> float:
    """Clip a scalar command to acceleration bounds."""

    return _clip(command, min_command, max_command)


def apply_rate_limit(
    command: Number,
    previous_command: Number,
    dt: Number,
    max_rate: Number | None = None,
    min_rate: Number | None = None,
) -> float:
    """Apply scalar command-rate limiting.

    If both ``max_rate`` and ``min_rate`` are omitted, the command is returned
    unchanged.
    """

    value = _as_float(command, "command")
    previous = _as_float(previous_command, "previous_command")
    step = _positive(dt, "dt")

    if max_rate is None and min_rate is None:
        return value

    if max_rate is None:
        upper_rate = float("inf")
    else:
        upper_rate = _positive(max_rate, "max_rate")

    if min_rate is None:
        lower_rate = -upper_rate
    else:
        lower_rate = _as_float(min_rate, "min_rate")

    if lower_rate > upper_rate:
        raise ValueError(
            f"min_rate must be <= max_rate, got {lower_rate} > {upper_rate}"
        )

    lower = previous + lower_rate * step
    upper = previous + upper_rate * step

    return _clip(value, lower, upper)


def project_command(
    command: Number,
    config: SaturationConfig | Mapping[str, Any] | None = None,
    previous_command: Number | None = None,
    dt: Number | None = None,
) -> SaturationResult:
    """Project a raw command into the admissible command set.

    The operation order is:

    1. amplitude saturation;
    2. optional rate limiting;
    3. final amplitude saturation for safety.

    Parameters
    ----------
    command:
        Raw acceleration command.
    config:
        Saturation configuration. If omitted, default acceleration bounds are
        used.
    previous_command:
        Previous command used for rate limiting.
    dt:
        Positive sample time used for rate limiting.
    """

    cfg = (
        SaturationConfig()
        if config is None
        else config
        if isinstance(config, SaturationConfig)
        else SaturationConfig.from_dict(config)
    )

    raw = _as_float(command, "command")
    saturated_command = cfg.clip(raw)
    saturated = _changed(saturated_command, raw)

    final = saturated_command
    rate_limited = False

    lower_bound: float | None = None
    upper_bound: float | None = None

    if previous_command is not None and dt is not None and cfg.has_rate_limit:
        previous = _as_float(previous_command, "previous_command")
        step = _positive(dt, "dt")

        assert cfg.min_rate is not None
        assert cfg.max_rate is not None

        lower_bound = previous + cfg.min_rate * step
        upper_bound = previous + cfg.max_rate * step

        limited = cfg.rate_clip(
            command=saturated_command,
            previous_command=previous,
            dt=step,
        )
        rate_limited = _changed(limited, saturated_command)
        final = limited

    final_clipped = cfg.clip(final)
    saturated = saturated or _changed(final_clipped, final)
    final = final_clipped

    return SaturationResult(
        command=final,
        raw_command=raw,
        saturated_command=saturated_command,
        previous_command=None if previous_command is None else float(previous_command),
        dt=None if dt is None else float(dt),
        saturated=saturated,
        rate_limited=rate_limited,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


# ------------------------------------------------------------------------------
# Batch and mapping operations
# ------------------------------------------------------------------------------

def project_command_mapping(
    commands: Mapping[int, Number],
    config: SaturationConfig | Mapping[str, Any] | None = None,
    previous_commands: Mapping[int, Number] | RateLimiterState | None = None,
    dt: Number | None = None,
    *,
    update_state: bool = True,
) -> dict[int, SaturationResult]:
    """Project a mapping of indexed commands.

    Parameters
    ----------
    commands:
        Mapping from vehicle/control index to raw command.
    config:
        Shared saturation configuration.
    previous_commands:
        Either a mapping of previous commands or ``RateLimiterState``.
    dt:
        Optional time step for rate limiting.
    update_state:
        If ``previous_commands`` is ``RateLimiterState``, update it with final
        projected commands.
    """

    cfg = (
        SaturationConfig()
        if config is None
        else config
        if isinstance(config, SaturationConfig)
        else SaturationConfig.from_dict(config)
    )

    results: dict[int, SaturationResult] = {}

    for raw_index, raw_command in commands.items():
        index = int(raw_index)

        if isinstance(previous_commands, RateLimiterState):
            previous = previous_commands.get(index, raw_command)
        elif previous_commands is None:
            previous = None
        else:
            previous = (
                float(previous_commands[index])
                if index in previous_commands
                else float(raw_command)
            )

        result = project_command(
            command=raw_command,
            config=cfg,
            previous_command=previous,
            dt=dt,
        )
        results[index] = result

        if isinstance(previous_commands, RateLimiterState) and update_state:
            previous_commands.set(index, result.command)

    return results


def command_results_to_mapping(
    results: Mapping[int, SaturationResult],
) -> dict[int, float]:
    """Extract final command mapping from saturation results."""

    return {int(index): result.command for index, result in results.items()}


def clip_command_mapping(
    commands: Mapping[int, Number],
    min_command: Number = SaturationConfig.DEFAULT_MIN_COMMAND,
    max_command: Number = SaturationConfig.DEFAULT_MAX_COMMAND,
) -> dict[int, float]:
    """Clip every value in an indexed command mapping."""

    return {
        int(index): clip_command(command, min_command, max_command)
        for index, command in commands.items()
    }


def project_command_sequence(
    commands: Sequence[Number],
    config: SaturationConfig | Mapping[str, Any] | None = None,
    previous_commands: Sequence[Number] | None = None,
    dt: Number | None = None,
) -> tuple[SaturationResult, ...]:
    """Project a sequence of commands while preserving order."""

    results: list[SaturationResult] = []

    for index, command in enumerate(commands):
        previous = (
            None
            if previous_commands is None or index >= len(previous_commands)
            else previous_commands[index]
        )
        results.append(
            project_command(
                command=command,
                config=config,
                previous_command=previous,
                dt=dt,
            )
        )

    return tuple(results)


def safe_brake_command(
    config: SaturationConfig | Mapping[str, Any] | None = None,
) -> float:
    """Return the strongest allowed braking command."""

    cfg = (
        SaturationConfig()
        if config is None
        else config
        if isinstance(config, SaturationConfig)
        else SaturationConfig.from_dict(config)
    )

    return cfg.min_command


def max_accel_command(
    config: SaturationConfig | Mapping[str, Any] | None = None,
) -> float:
    """Return the strongest allowed positive acceleration command."""

    cfg = (
        SaturationConfig()
        if config is None
        else config
        if isinstance(config, SaturationConfig)
        else SaturationConfig.from_dict(config)
    )

    return cfg.max_command


def is_saturated(
    command: Number,
    min_command: Number = SaturationConfig.DEFAULT_MIN_COMMAND,
    max_command: Number = SaturationConfig.DEFAULT_MAX_COMMAND,
    tol: Number = 1e-12,
) -> bool:
    """Return whether command would be clipped by the bounds."""

    clipped = clip_command(command, min_command, max_command)
    return _changed(clipped, command, tol=tol)


def normalize_command(
    command: Number,
    min_command: Number = SaturationConfig.DEFAULT_MIN_COMMAND,
    max_command: Number = SaturationConfig.DEFAULT_MAX_COMMAND,
) -> float:
    """Normalize command to approximately ``[-1, 1]`` over configured bounds.

    The midpoint maps to ``0``. The lower bound maps to ``-1`` and the upper
    bound maps to ``1``.
    """

    value = _as_float(command, "command")
    lo = _as_float(min_command, "min_command")
    hi = _as_float(max_command, "max_command")

    if lo >= hi:
        raise ValueError("min_command must be strictly smaller than max_command")

    midpoint = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)

    return (value - midpoint) / half_width


def denormalize_command(
    normalized_command: Number,
    min_command: Number = SaturationConfig.DEFAULT_MIN_COMMAND,
    max_command: Number = SaturationConfig.DEFAULT_MAX_COMMAND,
    *,
    clip_normalized: bool = True,
) -> float:
    """Map normalized command from ``[-1, 1]`` back to acceleration units."""

    value = _as_float(normalized_command, "normalized_command")

    if clip_normalized:
        value = _clip(value, -1.0, 1.0)

    lo = _as_float(min_command, "min_command")
    hi = _as_float(max_command, "max_command")

    if lo >= hi:
        raise ValueError("min_command must be strictly smaller than max_command")

    midpoint = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)

    return midpoint + half_width * value


__all__ = [
    "Number",
    "RateLimiterState",
    "SaturationConfig",
    "SaturationResult",
    "apply_rate_limit",
    "clip_command",
    "clip_command_mapping",
    "command_results_to_mapping",
    "denormalize_command",
    "is_saturated",
    "max_accel_command",
    "normalize_command",
    "project_command",
    "project_command_mapping",
    "project_command_sequence",
    "safe_brake_command",
]
