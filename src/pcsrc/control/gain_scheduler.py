# ==============================================================================
# src/pcsrc/control/gain_scheduler.py
#
# Gain scheduling utilities for pcsrc controllers.
#
# Purpose:
#   - Define gain tables and interpolation utilities for platoon controllers.
#   - Schedule local feedback gains based on speed, spacing error, relative
#     velocity, or arbitrary scalar operating variables.
#   - Provide deterministic, dependency-light logic suitable for real-time
#     simulation loops and controller wrappers.
#
# Typical use:
#   A CAV controller may use local feedback
#
#       u = k_e e + k_r r - k_a a
#
#   where gains are chosen according to an operating variable z, e.g.
#
#       z = ego velocity
#       z = spacing error
#       z = relative velocity
#
# Design notes:
#   - Uses only the Python standard library.
#   - Avoids NumPy so it remains cheap to import.
#   - Compatible with pcsrc.control.pc_src_controller.PCSRCFeedbackGains when
#     available.
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import isfinite
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from pcsrc.models.cav_dynamics import CAVErrorState


Number = int | float


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------

def _as_float(value: Number, name: str) -> float:
    """Convert ``value`` to finite float."""

    converted = float(value)

    if not isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")

    return converted


def _nonnegative(value: Number, name: str) -> float:
    """Convert ``value`` to non-negative finite float."""

    converted = _as_float(value, name)

    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")

    return converted


def _positive(value: Number, name: str) -> float:
    """Convert ``value`` to positive finite float."""

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


def _lerp(left: float, right: float, alpha: float) -> float:
    """Linear interpolation."""

    return (1.0 - alpha) * left + alpha * right


# ------------------------------------------------------------------------------
# Feedback gains
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScheduledFeedbackGains:
    """Feedback gains used by scheduled CAV controllers.

    Control convention:

        u = k_spacing * e + k_velocity * r - k_acceleration * a
    """

    k_spacing: float = 0.20
    k_velocity: float = 0.70
    k_acceleration: float = 0.00

    DEFAULT_K_SPACING: ClassVar[float] = 0.20
    DEFAULT_K_VELOCITY: ClassVar[float] = 0.70
    DEFAULT_K_ACCELERATION: ClassVar[float] = 0.00

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_spacing", _as_float(self.k_spacing, "k_spacing"))
        object.__setattr__(self, "k_velocity", _as_float(self.k_velocity, "k_velocity"))
        object.__setattr__(
            self,
            "k_acceleration",
            _as_float(self.k_acceleration, "k_acceleration"),
        )

    def evaluate(self, error_state: CAVErrorState) -> float:
        """Evaluate local feedback command."""

        return (
            self.k_spacing * error_state.spacing_error
            + self.k_velocity * error_state.relative_velocity
            - self.k_acceleration * error_state.ego_acceleration
        )

    def interpolate(
        self,
        other: "ScheduledFeedbackGains",
        alpha: Number,
    ) -> "ScheduledFeedbackGains":
        """Interpolate between this gain set and another."""

        a = _clip(alpha, 0.0, 1.0)

        return ScheduledFeedbackGains(
            k_spacing=_lerp(self.k_spacing, other.k_spacing, a),
            k_velocity=_lerp(self.k_velocity, other.k_velocity, a),
            k_acceleration=_lerp(
                self.k_acceleration,
                other.k_acceleration,
                a,
            ),
        )

    def scale(self, factor: Number) -> "ScheduledFeedbackGains":
        """Scale all gains by a scalar factor."""

        f = _as_float(factor, "factor")

        return ScheduledFeedbackGains(
            k_spacing=f * self.k_spacing,
            k_velocity=f * self.k_velocity,
            k_acceleration=f * self.k_acceleration,
        )

    def to_tuple(self) -> tuple[float, float, float]:
        """Return ``(k_spacing, k_velocity, k_acceleration)``."""

        return (self.k_spacing, self.k_velocity, self.k_acceleration)

    def to_dict(self) -> dict[str, float]:
        """Serialize gains to dictionary."""

        return asdict(self)

    def replace(self, **changes: Any) -> "ScheduledFeedbackGains":
        """Return copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduledFeedbackGains":
        """Create gains from mapping."""

        return cls(
            k_spacing=float(data.get("k_spacing", cls.DEFAULT_K_SPACING)),
            k_velocity=float(data.get("k_velocity", cls.DEFAULT_K_VELOCITY)),
            k_acceleration=float(
                data.get("k_acceleration", cls.DEFAULT_K_ACCELERATION)
            ),
        )


# ------------------------------------------------------------------------------
# Scheduling variable
# ------------------------------------------------------------------------------

class SchedulingVariable(str, Enum):
    """Built-in scalar scheduling variables."""

    EGO_VELOCITY = "ego_velocity"
    LEADER_VELOCITY = "leader_velocity"
    RELATIVE_VELOCITY = "relative_velocity"
    SPACING = "spacing"
    SPACING_ERROR = "spacing_error"
    EGO_ACCELERATION = "ego_acceleration"
    LEADER_ACCELERATION = "leader_acceleration"
    TIME = "time"
    CONSTANT = "constant"
    CUSTOM = "custom"


def scheduling_value(
    variable: SchedulingVariable | str,
    error_state: CAVErrorState,
    *,
    time: Number = 0.0,
    custom_value: Number | None = None,
) -> float:
    """Evaluate a built-in scheduling variable from local error state."""

    var = variable if isinstance(variable, SchedulingVariable) else SchedulingVariable(variable)

    if var is SchedulingVariable.EGO_VELOCITY:
        return error_state.ego_velocity

    if var is SchedulingVariable.LEADER_VELOCITY:
        return error_state.leader_velocity

    if var is SchedulingVariable.RELATIVE_VELOCITY:
        return error_state.relative_velocity

    if var is SchedulingVariable.SPACING:
        return error_state.spacing

    if var is SchedulingVariable.SPACING_ERROR:
        return error_state.spacing_error

    if var is SchedulingVariable.EGO_ACCELERATION:
        return error_state.ego_acceleration

    if var is SchedulingVariable.LEADER_ACCELERATION:
        return error_state.leader_acceleration

    if var is SchedulingVariable.TIME:
        return _nonnegative(time, "time")

    if var is SchedulingVariable.CONSTANT:
        return 0.0

    if var is SchedulingVariable.CUSTOM:
        if custom_value is None:
            raise ValueError("custom_value is required for CUSTOM scheduling")
        return _as_float(custom_value, "custom_value")

    raise ValueError(f"unsupported scheduling variable: {variable!r}")


# ------------------------------------------------------------------------------
# Gain schedule entries and tables
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GainSchedulePoint:
    """One point in a scalar gain schedule."""

    value: float
    gains: ScheduledFeedbackGains

    def __post_init__(self) -> None:
        gains = (
            self.gains
            if isinstance(self.gains, ScheduledFeedbackGains)
            else ScheduledFeedbackGains.from_dict(self.gains)  # type: ignore[arg-type]
        )

        object.__setattr__(self, "value", _as_float(self.value, "value"))
        object.__setattr__(self, "gains", gains)

    def to_dict(self) -> dict[str, Any]:
        """Serialize schedule point to dictionary."""

        return {
            "value": self.value,
            "gains": self.gains.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GainSchedulePoint":
        """Create schedule point from mapping."""

        return cls(
            value=float(data["value"]),
            gains=ScheduledFeedbackGains.from_dict(data.get("gains", {})),
        )


@dataclass(frozen=True, slots=True)
class GainScheduleTable:
    """Sorted scalar gain schedule table with linear interpolation."""

    points: tuple[GainSchedulePoint, ...]
    extrapolate: bool = False

    def __post_init__(self) -> None:
        if len(self.points) == 0:
            raise ValueError("points must contain at least one GainSchedulePoint")

        parsed = tuple(
            point
            if isinstance(point, GainSchedulePoint)
            else GainSchedulePoint.from_dict(point)  # type: ignore[arg-type]
            for point in self.points
        )

        parsed = tuple(sorted(parsed, key=lambda item: item.value))

        for left, right in zip(parsed, parsed[1:]):
            if left.value == right.value:
                raise ValueError(f"duplicate schedule point value: {left.value}")

        object.__setattr__(self, "points", parsed)
        object.__setattr__(self, "extrapolate", bool(self.extrapolate))

    @property
    def min_value(self) -> float:
        """Return minimum scheduling value."""

        return self.points[0].value

    @property
    def max_value(self) -> float:
        """Return maximum scheduling value."""

        return self.points[-1].value

    def lookup(self, value: Number) -> ScheduledFeedbackGains:
        """Lookup/interpolate gains at scalar scheduling value."""

        z = _as_float(value, "value")

        if len(self.points) == 1:
            return self.points[0].gains

        if z <= self.min_value:
            if not self.extrapolate:
                return self.points[0].gains
            left, right = self.points[0], self.points[1]
            alpha = (z - left.value) / (right.value - left.value)
            return left.gains.interpolate(right.gains, alpha)

        if z >= self.max_value:
            if not self.extrapolate:
                return self.points[-1].gains
            left, right = self.points[-2], self.points[-1]
            alpha = (z - left.value) / (right.value - left.value)
            return left.gains.interpolate(right.gains, alpha)

        for left, right in zip(self.points, self.points[1:]):
            if left.value <= z <= right.value:
                alpha = (z - left.value) / (right.value - left.value)
                return left.gains.interpolate(right.gains, alpha)

        # Should be unreachable due to boundary handling.
        return self.points[-1].gains

    def values(self) -> tuple[float, ...]:
        """Return schedule grid values."""

        return tuple(point.value for point in self.points)

    def gains(self) -> tuple[ScheduledFeedbackGains, ...]:
        """Return gains at schedule grid points."""

        return tuple(point.gains for point in self.points)

    def to_dict(self) -> dict[str, Any]:
        """Serialize table to dictionary."""

        return {
            "points": [point.to_dict() for point in self.points],
            "extrapolate": self.extrapolate,
        }

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[Number, ScheduledFeedbackGains | Mapping[str, Any]]],
        extrapolate: bool = False,
    ) -> "GainScheduleTable":
        """Create table from ``(value, gains)`` pairs."""

        return cls(
            points=tuple(
                GainSchedulePoint(
                    value=float(value),
                    gains=(
                        gains
                        if isinstance(gains, ScheduledFeedbackGains)
                        else ScheduledFeedbackGains.from_dict(gains)
                    ),
                )
                for value, gains in pairs
            ),
            extrapolate=extrapolate,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GainScheduleTable":
        """Create table from mapping."""

        return cls(
            points=tuple(
                GainSchedulePoint.from_dict(item)
                for item in data.get("points", [])
            ),
            extrapolate=bool(data.get("extrapolate", False)),
        )


# ------------------------------------------------------------------------------
# Scheduler configuration and implementation
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GainSchedulerConfig:
    """Configuration for scalar gain scheduler."""

    variable: SchedulingVariable = SchedulingVariable.EGO_VELOCITY
    table: GainScheduleTable = GainScheduleTable(
        points=(
            GainSchedulePoint(
                value=0.0,
                gains=ScheduledFeedbackGains(
                    k_spacing=0.18,
                    k_velocity=0.60,
                    k_acceleration=0.00,
                ),
            ),
            GainSchedulePoint(
                value=15.0,
                gains=ScheduledFeedbackGains(
                    k_spacing=0.20,
                    k_velocity=0.70,
                    k_acceleration=0.02,
                ),
            ),
            GainSchedulePoint(
                value=30.0,
                gains=ScheduledFeedbackGains(
                    k_spacing=0.24,
                    k_velocity=0.85,
                    k_acceleration=0.05,
                ),
            ),
        ),
        extrapolate=False,
    )
    enabled: bool = True

    def __post_init__(self) -> None:
        variable = (
            self.variable
            if isinstance(self.variable, SchedulingVariable)
            else SchedulingVariable(self.variable)
        )
        table = (
            self.table
            if isinstance(self.table, GainScheduleTable)
            else GainScheduleTable.from_dict(self.table)  # type: ignore[arg-type]
        )

        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "enabled", bool(self.enabled))

    def to_dict(self) -> dict[str, Any]:
        """Serialize scheduler config to dictionary."""

        return {
            "variable": self.variable.value,
            "table": self.table.to_dict(),
            "enabled": self.enabled,
        }

    def replace(self, **changes: Any) -> "GainSchedulerConfig":
        """Return copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GainSchedulerConfig":
        """Create scheduler config from mapping."""

        return cls(
            variable=SchedulingVariable(data.get("variable", SchedulingVariable.EGO_VELOCITY.value)),
            table=GainScheduleTable.from_dict(data.get("table", {})),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class GainScheduleResult:
    """Result of one gain-scheduler evaluation."""

    gains: ScheduledFeedbackGains
    variable: SchedulingVariable
    value: float
    enabled: bool

    def __post_init__(self) -> None:
        gains = (
            self.gains
            if isinstance(self.gains, ScheduledFeedbackGains)
            else ScheduledFeedbackGains.from_dict(self.gains)  # type: ignore[arg-type]
        )
        variable = (
            self.variable
            if isinstance(self.variable, SchedulingVariable)
            else SchedulingVariable(self.variable)
        )

        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "value", _as_float(self.value, "value"))
        object.__setattr__(self, "enabled", bool(self.enabled))

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""

        return {
            "gains": self.gains.to_dict(),
            "variable": self.variable.value,
            "value": self.value,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class GainScheduler:
    """Scalar gain scheduler with linear interpolation."""

    config: GainSchedulerConfig = GainSchedulerConfig()
    fallback_gains: ScheduledFeedbackGains = ScheduledFeedbackGains()

    def __post_init__(self) -> None:
        config = (
            self.config
            if isinstance(self.config, GainSchedulerConfig)
            else GainSchedulerConfig.from_dict(self.config)  # type: ignore[arg-type]
        )
        fallback = (
            self.fallback_gains
            if isinstance(self.fallback_gains, ScheduledFeedbackGains)
            else ScheduledFeedbackGains.from_dict(self.fallback_gains)  # type: ignore[arg-type]
        )

        object.__setattr__(self, "config", config)
        object.__setattr__(self, "fallback_gains", fallback)

    def schedule(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_value: Number | None = None,
    ) -> GainScheduleResult:
        """Evaluate scheduler and return gains plus metadata."""

        z = scheduling_value(
            variable=self.config.variable,
            error_state=error_state,
            time=time,
            custom_value=custom_value,
        )

        gains = self.config.table.lookup(z) if self.config.enabled else self.fallback_gains

        return GainScheduleResult(
            gains=gains,
            variable=self.config.variable,
            value=z,
            enabled=self.config.enabled,
        )

    def gains_for(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_value: Number | None = None,
    ) -> ScheduledFeedbackGains:
        """Return scheduled gains only."""

        return self.schedule(
            error_state=error_state,
            time=time,
            custom_value=custom_value,
        ).gains

    def command(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_value: Number | None = None,
    ) -> float:
        """Evaluate scheduled local feedback command."""

        return self.gains_for(
            error_state=error_state,
            time=time,
            custom_value=custom_value,
        ).evaluate(error_state)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scheduler."""

        return {
            "config": self.config.to_dict(),
            "fallback_gains": self.fallback_gains.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GainScheduler":
        """Create scheduler from mapping."""

        return cls(
            config=GainSchedulerConfig.from_dict(data.get("config", {})),
            fallback_gains=ScheduledFeedbackGains.from_dict(
                data.get("fallback_gains", {})
            ),
        )


# ------------------------------------------------------------------------------
# Optional adapter helpers
# ------------------------------------------------------------------------------

def to_pcsrc_feedback_gains(
    gains: ScheduledFeedbackGains,
) -> Any:
    """Convert scheduled gains to ``PCSRCFeedbackGains`` when available.

    The return type is ``Any`` to keep this module independent from importing
    ``pcsrc.control.pc_src_controller`` at module import time.
    """

    from pcsrc.control.pc_src_controller import PCSRCFeedbackGains

    return PCSRCFeedbackGains(
        k_spacing=gains.k_spacing,
        k_velocity=gains.k_velocity,
        k_acceleration=gains.k_acceleration,
    )


def scheduler_from_speed_breakpoints(
    breakpoints: Sequence[Number],
    gains: Sequence[ScheduledFeedbackGains | Mapping[str, Any]],
    *,
    extrapolate: bool = False,
) -> GainScheduler:
    """Create a speed-based scheduler from breakpoints and gain values."""

    if len(breakpoints) != len(gains):
        raise ValueError(
            f"breakpoints and gains must have same length, got "
            f"{len(breakpoints)} != {len(gains)}"
        )

    table = GainScheduleTable.from_pairs(
        zip(breakpoints, gains),
        extrapolate=extrapolate,
    )

    return GainScheduler(
        config=GainSchedulerConfig(
            variable=SchedulingVariable.EGO_VELOCITY,
            table=table,
            enabled=True,
        )
    )


def constant_gain_scheduler(
    gains: ScheduledFeedbackGains | Mapping[str, Any] | None = None,
) -> GainScheduler:
    """Create a scheduler that always returns the same gains."""

    parsed = (
        ScheduledFeedbackGains()
        if gains is None
        else gains
        if isinstance(gains, ScheduledFeedbackGains)
        else ScheduledFeedbackGains.from_dict(gains)
    )

    return GainScheduler(
        config=GainSchedulerConfig(
            variable=SchedulingVariable.CONSTANT,
            table=GainScheduleTable(
                points=(GainSchedulePoint(value=0.0, gains=parsed),),
                extrapolate=False,
            ),
            enabled=True,
        ),
        fallback_gains=parsed,
    )


def scheduled_command(
    error_state: CAVErrorState,
    scheduler: GainScheduler | Mapping[str, Any] | None = None,
    *,
    time: Number = 0.0,
    custom_value: Number | None = None,
) -> float:
    """Convenience wrapper for one scheduled command evaluation."""

    parsed = (
        GainScheduler()
        if scheduler is None
        else scheduler
        if isinstance(scheduler, GainScheduler)
        else GainScheduler.from_dict(scheduler)
    )

    return parsed.command(
        error_state=error_state,
        time=time,
        custom_value=custom_value,
    )


__all__ = [
    "GainSchedulePoint",
    "GainScheduleResult",
    "GainScheduleTable",
    "GainScheduler",
    "GainSchedulerConfig",
    "Number",
    "ScheduledFeedbackGains",
    "SchedulingVariable",
    "constant_gain_scheduler",
    "scheduled_command",
    "scheduler_from_speed_breakpoints",
    "scheduling_value",
    "to_pcsrc_feedback_gains",
]
