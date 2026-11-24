# ==============================================================================
# src/pcsrc/control/safety_filters.py
#
# Safety filters for pcsrc controllers.
#
# Purpose:
#   - Filter raw acceleration commands before they are applied to vehicle models.
#   - Enforce simple longitudinal safety constraints for mixed-platoon control.
#   - Provide collision-risk fallback braking and velocity-envelope protection.
#   - Support scalar commands, indexed command mappings, and rich diagnostics.
#
# Safety philosophy:
#   This module intentionally implements lightweight, deterministic filters rather
#   than relying on heavy online optimization. The goal is to provide a practical
#   runtime guard that can be used in simulation loops and controller wrappers.
#
# Typical workflow:
#
#       raw_commands = controller.command_mapping(model, state, dt)
#       filtered = filter_command_mapping(model, state, raw_commands, dt=dt)
#       applied_commands = safety_results_to_mapping(filtered)
#
# Design notes:
#   - Uses only Python standard library plus pcsrc model containers.
#   - Compatible with CAV/local predecessor-following error states.
#   - Does not require NumPy, SciPy, cvxpy, or control.
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from math import isfinite, sqrt
from typing import Any, ClassVar, Mapping, MutableMapping

from pcsrc.control.saturation import SaturationConfig, SaturationResult, project_command
from pcsrc.models.cav_dynamics import CAVErrorState
from pcsrc.models.mixed_platoon import MixedPlatoonModel, PlatoonState


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
    """Clip scalar to ``[lower, upper]``."""

    x = _as_float(value, "value")
    lo = _as_float(lower, "lower")
    hi = _as_float(upper, "upper")

    if lo > hi:
        raise ValueError(f"lower must be <= upper, got {lo} > {hi}")

    return min(max(x, lo), hi)


def _changed(left: Number, right: Number, tol: Number = 1e-12) -> bool:
    """Return whether two scalar values differ by more than tolerance."""

    return abs(float(left) - float(right)) > _nonnegative(tol, "tol")


def _get_vehicle_state(state: PlatoonState, index: int) -> Any:
    """Return vehicle state by index.

    Supports both a ``vehicle(index)`` method and direct ``vehicle_states`` access.
    """

    if hasattr(state, "vehicle"):
        return state.vehicle(index)

    return state.vehicle_states[index]


def _get_model_size(model: MixedPlatoonModel) -> int:
    """Return platoon size from model/config."""

    if hasattr(model, "size"):
        return int(model.size)

    if hasattr(model.config, "num_vehicles"):
        return int(model.config.num_vehicles)

    return int(len(model.config.vehicle_params))


def _desired_gap_for(model: MixedPlatoonModel, vehicle_index: int, ego_velocity: float) -> float:
    """Return desired gap for follower.

    Falls back to ``target_spacing`` when the richer model API is unavailable.
    """

    if hasattr(model.config, "desired_gap_for"):
        return float(model.config.desired_gap_for(vehicle_index, ego_velocity))

    return float(getattr(model.config, "target_spacing", 10.0))


def _spacing_for(model: MixedPlatoonModel, state: PlatoonState, vehicle_index: int) -> float:
    """Compute predecessor-follower spacing."""

    if hasattr(model, "get_spacing"):
        return float(model.get_spacing(state, vehicle_index))

    leader = _get_vehicle_state(state, vehicle_index - 1)
    ego = _get_vehicle_state(state, vehicle_index)
    leader_length = float(model.config.vehicle_params[vehicle_index - 1].length)

    return float(leader.position - ego.position - leader_length)


# ------------------------------------------------------------------------------
# Filter modes and statuses
# ------------------------------------------------------------------------------

class SafetyFilterMode(str, Enum):
    """Safety-filter operating mode."""

    PASS_THROUGH = "pass_through"
    MONITOR = "monitor"
    FILTER = "filter"
    EMERGENCY_BRAKE = "emergency_brake"


class SafetyFilterStatus(str, Enum):
    """Status of a filtered command."""

    OK = "ok"
    CLIPPED = "clipped"
    SAFE_BRAKE = "safe_brake"
    TIME_GAP_LIMITED = "time_gap_limited"
    VELOCITY_LIMITED = "velocity_limited"
    SATURATED = "saturated"
    RATE_LIMITED = "rate_limited"
    MISSING_VEHICLE = "missing_vehicle"


# ------------------------------------------------------------------------------
# Configuration and result containers
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SafetyFilterConfig:
    """Configuration for longitudinal command safety filtering.

    Attributes
    ----------
    min_spacing:
        Hard minimum allowed predecessor-follower spacing in meters.
    warning_spacing:
        Spacing threshold below which conservative braking may be applied.
    time_headway:
        Desired time-headway coefficient in seconds.
    min_time_gap:
        Minimum allowed time gap in seconds.
    max_closing_speed:
        Critical negative relative velocity threshold. Since relative velocity is
        ``leader_velocity - ego_velocity``, negative values indicate closing.
    reaction_time:
        Prediction horizon used by risk checks.
    min_velocity:
        Lower velocity envelope. Commands that would push below this value are
        limited.
    max_velocity:
        Upper velocity envelope. Commands that would push above this value are
        limited.
    brake_command:
        Optional fixed emergency brake command. If omitted, ``min_command`` from
        the saturation config is used.
    saturation:
        Command amplitude/rate saturation config.
    enabled:
        Whether filtering is enabled.
    mode:
        Filter operating mode.
    """

    min_spacing: float = 1.0
    warning_spacing: float = 3.0
    time_headway: float = 1.0
    min_time_gap: float = 0.5
    max_closing_speed: float = -6.0
    reaction_time: float = 0.5
    min_velocity: float = 0.0
    max_velocity: float = 40.0
    brake_command: float | None = None
    saturation: SaturationConfig = field(default_factory=SaturationConfig)
    enabled: bool = True
    mode: SafetyFilterMode = SafetyFilterMode.FILTER

    DEFAULT_MIN_SPACING: ClassVar[float] = 1.0
    DEFAULT_WARNING_SPACING: ClassVar[float] = 3.0

    def __post_init__(self) -> None:
        min_spacing = _nonnegative(self.min_spacing, "min_spacing")
        warning_spacing = _nonnegative(self.warning_spacing, "warning_spacing")

        if warning_spacing < min_spacing:
            warning_spacing = min_spacing

        min_velocity = _nonnegative(self.min_velocity, "min_velocity")
        max_velocity = _nonnegative(self.max_velocity, "max_velocity")

        if min_velocity > max_velocity:
            raise ValueError(
                f"min_velocity must be <= max_velocity, got "
                f"{min_velocity} > {max_velocity}"
            )

        saturation = (
            self.saturation
            if isinstance(self.saturation, SaturationConfig)
            else SaturationConfig.from_dict(self.saturation)  # type: ignore[arg-type]
        )
        mode = (
            self.mode
            if isinstance(self.mode, SafetyFilterMode)
            else SafetyFilterMode(self.mode)
        )

        object.__setattr__(self, "min_spacing", min_spacing)
        object.__setattr__(self, "warning_spacing", warning_spacing)
        object.__setattr__(self, "time_headway", _nonnegative(self.time_headway, "time_headway"))
        object.__setattr__(self, "min_time_gap", _nonnegative(self.min_time_gap, "min_time_gap"))
        object.__setattr__(self, "max_closing_speed", _as_float(self.max_closing_speed, "max_closing_speed"))
        object.__setattr__(self, "reaction_time", _nonnegative(self.reaction_time, "reaction_time"))
        object.__setattr__(self, "min_velocity", min_velocity)
        object.__setattr__(self, "max_velocity", max_velocity)
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "mode", mode)

        if self.brake_command is not None:
            object.__setattr__(
                self,
                "brake_command",
                _as_float(self.brake_command, "brake_command"),
            )

    @property
    def emergency_brake_command(self) -> float:
        """Return configured emergency braking command."""

        if self.brake_command is not None:
            return self.brake_command

        return self.saturation.min_command

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""

        return {
            "min_spacing": self.min_spacing,
            "warning_spacing": self.warning_spacing,
            "time_headway": self.time_headway,
            "min_time_gap": self.min_time_gap,
            "max_closing_speed": self.max_closing_speed,
            "reaction_time": self.reaction_time,
            "min_velocity": self.min_velocity,
            "max_velocity": self.max_velocity,
            "brake_command": self.brake_command,
            "saturation": self.saturation.to_dict(),
            "enabled": self.enabled,
            "mode": self.mode.value,
        }

    def replace(self, **changes: Any) -> "SafetyFilterConfig":
        """Return copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyFilterConfig":
        """Create config from mapping."""

        return cls(
            min_spacing=float(data.get("min_spacing", cls.DEFAULT_MIN_SPACING)),
            warning_spacing=float(data.get("warning_spacing", cls.DEFAULT_WARNING_SPACING)),
            time_headway=float(data.get("time_headway", 1.0)),
            min_time_gap=float(data.get("min_time_gap", 0.5)),
            max_closing_speed=float(data.get("max_closing_speed", -6.0)),
            reaction_time=float(data.get("reaction_time", 0.5)),
            min_velocity=float(data.get("min_velocity", 0.0)),
            max_velocity=float(data.get("max_velocity", 40.0)),
            brake_command=(
                None
                if data.get("brake_command", None) is None
                else float(data["brake_command"])
            ),
            saturation=SaturationConfig.from_dict(data.get("saturation", {})),
            enabled=bool(data.get("enabled", True)),
            mode=SafetyFilterMode(data.get("mode", SafetyFilterMode.FILTER.value)),
        )


@dataclass(frozen=True, slots=True)
class SafetyMetrics:
    """Safety-relevant metrics for one follower vehicle."""

    vehicle_index: int
    spacing: float
    desired_spacing: float
    spacing_error: float
    relative_velocity: float
    ego_velocity: float
    leader_velocity: float
    ego_acceleration: float
    leader_acceleration: float
    time_gap: float
    predicted_spacing: float
    stopping_margin: float

    def __post_init__(self) -> None:
        if self.vehicle_index <= 0:
            raise ValueError("vehicle_index must be a follower index > 0")

        for field_name in (
            "spacing",
            "desired_spacing",
            "spacing_error",
            "relative_velocity",
            "ego_velocity",
            "leader_velocity",
            "ego_acceleration",
            "leader_acceleration",
            "time_gap",
            "predicted_spacing",
            "stopping_margin",
        ):
            object.__setattr__(
                self,
                field_name,
                _as_float(getattr(self, field_name), field_name),
            )

    @property
    def closing_speed(self) -> float:
        """Return positive closing speed when ego is faster than leader."""

        return max(0.0, -self.relative_velocity)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""

        data = asdict(self)
        data["closing_speed"] = self.closing_speed
        return data


@dataclass(frozen=True, slots=True)
class SafetyFilterResult:
    """Result of filtering one command."""

    vehicle_index: int
    command: float
    raw_command: float
    filtered_command: float
    status: SafetyFilterStatus
    metrics: SafetyMetrics
    saturation_result: SaturationResult
    safe_brake: bool = False
    time_gap_limited: bool = False
    velocity_limited: bool = False
    clipped: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.vehicle_index <= 0:
            raise ValueError("vehicle_index must be a follower index > 0")

        status = (
            self.status
            if isinstance(self.status, SafetyFilterStatus)
            else SafetyFilterStatus(self.status)
        )

        object.__setattr__(self, "command", _as_float(self.command, "command"))
        object.__setattr__(self, "raw_command", _as_float(self.raw_command, "raw_command"))
        object.__setattr__(
            self,
            "filtered_command",
            _as_float(self.filtered_command, "filtered_command"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "safe_brake", bool(self.safe_brake))
        object.__setattr__(self, "time_gap_limited", bool(self.time_gap_limited))
        object.__setattr__(self, "velocity_limited", bool(self.velocity_limited))
        object.__setattr__(self, "clipped", bool(self.clipped))
        object.__setattr__(self, "reason", str(self.reason))

    @property
    def changed(self) -> bool:
        """Return whether filtering changed the raw command."""

        return (
            self.safe_brake
            or self.time_gap_limited
            or self.velocity_limited
            or self.clipped
            or _changed(self.command, self.raw_command)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""

        return {
            "vehicle_index": self.vehicle_index,
            "command": self.command,
            "raw_command": self.raw_command,
            "filtered_command": self.filtered_command,
            "status": self.status.value,
            "safe_brake": self.safe_brake,
            "time_gap_limited": self.time_gap_limited,
            "velocity_limited": self.velocity_limited,
            "clipped": self.clipped,
            "changed": self.changed,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
            "saturation_result": self.saturation_result.to_dict(),
        }


@dataclass(slots=True)
class SafetyFilterState:
    """Runtime memory for safety filtering."""

    previous_commands: MutableMapping[int, float] = field(default_factory=dict)
    intervention_count: int = 0
    emergency_count: int = 0

    def previous(self, vehicle_index: int, default: Number = 0.0) -> float:
        """Return previous command for vehicle."""

        return float(self.previous_commands.get(int(vehicle_index), float(default)))

    def update(self, vehicle_index: int, command: Number) -> None:
        """Store previous filtered command."""

        self.previous_commands[int(vehicle_index)] = _as_float(command, "command")

    def reset(self) -> None:
        """Reset runtime memory."""

        self.previous_commands.clear()
        self.intervention_count = 0
        self.emergency_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime state."""

        return {
            "previous_commands": {
                int(index): float(command)
                for index, command in self.previous_commands.items()
            },
            "intervention_count": int(self.intervention_count),
            "emergency_count": int(self.emergency_count),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyFilterState":
        """Create runtime state from mapping."""

        state = cls()
        for index, command in data.get("previous_commands", {}).items():
            state.previous_commands[int(index)] = float(command)

        state.intervention_count = int(data.get("intervention_count", 0))
        state.emergency_count = int(data.get("emergency_count", 0))

        return state


# ------------------------------------------------------------------------------
# Safety metric computation
# ------------------------------------------------------------------------------

def error_state_from_model(
    model: MixedPlatoonModel,
    state: PlatoonState,
    vehicle_index: int,
) -> CAVErrorState:
    """Compute local CAV-style error state for a follower vehicle."""

    size = _get_model_size(model)

    if vehicle_index <= 0 or vehicle_index >= size:
        raise IndexError(
            f"vehicle_index must be in [1, {size - 1}], got {vehicle_index}"
        )

    if hasattr(model, "validate_state"):
        model.validate_state(state)

    ego = _get_vehicle_state(state, vehicle_index)
    leader = _get_vehicle_state(state, vehicle_index - 1)

    spacing = _spacing_for(model, state, vehicle_index)
    desired_spacing = _desired_gap_for(model, vehicle_index, float(ego.velocity))

    return CAVErrorState(
        spacing=spacing,
        desired_spacing=desired_spacing,
        spacing_error=spacing - desired_spacing,
        relative_velocity=float(leader.velocity - ego.velocity),
        ego_velocity=float(ego.velocity),
        leader_velocity=float(leader.velocity),
        ego_acceleration=float(getattr(ego, "acceleration", 0.0)),
        leader_acceleration=float(getattr(leader, "acceleration", 0.0)),
    )


def stopping_margin(
    spacing: Number,
    ego_velocity: Number,
    leader_velocity: Number,
    *,
    min_spacing: Number = 1.0,
    ego_max_brake: Number = 3.0,
    leader_max_brake: Number = 3.0,
) -> float:
    """Compute a simple braking-distance safety margin.

    Positive margin indicates that the current spacing exceeds the approximate
    required differential stopping distance plus ``min_spacing``.
    """

    s = _as_float(spacing, "spacing")
    ve = max(0.0, _as_float(ego_velocity, "ego_velocity"))
    vl = max(0.0, _as_float(leader_velocity, "leader_velocity"))
    min_s = _nonnegative(min_spacing, "min_spacing")
    be = _positive(ego_max_brake, "ego_max_brake")
    bl = _positive(leader_max_brake, "leader_max_brake")

    ego_stop = ve * ve / (2.0 * be)
    leader_stop = vl * vl / (2.0 * bl)
    required = max(0.0, ego_stop - leader_stop) + min_s

    return s - required


def compute_safety_metrics(
    error_state: CAVErrorState,
    vehicle_index: int,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
) -> SafetyMetrics:
    """Compute safety metrics from local error state."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    ego_velocity = max(0.0, float(error_state.ego_velocity))
    time_gap = (
        float("inf")
        if ego_velocity <= 1e-9
        else float(error_state.spacing) / ego_velocity
    )

    predicted_spacing = (
        float(error_state.spacing)
        + float(error_state.relative_velocity) * cfg.reaction_time
    )

    margin = stopping_margin(
        spacing=error_state.spacing,
        ego_velocity=error_state.ego_velocity,
        leader_velocity=error_state.leader_velocity,
        min_spacing=cfg.min_spacing,
        ego_max_brake=abs(cfg.saturation.min_command)
        if cfg.saturation.min_command < 0.0
        else 3.0,
        leader_max_brake=abs(cfg.saturation.min_command)
        if cfg.saturation.min_command < 0.0
        else 3.0,
    )

    return SafetyMetrics(
        vehicle_index=vehicle_index,
        spacing=error_state.spacing,
        desired_spacing=error_state.desired_spacing,
        spacing_error=error_state.spacing_error,
        relative_velocity=error_state.relative_velocity,
        ego_velocity=error_state.ego_velocity,
        leader_velocity=error_state.leader_velocity,
        ego_acceleration=error_state.ego_acceleration,
        leader_acceleration=error_state.leader_acceleration,
        time_gap=time_gap,
        predicted_spacing=predicted_spacing,
        stopping_margin=margin,
    )


def is_emergency_condition(
    metrics: SafetyMetrics,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
) -> bool:
    """Return whether metrics indicate emergency braking."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    return (
        metrics.spacing <= cfg.min_spacing
        or metrics.predicted_spacing <= cfg.min_spacing
        or metrics.relative_velocity <= cfg.max_closing_speed
        or metrics.stopping_margin <= 0.0
    )


def is_time_gap_unsafe(
    metrics: SafetyMetrics,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
) -> bool:
    """Return whether time gap is below configured minimum."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    return metrics.ego_velocity > 1e-9 and metrics.time_gap < cfg.min_time_gap


# ------------------------------------------------------------------------------
# Command filtering logic
# ------------------------------------------------------------------------------

def velocity_guard_command(
    command: Number,
    ego_velocity: Number,
    dt: Number,
    *,
    min_velocity: Number = 0.0,
    max_velocity: Number = 40.0,
) -> float:
    """Limit command so one Euler step remains inside velocity envelope."""

    u = _as_float(command, "command")
    v = _as_float(ego_velocity, "ego_velocity")
    step = _positive(dt, "dt")
    v_min = _nonnegative(min_velocity, "min_velocity")
    v_max = _nonnegative(max_velocity, "max_velocity")

    if v_min > v_max:
        raise ValueError("min_velocity must be <= max_velocity")

    lower_u = (v_min - v) / step
    upper_u = (v_max - v) / step

    return _clip(u, lower_u, upper_u)


def time_gap_guard_command(
    command: Number,
    metrics: SafetyMetrics,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
) -> float:
    """Apply a conservative command cap when time gap is unsafe.

    The cap is intentionally simple: if the vehicle is too close or closing, do
    not allow positive acceleration, and increase braking as the predicted
    spacing approaches the hard minimum.
    """

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    u = _as_float(command, "command")

    unsafe_gap = is_time_gap_unsafe(metrics, cfg)
    close_spacing = metrics.spacing <= cfg.warning_spacing
    closing = metrics.relative_velocity < 0.0

    if not (unsafe_gap or close_spacing or closing):
        return u

    # No acceleration when headway is poor or the ego vehicle is closing.
    capped = min(u, 0.0)

    if metrics.predicted_spacing <= cfg.warning_spacing:
        span = max(cfg.warning_spacing - cfg.min_spacing, 1e-9)
        severity = _clip(
            (cfg.warning_spacing - metrics.predicted_spacing) / span,
            0.0,
            1.0,
        )
        capped = min(capped, severity * cfg.emergency_brake_command)

    return capped


def filter_command_for_error(
    vehicle_index: int,
    command: Number,
    error_state: CAVErrorState,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
    *,
    previous_command: Number | None = None,
    dt: Number | None = None,
) -> SafetyFilterResult:
    """Filter one raw command using local error-state safety checks."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    raw = _as_float(command, "command")
    metrics = compute_safety_metrics(error_state, vehicle_index, cfg)

    filtered = raw
    safe_brake = False
    time_gap_limited = False
    velocity_limited = False
    reason = "ok"

    if not cfg.enabled or cfg.mode is SafetyFilterMode.PASS_THROUGH:
        saturation = project_command(
            command=filtered,
            config=cfg.saturation,
            previous_command=previous_command,
            dt=dt,
        )

        return SafetyFilterResult(
            vehicle_index=vehicle_index,
            command=saturation.command,
            raw_command=raw,
            filtered_command=filtered,
            status=SafetyFilterStatus.OK
            if not saturation.changed
            else SafetyFilterStatus.SATURATED,
            metrics=metrics,
            saturation_result=saturation,
            clipped=saturation.changed,
            reason="pass_through",
        )

    emergency = (
        cfg.mode is SafetyFilterMode.EMERGENCY_BRAKE
        or is_emergency_condition(metrics, cfg)
    )

    if emergency:
        filtered = cfg.emergency_brake_command
        safe_brake = True
        reason = "emergency_brake"
    elif cfg.mode is SafetyFilterMode.FILTER:
        before_gap = filtered
        filtered = time_gap_guard_command(filtered, metrics, cfg)
        time_gap_limited = _changed(filtered, before_gap)

        if time_gap_limited:
            reason = "time_gap_or_spacing_guard"

        if dt is not None:
            before_velocity = filtered
            filtered = velocity_guard_command(
                filtered,
                metrics.ego_velocity,
                dt,
                min_velocity=cfg.min_velocity,
                max_velocity=cfg.max_velocity,
            )
            velocity_limited = _changed(filtered, before_velocity)

            if velocity_limited and reason == "ok":
                reason = "velocity_guard"

    elif cfg.mode is SafetyFilterMode.MONITOR:
        reason = "monitor"

    saturation = project_command(
        command=filtered,
        config=cfg.saturation,
        previous_command=previous_command,
        dt=dt,
    )

    clipped = saturation.changed

    if safe_brake:
        status = SafetyFilterStatus.SAFE_BRAKE
    elif time_gap_limited:
        status = SafetyFilterStatus.TIME_GAP_LIMITED
    elif velocity_limited:
        status = SafetyFilterStatus.VELOCITY_LIMITED
    elif saturation.rate_limited:
        status = SafetyFilterStatus.RATE_LIMITED
    elif saturation.saturated:
        status = SafetyFilterStatus.SATURATED
    elif clipped:
        status = SafetyFilterStatus.CLIPPED
    else:
        status = SafetyFilterStatus.OK

    return SafetyFilterResult(
        vehicle_index=vehicle_index,
        command=saturation.command,
        raw_command=raw,
        filtered_command=filtered,
        status=status,
        metrics=metrics,
        saturation_result=saturation,
        safe_brake=safe_brake,
        time_gap_limited=time_gap_limited,
        velocity_limited=velocity_limited,
        clipped=clipped,
        reason=reason,
    )


def filter_command_for_vehicle(
    model: MixedPlatoonModel,
    state: PlatoonState,
    vehicle_index: int,
    command: Number,
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
    *,
    previous_command: Number | None = None,
    dt: Number | None = None,
) -> SafetyFilterResult:
    """Filter command for one follower vehicle in a platoon state."""

    error_state = error_state_from_model(model, state, vehicle_index)

    return filter_command_for_error(
        vehicle_index=vehicle_index,
        command=command,
        error_state=error_state,
        config=config,
        previous_command=previous_command,
        dt=dt,
    )


def filter_command_mapping(
    model: MixedPlatoonModel,
    state: PlatoonState,
    commands: Mapping[int, Number],
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
    *,
    runtime: SafetyFilterState | None = None,
    dt: Number | None = None,
    update_runtime: bool = True,
) -> dict[int, SafetyFilterResult]:
    """Filter an indexed mapping of vehicle commands."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    results: dict[int, SafetyFilterResult] = {}

    for raw_index, raw_command in commands.items():
        index = int(raw_index)

        previous = (
            None
            if runtime is None
            else runtime.previous(index, default=raw_command)
        )

        try:
            result = filter_command_for_vehicle(
                model=model,
                state=state,
                vehicle_index=index,
                command=raw_command,
                config=cfg,
                previous_command=previous,
                dt=dt,
            )
        except (IndexError, ValueError):
            # If a command refers to an invalid follower, return a clipped command
            # with a missing-vehicle status rather than crashing a long simulation.
            dummy_error = CAVErrorState(
                spacing=0.0,
                desired_spacing=0.0,
                spacing_error=0.0,
                relative_velocity=0.0,
                ego_velocity=0.0,
                leader_velocity=0.0,
                ego_acceleration=0.0,
                leader_acceleration=0.0,
            )
            metrics = compute_safety_metrics(dummy_error, max(index, 1), cfg)
            saturation = project_command(
                command=raw_command,
                config=cfg.saturation,
                previous_command=previous,
                dt=dt,
            )
            result = SafetyFilterResult(
                vehicle_index=max(index, 1),
                command=saturation.command,
                raw_command=float(raw_command),
                filtered_command=float(raw_command),
                status=SafetyFilterStatus.MISSING_VEHICLE,
                metrics=metrics,
                saturation_result=saturation,
                clipped=saturation.changed,
                reason="missing_vehicle",
            )

        results[index] = result

        if runtime is not None and update_runtime:
            runtime.update(index, result.command)

            if result.changed:
                runtime.intervention_count += 1

            if result.safe_brake:
                runtime.emergency_count += 1

    return results


def safety_results_to_mapping(
    results: Mapping[int, SafetyFilterResult],
) -> dict[int, float]:
    """Extract final command mapping from safety-filter results."""

    return {int(index): result.command for index, result in results.items()}


# ------------------------------------------------------------------------------
# Stateful filter class
# ------------------------------------------------------------------------------

@dataclass(slots=True)
class SafetyFilter:
    """Stateful safety filter for mixed-platoon command mappings."""

    config: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)
    runtime: SafetyFilterState = field(default_factory=SafetyFilterState)

    def __post_init__(self) -> None:
        if not isinstance(self.config, SafetyFilterConfig):
            self.config = SafetyFilterConfig.from_dict(self.config)  # type: ignore[arg-type]

        if not isinstance(self.runtime, SafetyFilterState):
            self.runtime = SafetyFilterState()  # type: ignore[assignment]

    def reset(self) -> None:
        """Reset runtime memory."""

        self.runtime.reset()

    def filter_one(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        vehicle_index: int,
        command: Number,
        *,
        dt: Number | None = None,
    ) -> SafetyFilterResult:
        """Filter one vehicle command and update runtime memory."""

        previous = self.runtime.previous(vehicle_index, default=command)

        result = filter_command_for_vehicle(
            model=model,
            state=state,
            vehicle_index=vehicle_index,
            command=command,
            config=self.config,
            previous_command=previous,
            dt=dt,
        )

        self.runtime.update(vehicle_index, result.command)

        if result.changed:
            self.runtime.intervention_count += 1

        if result.safe_brake:
            self.runtime.emergency_count += 1

        return result

    def filter_mapping(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        commands: Mapping[int, Number],
        *,
        dt: Number | None = None,
    ) -> dict[int, SafetyFilterResult]:
        """Filter an indexed command mapping and update runtime memory."""

        return filter_command_mapping(
            model=model,
            state=state,
            commands=commands,
            config=self.config,
            runtime=self.runtime,
            dt=dt,
            update_runtime=True,
        )

    def command_mapping(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        commands: Mapping[int, Number],
        *,
        dt: Number | None = None,
    ) -> dict[int, float]:
        """Return filtered command mapping only."""

        return safety_results_to_mapping(
            self.filter_mapping(
                model=model,
                state=state,
                commands=commands,
                dt=dt,
            )
        )

    def __call__(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        commands: Mapping[int, Number],
        *,
        dt: Number | None = None,
    ) -> dict[int, float]:
        """Alias for ``command_mapping``."""

        return self.command_mapping(
            model=model,
            state=state,
            commands=commands,
            dt=dt,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize filter configuration and runtime state."""

        return {
            "config": self.config.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyFilter":
        """Create filter from dictionary."""

        return cls(
            config=SafetyFilterConfig.from_dict(data.get("config", {})),
            runtime=SafetyFilterState.from_dict(data.get("runtime", {})),
        )


# ------------------------------------------------------------------------------
# Convenience constructors and helpers
# ------------------------------------------------------------------------------

def make_default_safety_filter(
    min_spacing: Number = 1.0,
    warning_spacing: Number = 3.0,
    min_acceleration: Number = -3.0,
    max_acceleration: Number = 3.0,
    max_rate: Number | None = None,
) -> SafetyFilter:
    """Create a default stateful safety filter."""

    return SafetyFilter(
        config=SafetyFilterConfig(
            min_spacing=float(min_spacing),
            warning_spacing=float(warning_spacing),
            saturation=SaturationConfig(
                min_command=float(min_acceleration),
                max_command=float(max_acceleration),
                max_rate=None if max_rate is None else float(max_rate),
            ),
        )
    )


def emergency_brake_command(
    config: SafetyFilterConfig | Mapping[str, Any] | None = None,
) -> float:
    """Return emergency brake command for config."""

    cfg = (
        SafetyFilterConfig()
        if config is None
        else config
        if isinstance(config, SafetyFilterConfig)
        else SafetyFilterConfig.from_dict(config)
    )

    return cfg.emergency_brake_command


def safe_following_distance(
    ego_velocity: Number,
    *,
    min_spacing: Number = 1.0,
    time_headway: Number = 1.0,
) -> float:
    """Return simple constant-time-headway safe distance."""

    return _nonnegative(min_spacing, "min_spacing") + _nonnegative(
        time_headway,
        "time_headway",
    ) * max(0.0, _as_float(ego_velocity, "ego_velocity"))


def max_safe_speed_for_spacing(
    spacing: Number,
    *,
    min_spacing: Number = 1.0,
    time_headway: Number = 1.0,
) -> float:
    """Return maximum speed allowed by a time-headway spacing rule."""

    s = _as_float(spacing, "spacing")
    min_s = _nonnegative(min_spacing, "min_spacing")
    h = _positive(time_headway, "time_headway")

    return max(0.0, (s - min_s) / h)


def required_braking_for_spacing(
    spacing: Number,
    relative_velocity: Number,
    *,
    min_spacing: Number = 1.0,
) -> float:
    """Estimate braking deceleration needed to avoid closing over remaining gap.

    Returns a non-positive acceleration command. If relative velocity is not
    closing, returns ``0.0``.
    """

    s = _as_float(spacing, "spacing")
    r = _as_float(relative_velocity, "relative_velocity")
    min_s = _nonnegative(min_spacing, "min_spacing")

    closing_speed = max(0.0, -r)
    available_gap = max(1e-9, s - min_s)

    if closing_speed <= 0.0:
        return 0.0

    return -(closing_speed * closing_speed) / (2.0 * available_gap)


__all__ = [
    "Number",
    "SafetyFilter",
    "SafetyFilterConfig",
    "SafetyFilterMode",
    "SafetyFilterResult",
    "SafetyFilterState",
    "SafetyFilterStatus",
    "SafetyMetrics",
    "compute_safety_metrics",
    "emergency_brake_command",
    "error_state_from_model",
    "filter_command_for_error",
    "filter_command_for_vehicle",
    "filter_command_mapping",
    "is_emergency_condition",
    "is_time_gap_unsafe",
    "make_default_safety_filter",
    "max_safe_speed_for_spacing",
    "required_braking_for_spacing",
    "safe_following_distance",
    "safety_results_to_mapping",
    "stopping_margin",
    "time_gap_guard_command",
    "velocity_guard_command",
]
