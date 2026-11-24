# ==============================================================================
# src/pcsrc/control/pc_src_controller.py
#
# PC-SRC controller for mixed-platoon CAV control.
#
# Purpose:
#   - Provide the main controller used by pcsrc simulations and validation tools.
#   - Compute CAV acceleration commands from local predecessor-following error
#     states.
#   - Support nominal linear feedback, robust compensation, saturation, rate
#     limiting, and safety fallback.
#
# Naming:
#   PC-SRC may be interpreted by the project as a platoon-control / safe robust
#   control layer. This module intentionally keeps the implementation lightweight
#   and dependency-free, so it can be used in simulations without importing heavy
#   optimization or SOS-LMI solvers.
#
# Control convention:
#
#   For a CAV follower i with predecessor i-1:
#
#       e_i = s_i - d_i
#       r_i = v_{i-1} - v_i
#       a_i = ego realized acceleration
#
#   Nominal command:
#
#       u_i = k_e e_i + k_r r_i - k_a a_i
#
#   Optional robust correction:
#
#       u_i <- u_i + rho_e sat(e_i/e_scale)
#                   + rho_r sat(r_i/r_scale)
#                   - rho_a sat(a_i/a_scale)
#
#   Final command:
#
#       u_i <- rate_limit(saturate(u_i))
#
# Design notes:
#   - Uses only Python standard library plus pcsrc model modules.
#   - Avoids cvxpy/scipy/control imports.
#   - Designed to work with the earlier model files:
#       pcsrc.models.vehicle
#       pcsrc.models.mixed_platoon
#       pcsrc.models.cav_dynamics
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Any, ClassVar, Mapping, MutableMapping, Sequence

from pcsrc.models.cav_dynamics import CAVErrorState
from pcsrc.models.mixed_platoon import MixedPlatoonModel, PlatoonState, VehicleRole
from pcsrc.models.vehicle import ActuatorLimits


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
    """Clip a scalar to ``[lower, upper]``."""

    x = _as_float(value, "value")
    lo = _as_float(lower, "lower")
    hi = _as_float(upper, "upper")

    if lo > hi:
        raise ValueError(f"lower must be <= upper, got {lo} > {hi}")

    return min(max(x, lo), hi)


def _sat_unit(value: Number) -> float:
    """Unit saturation function."""

    return _clip(value, -1.0, 1.0)


# ------------------------------------------------------------------------------
# Controller modes and status
# ------------------------------------------------------------------------------

class PCSRCMode(str, Enum):
    """Operating mode of the PC-SRC controller."""

    NOMINAL = "nominal"
    ROBUST = "robust"
    SAFE_FALLBACK = "safe_fallback"


class PCSRCStatus(str, Enum):
    """Status returned by a control step."""

    OK = "ok"
    SATURATED = "saturated"
    RATE_LIMITED = "rate_limited"
    SAFE_BRAKE = "safe_brake"
    MISSING_CAV = "missing_cav"


# ------------------------------------------------------------------------------
# Configuration and state containers
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PCSRCFeedbackGains:
    """Linear feedback gains for local CAV control.

    Attributes
    ----------
    k_spacing:
        Gain on spacing error.
    k_velocity:
        Gain on predecessor-relative velocity.
    k_acceleration:
        Damping gain on ego acceleration.
    """

    k_spacing: float = 0.20
    k_velocity: float = 0.70
    k_acceleration: float = 0.00

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_spacing", _as_float(self.k_spacing, "k_spacing"))
        object.__setattr__(self, "k_velocity", _as_float(self.k_velocity, "k_velocity"))
        object.__setattr__(
            self,
            "k_acceleration",
            _as_float(self.k_acceleration, "k_acceleration"),
        )

    def evaluate(self, error: CAVErrorState) -> float:
        """Evaluate nominal local feedback command."""

        return (
            self.k_spacing * error.spacing_error
            + self.k_velocity * error.relative_velocity
            - self.k_acceleration * error.ego_acceleration
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize gains to dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PCSRCFeedbackGains":
        """Create gains from mapping."""

        return cls(
            k_spacing=float(data.get("k_spacing", 0.20)),
            k_velocity=float(data.get("k_velocity", 0.70)),
            k_acceleration=float(data.get("k_acceleration", 0.00)),
        )


@dataclass(frozen=True, slots=True)
class PCSRCRobustGains:
    """Bounded robust compensation gains.

    The robust term is deliberately implemented as a smooth-free saturated
    correction so it remains deterministic and dependency-free.
    """

    rho_spacing: float = 0.0
    rho_velocity: float = 0.0
    rho_acceleration: float = 0.0
    spacing_scale: float = 5.0
    velocity_scale: float = 2.0
    acceleration_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_spacing", _nonnegative(self.rho_spacing, "rho_spacing"))
        object.__setattr__(self, "rho_velocity", _nonnegative(self.rho_velocity, "rho_velocity"))
        object.__setattr__(
            self,
            "rho_acceleration",
            _nonnegative(self.rho_acceleration, "rho_acceleration"),
        )
        object.__setattr__(self, "spacing_scale", _positive(self.spacing_scale, "spacing_scale"))
        object.__setattr__(self, "velocity_scale", _positive(self.velocity_scale, "velocity_scale"))
        object.__setattr__(
            self,
            "acceleration_scale",
            _positive(self.acceleration_scale, "acceleration_scale"),
        )

    @property
    def enabled(self) -> bool:
        """Return whether any robust gain is active."""

        return (
            self.rho_spacing > 0.0
            or self.rho_velocity > 0.0
            or self.rho_acceleration > 0.0
        )

    def evaluate(self, error: CAVErrorState) -> float:
        """Evaluate bounded robust correction."""

        return (
            self.rho_spacing * _sat_unit(error.spacing_error / self.spacing_scale)
            + self.rho_velocity * _sat_unit(error.relative_velocity / self.velocity_scale)
            - self.rho_acceleration * _sat_unit(error.ego_acceleration / self.acceleration_scale)
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize robust gains to dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PCSRCRobustGains":
        """Create robust gains from mapping."""

        return cls(
            rho_spacing=float(data.get("rho_spacing", 0.0)),
            rho_velocity=float(data.get("rho_velocity", 0.0)),
            rho_acceleration=float(data.get("rho_acceleration", 0.0)),
            spacing_scale=float(data.get("spacing_scale", 5.0)),
            velocity_scale=float(data.get("velocity_scale", 2.0)),
            acceleration_scale=float(data.get("acceleration_scale", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class PCSRCSafetyConfig:
    """Safety fallback settings for PC-SRC."""

    enabled: bool = True
    collision_spacing: float = 1.0
    critical_spacing_error: float = -8.0
    critical_closing_speed: float = -8.0
    brake_command: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "collision_spacing",
            _nonnegative(self.collision_spacing, "collision_spacing"),
        )
        object.__setattr__(
            self,
            "critical_spacing_error",
            _as_float(self.critical_spacing_error, "critical_spacing_error"),
        )
        object.__setattr__(
            self,
            "critical_closing_speed",
            _as_float(self.critical_closing_speed, "critical_closing_speed"),
        )

        if self.brake_command is not None:
            object.__setattr__(
                self,
                "brake_command",
                _as_float(self.brake_command, "brake_command"),
            )

    def requires_braking(self, error: CAVErrorState) -> bool:
        """Return whether safety fallback should brake."""

        if not self.enabled:
            return False

        return (
            error.spacing <= self.collision_spacing
            or error.spacing_error <= self.critical_spacing_error
            or error.relative_velocity <= self.critical_closing_speed
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize safety config to dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PCSRCSafetyConfig":
        """Create safety config from mapping."""

        return cls(
            enabled=bool(data.get("enabled", True)),
            collision_spacing=float(data.get("collision_spacing", 1.0)),
            critical_spacing_error=float(data.get("critical_spacing_error", -8.0)),
            critical_closing_speed=float(data.get("critical_closing_speed", -8.0)),
            brake_command=(
                None
                if data.get("brake_command", None) is None
                else float(data["brake_command"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PCSRCControllerConfig:
    """Configuration for the PC-SRC controller.

    Attributes
    ----------
    feedback_gains:
        Nominal local linear feedback gains.
    robust_gains:
        Optional robust saturated correction gains.
    acceleration_limits:
        Output acceleration command bounds.
    max_command_rate:
        Optional symmetric command-rate limit in m/s³.
    controlled_indices:
        Optional explicit CAV indices. If ``None``, indices are inferred from the
        mixed-platoon model configuration.
    safety:
        Safety fallback configuration.
    mode:
        Requested controller mode.
    """

    feedback_gains: PCSRCFeedbackGains = PCSRCFeedbackGains()
    robust_gains: PCSRCRobustGains = PCSRCRobustGains()
    acceleration_limits: ActuatorLimits = ActuatorLimits(
        min_acceleration=-3.0,
        max_acceleration=3.0,
    )
    max_command_rate: float | None = None
    controlled_indices: tuple[int, ...] | None = None
    safety: PCSRCSafetyConfig = PCSRCSafetyConfig()
    mode: PCSRCMode = PCSRCMode.ROBUST

    DEFAULT_MIN_ACCELERATION: ClassVar[float] = -3.0
    DEFAULT_MAX_ACCELERATION: ClassVar[float] = 3.0

    def __post_init__(self) -> None:
        feedback_gains = (
            self.feedback_gains
            if isinstance(self.feedback_gains, PCSRCFeedbackGains)
            else PCSRCFeedbackGains.from_dict(self.feedback_gains)  # type: ignore[arg-type]
        )
        robust_gains = (
            self.robust_gains
            if isinstance(self.robust_gains, PCSRCRobustGains)
            else PCSRCRobustGains.from_dict(self.robust_gains)  # type: ignore[arg-type]
        )
        acceleration_limits = (
            self.acceleration_limits
            if isinstance(self.acceleration_limits, ActuatorLimits)
            else ActuatorLimits.from_dict(self.acceleration_limits)  # type: ignore[arg-type]
        )
        safety = (
            self.safety
            if isinstance(self.safety, PCSRCSafetyConfig)
            else PCSRCSafetyConfig.from_dict(self.safety)  # type: ignore[arg-type]
        )
        mode = self.mode if isinstance(self.mode, PCSRCMode) else PCSRCMode(self.mode)

        if self.max_command_rate is None:
            rate = None
        else:
            rate = _positive(self.max_command_rate, "max_command_rate")

        if self.controlled_indices is None:
            controlled = None
        else:
            controlled = tuple(int(index) for index in self.controlled_indices)
            if any(index <= 0 for index in controlled):
                raise ValueError("controlled_indices must contain follower indices > 0")

        object.__setattr__(self, "feedback_gains", feedback_gains)
        object.__setattr__(self, "robust_gains", robust_gains)
        object.__setattr__(self, "acceleration_limits", acceleration_limits)
        object.__setattr__(self, "max_command_rate", rate)
        object.__setattr__(self, "controlled_indices", controlled)
        object.__setattr__(self, "safety", safety)
        object.__setattr__(self, "mode", mode)

    def clip(self, command: Number) -> float:
        """Clip command to acceleration bounds."""

        return self.acceleration_limits.clip(command)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""

        return {
            "feedback_gains": self.feedback_gains.to_dict(),
            "robust_gains": self.robust_gains.to_dict(),
            "acceleration_limits": self.acceleration_limits.to_dict(),
            "max_command_rate": self.max_command_rate,
            "controlled_indices": (
                None
                if self.controlled_indices is None
                else list(self.controlled_indices)
            ),
            "safety": self.safety.to_dict(),
            "mode": self.mode.value,
        }

    def replace(self, **changes: Any) -> "PCSRCControllerConfig":
        """Return copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PCSRCControllerConfig":
        """Create config from mapping."""

        return cls(
            feedback_gains=PCSRCFeedbackGains.from_dict(data.get("feedback_gains", {})),
            robust_gains=PCSRCRobustGains.from_dict(data.get("robust_gains", {})),
            acceleration_limits=ActuatorLimits.from_dict(
                data.get(
                    "acceleration_limits",
                    {
                        "min_acceleration": cls.DEFAULT_MIN_ACCELERATION,
                        "max_acceleration": cls.DEFAULT_MAX_ACCELERATION,
                    },
                )
            ),
            max_command_rate=(
                None
                if data.get("max_command_rate", None) is None
                else float(data["max_command_rate"])
            ),
            controlled_indices=(
                None
                if data.get("controlled_indices", None) is None
                else tuple(int(index) for index in data["controlled_indices"])
            ),
            safety=PCSRCSafetyConfig.from_dict(data.get("safety", {})),
            mode=PCSRCMode(data.get("mode", PCSRCMode.ROBUST.value)),
        )


@dataclass(frozen=True, slots=True)
class PCSRCCommand:
    """Command result for one controlled CAV."""

    vehicle_index: int
    command: float
    raw_command: float
    nominal_command: float
    robust_correction: float
    saturated: bool
    rate_limited: bool
    safe_brake: bool
    status: PCSRCStatus
    error_state: CAVErrorState

    def __post_init__(self) -> None:
        if self.vehicle_index <= 0:
            raise ValueError("vehicle_index must be a follower index > 0")

        status = (
            self.status
            if isinstance(self.status, PCSRCStatus)
            else PCSRCStatus(self.status)
        )

        object.__setattr__(self, "command", _as_float(self.command, "command"))
        object.__setattr__(self, "raw_command", _as_float(self.raw_command, "raw_command"))
        object.__setattr__(
            self,
            "nominal_command",
            _as_float(self.nominal_command, "nominal_command"),
        )
        object.__setattr__(
            self,
            "robust_correction",
            _as_float(self.robust_correction, "robust_correction"),
        )
        object.__setattr__(self, "saturated", bool(self.saturated))
        object.__setattr__(self, "rate_limited", bool(self.rate_limited))
        object.__setattr__(self, "safe_brake", bool(self.safe_brake))
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        """Serialize command result to dictionary."""

        return {
            "vehicle_index": self.vehicle_index,
            "command": self.command,
            "raw_command": self.raw_command,
            "nominal_command": self.nominal_command,
            "robust_correction": self.robust_correction,
            "saturated": self.saturated,
            "rate_limited": self.rate_limited,
            "safe_brake": self.safe_brake,
            "status": self.status.value,
            "error_state": self.error_state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PCSRCControllerDiagnostics:
    """Diagnostics returned by a full control evaluation."""

    mode: PCSRCMode
    commands: tuple[PCSRCCommand, ...]
    controlled_indices: tuple[int, ...]
    dt: float | None = None

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, PCSRCMode) else PCSRCMode(self.mode)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(self, "controlled_indices", tuple(self.controlled_indices))

        if self.dt is not None:
            object.__setattr__(self, "dt", _positive(self.dt, "dt"))

    @property
    def any_saturated(self) -> bool:
        """Return whether any command was saturated."""

        return any(command.saturated for command in self.commands)

    @property
    def any_rate_limited(self) -> bool:
        """Return whether any command was rate limited."""

        return any(command.rate_limited for command in self.commands)

    @property
    def any_safe_brake(self) -> bool:
        """Return whether any command used safe braking."""

        return any(command.safe_brake for command in self.commands)

    @property
    def command_mapping(self) -> dict[int, float]:
        """Return ``{vehicle_index: command}`` mapping."""

        return {command.vehicle_index: command.command for command in self.commands}

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics to dictionary."""

        return {
            "mode": self.mode.value,
            "dt": self.dt,
            "controlled_indices": list(self.controlled_indices),
            "any_saturated": self.any_saturated,
            "any_rate_limited": self.any_rate_limited,
            "any_safe_brake": self.any_safe_brake,
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass(slots=True)
class PCSRCControllerState:
    """Runtime memory for rate limiting and warm-start diagnostics."""

    previous_commands: MutableMapping[int, float] = field(default_factory=dict)
    step_count: int = 0

    def reset(self) -> None:
        """Reset controller memory."""

        self.previous_commands.clear()
        self.step_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime state."""

        return {
            "previous_commands": {
                int(index): float(command)
                for index, command in self.previous_commands.items()
            },
            "step_count": int(self.step_count),
        }


# ------------------------------------------------------------------------------
# PC-SRC controller implementation
# ------------------------------------------------------------------------------

@dataclass(slots=True)
class PCSRCController:
    """Main PC-SRC controller for CAVs in a mixed platoon."""

    config: PCSRCControllerConfig = field(default_factory=PCSRCControllerConfig)
    runtime: PCSRCControllerState = field(default_factory=PCSRCControllerState)

    def __post_init__(self) -> None:
        if not isinstance(self.config, PCSRCControllerConfig):
            self.config = PCSRCControllerConfig.from_dict(self.config)  # type: ignore[arg-type]

        if not isinstance(self.runtime, PCSRCControllerState):
            self.runtime = PCSRCControllerState()  # type: ignore[assignment]

    @property
    def mode(self) -> PCSRCMode:
        """Return active controller mode."""

        return self.config.mode

    def reset(self) -> None:
        """Reset controller runtime memory."""

        self.runtime.reset()

    def controlled_indices(self, model: MixedPlatoonModel) -> tuple[int, ...]:
        """Return controlled vehicle indices for the given model."""

        if self.config.controlled_indices is not None:
            for index in self.config.controlled_indices:
                if index >= model.size:
                    raise IndexError(
                        f"controlled index {index} out of range for platoon size {model.size}"
                    )
            return self.config.controlled_indices

        indices: list[int] = []

        for index, role in enumerate(model.config.roles):
            if index == 0:
                continue
            if role is VehicleRole.CAV:
                indices.append(index)

        return tuple(indices)

    def error_state_for(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        vehicle_index: int,
    ) -> CAVErrorState:
        """Compute local CAV error state from mixed-platoon state."""

        model.validate_state(state)

        if vehicle_index <= 0 or vehicle_index >= model.size:
            raise IndexError(
                f"vehicle_index must be in [1, {model.size - 1}], got {vehicle_index}"
            )

        ego = state.vehicle(vehicle_index)
        leader = state.vehicle(vehicle_index - 1)
        leader_length = model.config.vehicle_params[vehicle_index - 1].length

        spacing = leader.position - ego.position - leader_length
        desired_spacing = model.config.desired_gap_for(vehicle_index, ego.velocity)

        return CAVErrorState(
            spacing=spacing,
            desired_spacing=desired_spacing,
            spacing_error=spacing - desired_spacing,
            relative_velocity=leader.velocity - ego.velocity,
            ego_velocity=ego.velocity,
            leader_velocity=leader.velocity,
            ego_acceleration=ego.acceleration,
            leader_acceleration=leader.acceleration,
        )

    def compute_command_for_error(
        self,
        vehicle_index: int,
        error_state: CAVErrorState,
        dt: Number | None = None,
    ) -> PCSRCCommand:
        """Compute one CAV command from an already formed error state."""

        nominal = self.config.feedback_gains.evaluate(error_state)

        robust = 0.0
        if self.config.mode is PCSRCMode.ROBUST and self.config.robust_gains.enabled:
            robust = self.config.robust_gains.evaluate(error_state)

        raw = nominal + robust
        safe_brake = False

        if (
            self.config.mode is PCSRCMode.SAFE_FALLBACK
            or self.config.safety.requires_braking(error_state)
        ):
            safe_brake = True
            raw = (
                self.config.safety.brake_command
                if self.config.safety.brake_command is not None
                else self.config.acceleration_limits.min_acceleration
            )

        saturated_command = self.config.clip(raw)
        saturated = abs(saturated_command - raw) > 1e-12

        final_command = saturated_command
        rate_limited = False

        if dt is not None and self.config.max_command_rate is not None:
            step = _positive(dt, "dt")
            previous = self.runtime.previous_commands.get(vehicle_index, final_command)
            max_delta = self.config.max_command_rate * step
            limited = _clip(
                final_command,
                previous - max_delta,
                previous + max_delta,
            )
            rate_limited = abs(limited - final_command) > 1e-12
            final_command = limited

        # A final clip is useful after rate limiting if the previous command was
        # loaded from external state and lies outside bounds.
        clipped_after_rate = self.config.clip(final_command)
        saturated = saturated or abs(clipped_after_rate - final_command) > 1e-12
        final_command = clipped_after_rate

        if safe_brake:
            status = PCSRCStatus.SAFE_BRAKE
        elif rate_limited:
            status = PCSRCStatus.RATE_LIMITED
        elif saturated:
            status = PCSRCStatus.SATURATED
        else:
            status = PCSRCStatus.OK

        self.runtime.previous_commands[vehicle_index] = final_command

        return PCSRCCommand(
            vehicle_index=vehicle_index,
            command=final_command,
            raw_command=raw,
            nominal_command=nominal,
            robust_correction=robust,
            saturated=saturated,
            rate_limited=rate_limited,
            safe_brake=safe_brake,
            status=status,
            error_state=error_state,
        )

    def compute_commands(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        dt: Number | None = None,
    ) -> PCSRCControllerDiagnostics:
        """Compute commands for all controlled CAVs.

        Returns
        -------
        PCSRCControllerDiagnostics
            Rich diagnostic object containing command mapping and per-vehicle
            local error states.
        """

        model.validate_state(state)
        indices = self.controlled_indices(model)

        commands: list[PCSRCCommand] = []

        for index in indices:
            error = self.error_state_for(model=model, state=state, vehicle_index=index)
            commands.append(
                self.compute_command_for_error(
                    vehicle_index=index,
                    error_state=error,
                    dt=dt,
                )
            )

        self.runtime.step_count += 1

        return PCSRCControllerDiagnostics(
            mode=self.config.mode,
            commands=tuple(commands),
            controlled_indices=indices,
            dt=None if dt is None else float(dt),
        )

    def command_mapping(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        dt: Number | None = None,
    ) -> dict[int, float]:
        """Return only the command mapping ``{vehicle_index: command}``."""

        return self.compute_commands(model=model, state=state, dt=dt).command_mapping

    def __call__(
        self,
        model: MixedPlatoonModel,
        state: PlatoonState,
        dt: Number | None = None,
    ) -> dict[int, float]:
        """Alias for ``command_mapping``."""

        return self.command_mapping(model=model, state=state, dt=dt)

    def to_dict(self) -> dict[str, Any]:
        """Serialize controller configuration and runtime state."""

        return {
            "config": self.config.to_dict(),
            "runtime": self.runtime.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PCSRCController":
        """Create controller from dictionary."""

        controller = cls(
            config=PCSRCControllerConfig.from_dict(data.get("config", {})),
        )

        runtime_data = data.get("runtime", {})
        previous = runtime_data.get("previous_commands", {})

        controller.runtime.previous_commands.update(
            {int(index): float(command) for index, command in previous.items()}
        )
        controller.runtime.step_count = int(runtime_data.get("step_count", 0))

        return controller


# ------------------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------------------

def make_default_pc_src_controller(
    mode: PCSRCMode | str = PCSRCMode.ROBUST,
    k_spacing: Number = 0.20,
    k_velocity: Number = 0.70,
    k_acceleration: Number = 0.00,
    min_acceleration: Number = -3.0,
    max_acceleration: Number = 3.0,
) -> PCSRCController:
    """Create a default PC-SRC controller."""

    return PCSRCController(
        config=PCSRCControllerConfig(
            feedback_gains=PCSRCFeedbackGains(
                k_spacing=float(k_spacing),
                k_velocity=float(k_velocity),
                k_acceleration=float(k_acceleration),
            ),
            acceleration_limits=ActuatorLimits(
                min_acceleration=float(min_acceleration),
                max_acceleration=float(max_acceleration),
            ),
            mode=PCSRCMode(mode),
        )
    )


def compute_pc_src_commands(
    model: MixedPlatoonModel,
    state: PlatoonState,
    controller_config: PCSRCControllerConfig | Mapping[str, Any] | None = None,
    dt: Number | None = None,
) -> dict[int, float]:
    """Convenience wrapper for one-shot PC-SRC command computation."""

    config = (
        PCSRCControllerConfig()
        if controller_config is None
        else controller_config
        if isinstance(controller_config, PCSRCControllerConfig)
        else PCSRCControllerConfig.from_dict(controller_config)
    )

    return PCSRCController(config=config).command_mapping(
        model=model,
        state=state,
        dt=dt,
    )


def pc_src_nominal_command(
    error_state: CAVErrorState,
    gains: PCSRCFeedbackGains | Mapping[str, Any] | None = None,
    limits: ActuatorLimits | Mapping[str, Any] | None = None,
) -> float:
    """Compute a clipped nominal PC-SRC command for one local error state."""

    parsed_gains = (
        PCSRCFeedbackGains()
        if gains is None
        else gains
        if isinstance(gains, PCSRCFeedbackGains)
        else PCSRCFeedbackGains.from_dict(gains)
    )

    parsed_limits = (
        ActuatorLimits(min_acceleration=-3.0, max_acceleration=3.0)
        if limits is None
        else limits
        if isinstance(limits, ActuatorLimits)
        else ActuatorLimits.from_dict(limits)
    )

    return parsed_limits.clip(parsed_gains.evaluate(error_state))


__all__ = [
    "Number",
    "PCSRCCommand",
    "PCSRCController",
    "PCSRCControllerConfig",
    "PCSRCControllerDiagnostics",
    "PCSRCControllerState",
    "PCSRCFeedbackGains",
    "PCSRCMode",
    "PCSRCRobustGains",
    "PCSRCSafetyConfig",
    "PCSRCStatus",
    "compute_pc_src_commands",
    "make_default_pc_src_controller",
    "pc_src_nominal_command",
]
