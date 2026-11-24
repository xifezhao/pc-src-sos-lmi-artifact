# ==============================================================================
# src/pcsrc/control/lut.py
#
# Look-up table utilities for pcsrc controllers.
#
# Purpose:
#   - Provide lightweight 1-D and 2-D LUT interpolation for controller gains or
#     commands.
#   - Support clipping/extrapolation policies and deterministic serialization.
#   - Offer convenience wrappers for CAV local error-state based command lookup.
#
# Typical use:
#   Offline synthesis may generate gain or command values on a grid, e.g.
#
#       u = LUT(e, r)
#
#   where:
#       e is spacing error,
#       r is predecessor-relative velocity.
#
#   This module evaluates such tables at runtime without requiring NumPy.
#
# Design notes:
#   - Uses only Python standard library.
#   - Avoids NumPy/SciPy/control imports.
#   - Compatible with pcsrc.models.cav_dynamics.CAVErrorState.
# ==============================================================================

from __future__ import annotations

import bisect
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
    """Convert value to a finite float."""

    converted = float(value)

    if not isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}")

    return converted


def _positive(value: Number, name: str) -> float:
    """Convert value to finite float and require positivity."""

    converted = _as_float(value, name)

    if converted <= 0.0:
        raise ValueError(f"{name} must be positive, got {converted}")

    return converted


def _nonnegative(value: Number, name: str) -> float:
    """Convert value to finite float and require non-negativity."""

    converted = _as_float(value, name)

    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {converted}")

    return converted


def _clip(value: Number, lower: Number, upper: Number) -> float:
    """Clip scalar to ``[lower, upper]``."""

    x = _as_float(value, "value")
    lo = _as_float(lower, "lower")
    hi = _as_float(upper, "upper")

    if lo > hi:
        raise ValueError(f"lower must be <= upper, got {lo} > {hi}")

    return min(max(x, lo), hi)


def _lerp(left: Number, right: Number, alpha: Number) -> float:
    """Linear interpolation between two scalar values."""

    l = _as_float(left, "left")
    r = _as_float(right, "right")
    a = _as_float(alpha, "alpha")

    return (1.0 - a) * l + a * r


def _ensure_increasing_grid(grid: Sequence[Number], name: str) -> tuple[float, ...]:
    """Validate and return a strictly increasing finite grid."""

    values = tuple(_as_float(item, f"{name}[{idx}]") for idx, item in enumerate(grid))

    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one point")

    for left, right in zip(values, values[1:]):
        if right <= left:
            raise ValueError(f"{name} must be strictly increasing")

    return values


def _is_uniform_grid(grid: Sequence[float], *, rel_tol: float = 1e-9) -> bool:
    """Return True when all grid intervals are equal within *rel_tol*.

    A uniform grid allows O(1) index computation via direct floor division,
    which is the basis of the paper's O(1) lookup-table claim.
    """

    if len(grid) < 2:
        return True

    step = grid[1] - grid[0]

    if step == 0.0:
        return False

    for i in range(2, len(grid)):
        if abs((grid[i] - grid[i - 1]) - step) > rel_tol * abs(step):
            return False

    return True


def _find_interval(
    grid: Sequence[float],
    value: float,
    *,
    extrapolate: bool,
) -> tuple[int, int, float, bool]:
    """Find interpolation interval in a sorted grid.

    Complexity
    ----------
    O(1) for uniform grids (equal-step grids, direct index computation).
    O(log N) for non-uniform grids (``bisect.bisect_right``).

    Returns
    -------
    tuple[int, int, float, bool]
        ``(left_index, right_index, alpha, clipped)``.
    """

    x = _as_float(value, "value")
    n = len(grid)

    if n == 1:
        return 0, 0, 0.0, x != grid[0]

    clipped = False

    # --- boundary handling (shared for both grid types) ---
    if x <= grid[0]:
        if extrapolate:
            alpha = (x - grid[0]) / (grid[1] - grid[0])
            return 0, 1, alpha, False
        clipped = x < grid[0]
        return 0, 0, 0.0, clipped

    if x >= grid[-1]:
        if extrapolate:
            alpha = (x - grid[-2]) / (grid[-1] - grid[-2])
            return n - 2, n - 1, alpha, False
        clipped = x > grid[-1]
        return n - 1, n - 1, 0.0, clipped

    # --- interior lookup ---
    if _is_uniform_grid(grid):
        # O(1): compute index directly from step size.
        step = (grid[-1] - grid[0]) / (n - 1)
        raw_idx = int((x - grid[0]) / step)
        idx = max(0, min(raw_idx, n - 2))
    else:
        # O(log N): binary search fallback for non-uniform grids.
        idx = bisect.bisect_right(grid, x, 0, n) - 1
        idx = max(0, min(idx, n - 2))

    left_value = grid[idx]
    right_value = grid[idx + 1]
    alpha = (x - left_value) / (right_value - left_value)
    return idx, idx + 1, alpha, False


def _matrix_shape(values: Sequence[Sequence[Number]]) -> tuple[int, int]:
    """Return rectangular matrix shape and validate finite entries."""

    rows = len(values)

    if rows == 0:
        raise ValueError("matrix must contain at least one row")

    cols = len(values[0])

    if cols == 0:
        raise ValueError("matrix must contain at least one column")

    for row_idx, row in enumerate(values):
        if len(row) != cols:
            raise ValueError("matrix rows must have equal length")

        for col_idx, item in enumerate(row):
            _as_float(item, f"matrix[{row_idx}][{col_idx}]")

    return rows, cols


# ------------------------------------------------------------------------------
# Enums and result containers
# ------------------------------------------------------------------------------

class LUTBoundaryMode(str, Enum):
    """Boundary behavior for LUT evaluation."""

    CLIP = "clip"
    EXTRAPOLATE = "extrapolate"


class LUTVariable(str, Enum):
    """Built-in variables that can be extracted from CAVErrorState."""

    SPACING = "spacing"
    DESIRED_SPACING = "desired_spacing"
    SPACING_ERROR = "spacing_error"
    RELATIVE_VELOCITY = "relative_velocity"
    EGO_VELOCITY = "ego_velocity"
    LEADER_VELOCITY = "leader_velocity"
    EGO_ACCELERATION = "ego_acceleration"
    LEADER_ACCELERATION = "leader_acceleration"
    TIME = "time"
    CUSTOM = "custom"


def lut_variable_value(
    variable: LUTVariable | str,
    error_state: CAVErrorState,
    *,
    time: Number = 0.0,
    custom_value: Number | None = None,
) -> float:
    """Extract a scalar LUT variable from a local CAV error state."""

    var = variable if isinstance(variable, LUTVariable) else LUTVariable(variable)

    if var is LUTVariable.SPACING:
        return error_state.spacing

    if var is LUTVariable.DESIRED_SPACING:
        return error_state.desired_spacing

    if var is LUTVariable.SPACING_ERROR:
        return error_state.spacing_error

    if var is LUTVariable.RELATIVE_VELOCITY:
        return error_state.relative_velocity

    if var is LUTVariable.EGO_VELOCITY:
        return error_state.ego_velocity

    if var is LUTVariable.LEADER_VELOCITY:
        return error_state.leader_velocity

    if var is LUTVariable.EGO_ACCELERATION:
        return error_state.ego_acceleration

    if var is LUTVariable.LEADER_ACCELERATION:
        return error_state.leader_acceleration

    if var is LUTVariable.TIME:
        return _nonnegative(time, "time")

    if var is LUTVariable.CUSTOM:
        if custom_value is None:
            raise ValueError("custom_value is required for CUSTOM LUT variable")
        return _as_float(custom_value, "custom_value")

    raise ValueError(f"unsupported LUT variable: {variable!r}")


@dataclass(frozen=True, slots=True)
class LUTLookupResult:
    """Diagnostic result of one LUT lookup."""

    value: float
    clipped: bool = False
    extrapolated: bool = False
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _as_float(self.value, "value"))
        object.__setattr__(self, "clipped", bool(self.clipped))
        object.__setattr__(self, "extrapolated", bool(self.extrapolated))

        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Serialize lookup result to dictionary."""

        return {
            "value": self.value,
            "clipped": self.clipped,
            "extrapolated": self.extrapolated,
            "metadata": self.metadata or {},
        }


# ------------------------------------------------------------------------------
# 1-D LUT
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LUT1D:
    """One-dimensional scalar look-up table with linear interpolation."""

    grid: tuple[float, ...]
    values: tuple[float, ...]
    boundary_mode: LUTBoundaryMode = LUTBoundaryMode.CLIP
    name: str = ""

    def __post_init__(self) -> None:
        grid = _ensure_increasing_grid(self.grid, "grid")
        values = tuple(
            _as_float(item, f"values[{idx}]")
            for idx, item in enumerate(self.values)
        )

        if len(values) != len(grid):
            raise ValueError(
                f"values length must match grid length, got "
                f"{len(values)} != {len(grid)}"
            )

        mode = (
            self.boundary_mode
            if isinstance(self.boundary_mode, LUTBoundaryMode)
            else LUTBoundaryMode(self.boundary_mode)
        )

        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "boundary_mode", mode)
        object.__setattr__(self, "name", str(self.name))

    @property
    def extrapolate(self) -> bool:
        """Return whether the LUT extrapolates outside the grid."""

        return self.boundary_mode is LUTBoundaryMode.EXTRAPOLATE

    @property
    def min_grid(self) -> float:
        """Return lower grid endpoint."""

        return self.grid[0]

    @property
    def max_grid(self) -> float:
        """Return upper grid endpoint."""

        return self.grid[-1]

    def lookup(self, x: Number, *, diagnostics: bool = False) -> float | LUTLookupResult:
        """Evaluate LUT at scalar ``x``."""

        left, right, alpha, clipped = _find_interval(
            self.grid,
            x,
            extrapolate=self.extrapolate,
        )

        if left == right:
            value = self.values[left]
        else:
            value = _lerp(self.values[left], self.values[right], alpha)

        if not diagnostics:
            return value

        return LUTLookupResult(
            value=value,
            clipped=clipped and not self.extrapolate,
            extrapolated=self.extrapolate and (x < self.min_grid or x > self.max_grid),
            metadata={
                "name": self.name,
                "x": float(x),
                "left_index": left,
                "right_index": right,
                "alpha": alpha,
            },
        )

    def __call__(self, x: Number) -> float:
        """Evaluate LUT at scalar ``x``."""

        return float(self.lookup(x))

    def map_values(self, func: Callable[[float], Number], name: str | None = None) -> "LUT1D":
        """Return a new LUT by applying a scalar function to all table values."""

        return LUT1D(
            grid=self.grid,
            values=tuple(_as_float(func(value), "mapped_value") for value in self.values),
            boundary_mode=self.boundary_mode,
            name=self.name if name is None else name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize 1-D LUT to dictionary."""

        return {
            "grid": list(self.grid),
            "values": list(self.values),
            "boundary_mode": self.boundary_mode.value,
            "name": self.name,
        }

    def replace(self, **changes: Any) -> "LUT1D":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[Number, Number]],
        *,
        boundary_mode: LUTBoundaryMode | str = LUTBoundaryMode.CLIP,
        name: str = "",
    ) -> "LUT1D":
        """Create 1-D LUT from ``(grid, value)`` pairs."""

        sorted_pairs = sorted(
            ((_as_float(x, "grid"), _as_float(y, "value")) for x, y in pairs),
            key=lambda item: item[0],
        )

        if not sorted_pairs:
            raise ValueError("pairs must contain at least one item")

        grid = tuple(item[0] for item in sorted_pairs)
        values = tuple(item[1] for item in sorted_pairs)

        return cls(
            grid=grid,
            values=values,
            boundary_mode=LUTBoundaryMode(boundary_mode),
            name=name,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LUT1D":
        """Create 1-D LUT from mapping."""

        return cls(
            grid=tuple(float(item) for item in data["grid"]),
            values=tuple(float(item) for item in data["values"]),
            boundary_mode=LUTBoundaryMode(data.get("boundary_mode", LUTBoundaryMode.CLIP.value)),
            name=str(data.get("name", "")),
        )


# ------------------------------------------------------------------------------
# 2-D LUT
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LUT2D:
    """Two-dimensional scalar LUT with bilinear interpolation.

    The value matrix shape must be:

        ``len(x_grid)`` rows by ``len(y_grid)`` columns.

    Thus:

        ``values[i][j]`` corresponds to ``x_grid[i], y_grid[j]``.
    """

    x_grid: tuple[float, ...]
    y_grid: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    boundary_mode: LUTBoundaryMode = LUTBoundaryMode.CLIP
    name: str = ""

    def __post_init__(self) -> None:
        x_grid = _ensure_increasing_grid(self.x_grid, "x_grid")
        y_grid = _ensure_increasing_grid(self.y_grid, "y_grid")

        _matrix_shape(self.values)

        matrix = tuple(
            tuple(_as_float(item, f"values[{i}][{j}]") for j, item in enumerate(row))
            for i, row in enumerate(self.values)
        )

        if len(matrix) != len(x_grid):
            raise ValueError(
                f"values row count must match x_grid length, got "
                f"{len(matrix)} != {len(x_grid)}"
            )

        for row in matrix:
            if len(row) != len(y_grid):
                raise ValueError(
                    f"each values row must match y_grid length, got "
                    f"{len(row)} != {len(y_grid)}"
                )

        mode = (
            self.boundary_mode
            if isinstance(self.boundary_mode, LUTBoundaryMode)
            else LUTBoundaryMode(self.boundary_mode)
        )

        object.__setattr__(self, "x_grid", x_grid)
        object.__setattr__(self, "y_grid", y_grid)
        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "boundary_mode", mode)
        object.__setattr__(self, "name", str(self.name))

    @property
    def extrapolate(self) -> bool:
        """Return whether the LUT extrapolates outside its grid."""

        return self.boundary_mode is LUTBoundaryMode.EXTRAPOLATE

    def lookup(
        self,
        x: Number,
        y: Number,
        *,
        diagnostics: bool = False,
    ) -> float | LUTLookupResult:
        """Evaluate LUT at ``(x, y)`` using bilinear interpolation."""

        xi0, xi1, ax, clipped_x = _find_interval(
            self.x_grid,
            x,
            extrapolate=self.extrapolate,
        )
        yi0, yi1, ay, clipped_y = _find_interval(
            self.y_grid,
            y,
            extrapolate=self.extrapolate,
        )

        if xi0 == xi1 and yi0 == yi1:
            value = self.values[xi0][yi0]
        elif xi0 == xi1:
            value = _lerp(self.values[xi0][yi0], self.values[xi0][yi1], ay)
        elif yi0 == yi1:
            value = _lerp(self.values[xi0][yi0], self.values[xi1][yi0], ax)
        else:
            q11 = self.values[xi0][yi0]
            q12 = self.values[xi0][yi1]
            q21 = self.values[xi1][yi0]
            q22 = self.values[xi1][yi1]

            low = _lerp(q11, q21, ax)
            high = _lerp(q12, q22, ax)
            value = _lerp(low, high, ay)

        if not diagnostics:
            return value

        out_of_x = x < self.x_grid[0] or x > self.x_grid[-1]
        out_of_y = y < self.y_grid[0] or y > self.y_grid[-1]

        return LUTLookupResult(
            value=value,
            clipped=(clipped_x or clipped_y) and not self.extrapolate,
            extrapolated=self.extrapolate and (out_of_x or out_of_y),
            metadata={
                "name": self.name,
                "x": float(x),
                "y": float(y),
                "x_left_index": xi0,
                "x_right_index": xi1,
                "y_left_index": yi0,
                "y_right_index": yi1,
                "alpha_x": ax,
                "alpha_y": ay,
            },
        )

    def __call__(self, x: Number, y: Number) -> float:
        """Evaluate LUT at ``(x, y)``."""

        return float(self.lookup(x, y))

    def map_values(
        self,
        func: Callable[[float], Number],
        name: str | None = None,
    ) -> "LUT2D":
        """Return a new 2-D LUT by applying a function to all values."""

        return LUT2D(
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            values=tuple(
                tuple(_as_float(func(value), "mapped_value") for value in row)
                for row in self.values
            ),
            boundary_mode=self.boundary_mode,
            name=self.name if name is None else name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize 2-D LUT to dictionary."""

        return {
            "x_grid": list(self.x_grid),
            "y_grid": list(self.y_grid),
            "values": [list(row) for row in self.values],
            "boundary_mode": self.boundary_mode.value,
            "name": self.name,
        }

    def replace(self, **changes: Any) -> "LUT2D":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LUT2D":
        """Create 2-D LUT from mapping."""

        return cls(
            x_grid=tuple(float(item) for item in data["x_grid"]),
            y_grid=tuple(float(item) for item in data["y_grid"]),
            values=tuple(
                tuple(float(item) for item in row)
                for row in data["values"]
            ),
            boundary_mode=LUTBoundaryMode(data.get("boundary_mode", LUTBoundaryMode.CLIP.value)),
            name=str(data.get("name", "")),
        )


# ------------------------------------------------------------------------------
# Gain LUTs and command LUTs
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LUTFeedbackGains:
    """Feedback gains obtained from one or more LUTs.

    Control convention:

        u = k_spacing * e + k_velocity * r - k_acceleration * a
    """

    k_spacing: float
    k_velocity: float
    k_acceleration: float = 0.0

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

    def to_dict(self) -> dict[str, float]:
        """Serialize gains to dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LUT1DGainSchedule:
    """Gain schedule backed by three 1-D LUTs."""

    variable: LUTVariable
    k_spacing_lut: LUT1D
    k_velocity_lut: LUT1D
    k_acceleration_lut: LUT1D | None = None

    def __post_init__(self) -> None:
        variable = self.variable if isinstance(self.variable, LUTVariable) else LUTVariable(self.variable)

        object.__setattr__(self, "variable", variable)

    def lookup(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_value: Number | None = None,
    ) -> LUTFeedbackGains:
        """Return feedback gains from 1-D LUTs."""

        z = lut_variable_value(
            self.variable,
            error_state,
            time=time,
            custom_value=custom_value,
        )

        return LUTFeedbackGains(
            k_spacing=float(self.k_spacing_lut.lookup(z)),
            k_velocity=float(self.k_velocity_lut.lookup(z)),
            k_acceleration=(
                0.0
                if self.k_acceleration_lut is None
                else float(self.k_acceleration_lut.lookup(z))
            ),
        )

    def command(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_value: Number | None = None,
    ) -> float:
        """Evaluate feedback command using LUT gains."""

        return self.lookup(
            error_state,
            time=time,
            custom_value=custom_value,
        ).evaluate(error_state)

    def to_dict(self) -> dict[str, Any]:
        """Serialize gain schedule."""

        return {
            "variable": self.variable.value,
            "k_spacing_lut": self.k_spacing_lut.to_dict(),
            "k_velocity_lut": self.k_velocity_lut.to_dict(),
            "k_acceleration_lut": (
                None
                if self.k_acceleration_lut is None
                else self.k_acceleration_lut.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LUT1DGainSchedule":
        """Create 1-D gain schedule from mapping."""

        return cls(
            variable=LUTVariable(data["variable"]),
            k_spacing_lut=LUT1D.from_dict(data["k_spacing_lut"]),
            k_velocity_lut=LUT1D.from_dict(data["k_velocity_lut"]),
            k_acceleration_lut=(
                None
                if data.get("k_acceleration_lut", None) is None
                else LUT1D.from_dict(data["k_acceleration_lut"])
            ),
        )


@dataclass(frozen=True, slots=True)
class LUT2DCommandTable:
    """Command table backed by a 2-D LUT and two error-state variables."""

    x_variable: LUTVariable = LUTVariable.SPACING_ERROR
    y_variable: LUTVariable = LUTVariable.RELATIVE_VELOCITY
    command_lut: LUT2D = LUT2D(
        x_grid=(-10.0, 0.0, 10.0),
        y_grid=(-5.0, 0.0, 5.0),
        values=(
            (-3.0, -2.0, -1.0),
            (-1.0, 0.0, 1.0),
            (1.0, 2.0, 3.0),
        ),
        boundary_mode=LUTBoundaryMode.CLIP,
        name="default_command_lut",
    )

    def __post_init__(self) -> None:
        x_variable = (
            self.x_variable
            if isinstance(self.x_variable, LUTVariable)
            else LUTVariable(self.x_variable)
        )
        y_variable = (
            self.y_variable
            if isinstance(self.y_variable, LUTVariable)
            else LUTVariable(self.y_variable)
        )
        command_lut = (
            self.command_lut
            if isinstance(self.command_lut, LUT2D)
            else LUT2D.from_dict(self.command_lut)  # type: ignore[arg-type]
        )

        object.__setattr__(self, "x_variable", x_variable)
        object.__setattr__(self, "y_variable", y_variable)
        object.__setattr__(self, "command_lut", command_lut)

    def lookup(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_x: Number | None = None,
        custom_y: Number | None = None,
        diagnostics: bool = False,
    ) -> float | LUTLookupResult:
        """Lookup command from error-state variables."""

        x = lut_variable_value(
            self.x_variable,
            error_state,
            time=time,
            custom_x=custom_x,
        )
        y = lut_variable_value(
            self.y_variable,
            error_state,
            time=time,
            custom_y=custom_y,
        )

        return self.command_lut.lookup(x, y, diagnostics=diagnostics)

    def command(
        self,
        error_state: CAVErrorState,
        *,
        time: Number = 0.0,
        custom_x: Number | None = None,
        custom_y: Number | None = None,
    ) -> float:
        """Return final command value from the 2-D command LUT."""

        return float(
            self.lookup(
                error_state,
                time=time,
                custom_x=custom_x,
                custom_y=custom_y,
                diagnostics=False,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize command table."""

        return {
            "x_variable": self.x_variable.value,
            "y_variable": self.y_variable.value,
            "command_lut": self.command_lut.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LUT2DCommandTable":
        """Create command table from mapping."""

        return cls(
            x_variable=LUTVariable(data.get("x_variable", LUTVariable.SPACING_ERROR.value)),
            y_variable=LUTVariable(data.get("y_variable", LUTVariable.RELATIVE_VELOCITY.value)),
            command_lut=LUT2D.from_dict(data["command_lut"]),
        )


# ------------------------------------------------------------------------------
# Convenience constructors and helpers
# ------------------------------------------------------------------------------

def make_linear_1d_lut(
    xmin: Number,
    xmax: Number,
    ymin: Number,
    ymax: Number,
    *,
    num: int = 2,
    boundary_mode: LUTBoundaryMode | str = LUTBoundaryMode.CLIP,
    name: str = "",
) -> LUT1D:
    """Create a simple linearly sampled 1-D LUT."""

    if num < 2:
        raise ValueError("num must be at least 2")

    x0 = _as_float(xmin, "xmin")
    x1 = _as_float(xmax, "xmax")
    y0 = _as_float(ymin, "ymin")
    y1 = _as_float(ymax, "ymax")

    if x1 <= x0:
        raise ValueError("xmax must be greater than xmin")

    grid = tuple(x0 + (x1 - x0) * i / (num - 1) for i in range(num))
    values = tuple(y0 + (y1 - y0) * i / (num - 1) for i in range(num))

    return LUT1D(
        grid=grid,
        values=values,
        boundary_mode=LUTBoundaryMode(boundary_mode),
        name=name,
    )


def make_default_spacing_velocity_command_lut() -> LUT2DCommandTable:
    """Create a default spacing-error / relative-velocity command LUT."""

    return LUT2DCommandTable()


def lookup_1d(
    grid: Sequence[Number],
    values: Sequence[Number],
    x: Number,
    *,
    boundary_mode: LUTBoundaryMode | str = LUTBoundaryMode.CLIP,
) -> float:
    """One-shot 1-D LUT lookup."""

    return float(
        LUT1D(
            grid=tuple(float(item) for item in grid),
            values=tuple(float(item) for item in values),
            boundary_mode=LUTBoundaryMode(boundary_mode),
        ).lookup(x)
    )


def lookup_2d(
    x_grid: Sequence[Number],
    y_grid: Sequence[Number],
    values: Sequence[Sequence[Number]],
    x: Number,
    y: Number,
    *,
    boundary_mode: LUTBoundaryMode | str = LUTBoundaryMode.CLIP,
) -> float:
    """One-shot 2-D LUT lookup."""

    return float(
        LUT2D(
            x_grid=tuple(float(item) for item in x_grid),
            y_grid=tuple(float(item) for item in y_grid),
            values=tuple(tuple(float(v) for v in row) for row in values),
            boundary_mode=LUTBoundaryMode(boundary_mode),
        ).lookup(x, y)
    )


def lut_command(
    error_state: CAVErrorState,
    table: LUT2DCommandTable | Mapping[str, Any] | None = None,
    *,
    time: Number = 0.0,
) -> float:
    """Convenience wrapper for evaluating a LUT command table."""

    parsed = (
        LUT2DCommandTable()
        if table is None
        else table
        if isinstance(table, LUT2DCommandTable)
        else LUT2DCommandTable.from_dict(table)
    )

    return parsed.command(error_state=error_state, time=time)


def to_pcsrc_feedback_gains(gains: LUTFeedbackGains) -> Any:
    """Convert LUT gains to ``PCSRCFeedbackGains`` when available."""

    from pcsrc.control.pc_src_controller import PCSRCFeedbackGains

    return PCSRCFeedbackGains(
        k_spacing=gains.k_spacing,
        k_velocity=gains.k_velocity,
        k_acceleration=gains.k_acceleration,
    )


__all__ = [
    "LUT1D",
    "LUT1DGainSchedule",
    "LUT2D",
    "LUT2DCommandTable",
    "LUTBoundaryMode",
    "LUTFeedbackGains",
    "LUTLookupResult",
    "LUTVariable",
    "Number",
    "lookup_1d",
    "lookup_2d",
    "lut_command",
    "lut_variable_value",
    "make_default_spacing_velocity_command_lut",
    "make_linear_1d_lut",
    "to_pcsrc_feedback_gains",
]
