# ==============================================================================
# src/pcsrc/control/interpolation.py
#
# Interpolation utilities for pcsrc controllers.
#
# Purpose:
#   - Provide lightweight interpolation primitives for controller deployment.
#   - Support 1-D linear interpolation and 2-D bilinear interpolation.
#   - Share boundary handling logic across LUT, gain scheduling, and policy
#     approximation modules.
#   - Avoid heavy dependencies such as NumPy/SciPy for real-time simulation loops.
#
# Design notes:
#   - Uses only the Python standard library.
#   - All grids are assumed to be strictly increasing.
#   - Boundary behavior can be configured as clipping or extrapolation.
#
# Conventions:
#   For a 1-D grid x[0] < x[1] < ... < x[n-1], interpolation uses
#
#       y = (1 - alpha) y_left + alpha y_right
#
#   where
#
#       alpha = (x_query - x_left) / (x_right - x_left)
#
#   For 2-D grids, values[i][j] corresponds to x_grid[i], y_grid[j].
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence


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


def _clip(value: Number, lower: Number, upper: Number) -> float:
    """Clip scalar to ``[lower, upper]``."""

    x = _as_float(value, "value")
    lo = _as_float(lower, "lower")
    hi = _as_float(upper, "upper")

    if lo > hi:
        raise ValueError(f"lower must be <= upper, got {lo} > {hi}")

    return min(max(x, lo), hi)


def _validate_grid(grid: Sequence[Number], name: str = "grid") -> tuple[float, ...]:
    """Validate a finite strictly increasing grid."""

    values = tuple(_as_float(item, f"{name}[{idx}]") for idx, item in enumerate(grid))

    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one point")

    for left, right in zip(values, values[1:]):
        if right <= left:
            raise ValueError(f"{name} must be strictly increasing")

    return values


def _validate_values_1d(
    values: Sequence[Number],
    expected_size: int,
    name: str = "values",
) -> tuple[float, ...]:
    """Validate finite 1-D values aligned with a grid."""

    parsed = tuple(_as_float(item, f"{name}[{idx}]") for idx, item in enumerate(values))

    if len(parsed) != expected_size:
        raise ValueError(
            f"{name} length must be {expected_size}, got {len(parsed)}"
        )

    return parsed


def _validate_values_2d(
    values: Sequence[Sequence[Number]],
    expected_rows: int,
    expected_cols: int,
    name: str = "values",
) -> tuple[tuple[float, ...], ...]:
    """Validate finite rectangular 2-D values."""

    if len(values) != expected_rows:
        raise ValueError(
            f"{name} row count must be {expected_rows}, got {len(values)}"
        )

    matrix: list[tuple[float, ...]] = []

    for i, row in enumerate(values):
        if len(row) != expected_cols:
            raise ValueError(
                f"{name}[{i}] length must be {expected_cols}, got {len(row)}"
            )

        matrix.append(
            tuple(_as_float(item, f"{name}[{i}][{j}]") for j, item in enumerate(row))
        )

    return tuple(matrix)


# ------------------------------------------------------------------------------
# Boundary mode and interval metadata
# ------------------------------------------------------------------------------

class BoundaryMode(str, Enum):
    """Boundary behavior for interpolation queries."""

    CLIP = "clip"
    EXTRAPOLATE = "extrapolate"


@dataclass(frozen=True, slots=True)
class Interval:
    """Interpolation interval metadata.

    Attributes
    ----------
    left:
        Left grid index.
    right:
        Right grid index.
    alpha:
        Interpolation coordinate. For clipping at endpoints, alpha is ``0``.
    clipped:
        Whether the query was clipped to a boundary.
    extrapolated:
        Whether the query is outside the grid and extrapolation is enabled.
    query:
        Original query value.
    effective_query:
        Query value after clipping, or original value for extrapolation.
    """

    left: int
    right: int
    alpha: float
    clipped: bool = False
    extrapolated: bool = False
    query: float = 0.0
    effective_query: float = 0.0

    def __post_init__(self) -> None:
        if self.left < 0 or self.right < 0:
            raise ValueError("interval indices must be non-negative")

        if self.left > self.right:
            raise ValueError("left index must be <= right index")

        object.__setattr__(self, "alpha", _as_float(self.alpha, "alpha"))
        object.__setattr__(self, "clipped", bool(self.clipped))
        object.__setattr__(self, "extrapolated", bool(self.extrapolated))
        object.__setattr__(self, "query", _as_float(self.query, "query"))
        object.__setattr__(
            self,
            "effective_query",
            _as_float(self.effective_query, "effective_query"),
        )

    @property
    def is_degenerate(self) -> bool:
        """Return whether interval refers to one grid point."""

        return self.left == self.right

    def to_dict(self) -> dict[str, Any]:
        """Serialize interval metadata."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class InterpolationResult:
    """Diagnostic result of an interpolation evaluation."""

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
        """Serialize result."""

        return {
            "value": self.value,
            "clipped": self.clipped,
            "extrapolated": self.extrapolated,
            "metadata": self.metadata or {},
        }


# ------------------------------------------------------------------------------
# Core interpolation primitives
# ------------------------------------------------------------------------------

def lerp(left: Number, right: Number, alpha: Number) -> float:
    """Linearly interpolate between two scalars."""

    l = _as_float(left, "left")
    r = _as_float(right, "right")
    a = _as_float(alpha, "alpha")

    return (1.0 - a) * l + a * r


def inverse_lerp(left: Number, right: Number, value: Number) -> float:
    """Return interpolation coordinate of ``value`` between ``left`` and ``right``."""

    l = _as_float(left, "left")
    r = _as_float(right, "right")
    x = _as_float(value, "value")

    if r == l:
        raise ValueError("left and right must be distinct for inverse_lerp")

    return (x - l) / (r - l)


def bilerp(
    q11: Number,
    q12: Number,
    q21: Number,
    q22: Number,
    alpha_x: Number,
    alpha_y: Number,
) -> float:
    """Bilinearly interpolate four corner values.

    Corner convention:

    - ``q11``: left-bottom / ``x0, y0``
    - ``q12``: left-top / ``x0, y1``
    - ``q21``: right-bottom / ``x1, y0``
    - ``q22``: right-top / ``x1, y1``
    """

    ax = _as_float(alpha_x, "alpha_x")
    ay = _as_float(alpha_y, "alpha_y")

    low = lerp(q11, q21, ax)
    high = lerp(q12, q22, ax)

    return lerp(low, high, ay)


def find_interval(
    grid: Sequence[Number],
    query: Number,
    *,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
) -> Interval:
    """Find interpolation interval for a scalar query.

    Parameters
    ----------
    grid:
        Strictly increasing grid.
    query:
        Query value.
    boundary_mode:
        ``CLIP`` clamps out-of-range queries to endpoints.
        ``EXTRAPOLATE`` uses the first or last segment outside the grid.

    Returns
    -------
    Interval
        Interval indices and interpolation coordinate.
    """

    values = _validate_grid(grid)
    x = _as_float(query, "query")
    mode = boundary_mode if isinstance(boundary_mode, BoundaryMode) else BoundaryMode(boundary_mode)

    if len(values) == 1:
        effective = values[0] if mode is BoundaryMode.CLIP else x

        return Interval(
            left=0,
            right=0,
            alpha=0.0,
            clipped=(x != values[0] and mode is BoundaryMode.CLIP),
            extrapolated=(x != values[0] and mode is BoundaryMode.EXTRAPOLATE),
            query=x,
            effective_query=effective,
        )

    if x <= values[0]:
        if mode is BoundaryMode.EXTRAPOLATE:
            alpha = inverse_lerp(values[0], values[1], x)

            return Interval(
                left=0,
                right=1,
                alpha=alpha,
                clipped=False,
                extrapolated=x < values[0],
                query=x,
                effective_query=x,
            )

        return Interval(
            left=0,
            right=0,
            alpha=0.0,
            clipped=x < values[0],
            extrapolated=False,
            query=x,
            effective_query=values[0],
        )

    if x >= values[-1]:
        last = len(values) - 1

        if mode is BoundaryMode.EXTRAPOLATE:
            alpha = inverse_lerp(values[last - 1], values[last], x)

            return Interval(
                left=last - 1,
                right=last,
                alpha=alpha,
                clipped=False,
                extrapolated=x > values[-1],
                query=x,
                effective_query=x,
            )

        return Interval(
            left=last,
            right=last,
            alpha=0.0,
            clipped=x > values[-1],
            extrapolated=False,
            query=x,
            effective_query=values[-1],
        )

    for idx in range(len(values) - 1):
        left = values[idx]
        right = values[idx + 1]

        if left <= x <= right:
            return Interval(
                left=idx,
                right=idx + 1,
                alpha=inverse_lerp(left, right, x),
                clipped=False,
                extrapolated=False,
                query=x,
                effective_query=x,
            )

    # Defensive fallback. Valid strictly increasing grids should never reach here.
    last = len(values) - 1
    return Interval(
        left=last,
        right=last,
        alpha=0.0,
        clipped=True,
        extrapolated=False,
        query=x,
        effective_query=values[-1],
    )


def interpolate_1d(
    grid: Sequence[Number],
    values: Sequence[Number],
    query: Number,
    *,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
    diagnostics: bool = False,
) -> float | InterpolationResult:
    """Evaluate 1-D linear interpolation."""

    x_grid = _validate_grid(grid)
    y_values = _validate_values_1d(values, len(x_grid))
    interval = find_interval(x_grid, query, boundary_mode=boundary_mode)

    if interval.is_degenerate:
        value = y_values[interval.left]
    else:
        value = lerp(
            y_values[interval.left],
            y_values[interval.right],
            interval.alpha,
        )

    if not diagnostics:
        return value

    return InterpolationResult(
        value=value,
        clipped=interval.clipped,
        extrapolated=interval.extrapolated,
        metadata={
            "interval": interval.to_dict(),
            "grid_min": x_grid[0],
            "grid_max": x_grid[-1],
        },
    )


def interpolate_2d(
    x_grid: Sequence[Number],
    y_grid: Sequence[Number],
    values: Sequence[Sequence[Number]],
    x_query: Number,
    y_query: Number,
    *,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
    diagnostics: bool = False,
) -> float | InterpolationResult:
    """Evaluate 2-D bilinear interpolation."""

    xs = _validate_grid(x_grid, "x_grid")
    ys = _validate_grid(y_grid, "y_grid")
    matrix = _validate_values_2d(values, len(xs), len(ys))

    ix = find_interval(xs, x_query, boundary_mode=boundary_mode)
    iy = find_interval(ys, y_query, boundary_mode=boundary_mode)

    if ix.is_degenerate and iy.is_degenerate:
        value = matrix[ix.left][iy.left]
    elif ix.is_degenerate:
        value = lerp(
            matrix[ix.left][iy.left],
            matrix[ix.left][iy.right],
            iy.alpha,
        )
    elif iy.is_degenerate:
        value = lerp(
            matrix[ix.left][iy.left],
            matrix[ix.right][iy.left],
            ix.alpha,
        )
    else:
        value = bilerp(
            q11=matrix[ix.left][iy.left],
            q12=matrix[ix.left][iy.right],
            q21=matrix[ix.right][iy.left],
            q22=matrix[ix.right][iy.right],
            alpha_x=ix.alpha,
            alpha_y=iy.alpha,
        )

    if not diagnostics:
        return value

    return InterpolationResult(
        value=value,
        clipped=ix.clipped or iy.clipped,
        extrapolated=ix.extrapolated or iy.extrapolated,
        metadata={
            "x_interval": ix.to_dict(),
            "y_interval": iy.to_dict(),
            "x_grid_min": xs[0],
            "x_grid_max": xs[-1],
            "y_grid_min": ys[0],
            "y_grid_max": ys[-1],
        },
    )


# ------------------------------------------------------------------------------
# Reusable interpolator classes
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LinearInterpolator1D:
    """Reusable 1-D linear interpolator."""

    grid: tuple[float, ...]
    values: tuple[float, ...]
    boundary_mode: BoundaryMode = BoundaryMode.CLIP
    name: str = ""

    def __post_init__(self) -> None:
        grid = _validate_grid(self.grid)
        values = _validate_values_1d(self.values, len(grid))
        mode = (
            self.boundary_mode
            if isinstance(self.boundary_mode, BoundaryMode)
            else BoundaryMode(self.boundary_mode)
        )

        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "boundary_mode", mode)
        object.__setattr__(self, "name", str(self.name))

    @property
    def min_grid(self) -> float:
        """Return minimum grid value."""

        return self.grid[0]

    @property
    def max_grid(self) -> float:
        """Return maximum grid value."""

        return self.grid[-1]

    def interval(self, query: Number) -> Interval:
        """Return interpolation interval for query."""

        return find_interval(
            self.grid,
            query,
            boundary_mode=self.boundary_mode,
        )

    def evaluate(
        self,
        query: Number,
        *,
        diagnostics: bool = False,
    ) -> float | InterpolationResult:
        """Evaluate interpolator."""

        result = interpolate_1d(
            grid=self.grid,
            values=self.values,
            query=query,
            boundary_mode=self.boundary_mode,
            diagnostics=diagnostics,
        )

        if diagnostics and isinstance(result, InterpolationResult):
            metadata = dict(result.metadata or {})
            metadata["name"] = self.name

            return InterpolationResult(
                value=result.value,
                clipped=result.clipped,
                extrapolated=result.extrapolated,
                metadata=metadata,
            )

        return result

    def __call__(self, query: Number) -> float:
        """Evaluate interpolator and return scalar."""

        return float(self.evaluate(query))

    def map_values(
        self,
        func: Callable[[float], Number],
        *,
        name: str | None = None,
    ) -> "LinearInterpolator1D":
        """Create a new interpolator by mapping values through ``func``."""

        return LinearInterpolator1D(
            grid=self.grid,
            values=tuple(
                _as_float(func(value), f"mapped_values[{idx}]")
                for idx, value in enumerate(self.values)
            ),
            boundary_mode=self.boundary_mode,
            name=self.name if name is None else name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize interpolator."""

        return {
            "grid": list(self.grid),
            "values": list(self.values),
            "boundary_mode": self.boundary_mode.value,
            "name": self.name,
        }

    def replace(self, **changes: Any) -> "LinearInterpolator1D":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[Number, Number]],
        *,
        boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
        name: str = "",
    ) -> "LinearInterpolator1D":
        """Create interpolator from ``(grid, value)`` pairs."""

        parsed = sorted(
            (
                (_as_float(x, "grid_value"), _as_float(y, "data_value"))
                for x, y in pairs
            ),
            key=lambda item: item[0],
        )

        if not parsed:
            raise ValueError("pairs must contain at least one item")

        return cls(
            grid=tuple(item[0] for item in parsed),
            values=tuple(item[1] for item in parsed),
            boundary_mode=BoundaryMode(boundary_mode),
            name=name,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinearInterpolator1D":
        """Create interpolator from dictionary."""

        return cls(
            grid=tuple(float(item) for item in data["grid"]),
            values=tuple(float(item) for item in data["values"]),
            boundary_mode=BoundaryMode(data.get("boundary_mode", BoundaryMode.CLIP.value)),
            name=str(data.get("name", "")),
        )


@dataclass(frozen=True, slots=True)
class BilinearInterpolator2D:
    """Reusable 2-D bilinear interpolator."""

    x_grid: tuple[float, ...]
    y_grid: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    boundary_mode: BoundaryMode = BoundaryMode.CLIP
    name: str = ""

    def __post_init__(self) -> None:
        x_grid = _validate_grid(self.x_grid, "x_grid")
        y_grid = _validate_grid(self.y_grid, "y_grid")
        values = _validate_values_2d(self.values, len(x_grid), len(y_grid))
        mode = (
            self.boundary_mode
            if isinstance(self.boundary_mode, BoundaryMode)
            else BoundaryMode(self.boundary_mode)
        )

        object.__setattr__(self, "x_grid", x_grid)
        object.__setattr__(self, "y_grid", y_grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "boundary_mode", mode)
        object.__setattr__(self, "name", str(self.name))

    def interval_x(self, query: Number) -> Interval:
        """Return x-axis interpolation interval."""

        return find_interval(
            self.x_grid,
            query,
            boundary_mode=self.boundary_mode,
        )

    def interval_y(self, query: Number) -> Interval:
        """Return y-axis interpolation interval."""

        return find_interval(
            self.y_grid,
            query,
            boundary_mode=self.boundary_mode,
        )

    def evaluate(
        self,
        x_query: Number,
        y_query: Number,
        *,
        diagnostics: bool = False,
    ) -> float | InterpolationResult:
        """Evaluate bilinear interpolator."""

        result = interpolate_2d(
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            values=self.values,
            x_query=x_query,
            y_query=y_query,
            boundary_mode=self.boundary_mode,
            diagnostics=diagnostics,
        )

        if diagnostics and isinstance(result, InterpolationResult):
            metadata = dict(result.metadata or {})
            metadata["name"] = self.name

            return InterpolationResult(
                value=result.value,
                clipped=result.clipped,
                extrapolated=result.extrapolated,
                metadata=metadata,
            )

        return result

    def __call__(self, x_query: Number, y_query: Number) -> float:
        """Evaluate interpolator and return scalar."""

        return float(self.evaluate(x_query, y_query))

    def map_values(
        self,
        func: Callable[[float], Number],
        *,
        name: str | None = None,
    ) -> "BilinearInterpolator2D":
        """Create a new interpolator by mapping all matrix values."""

        return BilinearInterpolator2D(
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            values=tuple(
                tuple(
                    _as_float(func(value), f"mapped_values[{i}][{j}]")
                    for j, value in enumerate(row)
                )
                for i, row in enumerate(self.values)
            ),
            boundary_mode=self.boundary_mode,
            name=self.name if name is None else name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize interpolator."""

        return {
            "x_grid": list(self.x_grid),
            "y_grid": list(self.y_grid),
            "values": [list(row) for row in self.values],
            "boundary_mode": self.boundary_mode.value,
            "name": self.name,
        }

    def replace(self, **changes: Any) -> "BilinearInterpolator2D":
        """Return a copy with selected fields replaced."""

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BilinearInterpolator2D":
        """Create interpolator from dictionary."""

        return cls(
            x_grid=tuple(float(item) for item in data["x_grid"]),
            y_grid=tuple(float(item) for item in data["y_grid"]),
            values=tuple(
                tuple(float(item) for item in row)
                for row in data["values"]
            ),
            boundary_mode=BoundaryMode(data.get("boundary_mode", BoundaryMode.CLIP.value)),
            name=str(data.get("name", "")),
        )


# ------------------------------------------------------------------------------
# Piecewise linear curves
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PiecewiseLinearCurve:
    """Named piecewise-linear scalar curve."""

    points: tuple[tuple[float, float], ...]
    boundary_mode: BoundaryMode = BoundaryMode.CLIP
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.points) == 0:
            raise ValueError("points must contain at least one point")

        parsed = tuple(
            (
                _as_float(point[0], f"points[{idx}][0]"),
                _as_float(point[1], f"points[{idx}][1]"),
            )
            for idx, point in enumerate(self.points)
        )
        parsed = tuple(sorted(parsed, key=lambda item: item[0]))

        for left, right in zip(parsed, parsed[1:]):
            if right[0] <= left[0]:
                raise ValueError("point x-values must be unique and increasing")

        mode = (
            self.boundary_mode
            if isinstance(self.boundary_mode, BoundaryMode)
            else BoundaryMode(self.boundary_mode)
        )

        object.__setattr__(self, "points", parsed)
        object.__setattr__(self, "boundary_mode", mode)
        object.__setattr__(self, "name", str(self.name))

    @property
    def grid(self) -> tuple[float, ...]:
        """Return x-values."""

        return tuple(point[0] for point in self.points)

    @property
    def values(self) -> tuple[float, ...]:
        """Return y-values."""

        return tuple(point[1] for point in self.points)

    @property
    def interpolator(self) -> LinearInterpolator1D:
        """Return equivalent 1-D interpolator."""

        return LinearInterpolator1D(
            grid=self.grid,
            values=self.values,
            boundary_mode=self.boundary_mode,
            name=self.name,
        )

    def evaluate(
        self,
        query: Number,
        *,
        diagnostics: bool = False,
    ) -> float | InterpolationResult:
        """Evaluate curve at query value."""

        return self.interpolator.evaluate(query, diagnostics=diagnostics)

    def __call__(self, query: Number) -> float:
        """Evaluate curve and return scalar."""

        return float(self.evaluate(query))

    def to_dict(self) -> dict[str, Any]:
        """Serialize curve."""

        return {
            "points": [[float(x), float(y)] for x, y in self.points],
            "boundary_mode": self.boundary_mode.value,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PiecewiseLinearCurve":
        """Create curve from dictionary."""

        return cls(
            points=tuple(
                (float(point[0]), float(point[1]))
                for point in data["points"]
            ),
            boundary_mode=BoundaryMode(data.get("boundary_mode", BoundaryMode.CLIP.value)),
            name=str(data.get("name", "")),
        )


# ------------------------------------------------------------------------------
# Grid generation and resampling helpers
# ------------------------------------------------------------------------------

def linspace(start: Number, stop: Number, num: int) -> tuple[float, ...]:
    """Return evenly spaced values over ``[start, stop]``."""

    if num <= 0:
        raise ValueError("num must be positive")

    a = _as_float(start, "start")
    b = _as_float(stop, "stop")

    if num == 1:
        return (a,)

    return tuple(a + (b - a) * i / (num - 1) for i in range(num))


def resample_1d(
    grid: Sequence[Number],
    values: Sequence[Number],
    new_grid: Sequence[Number],
    *,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
) -> tuple[float, ...]:
    """Resample 1-D data on a new grid."""

    interpolator = LinearInterpolator1D(
        grid=tuple(float(item) for item in grid),
        values=tuple(float(item) for item in values),
        boundary_mode=BoundaryMode(boundary_mode),
    )

    return tuple(interpolator(query) for query in new_grid)


def resample_2d(
    x_grid: Sequence[Number],
    y_grid: Sequence[Number],
    values: Sequence[Sequence[Number]],
    new_x_grid: Sequence[Number],
    new_y_grid: Sequence[Number],
    *,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
) -> tuple[tuple[float, ...], ...]:
    """Resample 2-D data on a new tensor-product grid."""

    interpolator = BilinearInterpolator2D(
        x_grid=tuple(float(item) for item in x_grid),
        y_grid=tuple(float(item) for item in y_grid),
        values=tuple(tuple(float(v) for v in row) for row in values),
        boundary_mode=BoundaryMode(boundary_mode),
    )

    return tuple(
        tuple(interpolator(x, y) for y in new_y_grid)
        for x in new_x_grid
    )


# ------------------------------------------------------------------------------
# Safety wrappers
# ------------------------------------------------------------------------------

def safe_interpolate_1d(
    grid: Sequence[Number],
    values: Sequence[Number],
    query: Number,
    *,
    default: Number = 0.0,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
) -> float:
    """Evaluate 1-D interpolation and return ``default`` on validation failure."""

    try:
        return float(
            interpolate_1d(
                grid=grid,
                values=values,
                query=query,
                boundary_mode=boundary_mode,
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return _as_float(default, "default")


def safe_interpolate_2d(
    x_grid: Sequence[Number],
    y_grid: Sequence[Number],
    values: Sequence[Sequence[Number]],
    x_query: Number,
    y_query: Number,
    *,
    default: Number = 0.0,
    boundary_mode: BoundaryMode | str = BoundaryMode.CLIP,
) -> float:
    """Evaluate 2-D interpolation and return ``default`` on validation failure."""

    try:
        return float(
            interpolate_2d(
                x_grid=x_grid,
                y_grid=y_grid,
                values=values,
                x_query=x_query,
                y_query=y_query,
                boundary_mode=boundary_mode,
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return _as_float(default, "default")


__all__ = [
    "BilinearInterpolator2D",
    "BoundaryMode",
    "InterpolationResult",
    "Interval",
    "LinearInterpolator1D",
    "Number",
    "PiecewiseLinearCurve",
    "bilerp",
    "find_interval",
    "interpolate_1d",
    "interpolate_2d",
    "inverse_lerp",
    "lerp",
    "linspace",
    "resample_1d",
    "resample_2d",
    "safe_interpolate_1d",
    "safe_interpolate_2d",
]
