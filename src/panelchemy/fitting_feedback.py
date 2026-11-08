"""Numerical locations for another search attempt; never acceptance certificates."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from panelchemy.curves import BezierSpan, Point2
from panelchemy.nearest_polyline import NearestPolyline

type FloatArray = NDArray[np.float64]
_SAMPLES = 17
_MAX_LOCATIONS = 8


def _value(values: FloatArray, index: int) -> float:
    return float(cast("np.float64", values[index]))


def _point(values: FloatArray, index: int) -> FloatArray:
    return cast("FloatArray", values[index])


@dataclass(frozen=True)
class Refinement:
    """Curve probes and original-source queries occupy different parameter spaces."""

    curve_parameters: tuple[float, ...] = ()
    source_parameters: tuple[float, ...] = ()


def _roots(polynomial: Polynomial) -> tuple[float, ...]:
    coefficients = np.asarray(polynomial.coef, dtype=np.float64)
    scale = float(cast("np.float64", np.max(np.abs(coefficients))))
    if scale == 0 or not isfinite(scale):
        return ()
    try:
        roots = np.asarray(
            Polynomial(coefficients / scale).roots(), dtype=np.complex128
        )
    except np.linalg.LinAlgError:
        return ()
    # Root filtering is search guidance. Any missed or inaccurate extremum is
    # still subject to the caller's unchanged continuous acceptance checks.
    return tuple(
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1e-8 and 0 < root.real < 1
    )


def curvature_locations(
    span: BezierSpan,
    offset_mm: float,
    inward_sign: int,
    *,
    fraction: float = 0.895,
) -> tuple[float, ...]:
    """Find local cubic curvature peaks and nearly stationary interior points."""
    if span.is_line or offset_mm == 0:
        return ()
    controls: FloatArray = np.asarray(span.control_points, dtype=np.float64)
    controls = controls - _point(controls, 0)
    scale = float(cast("np.float64", np.max(np.abs(controls))))
    if scale == 0 or not isfinite(scale):
        return ()
    controls /= scale
    axes = [
        Polynomial(
            [
                _value(axis, 0),
                3 * (_value(axis, 1) - _value(axis, 0)),
                3 * (_value(axis, 2) - 2 * _value(axis, 1) + _value(axis, 0)),
                _value(axis, 3)
                - 3 * _value(axis, 2)
                + 3 * _value(axis, 1)
                - _value(axis, 0),
            ]
        )
        for axis in (controls[:, 0], controls[:, 1])
    ]
    vx, vy = (axis.deriv() for axis in axes)
    speed = vx * vx + vy * vy
    cross = vx * vy.deriv() - vy * vx.deriv()
    critical = 2 * cross.deriv() * speed - 3 * cross * speed.deriv()
    selected: set[float] = set()
    for parameter in (0.0, 1.0, *_roots(critical)):
        squared = float(cast("np.float64", speed(parameter)))
        if squared > 0:
            ratio = (
                inward_sign
                * (offset_mm / scale)
                * float(cast("np.float64", cross(parameter)))
            )
            if ratio > fraction * squared**1.5:
                selected.add(parameter)
    speed_scale = float(cast("np.float64", np.max(np.abs(speed.coef))))
    for parameter in _roots(speed.deriv()):
        if (
            float(cast("np.float64", speed(parameter)))
            <= 64 * np.finfo(np.float64).eps * speed_scale
        ):
            selected.add(parameter)
    return tuple(sorted(selected))


def _peaks(values: FloatArray, limit: float) -> list[int]:
    candidates = [
        i
        for i in range(len(values))
        if _value(values, i) > limit
        and (i == 0 or _value(values, i) >= _value(values, i - 1))
        and (i + 1 == len(values) or _value(values, i) >= _value(values, i + 1))
    ]
    return sorted(candidates, key=lambda i: _value(values, i), reverse=True)[
        :_MAX_LOCATIONS
    ]


def distance_refinement(
    source: Sequence[Point2],
    source_parameters: FloatArray,
    spans: Sequence[BezierSpan],
    intervals: Sequence[tuple[float, float]],
    tolerance_mm: float,
    *,
    fraction: float = 0.9,
) -> Refinement:
    """Suggest forward probes and reverse queries beyond the search target.

    Reverse queries additionally refine their nearest curve-proxy segment: a
    source query alone cannot expose a proxy chord that shortcuts a curved arc.
    The sampled proxy has no authority to accept a candidate.
    """
    if (
        not spans
        or len(spans) != len(intervals)
        or source_parameters.shape != (len(source),)
        or len(source) < 2
        or not np.all(np.isfinite(source_parameters))
        or not np.all(np.diff(source_parameters) > 0)
        or not isfinite(tolerance_mm)
        or tolerance_mm <= 0
        or not isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("distance refinement requires finite increasing source data")
    grid = np.linspace(0, 1, _SAMPLES)
    positions: list[FloatArray] = []
    parameters: list[FloatArray] = []
    for index, (span, (start, stop)) in enumerate(zip(spans, intervals, strict=True)):
        points: FloatArray = np.asarray(span.control_points, dtype=np.float64)
        u = grid[:, None]
        a, b, c, d = (_point(points, i) for i in range(4))
        selected: FloatArray = (
            (1 - u) * a + u * d
            if span.is_line
            else (1 - u) ** 3 * a
            + 3 * (1 - u) ** 2 * u * b
            + 3 * (1 - u) * u**2 * c
            + u**3 * d
        )
        begin = int(index != 0)
        positions.append(selected[begin:])
        parameters.append((start + grid * (stop - start))[begin:])
    proxy = np.vstack(positions)
    curve_parameters = np.concatenate(parameters)
    source_array = np.asarray(source, dtype=np.float64)
    forward = proxy - NearestPolyline(source_array).project(proxy).points
    limit = (fraction * tolerance_mm) ** 2
    curve_locations = {
        _value(curve_parameters, index)
        for index in _peaks(np.sum(forward**2, axis=1), limit)
    }
    starts, gaps = source_parameters[:-1], np.diff(source_parameters)
    queries = np.unique(
        np.concatenate(
            (source_parameters, *(starts + gaps * u for u in (0.25, 0.5, 0.75)))
        )
    )
    targets = np.column_stack(
        [np.interp(queries, source_parameters, source_array[:, j]) for j in (0, 1)]
    )
    reverse = NearestPolyline(proxy).project(targets)
    source_locations: set[float] = set()
    for index in _peaks(np.sum((reverse.points - targets) ** 2, axis=1), limit):
        source_locations.add(_value(queries, index))
        at = int(cast("np.intp", reverse.segments[index]))
        u = _value(reverse.fractions, index)
        curve_locations.add(
            (1 - u) * _value(curve_parameters, at)
            + u * _value(curve_parameters, at + 1)
        )
    return Refinement(tuple(sorted(curve_locations)), tuple(sorted(source_locations)))
