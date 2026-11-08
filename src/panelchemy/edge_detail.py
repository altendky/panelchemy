"""Advisory physical-scale classification of raster boundary detail.

``fine_texture`` describes repeated geometry below the extrusion scale; it does
not decide artistic intent, change a contour, or authorize an error allowance.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import asin, atan2, dist, fsum, hypot, isfinite, sqrt
from typing import Literal

from panelchemy.curves import Point2

type Kind = Literal["raster_steps", "fine_texture", "structural", "uncertain"]
_EMPTY: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Sample:
    index: int
    kind: Kind
    reason: str
    chord_mm: float = 0.0
    path_excess_mm: float = 0.0
    coarse_path_excess_mm: float = 0.0
    backtracks: int = 0
    coarse_backtracks: int = 0
    normal_reversals: int = 0
    coarse_normal_reversals: int = 0
    max_backtrack_mm: float = 0.0
    max_event_mm: float = 0.0
    turn_rad: float = 0.0
    coarse_turn_rad: float = 0.0
    prominence_mm: float = 0.0


def _support(
    points: Sequence[Point2], index: int, radius: float, closed: bool
) -> tuple[list[Point2], int, bool]:
    origin = points[index]
    sides: list[list[Point2]] = []
    complete = True
    for direction in (-1, 1):
        side: list[Point2] = []
        previous = (0.0, 0.0)
        exited = False
        for step in range(1, len(points)):
            j = index + direction * step
            if not closed and not 0 <= j < len(points):
                break
            point = points[j % len(points)]
            current = point[0] - origin[0], point[1] - origin[1]
            if not all(isfinite(v) for v in current):
                break
            if hypot(*current) >= radius:
                delta = current[0] - previous[0], current[1] - previous[1]
                length = hypot(*delta)
                if not isfinite(length) or not length:
                    break
                unit = delta[0] / length, delta[1] / length
                normalized = previous[0] / radius, previous[1] / radius
                dot = normalized[0] * unit[0] + normalized[1] * unit[1]
                remainder = 1 - normalized[0] ** 2 - normalized[1] ** 2
                along = -dot + sqrt(max(0.0, dot**2 + remainder))
                exit_point = (
                    origin[0] + radius * (normalized[0] + along * unit[0]),
                    origin[1] + radius * (normalized[1] + along * unit[1]),
                )
                if not all(isfinite(v) for v in exit_point):
                    break
                side.append(exit_point)
                exited = True
                break
            side.append(point)
            previous = current
        complete &= exited
        sides.append(side)
    return [*reversed(sides[0]), origin, *sides[1]], len(sides[0]), complete


def _diameter(points: Sequence[Point2]) -> float:
    return max(
        (dist(a, b) for i, a in enumerate(points) for b in points[i + 1 :]), default=0.0
    )


def _backtracks(
    points: Sequence[Point2], progress: Sequence[float], pitch: float
) -> tuple[int, float, float]:
    count, amplitude, extent = 0, 0.0, 0.0
    i = 0
    while i + 1 < len(points):
        if progress[i + 1] >= progress[i]:
            i += 1
            continue
        start, low = progress[i], progress[i + 1]
        j = i + 1
        while j + 1 < len(points) and progress[j + 1] < start:
            j += 1
            low = min(low, progress[j])
        if j + 1 == len(points):
            break  # An unrecovered reversal is not a small repeated pulse.
        fraction = (start - progress[j]) / (progress[j + 1] - progress[j])
        recovery = tuple(
            points[j][k] + fraction * (points[j + 1][k] - points[j][k]) for k in (0, 1)
        )
        loss = start - low
        if loss > pitch / 2:
            count += 1
            amplitude = max(amplitude, loss)
            extent = max(
                extent, _diameter([*points[i : j + 1], (recovery[0], recovery[1])])
            )
        i = j + 1
    return count, amplitude, extent


def _normal_reversals(
    points: Sequence[Point2], normal: Sequence[float], pitch: float
) -> tuple[int, float, float]:
    # Pixel-diagonal hysteresis suppresses tiny plateau/roundoff reversals.
    threshold = sqrt(2) * pitch
    extrema = [0]
    peak, direction = 0, 0
    for i in range(1, len(points)):
        if direction == 0:
            if abs(normal[i] - normal[0]) >= threshold:
                direction = 1 if normal[i] > normal[0] else -1
                peak = i
        elif direction * (normal[i] - normal[peak]) >= 0:
            peak = i
        elif direction * (normal[peak] - normal[i]) >= threshold:
            extrema.append(peak)
            peak, direction = i, -direction
    if direction and abs(normal[peak] - normal[extrema[-1]]) >= threshold:
        extrema.append(peak)
    sizes = [_diameter(points[a : b + 1]) for a, b in pairwise(extrema)]
    amplitude = max(
        (abs(normal[b] - normal[a]) for a, b in pairwise(extrema)), default=0.0
    )
    return max(0, len(extrema) - 2), max(sizes, default=0.0), amplitude


@dataclass(frozen=True)
class _Measure:
    chord: float
    excess: float
    backtracks: int
    normal_reversals: int
    amplitude: float
    back_extent: float
    normal_extent: float
    turn: float
    prominence: float
    complete: bool
    normal_amplitude: float = 0.0


def _measure(
    points: Sequence[Point2], index: int, radius: float, pitch: float, closed: bool
) -> _Measure:
    path, middle, complete = _support(points, index, radius, closed)
    if not complete:
        return _Measure(0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    chord = dist(path[0], path[-1])
    if not chord or not isfinite(chord):
        return _Measure(0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    tangent = ((path[-1][0] - path[0][0]) / chord, (path[-1][1] - path[0][1]) / chord)
    progress = [
        (p[0] - path[0][0]) * tangent[0] + (p[1] - path[0][1]) * tangent[1]
        for p in path
    ]
    normal = [
        -(p[0] - path[0][0]) * tangent[1] + (p[1] - path[0][1]) * tangent[0]
        for p in path
    ]
    count, amplitude, back_extent = _backtracks(path, progress, pitch)
    reversals, normal_extent, normal_amplitude = _normal_reversals(path, normal, pitch)
    a = (
        (points[index][0] - path[0][0]) / radius,
        (points[index][1] - path[0][1]) / radius,
    )
    b = (
        (path[-1][0] - points[index][0]) / radius,
        (path[-1][1] - points[index][1]) / radius,
    )
    turn = atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    try:
        length = fsum(dist(a, b) for a, b in pairwise(path))
    except OverflowError:
        return _Measure(0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    return _Measure(
        chord,
        max(0, length - sqrt(2) * chord),
        count,
        reversals,
        amplitude,
        back_extent,
        normal_extent,
        turn,
        abs(normal[middle]),
        complete,
        normal_amplitude,
    )


def classify_chain(
    points: Sequence[Point2],
    *,
    pixel_pitch_mm: float,
    line_width_mm: float,
    protected_indices: frozenset[int] = _EMPTY,
) -> tuple[Sample, ...]:
    """Classify local geometry at one and two extrusion-width support radii.

    Monotone raster stairs have length at most sqrt(2) times their chord.
    Repeated sub-width excursions above that bound indicate fine texture.
    Persistent turns, short closed outlines and supplied feature barriers take
    priority. Classification is advisory: intended serrations can be texture too.
    """
    if len(points) < 2 or any(
        len(p) != 2 or not all(isfinite(v) for v in p) for p in points
    ):
        raise ValueError("edge classification requires a finite planar path")
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (pixel_pitch_mm, line_width_mm)
    ):
        raise ValueError("pixel pitch and line width must be positive and finite")
    if any(type(i) is not int or not 0 <= i < len(points) for i in protected_indices):
        raise ValueError("protected indices must belong to the path")
    if any(a == b for a, b in pairwise(points)):
        raise ValueError("edge classification requires distinct adjacent points")
    closed = points[0] == points[-1]
    unique = points[:-1] if closed else points
    try:
        perimeter = fsum(dist(a, b) for a, b in pairwise(points)) if closed else 0
    except OverflowError:
        perimeter = float("inf")
    # This enclosing-radius bound is linear in vertex count and invariant under
    # rotation/cyclic starting point. Do not compare every pair on a large ring.
    compact_ring = False
    if closed and isfinite(4 * line_width_mm):
        center = tuple(fsum(p[j] / len(unique) for p in unique) for j in (0, 1))
        compact_ring = (
            max(dist((center[0], center[1]), p) for p in unique) <= 2 * line_width_mm
        )
    short = (
        closed
        and isfinite(4 * line_width_mm)
        and ((isfinite(perimeter) and perimeter <= 4 * line_width_mm) or compact_ring)
    )
    protected = protected_indices | (
        frozenset({0}) if closed and len(points) - 1 in protected_indices else _EMPTY
    )
    samples: list[Sample] = []
    for i in range(len(unique)):
        if i in protected or short:
            samples.append(
                Sample(i, "structural", "protected feature or short closed outline")
            )
            continue
        if line_width_mm < 2 * pixel_pitch_mm or not isfinite(4 * line_width_mm):
            samples.append(
                Sample(i, "uncertain", "insufficient resolution or numerical range")
            )
            continue
        fine = _measure(unique, i, line_width_mm, pixel_pitch_mm, closed)
        coarse = _measure(unique, i, 2 * line_width_mm, pixel_pitch_mm, closed)
        uncertainty = 2 * asin(min(1.0, sqrt(2) * pixel_pitch_mm / line_width_mm))
        coarse_uncertainty = 2 * asin(
            min(1.0, sqrt(2) * pixel_pitch_mm / (2 * line_width_mm))
        )
        backward = (
            min(fine.backtracks, coarse.backtracks) >= 2
            and max(fine.back_extent, coarse.back_extent) <= line_width_mm
        )
        normal = (
            min(fine.normal_reversals, coarse.normal_reversals) >= 2
            and max(fine.normal_extent, coarse.normal_extent) <= line_width_mm
        )
        texture_amplitude = max(fine.amplitude, coarse.amplitude) if backward else 0.0
        if normal:
            texture_amplitude = max(
                texture_amplitude, fine.normal_amplitude, coarse.normal_amplitude
            )
        persistent = (
            fine.turn * coarse.turn > 0
            and abs(fine.turn) > uncertainty
            and abs(coarse.turn) > coarse_uncertainty
            and abs(fine.turn - coarse.turn) <= uncertainty + coarse_uncertainty
            and coarse.prominence - fine.prominence
            > max(sqrt(2) * pixel_pitch_mm, texture_amplitude)
        )
        isolated = (
            fine.normal_reversals == coarse.normal_reversals == 1
            and fine.prominence >= sqrt(2) * pixel_pitch_mm
            and abs(coarse.prominence - fine.prominence) <= sqrt(2) * pixel_pitch_mm
        )
        if (
            line_width_mm < 2 * pixel_pitch_mm
            or not fine.complete
            or not coarse.complete
        ):
            kind, reason = "uncertain", "insufficient resolution or physical support"
        elif min(fine.chord, coarse.chord) < line_width_mm or persistent or isolated:
            kind, reason = (
                "structural",
                "folded outline or persistent physical-scale turn",
            )
        elif min(fine.excess, coarse.excess) >= pixel_pitch_mm and (backward or normal):
            kind = "fine_texture"
            reason = (
                "repeated sub-width backward progress above the raster stair bound"
                if backward
                else "repeated sub-width normal excursions above the raster stair bound"
            )
        elif max(fine.excess, coarse.excess) < pixel_pitch_mm and not (
            fine.backtracks or coarse.backtracks
        ):
            kind, reason = "raster_steps", "grid-compatible forward progress"
        else:
            kind, reason = (
                "uncertain",
                "detail does not separate reliably across physical scales",
            )
        samples.append(
            Sample(
                i,
                kind,
                reason,
                fine.chord,
                fine.excess,
                coarse.excess,
                fine.backtracks,
                coarse.backtracks,
                fine.normal_reversals,
                coarse.normal_reversals,
                max(fine.amplitude, coarse.amplitude),
                max(fine.back_extent, coarse.back_extent)
                if kind == "fine_texture" and backward
                else max(fine.normal_extent, coarse.normal_extent)
                if kind == "fine_texture"
                else max(
                    fine.back_extent,
                    coarse.back_extent,
                    fine.normal_extent,
                    coarse.normal_extent,
                ),
                fine.turn,
                coarse.turn,
                fine.prominence,
            )
        )
    if closed:
        samples.append(replace(samples[0], index=len(points) - 1))
    return tuple(samples)
