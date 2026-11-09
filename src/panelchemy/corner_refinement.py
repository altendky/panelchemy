"""Construct a pointed alternative without authorizing its selection."""

# Reuse the fitter's exact coordinates and representable inherited-ray handles.
# pyright: reportPrivateUsage=false

from collections.abc import Iterator, Sequence
from fractions import Fraction
from itertools import pairwise
from math import atan2, hypot, isfinite
from typing import cast

from shapely.geometry import LineString, Point
from shapely.ops import substring

from panelchemy import curves as c
from panelchemy.outline_simplification import _handle


def _cross(a: c.RationalPoint, b: c.RationalPoint) -> Fraction:
    return a[0] * b[1] - a[1] * b[0]


def _finite_coordinate(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, Fraction))
        and isfinite(value)
    )


def forward_progress(span: c.BezierSpan) -> bool:
    """Prove positive interior chordwise progress using exact coefficients.

    This does not certify endpoint speed, curvature, source distance, or
    topology. Those remain independent acceptance checks for the caller.
    """
    controls = c._controls(span)
    direction = (controls[-1][0] - controls[0][0], controls[-1][1] - controls[0][1])
    a, b, d = (
        (b[0] - a[0]) * direction[0] + (b[1] - a[1]) * direction[1]
        for a, b in pairwise(controls)
    )
    # These are quadratic Bernstein coefficients of projected velocity. A
    # negative middle coefficient still has positive minimum iff a*d > b*b.
    # Strictness excludes a newly admitted interior stationary point.
    return any(direction) and a >= 0 and d >= 0 and (b >= 0 or a * d > b * b)


def smooth_join(before: c.BezierSpan, after: c.BezierSpan) -> bool:
    """Recognize an existing smooth join, including native straight edges."""
    try:
        anchor = before.control_points[-1]
        if anchor != after.control_points[0]:
            return False
        left = before.control_points[0 if before.is_line else 2]
        right = after.control_points[-1 if after.is_line else 1]
        u = anchor[0] - left[0], anchor[1] - left[1]
        v = right[0] - anchor[0], right[1] - anchor[1]
        lu, lv = hypot(*u), hypot(*v)
        if not isfinite(lu) or not isfinite(lv) or min(lu, lv) == 0:
            return False
        x, y, a, b = u[0] / lu, u[1] / lu, v[0] / lv, v[1] / lv
        return abs(atan2(x * b - y * a, x * a + y * b)) <= 1e-8
    except (IndexError, OverflowError, TypeError, ValueError):
        return False


def tangent_corner(span: c.BezierSpan) -> tuple[c.BezierSpan, c.BezierSpan] | None:
    """Intersect endpoint tangent rays and preserve their exact outside rays.

    The returned cubics meet at a C0 apex. Representable inherited-ray handles
    need not be one third of a side, so this construction alone proves neither
    forward progress nor curvature, source distance, or palette topology. The
    caller must certify those properties before selecting the alternative.
    """
    controls = span.control_points
    if span.is_line:
        return None
    try:
        if (
            len(controls) != 4
            or any(len(p) != 2 for p in controls)
            or not all(
                _finite_coordinate(value) for point in controls for value in point
            )
        ):
            return None
        p = tuple(c._rational(point) for point in controls)
        u = p[1][0] - p[0][0], p[1][1] - p[0][1]
        v = p[2][0] - p[3][0], p[2][1] - p[3][1]
        delta = p[3][0] - p[0][0], p[3][1] - p[0][1]
        determinant = _cross(u, v)
        if not determinant:
            return None
        a, b = _cross(delta, v) / determinant, _cross(delta, u) / determinant
        if a <= 0 or b <= 0:
            return None
        # Round the shared apex once; both incident cubics use this same point.
        apex = float(p[0][0] + a * u[0]), float(p[0][1] + a * u[1])
        if not all(isfinite(value) for value in apex) or apex in (
            controls[0],
            controls[-1],
        ):
            return None
        lengths = (
            hypot(apex[0] - controls[0][0], apex[1] - controls[0][1]),
            hypot(apex[0] - controls[-1][0], apex[1] - controls[-1][1]),
        )
        left = _handle(controls[0], controls[1], lengths[0] / 3)
        right = _handle(controls[-1], controls[-2], lengths[1] / 3)
        if left is None or right is None:
            return None
        inner_left = c._line(controls[0], apex).control_points[2]
        inner_right = c._line(apex, controls[-1]).control_points[1]
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(
        isfinite(value) for point in (inner_left, inner_right) for value in point
    ):
        return None
    return (
        c.BezierSpan(
            (controls[0], left, inner_left, apex), False, span.certified_error_mm
        ),
        c.BezierSpan(
            (apex, inner_right, right, controls[-1]), False, span.certified_error_mm
        ),
    )


def _lines(points: Sequence[c.Point2], error: float) -> tuple[c.BezierSpan, ...] | None:
    try:
        if (
            any(len(p) != 2 for p in points)
            or not all(_finite_coordinate(value) for p in points for value in p)
            or any(a == b for a, b in pairwise(points))
        ):
            return None
        exact = [c._rational(p) for p in points]
        for a, b, d in zip(exact, exact[1:], exact[2:], strict=False):
            u, v = (b[0] - a[0], b[1] - a[1]), (d[0] - b[0], d[1] - b[1])
            if _cross(u, v) == 0 and u[0] * v[0] + u[1] * v[1] < 0:
                return None
        selected = tuple(c._line(a, b, error) for a, b in pairwise(points))
        if all(
            isfinite(value)
            for span in selected
            for point in span.control_points
            for value in point
        ):
            return selected
    except (OverflowError, TypeError, ValueError):
        pass
    return None


def corner_variants(
    span: c.BezierSpan,
    *,
    free_start: bool = False,
    free_end: bool = False,
    apices: Sequence[c.Point2] = (),
) -> Iterator[tuple[c.BezierSpan, ...]]:
    """Offer bounded corner families, releasing only explicitly free end rays.

    The caller must classify existing C0 endpoints before setting the flags.
    Every alternative still needs forward, curvature, original-source, and
    complete palette certification; this iterator does not authorize selection.
    """
    if (
        type(cast("object", free_start)) is not bool
        or type(cast("object", free_end)) is not bool
    ):
        raise ValueError("corner ray freedom flags must be booleans")
    seen: set[tuple[c.BezierSpan, ...]] = set()
    inherited = tangent_corner(span)
    if inherited is not None:
        seen.add(inherited)
        yield inherited
    if not free_start and not free_end:
        return
    controls = span.control_points
    try:
        if (
            len(controls) != 4
            or any(len(p) != 2 for p in controls)
            or not all(_finite_coordinate(value) for p in controls for value in p)
        ):
            return
        start, end = controls[0], controls[-1]
        if start == end:
            return
        if free_start and free_end:
            chord = _lines((start, end), span.certified_error_mm)
            if chord is not None:
                seen.add(chord)
                yield chord
            left, _ = c._split(c._controls(span))
            midpoint = float(left[-1][0]), float(left[-1][1])
            choices = (*apices, midpoint, controls[1], controls[2])
        else:
            anchor, handle = (
                (end, controls[0 if span.is_line else 2])
                if free_start
                else (start, controls[-1 if span.is_line else 1])
            )
            length = hypot(end[0] - start[0], end[1] - start[1])
            choices = (
                handle,
                *(
                    _handle(anchor, handle, length * fraction)
                    for fraction in (0.25, 0.5, 1)
                ),
            )
        for apex in choices:
            if apex is None:
                continue
            selected = _lines((start, apex, end), span.certified_error_mm)
            if selected is not None and selected not in seen:
                seen.add(selected)
                yield selected
    except (OverflowError, TypeError, ValueError):
        return


def _distinct_neighbors(points: Sequence[c.Point2]) -> list[c.Point2]:
    selected: list[c.Point2] = []
    for point in points:
        if not selected or point != selected[-1]:
            selected.append(point)
    return selected


def source_corners(
    span: c.BezierSpan,
    source_paths: Sequence[Sequence[c.Point2]],
    fit_target_mm: float,
) -> Iterator[tuple[c.BezierSpan, ...]]:
    """Guide at most two corners from oriented raw arcs with fixed end positions.

    Only callers that permit both endpoint rays to change may select these
    alternatives. Projection, proximity/bounds guards, and simplification are
    construction guidance; original raw paths remain the independent distance
    certificate's inputs, and all radius/topology checks remain mandatory.
    """
    if not _finite_coordinate(fit_target_mm) or fit_target_mm <= 0:
        raise ValueError("source-corner fit target must be finite and positive")
    controls = span.control_points
    if (
        len(controls) != 4
        or any(len(p) != 2 for p in controls)
        or not all(_finite_coordinate(value) for point in controls for value in point)
    ):
        return
    start, end = controls[0], controls[-1]
    if start == end:
        return
    padding = 2 * fit_target_mm
    bounds = (
        min(p[0] for p in controls) - padding,
        min(p[1] for p in controls) - padding,
        max(p[0] for p in controls) + padding,
        max(p[1] for p in controls) + padding,
    )
    if not all(isfinite(value) for value in bounds):
        return
    seen: set[tuple[c.BezierSpan, ...]] = set()
    for points in source_paths:
        if (
            len(points) < 4
            or points[0] != points[-1]
            or any(len(p) != 2 for p in points)
            or not all(_finite_coordinate(value) for p in points for value in p)
        ):
            continue
        coordinates = _distinct_neighbors(points)
        if len(coordinates) < 2:
            continue
        path = LineString(coordinates)
        if not isfinite(path.length) or path.length == 0:
            continue
        anchors = Point(start), Point(end)
        if any(anchor.distance(path) > padding for anchor in anchors):
            continue
        first, last = (path.project(anchor) for anchor in anchors)
        if first == last:
            continue
        if first > last:
            chunk = [c._point(p) for p in substring(path, first, path.length).coords]
            chunk.extend(c._point(p) for p in list(substring(path, 0, last).coords)[1:])
        else:
            chunk = [c._point(p) for p in substring(path, first, last).coords]
        # A reversed/wrong whole-ring arc must not become a local candidate.
        if any(
            not (bounds[0] <= p[0] <= bounds[2] and bounds[1] <= p[1] <= bounds[3])
            for p in chunk
        ):
            continue
        chunk[0], chunk[-1] = start, end
        simplified = LineString(_distinct_neighbors(chunk)).simplify(
            fit_target_mm, preserve_topology=False
        )
        vertices = _distinct_neighbors([c._point(p) for p in simplified.coords])
        if not 3 <= len(vertices) <= 4:
            continue  # The chord is already offered by corner_variants.
        vertices[0], vertices[-1] = start, end
        selected = _lines(vertices, span.certified_error_mm)
        if selected is not None and selected not in seen:
            seen.add(selected)
            yield selected
