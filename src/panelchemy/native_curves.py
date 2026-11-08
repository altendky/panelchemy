"""Exact native representation of smooth runs of certified cubic spans.

Joining edges does not remove spline knots or alter the polynomial geometry.
Interior knots retain multiplicity three: geometric tangency is not a claim of
parametric C1 continuity. Analytic lines keep their original representation.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from math import atan2, cos, hypot, isfinite, pi, sin

from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int
from OCP.Geom import Geom_BSplineCurve
from OCP.gp import gp_Pnt

from panelchemy.curves import BezierSpan, CurvePolygon, CurveRing, Point2

type _SpanKey = tuple[bool, tuple[Point2, ...]]


def _key(span: BezierSpan) -> _SpanKey:
    points = (
        (span.control_points[0], span.control_points[-1])
        if span.is_line
        else span.control_points
    )
    return span.is_line, min(points, tuple(reversed(points)))


def protected_vertices(polygons: Sequence[CurvePolygon]) -> frozenset[Point2]:
    """Retain branches, point contacts, and changes in shared-ring ownership."""
    owners: dict[_SpanKey, set[int]] = defaultdict(set)
    incident: dict[Point2, set[_SpanKey]] = defaultdict(set)
    owner = 0
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            for span in ring.spans:
                key = _key(span)
                owners[key].add(owner)
                for point in (span.control_points[0], span.control_points[-1]):
                    incident[point].add(key)
            owner += 1
    return frozenset(
        point
        for point, edges in incident.items()
        if len(edges) != 2 or len({frozenset(owners[edge]) for edge in edges}) != 1
    )


def _smooth(before: BezierSpan, after: BezierSpan) -> bool:
    if before.is_line or after.is_line:
        return False
    a, b = before.control_points, after.control_points
    u = (a[3][0] - a[2][0], a[3][1] - a[2][1])
    v = (b[1][0] - b[0][0], b[1][1] - b[0][1])
    lu, lv = hypot(*u), hypot(*v)
    if not isfinite(lu) or not isfinite(lv) or min(lu, lv) == 0:
        return False
    ux, uy, vx, vy = u[0] / lu, u[1] / lu, v[0] / lv, v[1] / lv
    return abs(atan2(ux * vy - uy * vx, ux * vx + uy * vy)) <= 1e-8


def smooth_runs(
    ring: CurveRing, protected: frozenset[Point2]
) -> tuple[tuple[BezierSpan, ...], ...]:
    """Group G1 cubics cyclically, preserving real corners and protected points."""
    spans = ring.spans
    if not spans or any(
        span.control_points[-1] != spans[(i + 1) % len(spans)].control_points[0]
        for i, span in enumerate(spans)
    ):
        raise ValueError("native smooth runs need an exactly closed connected ring")
    breaks = {
        i
        for i, span in enumerate(spans)
        if span.control_points[0] in protected or not _smooth(spans[i - 1], span)
    }
    # Geometry selects the seam, rather than input array rotation. Protected
    # graph contacts guarantee that a shared endpoint remains a native vertex.
    anchor = min(breaks or range(len(spans)), key=lambda i: spans[i].control_points[0])
    selected = [spans[(anchor + i) % len(spans)] for i in range(len(spans))]
    if not breaks:
        return (tuple(selected),)
    runs: list[tuple[BezierSpan, ...]] = []
    current: list[BezierSpan] = []
    for offset, span in enumerate(selected):
        if current and (anchor + offset) % len(spans) in breaks:
            runs.append(tuple(current))
            current = []
        current.append(span)
    runs.append(tuple(current))
    return tuple(runs)


def draft_runs(
    ring: CurveRing, protected: frozenset[Point2]
) -> tuple[tuple[BezierSpan, ...], ...]:
    """Combine cubics only while their derivative directions fit a quarter turn.

    The draft kernel can lose a closing counter with wide-turn composite edges.
    Breaking at existing cubic boundaries avoids that case without changing any
    curve. A single wider-turn cubic stays intact. Small adjacent cubics on a
    common directional stretch still combine.
    """
    result: list[tuple[BezierSpan, ...]] = []
    for run in smooth_runs(ring, protected):
        current: list[BezierSpan] = []
        angles: list[float] = []
        reference = 0.0
        for span in run:
            p = span.control_points
            vectors = [(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(p)]
            directions = [atan2(y, x) for x, y in vectors if x != 0 or y != 0]
            if not current and directions:
                reference = directions[0]
            relative = [
                atan2(sin(a - reference), cos(a - reference)) for a in directions
            ]
            combined = angles + relative
            if current and combined and max(combined) - min(combined) > pi / 2 + 1e-12:
                result.append(tuple(current))
                current = []
                reference = directions[0] if directions else 0.0
                relative = [
                    atan2(sin(a - reference), cos(a - reference)) for a in directions
                ]
            current.append(span)
            angles = relative if len(current) == 1 else angles + relative
        if current:
            result.append(tuple(current))
    return tuple(result)


def composite_bspline(spans: Sequence[BezierSpan], z: float) -> Geom_BSplineCurve:
    """Encode unchanged cubics using unit intervals and read back every coefficient.

    The basis on each interval is exactly the cubic Bernstein basis. Identical
    poles, integer knots, and full internal multiplicities therefore certify
    the original piecewise polynomial, without a sampled-error approximation.
    """
    if not spans or any(span.is_line for span in spans):
        raise ValueError("an exact cubic composite needs nonempty curved spans")
    if not isfinite(z) or any(
        not isfinite(value)
        for span in spans
        for point in span.control_points
        for value in point
    ):
        raise ValueError("native cubic composite coordinates must be finite")
    if any(
        spans[i - 1].control_points[-1] != span.control_points[0]
        for i, span in enumerate(spans)
        if i
    ):
        raise ValueError("native cubic composite spans must join exactly")
    points = [*spans[0].control_points]
    for span in spans[1:]:
        points.extend(span.control_points[1:])
    poles = Array1_gp_Pnt(1, len(points))
    for i, (x, y) in enumerate(points, 1):
        poles.SetValue(i, gp_Pnt(x, y, z))
    knots = Array1_double(1, len(spans) + 1)
    multiplicities = Array1_int(1, len(spans) + 1)
    for i in range(1, len(spans) + 2):
        knots.SetValue(i, float(i - 1))
        multiplicities.SetValue(i, 4 if i in (1, len(spans) + 1) else 3)
    curve = Geom_BSplineCurve(poles, knots, multiplicities, 3, False)
    if (
        curve.Degree() != 3
        or curve.IsRational()
        or curve.IsPeriodic()
        or curve.NbPoles() != len(points)
        or curve.NbKnots() != len(spans) + 1
        or any(
            curve.Pole(i).Coord() != (*point, z) for i, point in enumerate(points, 1)
        )
        or any(
            curve.Knot(i) != float(i - 1)
            or curve.Multiplicity(i) != (4 if i in (1, len(spans) + 1) else 3)
            for i in range(1, len(spans) + 2)
        )
    ):
        raise ValueError("native cubic composite changed its polynomial representation")
    return curve
