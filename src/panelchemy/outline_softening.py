"""Conservative local corner-bridge candidates, awaiting global certification.

This module does not certify distance to artwork or preservation of topology.
Its caller must certify those properties before accepting the returned chain.
The radius bounds the spatial neighborhood changed, not a circular fillet.
"""

# Both fitting policies share the core's exact Bézier subdivision arithmetic.
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import hypot, isfinite, sqrt
from typing import cast

from panelchemy import curves as c

type Interval = tuple[float, float]
_EMPTY: frozenset[float] = frozenset()


@dataclass(frozen=True)
class SoftenedChain:
    spans: tuple[c.BezierSpan, ...]
    intervals: tuple[Interval, ...]
    dependencies: tuple[frozenset[float], ...]
    candidates: frozenset[float]
    applied: frozenset[float]
    max_support_mm: float = 0.0


@dataclass(frozen=True)
class _Bridge:
    corner: float
    before: int
    after: int
    fraction: Fraction
    span: c.BezierSpan
    support_mm: float


def _as_span(
    controls: c.RationalControls, original: c.BezierSpan, *, exact: bool
) -> c.BezierSpan | None:
    points = cast("c.Controls", tuple(tuple(float(v) for v in p) for p in controls))
    if exact:
        selected = (0, 3) if original.is_line else (0, 1, 2, 3)
        if any(c._rational(points[i]) != controls[i] for i in selected):
            return None
    if original.is_line:
        return c._line(points[0], points[-1], original.certified_error_mm)
    return c.BezierSpan(points, False, original.certified_error_mm)


def _trim(span: c.BezierSpan, start: Fraction, stop: Fraction) -> c.BezierSpan | None:
    if start == 0 and stop == 1:
        return span
    return _as_span(c._restrict(c._controls(span), start, stop), span, exact=True)


def _inside(
    controls: c.RationalControls, center: c.RationalPoint, radius: Fraction
) -> bool:
    return all(
        (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 <= radius**2 for p in controls
    )


def _tangent(span: c.BezierSpan, *, end: bool) -> c.RationalPoint:
    controls = c._controls(span)
    a, b = (
        (controls[0], controls[3])
        if span.is_line
        else ((controls[2], controls[3]) if end else (controls[0], controls[1]))
    )
    return b[0] - a[0], b[1] - a[1]


def _g1(before: c.BezierSpan, after: c.BezierSpan) -> bool:
    if before.control_points[-1] != after.control_points[0]:
        return False
    u, v = _tangent(before, end=True), _tangent(after, end=False)
    product = hypot(float(u[0]), float(u[1])) * hypot(float(v[0]), float(v[1]))
    return (
        product > 0
        and u[0] * v[0] + u[1] * v[1] > 0
        and abs(float(u[0] * v[1] - u[1] * v[0])) <= product * 1e-12
    )


def _candidate(
    spans: Sequence[c.BezierSpan],
    before: int,
    after: int,
    corner: float,
    radius_mm: float,
) -> _Bridge | None:
    original_left, original_right = spans[before], spans[after]
    center = c._rational(original_left.control_points[-1])
    radius = Fraction(radius_mm)
    for power in range(2, 18):
        fraction = Fraction(1, 1 << power)
        left = _trim(original_left, Fraction(0), 1 - fraction)
        right = _trim(original_right, fraction, Fraction(1))
        if left is None or right is None:
            continue
        removed_left = c._restrict(
            c._controls(original_left), 1 - fraction, Fraction(1)
        )
        removed_right = c._restrict(c._controls(original_right), Fraction(0), fraction)
        a, d = (
            c._rational(left.control_points[-1]),
            c._rational(right.control_points[0]),
        )
        u, v = _tangent(left, end=True), _tangent(right, end=False)
        # Dyadic factors preserve the binary lattice where representable. A
        # line's tangent uses its whole chord; a cubic uses its endpoint handle.
        ku = fraction * (Fraction(3, 4) if left.is_line else 2)
        kv = fraction * (Fraction(3, 4) if right.is_line else 2)
        controls: c.RationalControls = (
            a,
            (a[0] + ku * u[0], a[1] + ku * u[1]),
            (d[0] - kv * v[0], d[1] - kv * v[1]),
            d,
        )
        bridge = _as_span(
            controls,
            c.BezierSpan(
                original_left.control_points, False, original_left.certified_error_mm
            ),
            exact=False,
        )
        assert bridge is not None
        hulls = (removed_left, removed_right, c._controls(bridge))
        if (
            all(_inside(hull, center, radius) for hull in hulls)
            and _g1(left, bridge)
            and _g1(bridge, right)
        ):
            support = max(
                sqrt(float((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2))
                for hull in hulls
                for p in hull
            )
            return _Bridge(
                corner, before, after, fraction, bridge, min(radius_mm, support)
            )
    return None


def soften(
    spans: Sequence[c.BezierSpan],
    intervals: Sequence[Interval],
    corners: Sequence[float],
    radius_mm: float,
    *,
    excluded: frozenset[float] = _EMPTY,
    periodic: bool = False,
) -> SoftenedChain:
    """Build local candidates from the unchanged chain on every call.

    Excluding a corner restores its original incident geometry. Retained cubic
    subcurves must be exactly representable as floats; otherwise the affected
    candidate is left unapplied. Every output piece carries the corner IDs that
    changed it, allowing the caller to reject a failed candidate and rebuild.
    """
    if isinstance(radius_mm, bool) or not isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("softening radius must be finite and nonnegative")
    if not spans or len(spans) != len(intervals):
        raise ValueError("a nonempty chain needs one interval per span")
    if intervals[0][0] != 0 or intervals[-1][1] != 1:
        raise ValueError("source intervals must partition [0, 1]")
    if any(not isfinite(t) for pair in intervals for t in pair) or any(
        a >= b for a, b in intervals
    ):
        raise ValueError("source intervals must be finite and increasing")
    if any(
        intervals[i][1] != intervals[i + 1][0]
        or spans[i].control_points[-1] != spans[i + 1].control_points[0]
        for i in range(len(spans) - 1)
    ):
        raise ValueError("source intervals and spans must join exactly")
    if periodic and spans[-1].control_points[-1] != spans[0].control_points[0]:
        raise ValueError("a periodic chain must close exactly")
    junctions = {pair[0]: i for i, pair in enumerate(intervals) if i}
    if periodic:
        junctions[0.0] = 0
    normalized = {0.0 if periodic and t == 1 else t for t in corners}
    omitted = {0.0 if periodic and t == 1 else t for t in excluded}
    if any(not isfinite(t) or t < 0 or t > 1 for t in normalized):
        raise ValueError("corner parameters must be finite and within [0, 1]")
    selected = normalized - omitted - ({0.0, 1.0} if not periodic else set())
    if not selected <= junctions.keys():
        raise ValueError("a corner must coincide with an original span junction")
    candidates = frozenset(selected)
    plans: dict[float, _Bridge] = {}
    if radius_mm:
        for corner in sorted(selected):
            after = junctions[corner]
            plan = _candidate(spans, (after - 1) % len(spans), after, corner, radius_mm)
            if plan is not None:
                plans[corner] = plan

    # Two independent corner trims may affect one span. Their quarter caps
    # cannot overlap; if their combined exact subdivision is unrepresentable,
    # restore both candidates instead of perturbing an unrelated retained arc.
    while True:
        starts = {p.after: p for p in plans.values()}
        stops = {p.before: p for p in plans.values()}
        retained: list[c.BezierSpan] = []
        retained_intervals: list[Interval] = []
        retained_dependencies: list[frozenset[float]] = []
        rejected: set[float] = set()
        for i, (span, interval) in enumerate(zip(spans, intervals, strict=True)):
            start, stop = starts.get(i), stops.get(i)
            lo = start.fraction if start else Fraction(0)
            hi = 1 - stop.fraction if stop else Fraction(1)
            selected_span = _trim(span, lo, hi)
            dependencies = frozenset(p.corner for p in (start, stop) if p is not None)
            if selected_span is None:
                rejected.update(dependencies)
                continue
            a, b = map(Fraction, interval)
            retained.append(selected_span)
            retained_intervals.append(
                (float(a + (b - a) * lo), float(a + (b - a) * hi))
            )
            retained_dependencies.append(dependencies)
        if rejected:
            for corner in rejected:
                _ = plans.pop(corner)
            continue
        break

    output: list[c.BezierSpan] = []
    times: list[Interval] = []
    dependencies_out: list[frozenset[float]] = []

    def emit(
        span: c.BezierSpan, interval: Interval, dependencies: frozenset[float]
    ) -> None:
        output.append(span)
        times.append(interval)
        dependencies_out.append(dependencies)

    seam = plans.get(0.0) if periodic else None
    seam_end: c.BezierSpan | None = None
    if seam is not None:
        left_start, right_stop = retained_intervals[-1][1], retained_intervals[0][0]
        seam_parameter = (1 - Fraction(left_start)) / (
            1 - Fraction(left_start) + Fraction(right_stop)
        )
        left_controls, right_controls = c._split(c._controls(seam.span), seam_parameter)
        seam_end = _as_span(left_controls, seam.span, exact=False)
        seam_start = _as_span(right_controls, seam.span, exact=False)
        assert seam_end is not None and seam_start is not None
        center = c._rational(spans[0].control_points[0])
        if not (
            _g1(retained[-1], seam_end)
            and _g1(seam_end, seam_start)
            and _g1(seam_start, retained[0])
            and _inside(c._controls(seam_end), center, Fraction(radius_mm))
            and _inside(c._controls(seam_start), center, Fraction(radius_mm))
        ):
            return soften(
                spans,
                intervals,
                corners,
                radius_mm,
                excluded=excluded | {0.0},
                periodic=periodic,
            )
        emit(seam_start, (0.0, right_stop), frozenset({0.0}))
    for i, (span, interval, deps) in enumerate(
        zip(retained, retained_intervals, retained_dependencies, strict=True)
    ):
        emit(span, interval, deps)
        plan = stops.get(i)
        if plan is not None and plan.corner != 0:
            emit(
                plan.span,
                (interval[1], retained_intervals[plan.after][0]),
                frozenset({plan.corner}),
            )
    if seam is not None:
        assert seam_end is not None
        emit(seam_end, (retained_intervals[-1][1], 1.0), frozenset({0.0}))
    # Even valid distinct source parameters can be adjacent floats. A trim
    # boundary may then round onto its neighbor, collapsing a bridge's source
    # interval. Reject its owners before a distance certificate divides by that
    # interval, retaining every unaffected candidate and the original geometry.
    collapsed: set[float] = set()
    for interval, owners in zip(times, dependencies_out, strict=True):
        if interval[0] >= interval[1]:
            collapsed.update(owners)
    for i in range(len(times) - 1):
        if times[i][1] != times[i + 1][0]:
            collapsed.update(dependencies_out[i] | dependencies_out[i + 1])
    if collapsed:
        restored = soften(
            spans,
            intervals,
            corners,
            radius_mm,
            excluded=excluded | collapsed,
            periodic=periodic,
        )
        return replace(restored, candidates=candidates | restored.candidates)
    return SoftenedChain(
        tuple(output),
        tuple(times),
        tuple(dependencies_out),
        candidates,
        frozenset(plans),
        max((p.support_mm for p in plans.values()), default=0.0),
    )
