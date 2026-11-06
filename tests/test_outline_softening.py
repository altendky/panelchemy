"""Local bridge candidates retain exact tails and bounded, tangent geometry."""

from fractions import Fraction
from itertools import pairwise
from math import hypot, nextafter
from typing import cast

import pytest

from panelchemy import curves as c
from panelchemy.outline_softening import SoftenedChain, soften


def _line(a: c.Point2, b: c.Point2) -> c.BezierSpan:
    return c.BezierSpan(
        (
            a,
            ((2 * a[0] + b[0]) / 3, (2 * a[1] + b[1]) / 3),
            ((a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3),
            b,
        ),
        True,
    )


def _point(span: c.BezierSpan, t: float) -> c.Point2:
    if span.is_line:
        a, d = span.control_points[0], span.control_points[-1]
        return (1 - t) * a[0] + t * d[0], (1 - t) * a[1] + t * d[1]
    p = span.control_points
    weights = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3)
    return cast(
        "c.Point2",
        tuple(sum(w * q[j] for w, q in zip(weights, p, strict=True)) for j in (0, 1)),
    )


def _smooth(before: c.BezierSpan, after: c.BezierSpan) -> None:
    assert before.control_points[-1] == after.control_points[0]
    a, b = before.control_points, after.control_points
    u = (
        a[3][0] - a[0 if before.is_line else 2][0],
        a[3][1] - a[0 if before.is_line else 2][1],
    )
    v = (
        b[3 if after.is_line else 1][0] - b[0][0],
        b[3 if after.is_line else 1][1] - b[0][1],
    )
    product = hypot(*u) * hypot(*v)
    assert product > 0
    assert u[0] * v[0] + u[1] * v[1] > 0
    assert abs(u[0] * v[1] - u[1] * v[0]) <= product * 1e-11


def _partition(chain: SoftenedChain) -> None:
    assert len(chain.spans) == len(chain.intervals) == len(chain.dependencies)
    assert chain.intervals[0][0] == 0
    assert chain.intervals[-1][1] == 1
    assert all(a < b for a, b in chain.intervals)
    assert all(before[1] == after[0] for before, after in pairwise(chain.intervals))
    assert all(
        before.control_points[-1] == after.control_points[0]
        for before, after in pairwise(chain.spans)
    )


def _elbow() -> tuple[tuple[c.BezierSpan, ...], tuple[tuple[float, float], ...]]:
    return (_line((0, 0), (2, 0)), _line((2, 0), (2, 2))), ((0, 0.5), (0.5, 1))


def test_right_angle_bridge_is_local_smooth_and_keeps_distant_lines_exact() -> None:
    spans, intervals = _elbow()
    result = soften(spans, intervals, [0.5], 0.25)
    _partition(result)
    assert result.applied == result.candidates == frozenset({0.5})
    assert len(result.spans) == 3
    assert result.max_support_mm <= 0.25
    assert result.spans[0].control_points[0] == (0, 0)
    assert result.spans[-1].control_points[-1] == (2, 2)
    assert result.spans[0].is_line and result.spans[-1].is_line
    assert result.spans[0].control_points[-1][1] == 0
    assert result.spans[-1].control_points[0][0] == 2
    for before, after in pairwise(result.spans):
        _smooth(before, after)
    bridge = result.spans[1]
    for p in bridge.control_points:
        dx, dy = Fraction(p[0]) - 2, Fraction(p[1])
        assert dx**2 + dy**2 <= Fraction(0.25) ** 2
    # Both removed straight tails stay inside the same spatial neighborhood.
    assert hypot(result.spans[0].control_points[-1][0] - 2, 0) <= 0.25
    assert hypot(0, result.spans[-1].control_points[0][1]) <= 0.25
    assert all(deps == frozenset({0.5}) for deps in result.dependencies)


def test_curved_incident_spans_keep_exact_subcurves_outside_the_bridge() -> None:
    spans = (
        c.BezierSpan(((0, 0), (1, 0), (2, 0), (2, 1)), False),
        c.BezierSpan(((2, 1), (3, 1), (3, 2), (3, 3)), False),
    )
    result = soften(spans, ((0, 0.5), (0.5, 1)), [0.5], 0.5)
    assert result.applied == frozenset({0.5})
    _partition(result)
    for before, after in pairwise(result.spans):
        _smooth(before, after)
    first_stop = result.intervals[0][1] / 0.5
    last_start = (result.intervals[-1][0] - 0.5) / 0.5
    for t in (0.0, 0.125, 0.5, 0.875, 1.0):
        assert _point(result.spans[0], t) == pytest.approx(
            _point(spans[0], t * first_stop), abs=1e-14
        )
        assert _point(result.spans[-1], t) == pytest.approx(
            _point(spans[1], last_start + t * (1 - last_start)), abs=1e-14
        )


def test_periodic_seam_splits_a_bridge_without_breaking_source_interval_order() -> None:
    vertices = ((0, 0), (2, 0), (2, 2), (0, 2), (0, 0))
    spans = tuple(_line(a, b) for a, b in pairwise(vertices))
    intervals = ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1))
    result = soften(spans, intervals, [0.0, 0.25, 0.5, 0.75, 1.0], 0.25, periodic=True)
    assert result.applied == frozenset({0.0, 0.25, 0.5, 0.75})
    _partition(result)
    for before, after in pairwise((*result.spans, result.spans[0])):
        _smooth(before, after)
    assert result.spans[0].control_points[0] == result.spans[-1].control_points[-1]
    assert result.dependencies[0] == result.dependencies[-1] == frozenset({0.0})
    assert result.intervals[0][1] < 0.25
    assert result.intervals[-1][0] > 0.75


def test_shared_boundary_reversal_produces_identical_reversed_geometry() -> None:
    spans, intervals = _elbow()
    forward = soften(spans, intervals, [0.5], 0.25)
    backward_spans = tuple(
        c.BezierSpan(
            cast("c.Controls", tuple(reversed(span.control_points))), span.is_line
        )
        for span in reversed(spans)
    )
    backward = soften(backward_spans, intervals, [0.5], 0.25)
    assert backward.applied == forward.applied
    assert backward.intervals == tuple(
        (1 - b, 1 - a) for a, b in reversed(forward.intervals)
    )
    for before, after in zip(forward.spans, reversed(backward.spans), strict=True):
        assert before.control_points == tuple(reversed(after.control_points))


def test_exclusion_and_zero_radius_restore_the_original_chain() -> None:
    spans, intervals = _elbow()
    for result in (
        soften(spans, intervals, [0.5], 0.25, excluded=frozenset({0.5})),
        soften(spans, intervals, [0.5], 0),
        soften(spans, intervals, [0.0, 1.0], 0.25),
    ):
        assert result.spans == spans
        assert result.intervals == intervals
        assert not result.applied
        assert all(not deps for deps in result.dependencies)


def test_neighboring_bridges_leave_a_positive_unchanged_middle_segment() -> None:
    vertices = ((0, 0), (2, 0), (2, 2), (4, 2))
    spans = tuple(_line(a, b) for a, b in pairwise(vertices))
    result = soften(spans, ((0.0, 0.25), (0.25, 0.75), (0.75, 1.0)), [0.25, 0.75], 10)
    assert result.applied == frozenset({0.25, 0.75})
    _partition(result)
    middle = next(
        s
        for s, deps in zip(result.spans, result.dependencies, strict=True)
        if len(deps) == 2
    )
    assert middle.is_line
    assert middle.control_points[0][0] == middle.control_points[-1][0] == 2
    assert middle.control_points[-1][1] - middle.control_points[0][1] >= 1


def test_uncertifiable_tiny_support_leaves_original_geometry_unapplied() -> None:
    spans, intervals = _elbow()
    result = soften(spans, intervals, [0.5], 1e-12)
    assert result.candidates == frozenset({0.5})
    assert not result.applied
    assert result.spans == spans


@pytest.mark.parametrize("radius", [-1.0, float("inf"), float("nan"), True])
def test_invalid_radius_is_rejected(radius: float) -> None:
    spans, intervals = _elbow()
    with pytest.raises(ValueError, match="radius"):
        _ = soften(spans, intervals, [0.5], radius)


def test_nonperiodic_contact_endpoints_are_not_rounded() -> None:
    spans, intervals = _elbow()
    result = soften(spans, intervals, [0.0, 0.5, 1.0], 0.25)
    assert result.applied == frozenset({0.5})
    assert result.spans[0].control_points[0] == spans[0].control_points[0]
    assert result.spans[-1].control_points[-1] == spans[-1].control_points[-1]


def test_adjacent_source_parameters_cannot_collapse_a_bridge_interval() -> None:
    vertices = ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))
    spans = tuple(_line(a, b) for a, b in pairwise(vertices))
    a = 0.5
    b = nextafter(a, 1)
    c_ = nextafter(b, 1)
    intervals = ((0.0, a), (a, b), (b, c_), (c_, 1.0))
    result = soften(spans, intervals, [b], 0.25)
    _partition(result)
    assert result.candidates == frozenset({b})
    assert not result.applied
    assert result.spans == spans
    assert result.intervals == intervals
