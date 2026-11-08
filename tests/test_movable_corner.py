"""One inferred corner simplifies a raster turn without moving splice rays."""

# pyright: reportPrivateUsage=false

from fractions import Fraction
from itertools import pairwise
from math import cos, dist, radians, sin
from typing import cast

import numpy as np
import pytest

from panelchemy import (
    boundary_tapers,
    curvature,
    curvature_fitting,
    geometric_distance,
    local_windows,
)
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting


@pytest.mark.parametrize(
    "controls,expected",
    [
        (((0, 0), (0.75, 0.5), (0.25, -0.5), (1, 0)), True),
        (((0, 0), (1.25, 0.5), (-0.25, -0.5), (1, 0)), False),
        (((0, 0), (1, 0), (0, 0), (1, 0)), False),
        (((0, 0), (1, 1), (0, 0), (1, 0)), False),
        (((0, 0), (0, 0), (0.5, 0), (1, 0)), True),
        (((0, 0), (-0.25, 0), (0.5, 0), (1, 0)), False),
        (((0, 0), (0, 1), (1, 1), (0, 0)), False),
    ],
)
def test_exact_forward_progress_rejects_reversal_and_interior_stationary_points(
    controls: tuple[tuple[float, float], ...], expected: bool
) -> None:
    for selected in (controls, tuple(reversed(controls))):
        span = c.BezierSpan(cast("c.Controls", selected), False)
        assert fitting._forward_progress(span) is expected


def test_exact_forward_progress_does_not_change_initial_control_polygon_guard() -> None:
    span = c.BezierSpan(((0, 0), (0.75, 0.5), (0.25, -0.5), (1, 0)), False)
    assert fitting._forward_progress(span)
    assert not c._monotone(span)


def example(angle: float = 0) -> tuple[fitting._Fit, local_windows.Guide]:
    a = radians(angle)

    def point(p: c.Point2) -> c.Point2:
        # Ordinary spline conversion canonicalizes poles on this grid before
        # local repair inherits their exactly representable tangent rays.
        grid = 2.0**-30
        return (
            round((p[0] * cos(a) - p[1] * sin(a)) / grid) * grid,
            round((p[0] * sin(a) + p[1] * cos(a)) / grid) * grid,
        )

    raw = tuple(
        map(
            point,
            (
                (-1, 0.2),
                (0, 0),
                (0.5, 0),
                (0.5, -0.17),
                (0.67, -0.17),
                (0.67, -0.34),
                (0.5, -0.34),
                (0.5, -0.51),
                (0, -0.51),
                (-1, -0.7),
            ),
        )
    )
    first = c.BezierSpan(
        cast(
            "c.Controls",
            tuple(map(point, ((0, 0.05), (0.33, 0.01), (0.64, -0.11), (0.64, -0.255)))),
        ),
        False,
    )
    second = c.BezierSpan(
        cast(
            "c.Controls",
            tuple(
                map(point, ((0.64, -0.255), (0.64, -0.4), (0.33, -0.54), (0, -0.56)))
            ),
        ),
        False,
    )
    spans = [
        c._line(raw[0], first.control_points[0]),
        first,
        second,
        c._line(second.control_points[-1], raw[-1]),
    ]
    lengths = [dist(p, q) for p, q in pairwise(raw)]
    distances = np.asarray([0, *np.cumsum(lengths)])
    parameters = distances / fitting._scalar(distances, -1)
    start, end = (
        float(fitting._scalar(parameters, 1)),
        float(fitting._scalar(parameters, -2)),
    )
    knots = [0.0, start, (start + end) / 2, end, 1.0]
    fit = fitting._Fit(
        c._Chain(raw, False),
        parameters,
        [start, end],
        [],
        [c._Piece(0, i, i + 1, s) for i, s in enumerate(spans)],
        list(pairwise(knots)),
    )
    return fit, local_windows.prepare(
        local_windows.Window(1, 2), raw, parameters, spans, fit.intervals
    )


@pytest.mark.parametrize("angle", [0, 37, 120])
def test_one_movable_corner_keeps_source_radius_outside_curves_and_exact_rays(
    angle: float,
) -> None:
    fit, guide = example(angle)
    limit = boundary_tapers.BoundaryTaper(2, 0, 0.35, -1)
    result, report = fitting._local_window_refit(fit, 0, guide, limit, 0.2, 2, apex=0.5)
    assert result is not None and report["basis"] == "movable-apex"
    assert len(result.pieces) == 4
    assert result.pieces[0].span is fit.pieces[0].span
    assert result.pieces[-1].span is fit.pieces[-1].span
    assert result.intervals[0] is fit.intervals[0]
    assert result.intervals[-1] is fit.intervals[-1]
    apex = result.pieces[1].span.control_points[-1]
    assert apex == result.pieces[2].span.control_points[0]
    incoming = result.pieces[1].span.control_points[-2]
    outgoing = result.pieces[2].span.control_points[1]
    u, v = (
        (apex[0] - incoming[0], apex[1] - incoming[1]),
        (outgoing[0] - apex[0], outgoing[1] - apex[1]),
    )
    assert u[0] * v[1] != u[1] * v[0]
    assert all(dist(apex, p) > 1e-5 for p in fit.chain.points)
    assert geometric_distance.within_distance(
        fit.chain.points, [p.span for p in result.pieces], 0.2
    )
    assert all(
        curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
        for p in result.pieces
    )
    for index, anchor, inherited, pole in (
        (1, fit.pieces[1].span.control_points[0], guide.inherited_handles[0], 1),
        (2, fit.pieces[2].span.control_points[-1], guide.inherited_handles[1], 2),
    ):
        position = result.pieces[index].span.control_points[pole]
        x, y = Fraction(float(anchor[0])), Fraction(float(anchor[1]))
        u, v = Fraction(inherited[0]) - x, Fraction(inherited[1]) - y
        s, t = Fraction(position[0]) - x, Fraction(position[1]) - y
        assert u * t == v * s and u * s + v * t > 0


def test_inferred_apex_cannot_remove_a_retained_interior_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, guide = example()
    fit.corners.append((guide.start_t + guide.end_t) / 2)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a single apex must not replace another retained interior corner")

    monkeypatch.setattr(curvature_fitting, "refit", unexpected)
    result, report = fitting._local_window_refit(
        fit, 0, guide, boundary_tapers.BoundaryTaper(2, 0, 0.35, -1), 0.2, 1, apex=0.5
    )
    assert result is None and report["status"] == "protected-corner"


def test_apex_does_not_relax_the_source_budget() -> None:
    fit, guide = example()
    result, _ = fitting._local_window_refit(
        fit, 0, guide, boundary_tapers.BoundaryTaper(2, 0, 0.35, -1), 0.01, 1, apex=0.5
    )
    assert result is None


@pytest.mark.parametrize("active", [[], [0], [1]])
def test_free_corner_rays_require_both_active_endpoints(
    monkeypatch: pytest.MonkeyPatch, active: list[int]
) -> None:
    fit, guide = example()
    ends = [guide.start_t, guide.end_t_original]
    fit.corners = [ends[i] for i in active]
    fit.source_corners = tuple(ends)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("inactive source corners cannot authorize a free splice ray")

    monkeypatch.setattr(curvature_fitting, "refit", unexpected)
    result, report = fitting._local_window_refit(
        fit,
        0,
        guide,
        boundary_tapers.BoundaryTaper(2, 0, 0.05, -1),
        0.2,
        1,
        compact=True,
        free_corner_rays=True,
    )
    assert result is None and report["status"] == "endpoint-not-corner"


def test_free_corner_run_keeps_pins_outside_curves_and_strict_certificates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, guide = example()
    limit = boundary_tapers.BoundaryTaper(2, 0, 0.05, -1)

    # Only the corner-bounded family may use the exact progress proof in place
    # of the sufficient forward-control-polygon restriction.
    def reject_control_polygon(_span: c.BezierSpan) -> bool:
        return False

    monkeypatch.setattr(c, "_monotone", reject_control_polygon)
    result, _ = fitting._local_window_refit(
        fit, 0, guide, limit, 0.2, 1, compact=True, free_corner_rays=True
    )
    assert result is not None
    assert result.pieces[0].span is fit.pieces[0].span
    assert result.pieces[-1].span is fit.pieces[-1].span
    assert (
        result.pieces[1].span.control_points[0] == fit.pieces[1].span.control_points[0]
    )
    assert (
        result.pieces[-2].span.control_points[-1]
        == fit.pieces[-2].span.control_points[-1]
    )
    assert geometric_distance.within_distance(
        fit.chain.points, [p.span for p in result.pieces], 0.2
    )
    assert all(
        curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
        for p in result.pieces
    )
    ordinary, _ = fitting._local_window_refit(
        fit, 0, guide, limit, 0.2, 0.1, compact=True
    )
    assert ordinary is None


@pytest.mark.parametrize(
    "apex,compact", [(0.0, False), (1.0, False), (float("nan"), False), (0.5, True)]
)
def test_invalid_apex_family_is_rejected(apex: float, compact: bool) -> None:
    fit, guide = example()
    with pytest.raises(ValueError, match="local apex"):
        _ = fitting._local_window_refit(
            fit,
            0,
            guide,
            boundary_tapers.BoundaryTaper(2, 0, 0.35, -1),
            0.2,
            1,
            apex=apex,
            compact=compact,
        )


def test_movable_apex_at_a_wrapped_seam_keeps_zero_source_provenance() -> None:
    original, _ = example()
    raw = original.chain.points
    seam = ((raw[4][0] + raw[5][0]) / 2, (raw[4][1] + raw[5][1]) / 2)
    points = (seam, *raw[5:], *raw[:5], seam)
    spans = [
        original.pieces[2].span,
        original.pieces[3].span,
        c._line(raw[-1], raw[0]),
        original.pieces[0].span,
        original.pieces[1].span,
    ]
    distances = np.asarray([0, *np.cumsum([dist(a, b) for a, b in pairwise(points)])])
    parameters = distances / fitting._scalar(distances, -1)
    stops = [
        0.0,
        *[
            float(fitting._scalar(parameters, points.index(p)))
            for p in (raw[-2], raw[-1], raw[0], raw[1])
        ],
        1.0,
    ]
    fit = fitting._Fit(
        c._Chain(points, False, True),
        parameters,
        [stops[1], stops[-2]],
        [],
        [c._Piece(0, i, i + 1, s) for i, s in enumerate(spans)],
        list(pairwise(stops)),
    )
    guide = local_windows.prepare(
        local_windows.Window(4, 2),
        points,
        parameters,
        spans,
        fit.intervals,
        periodic=True,
    )
    assert guide.seam is not None
    result, _ = fitting._local_window_refit(
        fit,
        0,
        guide,
        boundary_tapers.BoundaryTaper(2, 0, 0.35, -1),
        0.2,
        2,
        apex=guide.seam,
    )
    assert result is not None
    assert 0.0 in result.corners and 1.0 not in result.corners
    assert result.intervals[0][0] == 0 and result.intervals[-1][1] == 1
    assert (
        result.pieces[0].span.control_points[0]
        == result.pieces[-1].span.control_points[-1]
    )
