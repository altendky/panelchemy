"""Physical-scale outlines before CAD: corners, smooth joins, and shared topology."""

from fractions import Fraction
from itertools import pairwise
from math import atan2, degrees, hypot

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from panelchemy.approximation import boundary_within_tolerance
from panelchemy.contours import PanelContours, vectorize
from panelchemy.curves import (
    BezierSpan,
    CurveFitResult,
    CurveRing,
    Point2,
    _Chain,  # pyright: ignore[reportPrivateUsage]
    _line,  # pyright: ignore[reportPrivateUsage]
    _Piece,  # pyright: ignore[reportPrivateUsage]
)
from panelchemy.outline_fitting import (
    _certified,  # pyright: ignore[reportPrivateUsage]
    _Fit,  # pyright: ignore[reportPrivateUsage]
    _pieces,  # pyright: ignore[reportPrivateUsage]
    fit_outlines,
)
from panelchemy.separation import Labels, SeparationConfig, SeparationResult

_SIZE_MM = 12.8
_TOLERANCE_MM = 0.2


def _source(
    *, orange: list[Polygon] | None = None, white: list[Polygon] | None = None
) -> PanelContours:
    orange_parts = [orient(polygon, sign=1) for polygon in orange or []]
    white_parts = [orient(polygon, sign=1) for polygon in white or []]
    silhouette = box(0, 0, _SIZE_MM, _SIZE_MM)
    return PanelContours(
        silhouette,
        orange_parts,
        white_parts,
        0.1,
        (0, 0, 128, 128),
        sum(
            len(polygon.exterior.coords)
            - 1
            + sum(len(ring.coords) - 1 for ring in polygon.interiors)
            for polygon in (silhouette, *orange_parts, *white_parts)
        ),
    )


def test_single_span_chain_pins_both_nonbinary_junctions_exactly() -> None:
    start, end = (1.1, 2.2), (3.3, 4.4)
    fit = _Fit(
        _Chain((start, end), fixed=False),
        np.array([0.0, 1.0]),
        [],
        [],
        [],
        [],
    )
    pieces, _ = _pieces(fit, 0, _TOLERANCE_MM)
    assert len(pieces) == 1
    assert pieces[0].span.control_points[0] == start
    assert pieces[0].span.control_points[-1] == end


def test_shared_one_span_boundary_keeps_both_closed_rings_connected() -> None:
    original = _source(
        orange=[box(1.1, 1.1, 6.3, 10.1)],
        white=[box(6.3, 1.1, 10.1, 10.1)],
    )
    result = fit_outlines(original, _TOLERANCE_MM)
    _check_geometry(original, result)
    orange = result.contours.orange[0].exterior.spans
    white = result.contours.white[0].exterior.spans
    shared = [
        a
        for a in orange
        if any(a.control_points == tuple(reversed(b.control_points)) for b in white)
    ]
    assert len(shared) == 1
    assert {shared[0].control_points[0], shared[0].control_points[-1]} == {
        (6.3, 1.1),
        (6.3, 10.1),
    }


def test_periodic_chain_closes_exactly_at_a_deliberate_corner() -> None:
    points = ((1.1, 1.1), (3.3, 1.1), (3.3, 3.3), (1.1, 3.3), (1.1, 1.1))
    fit = _Fit(
        _Chain(points, fixed=False, free_loop=True),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        [0.0, 0.25, 0.5, 0.75],
        [0.25, 0.5, 0.75],
        [],
        [],
    )
    fit.pieces, fit.intervals = _pieces(fit, 0, _TOLERANCE_MM)
    ring = CurveRing(tuple(piece.span for piece in fit.pieces))
    assert ring.spans[-1].control_points[-1] == ring.spans[0].control_points[0]
    assert all(angle > 80 for _, angle in _joins(ring))
    assert all(_certified(fit, piece, _TOLERANCE_MM) for piece in fit.pieces)


def test_error_certificate_uses_exact_original_source_at_moved_knots() -> None:
    fit = _Fit(
        _Chain(((0.0, 0.0), (1 / 3, 1.0)), fixed=False),
        np.array([0.0, 1.0]),
        [],
        [],
        [],
        [(0.375, 0.75)],
    )
    piece = _Piece(0, 0, 1, _line((0.125, 0.375), (0.25, 0.75)))
    tolerance = 1e-20
    x, y = (Fraction(value) for value in piece.span.control_points[-1])
    dx, dy = (Fraction(value) for value in fit.chain.points[-1])
    distance_squared = (x * dy - y * dx) ** 2 / (dx * dx + dy * dy)
    # Rounded interpolation puts the candidate exactly on a replacement line.
    # The endpoint's true distance to the original source is about 1.32e-17 mm.
    assert distance_squared > Fraction(tolerance) ** 2
    assert not _certified(fit, piece, tolerance)


def test_exact_knot_normalization_accepts_an_unchanged_source_line() -> None:
    fit = _Fit(
        _Chain(((0.0, 0.0), (0.2, 0.0), (1.0, 0.0)), fixed=False),
        np.array([0.0, 0.2, 1.0]),
        [],
        [],
        [],
        [(0.1, 0.3)],
    )
    piece = _Piece(0, 0, 1, _line((0.1, 0.0), (0.3, 0.0)))
    assert _certified(fit, piece, 1e-20)


def _raster(labels: Labels) -> PanelContours:
    silhouette = np.ones(labels.shape, dtype=np.bool_)
    return vectorize(
        SeparationResult(
            labels, silhouette, SeparationConfig(), "explicit-mask", [], 1
        ),
        _SIZE_MM,
    )


def _disc_labels() -> Labels:
    return np.array(
        [
            [2 if (x - 64) ** 2 + (y - 64) ** 2 < 40**2 else 0 for x in range(128)]
            for y in range(128)
        ],
        dtype=np.uint8,
    )


def _tangent(span: BezierSpan, *, end: bool) -> Point2:
    a, b, c, d = span.control_points
    if span.is_line:
        direction = (d[0] - a[0], d[1] - a[1])
    elif end:
        direction = (d[0] - c[0], d[1] - c[1])
    else:
        direction = (b[0] - a[0], b[1] - a[1])
    assert hypot(*direction) > 1e-10, "a zero derivative cannot establish a G1 join"
    return direction


def _joins(ring: CurveRing) -> list[tuple[Point2, float]]:
    assert ring.spans
    joins: list[tuple[Point2, float]] = []
    for previous, following in pairwise((*ring.spans, ring.spans[0])):
        assert previous.control_points[-1] == following.control_points[0]
        incoming, outgoing = (
            _tangent(previous, end=True),
            _tangent(following, end=False),
        )
        angle = degrees(
            atan2(
                abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]),
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1],
            )
        )
        joins.append((previous.control_points[-1], angle))
    return joins


def _check_geometry(original: PanelContours, result: CurveFitResult) -> None:
    assert result.proxy.silhouette.equals(original.silhouette)
    assert result.contours.mm_per_pixel == original.mm_per_pixel
    assert result.contours.source_bounds_px == original.source_bounds_px
    assert len(result.contours.orange) == len(original.orange)
    assert len(result.contours.white) == len(original.white)
    assert 0 <= result.proxy_deviation_mm < _TOLERANCE_MM
    for originals, proxies, curves in (
        (original.orange, result.proxy.orange, result.contours.orange),
        (original.white, result.proxy.white, result.contours.white),
    ):
        for source, proxy, curve in zip(originals, proxies, curves, strict=True):
            assert proxy.is_valid and proxy.area > 0
            assert len(proxy.interiors) == len(curve.interiors) == len(source.interiors)
            assert proxy.exterior.is_ccw
            assert all(not ring.is_ccw for ring in proxy.interiors)
            assert original.silhouette.covers(proxy)
            assert boundary_within_tolerance(
                source.boundary,
                proxy.boundary,
                _TOLERANCE_MM + result.proxy_deviation_mm,
            )
            for ring in (curve.exterior, *curve.interiors):
                _ = _joins(ring)


def test_raster_diagonal_fits_whole_edges_without_retaining_pixel_stairs() -> None:
    labels: Labels = np.array(
        [
            [2 if 20 <= x <= 100 and abs(y - x - 3) <= 8 else 0 for x in range(128)]
            for y in range(128)
        ],
        dtype=np.uint8,
    )
    original = _raster(labels)
    result = fit_outlines(original, _TOLERANCE_MM, corner_scale_mm=1)
    _check_geometry(original, result)
    ring = result.contours.white[0].exterior
    assert len(ring.spans) < (len(original.white[0].exterior.coords) - 1) / 3
    # Only meaningful cap corners may break tangent continuity; the many
    # one-pixel horizontal/vertical turns must not become outline corners.
    assert sum(angle > 5 for _, angle in _joins(ring)) <= 6


@pytest.fixture
def fitted_disc() -> tuple[PanelContours, CurveFitResult]:
    original = _raster(_disc_labels())
    return original, fit_outlines(original, _TOLERANCE_MM, corner_scale_mm=1)


def test_raster_circle_has_smooth_joins_including_the_closed_seam(
    fitted_disc: tuple[PanelContours, CurveFitResult],
) -> None:
    original, result = fitted_disc
    _check_geometry(original, result)
    ring = result.contours.white[0].exterior
    assert any(not span.is_line for span in ring.spans)
    assert len(ring.spans) < (len(original.white[0].exterior.coords) - 1) / 2
    assert all(angle < 1e-4 for _, angle in _joins(ring))


def test_ordinary_smooth_joins_can_leave_the_source_pixel_vertices(
    fitted_disc: tuple[PanelContours, CurveFitResult],
) -> None:
    original, result = fitted_disc
    source_vertices = list(original.white[0].exterior.coords)
    assert any(
        min(hypot(point[0] - x, point[1] - y) for x, y in source_vertices) > 1e-6
        for point, _ in _joins(result.contours.white[0].exterior)
    )


def test_a_substantial_sharp_terminal_point_remains_a_true_corner() -> None:
    tip = (10.8, 6.4)
    original = _source(white=[Polygon([(2, 2), tip, (2, 10.8)])])
    result = fit_outlines(original, _TOLERANCE_MM, corner_scale_mm=1)
    _check_geometry(original, result)
    joins = _joins(result.contours.white[0].exterior)
    point, angle = min(
        joins, key=lambda join: hypot(join[0][0] - tip[0], join[0][1] - tip[1])
    )
    assert hypot(point[0] - tip[0], point[1] - tip[1]) <= _TOLERANCE_MM
    assert angle > 60


def test_shared_boundaries_reuse_identical_spans_with_reverse_orientation() -> None:
    white = _raster(_disc_labels()).white[0]
    orange = box(1, 1, 11.8, 11.8).difference(white)
    assert isinstance(orange, Polygon)
    original = _source(orange=[orange], white=[white])
    result = fit_outlines(original, _TOLERANCE_MM, corner_scale_mm=1)
    _check_geometry(original, result)
    orange_ring = result.contours.orange[0].interiors[0]
    white_ring = result.contours.white[0].exterior
    assert {span.control_points for span in orange_ring.spans} == {
        tuple(reversed(span.control_points)) for span in white_ring.spans
    }
    assert result.proxy.orange[0].intersection(result.proxy.white[0]).area == 0
    assert (
        result.proxy.orange[0]
        .boundary.intersection(result.proxy.white[0].boundary)
        .equals(result.proxy.white[0].exterior)
    )
    assert all(angle < 1e-4 for _, angle in _joins(white_ring))


def test_a_tiny_counter_survives_instead_of_becoming_feature_cleanup() -> None:
    outer = [(2, 2), (10.8, 2), (10.8, 10.8), (2, 10.8)]
    counter = [(6.3, 6.3), (6.5, 6.3), (6.4, 6.5)]
    original = _source(white=[Polygon(outer, [counter])])
    result = fit_outlines(original, _TOLERANCE_MM, corner_scale_mm=1)
    _check_geometry(original, result)
    assert len(result.contours.white) == 1
    assert len(result.contours.white[0].interiors) == 1
    assert Polygon(result.proxy.white[0].interiors[0]).area > 0


def test_nearest_neighbor_upscale_does_not_change_physical_fit(
    fitted_disc: tuple[PanelContours, CurveFitResult],
) -> None:
    original, first = fitted_disc
    doubled: Labels = np.repeat(np.repeat(_disc_labels(), 2, axis=0), 2, axis=1)
    upscaled = _raster(doubled)
    assert upscaled.white[0].equals(original.white[0])
    second = fit_outlines(upscaled, _TOLERANCE_MM, corner_scale_mm=1)
    _check_geometry(upscaled, second)
    assert len(first.contours.white[0].exterior.spans) == len(
        second.contours.white[0].exterior.spans
    )
    assert boundary_within_tolerance(
        first.proxy.white[0].boundary, second.proxy.white[0].boundary, 1e-7
    )
    assert all(angle < 1e-4 for _, angle in _joins(second.contours.white[0].exterior))
