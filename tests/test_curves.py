"""Continuous cubic bounds, shared geometry, and local topology refinement."""

from itertools import pairwise
from typing import Literal, cast

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from panelchemy.contours import PanelContours, vectorize
from panelchemy.curves import (
    BezierSpan,
    CurveRing,
    fit_contours,
    polygon_lines,
    span_within_tolerance,
)
from panelchemy.separation import Labels, Mask, SeparationConfig, SeparationResult


def source(
    orange: list[Polygon] | None = None, white: list[Polygon] | None = None
) -> PanelContours:
    silhouette = box(0, 0, 64, 64)
    parts = [orient(p, sign=1) for p in [*(orange or []), *(white or [])]]
    split = len(orange or [])
    return PanelContours(
        silhouette,
        parts[:split],
        parts[split:],
        1,
        (0, 0, 64, 64),
        sum(
            len(p.exterior.coords) - 1 + sum(len(r.coords) - 1 for r in p.interiors)
            for p in [silhouette, *parts]
        ),
    )


def disc() -> PanelContours:
    labels: Labels = np.array(
        [
            [
                2
                if 7**2 <= (x - 32) ** 2 + (y - 32) ** 2 < 20**2
                or (5 <= x < 8 and 5 <= y < 8)
                else 0
                for x in range(64)
            ]
            for y in range(64)
        ],
        dtype=np.uint8,
    )
    silhouette = cast("Mask", np.ones((64, 64), dtype=np.bool_))
    return vectorize(
        SeparationResult(
            labels, silhouette, SeparationConfig(), "explicit-mask", [], 1
        ),
        64,
    )


def closed(ring: CurveRing) -> None:
    assert ring.spans
    for left, right in pairwise((*ring.spans, ring.spans[0])):
        assert left.control_points[-1] == right.control_points[0]


def test_zero_tolerance_retains_exact_footprints_and_line_segments() -> None:
    original = disc()
    result = fit_contours(original, 0)
    assert result.proxy.silhouette is original.silhouette
    assert result.proxy_deviation_mm == 0
    assert len(result.contours.white) == 2
    for before, after in zip(original.white, result.proxy.white, strict=True):
        assert before.equals(after)
    for polygon in (result.contours.silhouette, *result.contours.white):
        for ring in (polygon.exterior, *polygon.interiors):
            closed(ring)
            assert all(span.is_line for span in ring.spans)


def test_raster_disc_fits_real_curves_without_losing_counter_or_island() -> None:
    original = disc()
    result = fit_contours(original, 1, mode="curves")
    assert result.contours.edge_count < original.vertex_count
    assert cast("int", result.diagnostics["selected_unique_curved_spans"]) > 0
    assert len(result.contours.white) == 2
    assert sum(len(p.interiors) for p in result.contours.white) == 1
    assert result.proxy_deviation_mm == 1 / 64
    assert result.proxy.silhouette is original.silhouette
    assert result.diagnostics["whole_ring_rollbacks"] == 0
    assert " C " in result.comparison_svg
    for polygon in result.contours.white:
        for ring in (polygon.exterior, *polygon.interiors):
            closed(ring)
    assert all(p.is_valid and p.area > 0 for p in result.proxy.white)


@pytest.mark.parametrize("mode", ["curves", "lines"])
def test_shared_orange_white_curves_are_identical_under_reversal(
    mode: Literal["curves", "lines"],
) -> None:
    white = max(disc().white, key=lambda polygon: polygon.area)
    # A hole in the white annulus remains black; the surrounding orange region
    # uses the white outer ring as its own counter boundary.
    orange = box(2, 2, 62, 62).difference(Polygon(white.exterior.coords))
    assert isinstance(orange, Polygon)
    result = fit_contours(source(orange=[orange], white=[white]), 1, mode=mode)
    left = result.contours.orange[0].interiors[0]
    right = result.contours.white[0].exterior
    assert {span.control_points for span in left.spans} == {
        tuple(reversed(span.control_points)) for span in right.spans
    }
    assert any(not span.is_line for span in right.spans) == (mode == "curves")
    assert result.proxy.orange[0].intersection(result.proxy.white[0]).area == 0
    assert (
        result.proxy.orange[0]
        .boundary.intersection(result.proxy.white[0].boundary)
        .equals(result.proxy.white[0].exterior)
    )


def test_mid_edge_contact_is_pinned_and_source_relationship_retained() -> None:
    orange = box(10, 10, 20, 20)
    white = Polygon([(20, 15), (25, 12), (25, 18)])
    result = fit_contours(source(orange=[orange], white=[white]), 2)
    for polygon in (result.contours.orange[0], result.contours.white[0]):
        endpoints = {span.control_points[0] for span in polygon.exterior.spans}
        assert (20, 15) in endpoints
    assert result.proxy.orange[0].relate(result.proxy.white[0]) == orange.relate(white)


def test_continuous_certificate_rejects_unsampled_cubic_excursion() -> None:
    # Both endpoint samples lie on the source; the interior arch does not.
    span = BezierSpan(((0, 0), (0, 10), (1, 10), (1, 0)), False)
    assert not span_within_tolerance(span, [(0, 0), (1, 0)], 1)
    assert span_within_tolerance(span, [(0, 0), (1, 0)], 10)


def test_continuous_certificate_checks_source_between_fitted_endpoints() -> None:
    span = BezierSpan(((0, 0), (1 / 3, 0), (2 / 3, 0), (1, 0)), True)
    assert not span_within_tolerance(span, [(0, 0), (0.5, 2), (1, 0)], 0.1)
    assert span_within_tolerance(span, [(0, 0), (0.5, 2), (1, 0)], 2.1)


def test_nearby_counter_and_island_keep_their_embedding_at_large_budget() -> None:
    outer = Polygon(
        [(5, 5), (45, 5), (45, 45), (5, 45)], [[(15, 15), (15, 35), (35, 35), (35, 15)]]
    )
    island = box(15.001, 24, 16, 25)
    original = source(white=[outer, island])
    result = fit_contours(original, 10)
    assert len(result.contours.white) == 2
    assert len(result.contours.white[0].interiors) == 1
    assert result.proxy.white[0].relate(result.proxy.white[1]) == original.white[
        0
    ].relate(original.white[1])
    assert result.proxy.white[0].intersection(result.proxy.white[1]).area == 0


def test_local_refinement_keeps_a_nearby_island_without_ring_rollback() -> None:
    original = source(white=[box(5, 5, 25, 25), box(25.001, 14, 25.01, 14.01)])
    result = fit_contours(original, 100)
    assert cast("int", result.diagnostics["local_topology_refinements"]) > 0
    assert result.diagnostics["whole_ring_rollbacks"] == 0
    assert len(result.contours.white) == 2
    assert result.proxy.white[0].intersection(result.proxy.white[1]).is_empty
    assert result.contours.edge_count < original.vertex_count


def test_proxy_homotopy_preserves_tiny_island_in_a_curved_counter() -> None:
    annulus = max(disc().white, key=lambda polygon: polygon.area)
    island = box(26.00001, 32, 26.001, 32.001)
    assert Polygon(annulus.interiors[0]).contains(island)
    result = fit_contours(source(white=[annulus, island]), 1)
    assert len(result.contours.white) == 2
    assert result.proxy.white[0].relate(result.proxy.white[1]) == annulus.relate(island)
    assert "homotopy" in cast("str", result.diagnostics["topology_certificate"])
    assert cast("int", result.diagnostics["selected_unique_curved_spans"]) > 0


def test_exact_polygon_helper_uses_endpoints_for_native_lines() -> None:
    polygon = box(1, 2, 5, 8)
    curves = polygon_lines(polygon)
    assert len(curves.exterior.spans) == 4
    closed(curves.exterior)
    assert {span.control_points[0] for span in curves.exterior.spans} == set(
        polygon.exterior.coords
    )


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "1"])
def test_invalid_tolerance_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = fit_contours(source(), cast("float", value))


def test_no_color_features_produces_only_exact_silhouette() -> None:
    result = fit_contours(source(), 0.2)
    assert result.contours.orange == result.contours.white == ()
    assert result.contours.edge_count == 4
    assert result.proxy_deviation_mm == 0


def test_line_mode_keeps_registration_counters_and_only_certified_lines() -> None:
    original = disc()
    result = fit_contours(original, 1, mode="lines")
    assert result.diagnostics["method"] == "shared-chain line fitting"
    assert result.diagnostics["selected_unique_curved_spans"] == 0
    assert result.proxy_deviation_mm == 0
    assert result.contours.mm_per_pixel == original.mm_per_pixel
    assert result.contours.source_bounds_px == original.source_bounds_px
    assert result.proxy.silhouette is original.silhouette
    assert len(result.contours.white) == 2
    assert sum(len(p.interiors) for p in result.contours.white) == 1
    assert result.contours.edge_count < original.vertex_count
    assert result.diagnostics["continuous_boundary_error_upper_bound_mm"] == 1
    for polygon in (result.contours.silhouette, *result.contours.white):
        for ring in (polygon.exterior, *polygon.interiors):
            closed(ring)
            assert all(
                span.is_line and span.certified_error_mm <= 1 for span in ring.spans
            )
    assert all(p.is_valid and p.area > 0 for p in result.proxy.white)


@pytest.mark.parametrize("mode", ["other", None, False])
def test_unknown_contour_mode_is_rejected(mode: object) -> None:
    with pytest.raises(ValueError, match="contour mode"):
        _ = fit_contours(source(), 1, mode=cast("Literal['curves', 'lines']", mode))
