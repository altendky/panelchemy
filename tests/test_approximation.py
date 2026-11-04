"""Approximation budgets preserve individual islands, counters, and contacts."""

from typing import cast

import pytest
from shapely import union_all
from shapely.affinity import translate
from shapely.geometry import LinearRing, LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy import approximation
from panelchemy.approximation import approximate
from panelchemy.contours import PanelContours


def vertex_count(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def contours(
    *, orange: list[Polygon] | None = None, white: list[Polygon] | None = None
) -> PanelContours:
    orange = orange or []
    white = white or []
    silhouette = box(0, 0, 20, 20)
    return PanelContours(
        silhouette,
        orange,
        white,
        1.0,
        (0, 0, 20, 20),
        vertex_count(silhouette) + sum(vertex_count(p) for p in [*orange, *white]),
    )


def staircase() -> Polygon:
    polygon = union_all([box(1, row, row + 3, row + 1) for row in range(1, 11)])
    assert isinstance(polygon, Polygon)
    return polygon


def test_zero_tolerance_returns_exact_source_without_changes() -> None:
    source = contours(white=[staircase()])
    result = approximate(source, 0)
    assert result.contours is source
    assert result.diagnostics["method"] == "exact"
    assert result.diagnostics["changed_ring_count"] == 0
    assert result.diagnostics["certified_boundary_deviation_upper_bound_mm"] == 0


def test_comparison_uses_the_configured_artwork_palette() -> None:
    source = contours(orange=[box(1, 1, 3, 3)], white=[box(5, 1, 7, 3)])
    result = approximate(source, 0, palette=((10, 20, 30), (40, 50, 60), (70, 80, 90)))
    assert 'fill="#0a141e"' in result.comparison_svg
    assert 'fill="#28323c"' in result.comparison_svg
    assert 'fill="#46505a"' in result.comparison_svg
    assert 'stroke="#1166cc"' in result.comparison_svg
    assert 'stroke="#e07700"' in result.comparison_svg


def test_staircase_reduces_vertices_with_physical_boundary_budget() -> None:
    source = contours(white=[staircase()])
    result = approximate(source, 0.65)
    selected = result.contours.white[0]
    assert result.contours.silhouette is source.silhouette
    assert result.contours.vertex_count < source.vertex_count
    assert selected.boundary.difference(source.white[0].boundary.buffer(0.65)).is_empty
    assert source.white[0].boundary.difference(selected.boundary.buffer(0.65)).is_empty
    assert result.diagnostics["certified_boundary_deviation_upper_bound_mm"] == 0.65
    assert "Original contours" in result.comparison_svg
    assert "Selected approximation" in result.comparison_svg
    assert "Outline overlay" in result.comparison_svg
    color_changes = cast(
        "dict[str, dict[str, float]]", result.diagnostics["color_changes"]
    )
    assert color_changes["white"]["symmetric_difference_area_mm2"] > 0
    assert (
        color_changes["white"]["source_perimeter_mm"]
        > color_changes["white"]["selected_perimeter_mm"]
    )
    ring_changes = cast(
        "list[dict[str, object]]", result.diagnostics["observed_ring_changes"]
    )
    assert ring_changes
    assert result.diagnostics["observed_discrete_distances_are_certificates"] is False


def test_tiny_island_and_counter_remain_present_even_with_large_tolerance() -> None:
    source = contours(
        white=[
            Polygon(
                box(1, 1, 9, 9).exterior.coords, [box(3, 3, 3.02, 3.02).exterior.coords]
            ),
            box(11, 1, 11.01, 1.01),
        ]
    )
    result = approximate(source, 2)
    assert len(result.contours.white) == 2
    assert len(result.contours.white[0].interiors) == 1
    assert all(
        polygon.is_valid and polygon.area > 0 for polygon in result.contours.white
    )
    assert Polygon(result.contours.white[0].interiors[0]).area > 0
    assert result.diagnostics["counter_count"] == 1


def test_diagonal_hole_contact_is_pinned() -> None:
    source_polygon = Polygon(
        box(1, 1, 7, 7).exterior.coords,
        [box(2, 2, 3, 3).exterior.coords, box(3, 3, 4, 4).exterior.coords],
    )
    result = approximate(contours(white=[source_polygon]), 0.8)
    selected = result.contours.white[0]
    assert len(selected.interiors) == 2
    assert (
        selected.interiors[0]
        .intersection(selected.interiors[1])
        .equals(source_polygon.interiors[0].intersection(source_polygon.interiors[1]))
    )
    assert selected.is_valid


def test_same_and_cross_color_shared_boundaries_do_not_overlap() -> None:
    white = staircase()
    remainder = box(1, 1, 13, 11).difference(white)
    assert isinstance(remainder, Polygon)
    source = contours(orange=[remainder], white=[white])
    result = approximate(source, 0.65)
    selected_orange, selected_white = (
        result.contours.orange[0],
        result.contours.white[0],
    )
    assert selected_orange.intersection(selected_white).area == 0
    assert selected_orange.relate(selected_white) == remainder.relate(white)


def test_continuous_budget_rejects_the_discrete_hausdorff_counterexample() -> None:
    original = LineString([(130, 0), (0, 0), (0, 150)])
    selected = LineString([(10, 10), (10, 150), (130, 10)])
    assert original.hausdorff_distance(selected) < 20
    # This point is on the selected segment but is not sampled by the discrete
    # distance between the two original vertex lists.
    assert selected.covers(Point(70, 80))
    assert original.distance(Point(70, 80)) > 20
    assert not approximation.boundary_within_tolerance(original, selected, 20)


def test_invalid_ring_proposal_is_retained_exactly_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = contours(white=[box(1, 1, 3, 3)])

    def bad_ring(
        ring: LinearRing, _tolerance: float, _anchors: set[tuple[float, float]]
    ) -> LinearRing:
        return LinearRing(translate(ring, xoff=1).coords)

    monkeypatch.setattr(approximation, "_simplify_ring", bad_ring)
    result = approximate(source, 0.1)
    assert result.contours.white[0].equals(source.white[0])
    assert result.diagnostics["fallback_ring_count"] == 1
    assert "boundary deviation" in str(result.diagnostics["fallback_rings"])


def test_new_cross_color_overlap_restores_exact_rings_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = contours(orange=[box(3.1, 1, 5.1, 3)], white=[box(1, 1, 3, 3)])

    def overlapping_ring(
        ring: LinearRing, _tolerance: float, _anchors: set[tuple[float, float]]
    ) -> LinearRing:
        if ring.bounds == (1, 1, 3, 3):
            return LinearRing(translate(ring, xoff=0.4).coords)
        return ring

    monkeypatch.setattr(approximation, "_simplify_ring", overlapping_ring)
    result = approximate(source, 0.5)
    assert result.contours.orange[0].intersection(result.contours.white[0]).area == 0
    assert result.contours.white[0].equals(source.white[0])
    assert result.diagnostics["fallback_ring_count"] == 1
    assert "relationships" in str(result.diagnostics["fallback_rings"])


def test_small_tolerance_never_uses_a_fixed_absolute_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = contours(white=[box(1, 1, 3, 3)])

    def displaced_ring(
        ring: LinearRing, _tolerance: float, _anchors: set[tuple[float, float]]
    ) -> LinearRing:
        return LinearRing(translate(ring, xoff=5e-10).coords)

    monkeypatch.setattr(approximation, "_simplify_ring", displaced_ring)
    result = approximate(source, 1e-12)
    assert result.contours.white[0].equals(source.white[0])
    assert result.diagnostics["fallback_ring_count"] == 1


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, 10**1000])
def test_invalid_tolerance_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _ = approximate(contours(), cast("float", value))


def test_a_failed_counter_proposal_does_not_undo_an_accepted_outer_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = staircase()
    source = contours(
        white=[
            Polygon(outer.exterior.coords, [box(1.4, 2.3, 1.6, 2.5).exterior.coords])
        ]
    )
    simplify_ring = LinearRing.simplify

    def bad_counter(
        ring: LinearRing, tolerance: float, preserve_topology: bool = True
    ) -> BaseGeometry:
        if ring.bounds == (1.4, 2.3, 1.6, 2.5):
            return LinearRing(translate(ring, xoff=3).coords)
        return simplify_ring(ring, tolerance, preserve_topology=preserve_topology)

    monkeypatch.setattr(LinearRing, "simplify", bad_counter)
    result = approximate(source, 0.65)
    assert len(result.contours.white[0].exterior.coords) < len(
        source.white[0].exterior.coords
    )
    assert result.contours.white[0].interiors[0].equals(source.white[0].interiors[0])
    assert result.diagnostics["fallback_ring_count"] == 1
