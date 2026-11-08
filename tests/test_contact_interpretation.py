"""Small point-contact tips are explicit, bounded interpretations of raster cells."""

from dataclasses import replace
from itertools import permutations
from typing import cast

import pytest
from shapely import affinity, union_all
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy.contact_interpretation import (
    ContactInterpretation,
    PointContact,
    before_graph,
)
from panelchemy.contours import PanelContours


def source(
    white: list[Polygon],
    orange: list[Polygon] | None = None,
    silhouette: Polygon | None = None,
) -> PanelContours:
    orange = orange or []
    silhouette = silhouette if silhouette is not None else box(0, 0, 20, 20)
    polygons = [silhouette, *orange, *white]
    count = sum(
        len(p.exterior.coords) - 1 + sum(len(r.coords) - 1 for r in p.interiors)
        for p in polygons
    )
    return PanelContours(silhouette, orange, white, 0.1, (0, 0, 200, 200), count)


def interpret(contours: PanelContours) -> ContactInterpretation:
    return before_graph(contours, scale_mm=0.5, tolerance_mm=0.2)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("color", ["orange", "white"])
def test_corner_tip_joins_one_simple_polygon_with_an_apex(
    angle: int, reflected: bool, color: str
) -> None:
    def transform[T: BaseGeometry](geometry: T) -> T:
        return affinity.rotate(
            affinity.scale(geometry, -1 if reflected else 1, 1, origin=(6, 6)),
            angle,
            origin=(6, 6),
        )

    parent = transform(box(4, 4, 8, 8))
    tip = transform(box(8, 8, 8.2, 8.2))
    apex = transform(Point(8.2, 8.2))
    original = source([parent, tip]) if color == "white" else source([], [parent, tip])
    original_parts = original.white if color == "white" else original.orange
    before = [p.wkb for p in original_parts]
    result = interpret(original)
    selected = result.contours.white if color == "white" else result.contours.orange
    assert len(selected) == 1 and selected[0].is_valid
    assert not selected[0].interiors
    assert selected[0].covers(parent) and selected[0].covers(tip)
    assert selected[0].area - parent.area - tip.area == pytest.approx(0.05**2)
    assert (
        selected[0].boundary.hausdorff_distance(
            union_all([parent.boundary, tip.boundary])
        )
        <= 0.05
    )
    assert result.tip_corners == ((apex.x, apex.y),)
    groups = result.white_sources if color == "white" else result.orange_sources
    assert groups == ((0, 1),)
    assert selected[0].exterior.is_ccw
    assert result.contours.silhouette is original.silhouette
    assert result.contours.source_bounds_px == original.source_bounds_px
    assert result.contours.mm_per_pixel == original.mm_per_pixel
    assert [p.wkb for p in original_parts] == before
    assert original.white == ([parent, tip] if color == "white" else [])


def test_two_children_join_one_parent_without_altering_an_unrelated_part() -> None:
    parent = box(4, 4, 8, 8)
    first, second = box(8, 8, 8.2, 8.2), box(3.8, 3.8, 4, 4)
    unrelated = box(12, 12, 13, 13)
    orange = box(1, 1, 2, 2)
    original = source([parent, first, second, unrelated], [orange])
    result = interpret(original)
    assert len(result.contours.white) == 2
    assert result.white_sources == ((0, 1, 2), (3,))
    assert result.orange_sources == ((0,),)
    assert result.contours.white[1] is unrelated
    assert result.contours.orange[0] is orange
    assert result.tip_corners == ((3.8, 3.8), (8.2, 8.2))
    assert result.contours.vertex_count == sum(
        len(p.exterior.coords) - 1
        for p in [
            result.contours.silhouette,
            *result.contours.orange,
            *result.contours.white,
        ]
    )


def test_input_order_does_not_change_interpreted_geometry() -> None:
    parts = [box(4, 4, 8, 8), box(8, 8, 8.2, 8.2), box(3.8, 3.8, 4, 4)]
    expected = interpret(source(parts))
    for ordering in permutations(parts):
        result = interpret(source(list(ordering)))
        assert result.contours.white[0].wkb == expected.contours.white[0].wkb
        assert result.tip_corners == expected.tip_corners
        assert result.white_sources == ((0, 1, 2),)


def test_selected_subset_keeps_another_child_separate() -> None:
    parent = box(4, 4, 8, 8)
    first, second = box(8, 8, 8.2, 8.2), box(3.8, 3.8, 4, 4)
    original = source([parent, first, second])
    all_contacts = interpret(original).contacts
    assert {contact.tip for contact in all_contacts} == {1, 2}
    result = before_graph(
        original,
        scale_mm=0.5,
        tolerance_mm=0.2,
        selected_tips=frozenset({("white", 1)}),
    )
    assert len(result.contours.white) == 2
    assert result.white_sources == ((2,), (0, 1))
    assert result.contours.white[0] is second
    assert result.contacts == (PointContact("white", 1, 0, (8, 8), (8.2, 8.2)),)
    assert result.tip_corners == ((8.2, 8.2),)
    empty = before_graph(
        original, scale_mm=0.5, tolerance_mm=0.2, selected_tips=frozenset()
    )
    assert empty.contours is original
    assert empty.white_sources == ((0,), (1,), (2,))
    assert empty.contacts == () and empty.tip_corners == ()


def test_contact_subset_is_scoped_to_its_color() -> None:
    original = source(
        [box(10, 10, 14, 14), box(14, 14, 14.2, 14.2)],
        [box(4, 4, 8, 8), box(8, 8, 8.2, 8.2)],
    )
    result = before_graph(
        original,
        scale_mm=0.5,
        tolerance_mm=0.2,
        selected_tips=frozenset({("orange", 1)}),
    )
    assert result.orange_sources == ((0, 1),)
    assert result.white_sources == ((0,), (1,))
    assert result.contours.white is original.white
    assert result.contacts == (PointContact("orange", 1, 0, (8, 8), (8.2, 8.2)),)


def test_parent_counter_is_preserved() -> None:
    parent = Polygon(box(4, 4, 8, 8).exterior.coords, [box(5, 5, 7, 7).exterior.coords])
    result = interpret(source([parent, box(8, 8, 8.2, 8.2)]))
    assert len(result.contours.white) == 1
    assert len(result.contours.white[0].interiors) == 1
    assert Polygon(result.contours.white[0].interiors[0]).equals(box(5, 5, 7, 7))


@pytest.mark.parametrize(
    "case",
    ["free", "edge", "equal", "both-small", "long", "multiple", "multi-point", "hole"],
)
def test_other_source_relationships_are_not_interpreted(case: str) -> None:
    parent, tip = box(4, 4, 8, 8), box(8, 8, 8.2, 8.2)
    extras: list[Polygon] = []
    scale = 0.5
    if case == "free":
        tip = box(8.1, 8.1, 8.3, 8.3)
    elif case == "edge":
        tip = box(8, 6, 8.2, 6.2)
    elif case == "equal":
        parent = box(7.8, 7.8, 8, 8)
    elif case == "both-small":
        parent = box(7.6, 7.6, 8, 8)
    elif case == "long":
        tip = box(8, 8, 8.1, 10)
    elif case == "multiple":
        extras = [box(8.2, 8.2, 12, 12)]
    elif case == "multi-point":
        parent = Polygon(
            [(4, 4), (8, 4), (8, 8), (7, 8), (7, 5), (5, 5), (5, 8), (4, 8)]
        )
        tip, scale = box(5, 8, 7, 8.2), 2.1
    else:
        tip = Polygon(
            tip.exterior.coords, [box(8.05, 8.05, 8.15, 8.15).exterior.coords]
        )
    original = source([parent, tip, *extras])
    result = before_graph(original, scale_mm=scale, tolerance_mm=0.2)
    assert result.contours is original
    assert result.white_sources == tuple((i,) for i in range(len(original.white)))
    assert result.tip_corners == ()


@pytest.mark.parametrize(
    "case", ["silhouette", "foreign-bridge", "foreign-tip", "same-color-bridge"]
)
def test_other_boundaries_prevent_a_join(case: str) -> None:
    parent, tip = box(4, 4, 8, 8), box(8, 8, 8.2, 8.2)
    near_bridge = box(7.98, 8.005, 7.995, 8.025)
    white = [parent, tip]
    orange: list[Polygon] = []
    silhouette = box(0, 0, 20, 20)
    if case == "silhouette":
        silhouette = box(0, 0, 8.2, 20)
    elif case == "foreign-bridge":
        orange = [near_bridge]
    elif case == "foreign-tip":
        orange = [box(8.2, 8.2, 8.3, 8.3)]
    else:
        white.append(near_bridge)
    original = source(white, orange, silhouette)
    assert interpret(original).contours is original


def test_bridge_size_uses_the_existing_tolerance_when_smaller_than_half_pixel() -> None:
    original = source([box(4, 4, 8, 8), box(8, 8, 8.2, 8.2)])
    result = before_graph(original, scale_mm=0.5, tolerance_mm=0.01)
    assert len(result.contours.white) == 1
    assert result.contours.white[0].area - sum(
        p.area for p in original.white
    ) == pytest.approx(0.01**2)
    assert (
        result.contours.white[0].boundary.hausdorff_distance(
            union_all([p.boundary for p in original.white])
        )
        <= 0.01
    )


@pytest.mark.parametrize("field", ["scale_mm", "tolerance_mm"])
@pytest.mark.parametrize("value", [True, 0, -0.1, float("inf"), float("nan")])
def test_invalid_physical_limits_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        _ = before_graph(
            source([]),
            scale_mm=value if field == "scale_mm" else 0.5,
            tolerance_mm=value if field == "tolerance_mm" else 0.2,
        )


def test_invalid_source_geometry_and_pitch_are_rejected() -> None:
    original = source([box(4, 4, 8, 8), box(7, 7, 9, 9)])
    with pytest.raises(ValueError, match="disjoint interiors"):
        _ = interpret(original)
    with pytest.raises(ValueError, match="contained polygons"):
        _ = interpret(source([box(19, 19, 21, 21)]))
    with pytest.raises(ValueError, match="positive and finite"):
        _ = interpret(replace(source([]), mm_per_pixel=cast("float", True)))


def chain_source(count: int = 3) -> PanelContours:
    return source(
        [
            box(4, 4, 8, 8),
            *(
                box(8 + 0.2 * i, 8 + 0.2 * i, 8 + 0.2 * (i + 1), 8 + 0.2 * (i + 1))
                for i in range(count)
            ),
        ]
    )


@pytest.mark.parametrize("selected", [None, 1, 2, 3])
def test_point_connected_chain_is_one_selected_group(selected: int | None) -> None:
    original = chain_source()
    result = before_graph(
        original,
        scale_mm=0.5,
        tolerance_mm=0.2,
        selected_tips=None if selected is None else frozenset({("white", selected)}),
    )
    assert len(result.contours.white) == 1
    assert result.white_sources == ((0, 1, 2, 3),)
    (contact,) = result.contacts
    assert contact.tip == 1 and contact.parent == 0
    assert contact.members == (1, 2, 3) and contact.points == ((8, 8),)
    assert contact.point == (8, 8) and contact.apex == (8.6, 8.6)
    assert all(result.contours.white[0].covers(p) for p in original.white)
    detail = cast("dict[str, object]", result.diagnostics["contact_interpretation"])
    assert detail["joined_tips"] == 3 and detail["joined_groups"] == 1


@pytest.mark.parametrize("case", ["extent", "parents", "area", "edge", "hole"])
def test_contact_graph_rejects_unbounded_or_ambiguous_clusters(case: str) -> None:
    original = chain_source(6 if case == "extent" else 3)
    parts = list(original.white)
    if case == "parents":
        parts.append(box(8.6, 8.6, 12, 12))
    elif case == "area":
        parts[0] = box(4, 7.99, 8, 8)
    elif case == "edge":
        parts = [parts[0], box(8, 8, 8.2, 8.2), box(8.2, 8, 8.4, 8.2)]
    elif case == "hole":
        parts[1] = Polygon(
            parts[1].exterior.coords, [box(8.05, 8.05, 8.15, 8.15).exterior.coords]
        )
    original = source(parts)
    result = interpret(original)
    assert result.contours is original and not result.contacts


def double_contact_source(
    *, depth: float = 0.2, old_hole: bool = False
) -> PanelContours:
    parent = Polygon(
        [
            (4, 4),
            (8, 4),
            (8, 8),
            (6.2, 8),
            (6.2, 8 - depth),
            (6, 8 - depth),
            (6, 8),
            (4, 8),
        ],
        [box(4.5, 4.5, 5, 5).exterior.coords] if old_hole else [],
    )
    return source([parent, box(6, 8, 6.2, 8.2)])


@pytest.mark.parametrize("old_hole", [False, True])
def test_double_contact_fills_only_its_new_small_counter(old_hole: bool) -> None:
    original = double_contact_source(old_hole=old_hole)
    result = interpret(original)
    assert len(result.contours.white) == 1 and result.white_sources == ((0, 1),)
    selected = result.contours.white[0]
    assert len(selected.interiors) == int(old_hole)
    if old_hole:
        assert Polygon(selected.interiors[0]).equals(box(4.5, 4.5, 5, 5))
    assert selected.covers(box(6, 7.8, 6.2, 8))
    (contact,) = result.contacts
    assert contact.members == (1,) and contact.points == ((6, 8), (6.2, 8))
    assert contact.point == contact.points[0]
    detail = cast("dict[str, object]", result.diagnostics["contact_interpretation"])
    (join,) = cast("list[dict[str, object]]", detail["joins"])
    assert join["source_members"] == (1,) and join["filled_contact_counters"] == 1
    assert 0 < cast("float", join["filled_contact_counter_area_mm2"]) < 0.04


@pytest.mark.parametrize("case", ["large", "foreign", "unrelated", "silhouette-hole"])
def test_new_counter_fill_cannot_consume_other_boundaries(case: str) -> None:
    original = double_contact_source(depth=0.7 if case == "large" else 0.2)
    obstacle = box(6.08, 7.86, 6.12, 7.90)
    if case == "foreign":
        original = source(original.white, [obstacle])
    elif case == "unrelated":
        original = source([*original.white, obstacle])
    elif case == "silhouette-hole":
        silhouette = Polygon(
            box(0, 0, 20, 20).exterior.coords, [obstacle.exterior.coords]
        )
        original = source(original.white, silhouette=silhouette)
    result = interpret(original)
    assert result.contours is original and not result.contacts


@pytest.mark.parametrize("disappears", [False, True])
def test_contact_cannot_split_or_remove_an_original_parent_hole(
    disappears: bool,
) -> None:
    hole = box(5.95, 5.95, 6.05, 6.05) if disappears else box(5.9, 5.5, 6.1, 6.5)
    reach = 0.05 if disappears else 0.1
    tip = Polygon([(6 - reach, 6), (6, 6 - reach), (6 + reach, 6), (6, 6 + reach)])
    parent = Polygon(box(4, 4, 8, 8).exterior.coords, [hole.exterior.coords])
    original = source([parent, tip])
    assert parent.intersection(tip).geom_type == "MultiPoint"
    result = interpret(original)
    assert result.contours is original and not result.contacts
    detail = cast("dict[str, object]", result.diagnostics["contact_interpretation"])
    skipped = cast("dict[str, dict[str, int]]", detail["skipped"])
    assert skipped["white"]["parent_hole_changed"] == 1


@pytest.mark.parametrize("cross_color", [False, True])
def test_added_geometry_conflicts_retain_both_source_groups(cross_color: bool) -> None:
    first = [box(2, 2, 4, 4), box(4, 4, 4.015, 4.015)]
    second = [box(4.06, 4.035, 6, 6), box(4.045, 4.02, 4.06, 4.035)]
    original = source(first, second) if cross_color else source([*first, *second])
    result = interpret(original)
    assert result.contours is original and not result.contacts
    detail = cast("dict[str, object]", result.diagnostics["contact_interpretation"])
    skipped = cast("dict[str, dict[str, int]]", detail["skipped"])
    key = "cross_color_bridge_conflict" if cross_color else "bridge_conflict"
    assert skipped["white"][key] > 0
