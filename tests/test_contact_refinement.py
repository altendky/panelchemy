"""Point-contact cleanup keeps good curves and all ordinary proof gates."""

# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from dataclasses import replace
from math import cos, radians, sin, tan
from typing import cast

import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon, box

from panelchemy import boundary_tapers, geometric_distance
from panelchemy import contact_interpretation as ci
from panelchemy import contact_refinement as cr
from panelchemy import curves as c
from panelchemy.contours import PanelContours

_RUNS = (0.0, tan(radians(10)), 2 * tan(radians(10)))


@pytest.mark.parametrize(
    "points,expected",
    [
        ([(0, 0), (1, 0), (0.5, 0)], True),
        ([(0, 0), (0, 0), (1, 1)], True),
        ([(100, 100), (101, 100), (100.5, 100 + 2**-46)], True),
        ([(100, 100), (101, 100), (100.5, 100 + 1e-8)], False),
        ([(0, 0), (0.5, 0), (1, 0)], False),
        ([(0, 0), (1, 0), (1, 1)], False),
    ],
)
def test_attachment_guard_distinguishes_retracing_from_resolved_sharp_tips(
    points: list[c.Point2], expected: bool
) -> None:
    assert cr._degenerate(points) is expected
    assert cr._degenerate(list(reversed(points))) is expected


@pytest.mark.parametrize("angle", [0, 37, 120])
def test_backtracking_guard_accounts_for_rotated_translated_float_coordinates(
    angle: float,
) -> None:
    a = radians(angle)
    points = [(100 + x * cos(a), 70 + x * sin(a)) for x in (0, 0.25, -0.25)]
    assert cr._degenerate(points)


def example(
    *, good: bool = False
) -> tuple[PanelContours, c.CurveContours, PanelContours]:
    before = c.BezierSpan(((2, -1), (1.2, -0.5), (0.3, -0.1), (0, 0)), False)
    bad = c.BezierSpan(((0, 0), (-0.02, -0.3), (-0.12, -0.24), (-0.2, -0.15)), False)
    after = c.BezierSpan(((-0.2, -0.15), (-0.5, 0.2), (-0.8, 0.8), (-1, 1)), False)
    spans = (
        c._line((-2, -2), (2, -2)),
        c._line((2, -2), (2, -1)),
        before,
        c._line((0, 0), (-0.2, -0.15)) if good else bad,
        after,
        c._line((-1, 1), (-2, 1)),
        c._line((-2, 1), (-2, -2)),
    )
    points: list[c.Point2] = []
    for span in spans:
        vertices, _ = c._flatten(span, 0.00001, set())
        points.extend(vertices[:-1])
    parent, tip = Polygon(points), box(-0.18, 0, 0, 0.18)
    assert parent.is_valid and parent.intersection(tip).geom_type == "Point"
    source = PanelContours(
        box(-3, -3, 3, 3), [], [parent, tip], 0.09, (0, 0, 66, 66), 0
    )
    contours = c.CurveContours(
        c.polygon_lines(source.silhouette),
        (),
        (c.CurvePolygon(c.CurveRing(spans)), c.polygon_lines(tip)),
        source.mm_per_pixel,
        source.source_bounds_px,
    )
    return source, contours, source


def refine(
    source: PanelContours,
    contours: c.CurveContours,
    proxy: PanelContours,
    tolerance: float = 0.2,
) -> tuple[c.CurveContours, PanelContours, dict[str, object]]:
    return cr.refine(
        source,
        contours,
        proxy,
        scale_mm=0.25,
        tolerance_mm=tolerance,
        material_runs=_RUNS,
        proxy_tolerance_mm=0.001,
    )


def test_notch_search_skips_a_collapsed_candidate_and_preserves_outside_curves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, _ = example()
    parent = contours.white[0]
    contact = ci.PointContact("white", 1, 0, (0, 0), (-0.18, 0.18))
    valid_notch = (-0.25, -0.1)

    def notches(*_args: object) -> list[c.Point2]:
        return [(0.18, -0.18), valid_notch]

    monkeypatch.setattr(cr, "_notches", notches)
    candidate = next(
        cr._candidates(
            contact,
            parent,
            source.white[0],
            {cr._key(parent.exterior.spans[3])},
            0.25,
            0.2,
        )
    )
    spans = candidate.exterior.spans
    assert [s.control_points[0] for s in spans[3:6]] == [
        (0, 0),
        contact.apex,
        valid_notch,
    ]
    assert all(
        any(s is q for q in spans)
        for i, s in enumerate(parent.exterior.spans)
        if i != 3
    )
    assert not cr._degenerate(
        [
            spans[3].control_points[0],
            spans[4].control_points[0],
            spans[5].control_points[0],
            spans[5].control_points[-1],
        ]
    )


def test_tangent_grid_candidates_use_the_same_degeneracy_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, _ = example()
    parent = contours.white[0]
    contact = ci.PointContact("white", 1, 0, (0, 0), (-0.18, 0.18))
    checked: list[list[c.Point2]] = []

    def reject(points: list[c.Point2]) -> bool:
        checked.append(points)
        return True

    def notches(*_args: object) -> list[c.Point2]:
        return []

    monkeypatch.setattr(cr, "_notches", notches)
    monkeypatch.setattr(cr, "_degenerate", reject)
    candidates = list(
        cr._candidates(
            contact,
            parent,
            source.white[0],
            {cr._key(parent.exterior.spans[3])},
            0.25,
            0.2,
        )
    )
    assert not candidates and checked


def test_point_attachment_becomes_two_corners_without_refitting_good_curves() -> None:
    source, contours, proxy = example()
    with pytest.raises(boundary_tapers.CurvatureLimitError):
        boundary_tapers.require_curvature(contours, _RUNS)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert report["changed"] is True
    assert len(result.white) == 1
    boundary_tapers.require_curvature(result, _RUNS)
    assert selected_proxy.white[0].is_valid and not selected_proxy.white[0].interiors
    before = contours.white[0].exterior.spans
    after = result.white[0].exterior.spans
    assert all(span in after for i, span in enumerate(before) if i != 3)
    assert before[3] not in after
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["raw_source_distance_certified"] is True
    assert detail["white_sources"] == ((0, 1),)
    assert len(contours.white) == len(source.white) == 2
    assert selected_proxy.silhouette is source.silhouette


def test_already_good_point_contact_is_unchanged() -> None:
    source, contours, proxy = example(good=True)
    boundary_tapers.require_curvature(contours, _RUNS)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False


def test_original_source_budget_cannot_be_bypassed_for_attachment() -> None:
    source, contours, proxy = example()
    result, selected_proxy, report = refine(source, contours, proxy, 0.001)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False


def test_locally_certified_attachment_still_requires_global_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = example()

    def rejected(*_args: object) -> None:
        return None

    monkeypatch.setattr(cr, "_assemble", rejected)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["blocked"]


def test_unrelated_existing_radius_failure_is_retained() -> None:
    source, contours, _proxy = example()
    distant = c.CurvePolygon(
        c.CurveRing(
            tuple(
                replace(
                    span,
                    control_points=cast(
                        "c.Controls", tuple((x + 6, y) for x, y in span.control_points)
                    ),
                )
                for span in contours.white[0].exterior.spans
            )
        )
    )
    silhouette = box(-3, -3, 9, 3)
    source = replace(
        source,
        silhouette=silhouette,
        white=[*source.white, translate(source.white[0], xoff=6)],
    )
    contours = replace(
        contours,
        silhouette=c.polygon_lines(silhouette),
        white=(*contours.white, distant),
    )
    result, _, report = refine(source, contours, source)
    assert report["changed"] is True
    assert distant in result.white
    with pytest.raises(boundary_tapers.CurvatureLimitError) as error:
        boundary_tapers.require_curvature(result, _RUNS)
    assert len(error.value.failures) == 1


def test_contact_search_exhaustion_is_explicit_and_preserves_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = example()
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 3 else 13.0

    monkeypatch.setattr(cr, "monotonic", clock)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False
    detail = cast("dict[str, object]", report["contact_interpretation"])
    blocked = cast("list[dict[str, object]]", detail["blocked"])
    assert blocked[0]["reason"] == "time-budget"


def hole_example() -> tuple[PanelContours, c.CurveContours, PanelContours]:
    _, original, _ = example()
    before, bad, after = original.white[0].exterior.spans[2:5]
    hole = c.CurveRing(
        (
            before,
            bad,
            after,
            c._line((-1, 1), (-2, 2)),
            c._line((-2, 2), (2, 2)),
            c._line((2, 2), (2, -1)),
        )
    )
    points: list[c.Point2] = []
    for span in hole.spans:
        vertices, _ = c._flatten(span, 0.00001, set())
        points.extend(vertices[:-1])
    untouched_hole = box(2.3, -2.5, 2.7, -2.1)
    parent = Polygon(
        box(-3, -3, 3, 3).exterior.coords, [points, untouched_hole.exterior.coords]
    )
    tip = box(-0.18, 0, 0, 0.18)
    assert parent.is_valid and parent.intersection(tip).geom_type == "Point"
    source = PanelContours(
        box(-4, -4, 4, 4), [], [parent, tip], 0.09, (0, 0, 88, 88), 0
    )
    contours = c.CurveContours(
        c.polygon_lines(source.silhouette),
        (),
        (
            c.CurvePolygon(
                c.polygon_lines(parent).exterior,
                (hole, c.polygon_lines(untouched_hole).exterior),
            ),
            c.polygon_lines(tip),
        ),
        source.mm_per_pixel,
        source.source_bounds_px,
    )
    return source, contours, source


def test_hole_attachment_is_repaired_with_all_other_polynomials_unchanged() -> None:
    source, contours, proxy = hole_example()
    parent = contours.white[0]
    failing = parent.interiors[0].spans[1]
    with pytest.raises(boundary_tapers.CurvatureLimitError):
        boundary_tapers.require_curvature(contours, _RUNS)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert report["changed"] is True
    assert len(result.white) == 1
    boundary_tapers.require_curvature(result, _RUNS)
    selected = result.white[0]
    assert selected.exterior.spans == parent.exterior.spans
    assert selected.interiors[1].spans == parent.interiors[1].spans
    assert all(
        span in selected.interiors[0].spans
        for span in parent.interiors[0].spans
        if span != failing
    )
    assert failing not in selected.interiors[0].spans
    assert selected_proxy.white[0].is_valid
    assert len(selected_proxy.white[0].interiors) == 2
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["raw_source_distance_certified"] is True
    assert detail["white_sources"] == ((0, 1),)


def test_hole_attachment_cannot_bypass_original_source_error() -> None:
    source, contours, proxy = hole_example()
    result, selected_proxy, report = refine(source, contours, proxy, tolerance=0.001)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False


def test_hole_attachment_still_requires_global_palette_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = hole_example()

    def rejected(*_args: object) -> None:
        return None

    monkeypatch.setattr(cr, "_assemble", rejected)
    result, selected_proxy, report = refine(source, contours, proxy)
    assert result is contours and selected_proxy is proxy
    assert report["changed"] is False


def test_reference_hole_permutation_keeps_identity_and_geometry() -> None:
    source, _, _ = hole_example()
    reference = ci.before_graph(
        source,
        scale_mm=0.25,
        tolerance_mm=0.2,
        selected_tips=frozenset((("white", 1),)),
    )
    polygon = reference.contours.white[0]
    scrambled = Polygon(
        polygon.exterior.coords, [ring.coords for ring in reversed(polygon.interiors)]
    )
    reference = replace(
        reference, contours=replace(reference.contours, white=[scrambled])
    )
    ordered = cr._ordered_reference(source, reference)
    assert ordered is not None and ordered.white[0].equals(polygon)
    assert all(
        Polygon(old).relate_pattern(Polygon(new), "T********")
        for old, new in zip(
            source.white[0].interiors, ordered.white[0].interiors, strict=True
        )
    )
    assert sorted(r.wkb for r in ordered.white[0].interiors) == sorted(
        r.wkb for r in scrambled.interiors
    )


def test_changed_hole_source_group_composes_with_untouched_ring_certificates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = hole_example()
    original = geometric_distance.within_paths
    calls: list[tuple[Sequence[Sequence[c.Point2]], float]] = []

    def checked(
        paths: Sequence[Sequence[c.Point2]],
        spans: Sequence[c.BezierSpan],
        tolerance: float,
    ) -> bool:
        calls.append((paths, tolerance))
        return original(paths, spans, tolerance)

    monkeypatch.setattr(geometric_distance, "within_paths", checked)
    result, _, report = refine(source, contours, proxy)
    assert report["changed"] is True
    assert all(tolerance == 0.2 for _, tolerance in calls)
    grouped = [paths for paths, _ in calls if len(paths) == 2]
    assert grouped
    assert all(tuple(source.white[1].exterior.coords) in paths for paths in grouped)
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["whole_union_source_checks"] == 0
    boundary_tapers.require_curvature(result, _RUNS)


def test_conservative_pairing_failure_keeps_the_original_union_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = hole_example()
    original = geometric_distance.within_paths

    def unproved_pairs(
        paths: Sequence[Sequence[c.Point2]],
        spans: Sequence[c.BezierSpan],
        tolerance: float,
    ) -> bool:
        # Pairing is inconclusive; the distinct whole-parent union still gets
        # its real certificate, using the same source tolerance.
        return len(paths) >= 4 and original(paths, spans, tolerance)

    monkeypatch.setattr(geometric_distance, "within_paths", unproved_pairs)
    result, _, report = refine(source, contours, proxy)
    assert report["changed"] is True
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert cast("int", detail["whole_union_source_checks"]) > 0
    boundary_tapers.require_curvature(result, _RUNS)


def test_source_cache_tracks_original_group_and_exact_selected_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, _ = example()
    ring = contours.white[0].exterior
    paths = (tuple((float(x), float(y)) for x, y in source.white[0].exterior.coords),)
    calls = 0

    def certified(*_args: object) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(geometric_distance, "within_paths", certified)
    cache: dict[cr._DistanceKey, bool] = {}
    assert cr._source_certificate(paths, ring.spans, 0.2, cache)
    assert cr._source_certificate(paths, replace(ring).spans, 0.2, cache)
    assert calls == 1
    first = ring.spans[0]
    moved = replace(
        first,
        control_points=cast(
            "c.Controls", tuple((x + 0.1, y) for x, y in first.control_points)
        ),
    )
    changed = replace(ring, spans=(moved, *ring.spans[1:]))
    assert cr._source_certificate(paths, changed.spans, 0.2, cache)
    assert cr._source_certificate(
        (
            *paths,
            tuple((float(x), float(y)) for x, y in source.white[1].exterior.coords),
        ),
        changed.spans,
        0.2,
        cache,
    )
    assert cr._source_certificate(paths, changed.spans, 0.1, cache)
    assert calls == 4
    grouped = (
        *paths,
        tuple((float(x), float(y)) for x, y in source.white[1].exterior.coords),
    )
    assert cr._source_certificate(tuple(reversed(grouped)), changed.spans, 0.2, cache)
    assert cr._source_certificate(grouped, tuple(reversed(changed.spans)), 0.2, cache)
    assert calls == 6


def test_identical_holeless_pair_and_union_reuse_a_failed_exact_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = example()
    assert not contours.white[0].interiors
    calls: list[cr._DistanceKey] = []

    def rejected(
        paths: Sequence[Sequence[c.Point2]],
        spans: Sequence[c.BezierSpan],
        tolerance: float,
    ) -> bool:
        calls.append(
            (
                tuple(tuple(path) for path in paths),
                tuple((span.is_line, span.control_points) for span in spans),
                tolerance,
            )
        )
        return False

    monkeypatch.setattr(geometric_distance, "within_paths", rejected)
    result, _, report = refine(source, contours, proxy)
    assert result is contours and report["changed"] is False
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert cast("int", detail["whole_union_source_checks"]) > 0
    assert any(len(paths) == 2 for paths, _, _ in calls)
    assert len(calls) == len(set(calls))


def test_baseline_preparation_has_explicit_time_accounting_outside_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = hole_example()
    elapsed = 0.0

    def certificate(*_args: object) -> bool:
        nonlocal elapsed
        elapsed += 50.0
        return True

    monkeypatch.setattr(cr, "_source_certificate", certificate)
    monkeypatch.setattr(cr, "monotonic", lambda: elapsed)
    _, _, report = refine(source, contours, proxy)
    assert report["changed"] is True
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert cast("float", detail["source_preparation_seconds"]) >= 150.0


def double_contact_example() -> tuple[PanelContours, c.CurveContours, PanelContours]:
    parent = Polygon(
        [
            (0, 0),
            (4, 0),
            (4, 3),
            (2.125, 3),
            (2.125, 2.875),
            (2, 2.875),
            (2, 3),
            (0, 3),
        ],
        [
            box(0.5, 0.5, 0.75, 0.75).exterior.coords,
            box(3, 0.5, 3.25, 0.75).exterior.coords,
        ],
    )
    tip = box(2, 3, 2.125, 3.125)
    source = PanelContours(
        box(-1, -1, 5, 4), [], [parent, tip], 0.0625, (0, 0, 96, 80), 0
    )
    contours = c.CurveContours(
        c.polygon_lines(source.silhouette),
        (),
        tuple(c.polygon_lines(p) for p in source.white),
        source.mm_per_pixel,
        source.source_bounds_px,
    )
    return source, contours, source


@pytest.mark.parametrize("rotation", [0, 4, 5])
def test_short_contact_arc_chord_handles_wrapping_and_preserves_outside_coefficients(
    rotation: int,
) -> None:
    source, contours, _ = double_contact_example()
    parent = contours.white[0]
    spans = parent.exterior.spans
    parent = replace(
        parent, exterior=c.CurveRing((*spans[rotation:], *spans[:rotation]))
    )
    contact = ci.PointContact(
        "white", 1, 0, (2, 3), (2.125, 3.125), members=(1,), points=((2, 3), (2.125, 3))
    )
    candidates = list(cr._short_arc_candidates(contact, parent, 0.25, 0))
    assert candidates
    candidate = candidates[0]
    old = parent.exterior.spans
    changed = [s for s in old if s not in candidate.exterior.spans]
    assert len(changed) == 3
    assert all(
        any(s is q for q in candidate.exterior.spans) for s in old if s not in changed
    )
    assert candidate.interiors is parent.interiors
    assert geometric_distance.within_paths(
        cr._paths([Polygon(source.white[0].exterior), source.white[1]]),
        candidate.exterior.spans,
        0.2,
    )
    assert not list(cr._short_arc_candidates(contact, parent, 0.1, 0))


def test_all_line_double_contact_is_cleaned_without_radius_failure() -> None:
    source, contours, proxy = double_contact_example()
    assert cr._bad(contours, _RUNS) == set()
    result, selected, report = refine(source, contours, proxy)
    assert report["changed"] and len(result.white) == 1
    assert len(result.white[0].interiors) == 2
    assert result.white[0].interiors == contours.white[0].interiors
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["white_sources"] == ((0, 1),)
    assert selected.white[0].is_valid
    boundary_tapers.require_curvature(result, _RUNS)


def test_complex_contact_still_requires_original_source_and_palette_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = double_contact_example()
    result, selected, report = refine(source, contours, proxy, tolerance=0.01)
    assert result is contours and selected is proxy and report["changed"] is False

    def rejected(*_args: object) -> None:
        return None

    monkeypatch.setattr(cr, "_assemble", rejected)
    result, selected, report = refine(source, contours, proxy)
    assert result is contours and selected is proxy and report["changed"] is False


def test_grouped_attachment_certifies_every_original_member_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, _ = example(good=True)
    tips = [
        box(-0.125, 0, 0, 0.125),
        box(-0.25, 0.125, -0.125, 0.25),
        box(-0.375, 0.25, -0.25, 0.375),
    ]
    source = replace(source, white=[source.white[0], *tips])
    contours = replace(
        contours, white=(contours.white[0], *(c.polygon_lines(p) for p in tips))
    )
    calls: list[tuple[cr._Path, ...]] = []
    certificate = geometric_distance.within_paths

    def checked(
        paths: Sequence[Sequence[c.Point2]],
        spans: Sequence[c.BezierSpan],
        tolerance: float,
    ) -> bool:
        calls.append(tuple(tuple(path) for path in paths))
        return certificate(paths, spans, tolerance)

    monkeypatch.setattr(geometric_distance, "within_paths", checked)
    result, _, report = refine(source, contours, source)
    original_tip_paths = [tuple(path) for path in cr._paths(tips)]
    assert any(all(path in group for path in original_tip_paths) for group in calls)
    assert report["changed"] is True and len(result.white) == 1
    detail = cast("dict[str, object]", report["contact_interpretation"])
    assert detail["white_sources"] == ((0, 1, 2, 3),)
