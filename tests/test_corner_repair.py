"""Corner repair keeps original-source, palette, and curvature proof gates."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from fractions import Fraction
from math import radians, tan
from typing import cast

import pytest
from shapely.geometry import Polygon, box

from panelchemy import boundary_tapers as bt
from panelchemy import contact_refinement as cr
from panelchemy import corner_refinement as corner
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours

_RUNS = (0.0, tan(radians(10)), 2 * tan(radians(10)))


def _example(
    *, obstacle: bool = False, hole: bool = False
) -> tuple[PanelContours, c.CurveContours, PanelContours]:
    before = c.BezierSpan(((2, -1), (1.2, -0.5), (0.3, -0.1), (0, 0)), False)
    bad = c.BezierSpan(((0, 0), (-0.02, -0.3), (-0.12, -0.24), (-0.2, -0.15)), False)
    after = c.BezierSpan(((-0.2, -0.15), (-0.5, 0.2), (-0.8, 0.8), (-1, 1)), False)
    spans = (
        c._line((-2, -2), (2, -2)),
        c._line((2, -2), (2, -1)),
        before,
        bad,
        after,
        c._line((-1, 1), (-2, 1)),
        c._line((-2, 1), (-2, -2)),
    )
    points: list[c.Point2] = []
    for span in spans:
        vertices, _ = c._flatten(span, 0.00001, set())
        points.extend(vertices[:-1])
    holes = [box(-1.5, -1.5, -1, -1)] if hole else []
    parent = Polygon(points, [p.exterior.coords for p in holes])
    white = [parent]
    curves = [
        c.CurvePolygon(
            c.CurveRing(spans), tuple(c.polygon_lines(p).exterior for p in holes)
        )
    ]
    silhouette = box(-3, -3, 3, 3)
    orange = (
        [Polygon(silhouette.exterior.coords, [parent.exterior.coords])]
        if obstacle
        else []
    )
    source = PanelContours(silhouette, orange, white, 0.1, (0, 0, 60, 60), 0)
    orange_curves = (
        (
            c.CurvePolygon(
                c.polygon_lines(silhouette).exterior,
                (
                    c.CurveRing(
                        tuple(
                            replace(
                                s,
                                control_points=cast(
                                    "c.Controls", tuple(reversed(s.control_points))
                                ),
                            )
                            for s in reversed(spans)
                        )
                    ),
                ),
            ),
        )
        if obstacle
        else ()
    )
    selected = c.CurveContours(
        c.polygon_lines(source.silhouette),
        orange_curves,
        tuple(curves),
        source.mm_per_pixel,
        source.source_bounds_px,
    )
    return source, selected, source


def _refine(
    source: PanelContours,
    contours: c.CurveContours,
    proxy: PanelContours,
    *,
    repair: bool = True,
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
        repair_corners=repair,
    )


def test_explicit_corner_repair_fixes_a_bad_cubic_without_point_contacts() -> None:
    source, contours, proxy = _example(hole=True)
    with pytest.raises(bt.CurvatureLimitError):
        bt.require_curvature(contours, _RUNS)
    selected, selected_proxy, report = _refine(source, contours, proxy)
    assert report["changed"] is True
    assert cast("dict[str, int]", report["corner_refinement"])["repaired"] == 1
    assert (
        cast("dict[str, object]", report["contact_interpretation"])["joined_tips"] == 0
    )
    bt.require_curvature(selected, _RUNS)
    assert len(selected.white) == 1
    assert len(selected.white[0].interiors) == 1
    assert selected.white[0].interiors[0] == contours.white[0].interiors[0]
    assert selected_proxy.white[0].is_valid
    assert selected_proxy.silhouette is source.silhouette


def test_corner_repair_remains_explicit_for_existing_contact_callers() -> None:
    source, contours, proxy = _example()
    selected, selected_proxy, report = cr.refine(
        source,
        contours,
        proxy,
        scale_mm=0.25,
        tolerance_mm=0.2,
        material_runs=_RUNS,
        proxy_tolerance_mm=0.001,
    )
    assert selected is contours and selected_proxy is proxy
    assert report["changed"] is False
    assert cast("dict[str, int]", report["corner_refinement"])["attempted"] == 0
    with pytest.raises(bt.CurvatureLimitError):
        bt.require_curvature(selected, _RUNS)


def test_corner_repair_does_not_bypass_a_tight_original_source_limit() -> None:
    source, contours, proxy = _example()
    selected, selected_proxy, report = _refine(source, contours, proxy, tolerance=0.001)
    assert selected is contours and selected_proxy is proxy
    assert report["changed"] is False
    assert cast("dict[str, int]", report["corner_refinement"])["source_rejected"] >= 1


def test_real_palette_overlap_rejects_corner_without_removing_islands_or_holes() -> (
    None
):
    source, contours, proxy = _example(obstacle=True, hole=True)
    ring = contours.white[0].exterior
    candidate_ring = next(cr._corner_rings(ring, 3))
    candidate = replace(contours.white[0], exterior=candidate_ring)
    assert (
        cr._assemble(
            [contours.silhouette, *contours.orange, candidate],
            source,
            0.001,
            proofs=c.ProofCache(),
        )
        is None
    )
    selected, selected_proxy, report = _refine(source, contours, proxy)
    assert selected is contours and selected_proxy is proxy
    assert report["changed"] is False
    assert selected.orange[0] is contours.orange[0]
    assert selected.white[0].interiors == contours.white[0].interiors


def test_single_corner_repair_preserves_every_outside_curve_and_endpoint() -> None:
    source, contours, proxy = _example()
    selected, _, _ = _refine(source, contours, proxy)
    before = contours.white[0].exterior.spans
    after = selected.white[0].exterior.spans
    assert all(any(span is old for span in after) for old in (*before[:3], *before[4:]))
    replacement = tuple(span for span in after if span not in before)
    assert replacement[0].control_points[0] == before[3].control_points[0]
    assert replacement[-1].control_points[-1] == before[3].control_points[-1]


def _ray(a: c.Point2, b: c.Point2) -> tuple[Fraction, Fraction]:
    return Fraction(b[0]) - Fraction(a[0]), Fraction(b[1]) - Fraction(a[1])


def _same_ray(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> bool:
    return a[0] * b[1] == a[1] * b[0] and a[0] * b[0] + a[1] * b[1] > 0


def test_single_window_keeps_exact_outer_g1_rays() -> None:
    bad = c.BezierSpan(((0, 0), (0, -0.25), (-0.125, -0.25), (-0.25, -0.125)), False)
    before = c._line((0, 0.25), (0, 0))
    after = c._line((-0.25, -0.125), (-0.375, 0))
    ring = c.CurveRing((before, bad, after, c._line((-0.375, 0), (0, 0.25))))
    assert corner.smooth_join(before, bad) and corner.smooth_join(bad, after)
    candidate = next(cr._corner_rings(ring, 1))
    replacements = tuple(s for s in candidate.spans if s not in ring.spans)
    assert len(replacements) == 2
    assert all(
        any(s is old for s in candidate.spans)
        for old in (before, after, ring.spans[-1])
    )
    assert _same_ray(
        _ray(*bad.control_points[:2]), _ray(*replacements[0].control_points[:2])
    )
    assert _same_ray(
        _ray(*bad.control_points[-2:]), _ray(*replacements[-1].control_points[-2:])
    )


@pytest.mark.parametrize("rotation", range(4))
def test_neighbor_windows_preserve_existing_corners_and_wrap_safely(
    rotation: int,
) -> None:
    bad = c.BezierSpan(((0, 0), (0, -0.25), (-0.125, -0.25), (-0.25, -0.125)), False)
    before = c._line((0, 0.25), (0, 0))
    after = c._line((-0.25, -0.125), (-0.375, 0))
    closing = c._line((-0.375, 0), (0, 0.25))
    original = (before, bad, after, closing)
    spans = (*original[rotation:], *original[:rotation])
    ring = c.CurveRing(spans)
    candidates = list(cr._corner_rings(ring, spans.index(bad)))
    assert candidates
    removed_groups: set[frozenset[int]] = set()
    old_corners = {
        a.control_points[-1]
        for a, b in zip(spans, (*spans[1:], spans[0]), strict=True)
        if not corner.smooth_join(a, b)
    }
    for candidate in candidates:
        missing = frozenset(
            i
            for i, old in enumerate(original)
            if not any(old is current for current in candidate.spans)
        )
        removed_groups.add(missing)
        assert missing in {frozenset({1}), frozenset({0, 1}), frozenset({1, 2})}
        assert old_corners <= {s.control_points[0] for s in candidate.spans}
        assert all(
            a.control_points[-1] == b.control_points[0]
            for a, b in zip(
                candidate.spans, (*candidate.spans[1:], candidate.spans[0]), strict=True
            )
        )
    assert frozenset({0, 1}) in removed_groups
    assert frozenset({1, 2}) in removed_groups


def test_a_neighbor_window_cannot_erase_its_existing_interior_corner() -> None:
    bad = c.BezierSpan(((0, 0), (0, -0.25), (-0.125, -0.25), (-0.25, -0.125)), False)
    before = c._line((0.25, 0), (0, 0))
    after = c._line((-0.25, -0.125), (-0.375, 0))
    closing = c._line((-0.375, 0), (0.25, 0))
    ring = c.CurveRing((before, bad, after, closing))
    assert not corner.smooth_join(before, bad)
    candidates = list(cr._corner_rings(ring, 1))
    assert candidates
    assert all(any(s is before for s in candidate.spans) for candidate in candidates)
    assert all(
        (0, 0) in {s.control_points[0] for s in candidate.spans}
        for candidate in candidates
    )


@pytest.mark.parametrize("rotation", range(4))
def test_smooth_point_contact_cannot_be_removed_by_a_neighbor_window(
    rotation: int,
) -> None:
    bad = c.BezierSpan(((0, 0), (0, -0.25), (-0.125, -0.25), (-0.25, -0.125)), False)
    before = c._line((0, 0.25), (0, 0))
    after = c._line((-0.25, -0.125), (-0.375, 0))
    closing = c._line((-0.375, 0), (0, 0.25))
    original = (before, bad, after, closing)
    spans = (*original[rotation:], *original[:rotation])
    ring = c.CurveRing(spans)
    protected = frozenset({bad.control_points[0], bad.control_points[-1]})
    assert corner.smooth_join(before, bad) and corner.smooth_join(bad, after)
    candidates = list(
        cr._corner_rings(ring, spans.index(bad), protected_points=protected)
    )
    assert candidates
    for candidate in candidates:
        assert protected <= {span.control_points[0] for span in candidate.spans}
        assert all(
            any(span is old for span in candidate.spans) for old in (before, after)
        )


def test_corner_pass_protects_vertices_owned_by_other_rings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, contours, proxy = _example(obstacle=True)
    observed: list[frozenset[c.Point2]] = []

    def inspect(
        _ring: c.CurveRing,
        _index: int,
        _source_paths: object = (),
        _tolerance_mm: float = 0.0,
        *,
        protected_points: frozenset[c.Point2],
    ) -> tuple[c.CurveRing, ...]:
        observed.append(protected_points)
        return ()

    monkeypatch.setattr(cr, "_corner_rings", inspect)
    _ = _refine(source, contours, proxy)
    bad = contours.white[0].exterior.spans[3]
    endpoints = {bad.control_points[0], bad.control_points[-1]}
    assert observed and all(endpoints <= protected for protected in observed)


def test_fitting_automatically_enables_proved_corner_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PanelContours(
        box(0, 0, 20, 20), [], [box(2, 2, 18, 18)], 0.1, (0, 0, 200, 200), 8
    )
    calls: list[bool] = []

    def refine(
        original: PanelContours,
        contours: c.CurveContours,
        proxy: PanelContours,
        *,
        repair_corners: bool,
        **_kwargs: object,
    ) -> tuple[c.CurveContours, PanelContours, dict[str, object]]:
        assert original == source
        bt.require_curvature(contours, _RUNS)
        calls.append(repair_corners)
        return contours, proxy, {"changed": False}

    monkeypatch.setattr(cr, "refine", refine)
    fitted = fitting.fit_outlines(
        source,
        0.02,
        material_taper_runs_mm=_RUNS,
        point_contact_scale_mm=0.5,
    )
    assert calls == [True]
    assert fitted.proxy.white[0].is_valid
    bt.require_curvature(fitted.contours, _RUNS)


@pytest.mark.parametrize(
    "smooth_start,smooth_end",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_raw_source_corner_fallback_requires_two_existing_outer_corners(
    monkeypatch: pytest.MonkeyPatch,
    smooth_start: bool,
    smooth_end: bool,
) -> None:
    bad = c.BezierSpan(((0, 0), (0, -0.25), (-0.125, -0.25), (-0.25, -0.125)), False)
    before = c._line((0, 0.25) if smooth_start else (0.25, 0), (0, 0))
    end = (-0.375, 0) if smooth_end else (-0.5, -0.125)
    after = c._line((-0.25, -0.125), end)
    closing = c._line(end, before.control_points[0])
    ring = c.CurveRing((before, bad, after, closing))
    corners = {
        a.control_points[-1]
        for a, b in zip(ring.spans, (*ring.spans[1:], ring.spans[0]), strict=True)
        if not corner.smooth_join(a, b)
    }
    source_paths = (((0.0, 0.0), (0.0, -0.25), (-0.25, -0.125)),)
    guides: list[c.BezierSpan] = []

    def source_corners(
        guide: c.BezierSpan,
        paths: object,
        tolerance_mm: float,
    ) -> tuple[()]:
        assert paths == source_paths
        assert tolerance_mm == 0.9 * 0.2
        guides.append(guide)
        return ()

    monkeypatch.setattr(corner, "source_corners", source_corners)
    _ = list(cr._corner_rings(ring, 1, source_paths, 0.2))
    assert all(
        guide.control_points[0] in corners and guide.control_points[-1] in corners
        for guide in guides
    )
    if smooth_start and smooth_end:
        assert not guides
    else:
        assert guides


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_corner_repair_flag_requires_a_boolean(value: object) -> None:
    source, contours, proxy = _example()
    with pytest.raises(ValueError, match="repair_corners must be a boolean"):
        _ = cr.refine(
            source,
            contours,
            proxy,
            scale_mm=0.25,
            tolerance_mm=0.2,
            material_runs=_RUNS,
            proxy_tolerance_mm=0.001,
            repair_corners=cast("bool", value),
        )
