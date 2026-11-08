"""Attached pixel fragments become one trace without spending extra accuracy."""

# Synthetic curves also exercise the internal retained-tip constraint.
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from math import hypot, radians, tan
from typing import cast

import pytest
from shapely.geometry import Polygon, box

from panelchemy import boundary_tapers, geometric_distance
from panelchemy.contours import PanelContours
from panelchemy.curves import BezierSpan, Point2
from panelchemy.outline_fitting import fit_outlines


def artwork() -> PanelContours:
    parent = Polygon(
        [(2, 2), (6, 2), (6, 4), (4, 4), (4, 3.65), (3.6, 3.65), (3.6, 4), (2, 4)]
    )
    tip = box(3.8, 4, 4, 4.2)
    other_tip = box(1.8, 4, 2, 4.2)
    return PanelContours(
        box(0, 0, 8, 8), [], [parent, tip, other_tip], 0.1, (0, 0, 80, 80), 20
    )


def test_attached_tip_and_notch_fit_as_one_valid_feature() -> None:
    source = artwork()
    snapshots = [p.wkb for p in source.white]
    slope = tan(radians(10))
    runs = (0.0, slope, 2 * slope)
    fitted = fit_outlines(
        source,
        0.2,
        point_contact_scale_mm=0.5,
        corner_scale_mm=1.0,
        corner_prominence_mm=0.25,
        corner_softening_mm=0.25,
        fairing_mm=0.25,
        span_simplification_mm=0.1,
        material_taper_runs_mm=runs,
    )
    assert len(fitted.contours.white) == 2
    parent_index = max(range(2), key=lambda i: fitted.proxy.white[i].area)
    assert (
        fitted.proxy.white[parent_index].is_valid
        and not fitted.proxy.white[parent_index].interiors
    )
    region = fitted.contours.white[parent_index]
    for corner in ((3.8, 4.2), (4.0, 3.65)):
        index, nearest = min(
            enumerate(region.exterior.spans),
            key=lambda pair: hypot(
                pair[1].control_points[0][0] - corner[0],
                pair[1].control_points[0][1] - corner[1],
            ),
        )
        point = nearest.control_points[0]
        assert hypot(point[0] - corner[0], point[1] - corner[1]) < 0.2
        before = region.exterior.spans[index - 1].control_points
        after = nearest.control_points
        u = (before[3][0] - before[2][0], before[3][1] - before[2][1])
        v = (after[1][0] - after[0][0], after[1][1] - after[0][1])
        assert u[0] * v[1] != u[1] * v[0]  # Real corners, not rounded bridges.
    spans = [
        span for ring in (region.exterior, *region.interiors) for span in ring.spans
    ]
    paths = [
        [cast("Point2", tuple(point)) for point in polygon.exterior.coords]
        for polygon in source.white[:2]
    ]
    assert geometric_distance.within_paths(paths, spans, 0.2)
    boundary_tapers.require_curvature(fitted.contours, runs)
    diagnostics = cast(
        "dict[str, object]", fitted.diagnostics["contact_interpretation"]
    )
    assert diagnostics["raw_source_distance_certified"] is True
    assert [p.wkb for p in source.white] == snapshots
    assert "Original blue" in fitted.comparison_svg


def test_zero_tolerance_keeps_exact_original_components() -> None:
    fitted = fit_outlines(artwork(), 0, point_contact_scale_mm=0.5)
    assert len(fitted.contours.white) == 3


def test_raw_source_proof_cannot_be_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = geometric_distance.within_paths
    calls = 0

    def reject_merged(
        paths: Sequence[Sequence[Point2]],
        spans: Sequence[BezierSpan],
        tolerance_mm: float,
    ) -> bool:
        nonlocal calls
        if len(paths) > 1:
            calls += 1
            return False
        return original(paths, spans, tolerance_mm)

    monkeypatch.setattr(geometric_distance, "within_paths", reject_merged)
    slope = tan(radians(10))
    fitted = fit_outlines(
        artwork(),
        0.2,
        point_contact_scale_mm=0.5,
        material_taper_runs_mm=(0.0, slope, 2 * slope),
    )
    assert calls > 0
    assert len(fitted.contours.white) == 3
