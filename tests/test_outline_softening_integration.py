"""Local corner edits pass the original continuous geometry/topology gates."""

from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from math import hypot
from typing import cast

import pytest
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient

from panelchemy import curves
from panelchemy.approximation import boundary_within_tolerance
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours
from panelchemy.outline_fitting import fit_outlines


def source(
    *, white: list[Polygon], orange: list[Polygon] | None = None
) -> PanelContours:
    parts = [box(0, 0, 16, 16), *(orange or []), *white]
    return PanelContours(
        parts[0],
        [orient(p, sign=1) for p in orange or []],
        [orient(p, sign=1) for p in white],
        0.1,
        (0, 0, 160, 160),
        sum(len(r.coords) - 1 for p in parts for r in (p.exterior, *p.interiors)),
    )


def test_rectangle_corners_become_tangent_continuous_with_bounded_source_error() -> (
    None
):
    original = source(white=[box(3, 3, 13, 13)])
    fitted = fit_outlines(original, 0.2, corner_softening_mm=0.25)
    assert fitted.diagnostics["softened_corners"] == 4
    spans = fitted.contours.white[0].exterior.spans
    for before, after in pairwise((*spans, spans[0])):
        p = before.control_points[-1]
        assert p == after.control_points[0]
        a = before.control_points[0 if before.is_line else 2]
        b = after.control_points[3 if after.is_line else 1]
        u, v = (p[0] - a[0], p[1] - a[1]), (b[0] - p[0], b[1] - p[1])
        assert hypot(*u) > 0 and hypot(*v) > 0
        assert abs(u[0] * v[1] - u[1] * v[0]) < 1e-10 * hypot(*u) * hypot(*v)
        assert u[0] * v[0] + u[1] * v[1] > 0
    assert boundary_within_tolerance(
        original.white[0].boundary,
        fitted.proxy.white[0].boundary,
        0.2 + fitted.proxy_deviation_mm,
    )
    assert fitted.proxy.silhouette.equals(original.silhouette)


def test_a_large_nozzle_does_not_make_a_small_retained_island_unfit_able() -> None:
    original = source(white=[box(5, 5, 5.6, 5.6)])
    profile = ConstructionProfile(nozzle_diameter_mm=0.8)
    fitted = fit_outlines(
        original,
        0.2,
        corner_scale_mm=2 * profile.assumed_line_width_mm,
        corner_prominence_mm=profile.tracing_radius_mm,
        corner_softening_mm=profile.tracing_radius_mm,
        fairing_mm=profile.tracing_radius_mm,
    )
    assert len(fitted.proxy.white) == 1
    assert fitted.proxy.white[0].is_valid
    assert fitted.proxy.white[0].area > 0
    assert boundary_within_tolerance(
        original.white[0].boundary,
        fitted.proxy.white[0].boundary,
        0.2 + fitted.proxy_deviation_mm,
    )


def test_point_contact_is_pinned_while_other_corners_can_soften() -> None:
    original = source(white=[box(2, 2, 6, 6)], orange=[box(6, 6, 11, 11)])
    fitted = fit_outlines(original, 0.2, corner_softening_mm=0.25)
    assert cast("int", fitted.diagnostics["softened_corners"]) > 0
    assert (
        fitted.proxy.white[0].intersection(fitted.proxy.orange[0]).equals(Point(6, 6))
    )


def test_shared_color_boundary_gets_the_same_softening_in_reverse() -> None:
    white = box(5, 5, 9, 9)
    orange = box(2, 2, 12, 12).difference(white)
    assert isinstance(orange, Polygon)
    fitted = fit_outlines(
        source(white=[white], orange=[orange]), 0.2, corner_softening_mm=0.25
    )
    a = fitted.contours.white[0].exterior.spans
    b = fitted.contours.orange[0].interiors[0].spans
    assert {s.control_points for s in a} == {
        tuple(reversed(s.control_points)) for s in b
    }
    assert fitted.proxy.white[0].intersection(fitted.proxy.orange[0]).area == 0


def test_tight_source_error_budget_retains_certified_sharp_corners() -> None:
    original = source(white=[box(3, 3, 13, 13)])
    baseline = fit_outlines(original, 1e-8)
    fitted = fit_outlines(original, 1e-8, corner_softening_mm=0.25)
    assert cast("int", fitted.diagnostics["softening_blocked"]) > 0
    assert fitted.diagnostics["softened_corners"] == 0
    assert fitted.contours == baseline.contours


@pytest.mark.parametrize(
    "message",
    [
        "curve proxy cannot meet its continuous error budget",
        "rounded curve proxy cannot preserve monotone embedding",
    ],
)
def test_candidate_proxy_rejection_preserves_accepted_baseline(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    original = source(white=[box(3, 3, 13, 13)])
    baseline = fit_outlines(original, 0.2)
    assemble = curves._assemble  # pyright: ignore[reportPrivateUsage]
    original_count = cast("int", baseline.diagnostics["selected_unique_spans"])

    def reject_candidate(
        pieces: Sequence[curves._Piece],  # pyright: ignore[reportPrivateUsage]
        uses: Sequence[Sequence[tuple[int, bool]]],
        ranges: Sequence[range],
        original_source: PanelContours,
        proxy_tolerance: float,
        topology_cuts: Sequence[set[Fraction]],
    ):
        if len(pieces) > original_count:
            raise ValueError(message)
        return assemble(
            pieces, uses, ranges, original_source, proxy_tolerance, topology_cuts
        )

    monkeypatch.setattr(curves, "_assemble", reject_candidate)
    fitted = fit_outlines(original, 0.2, corner_softening_mm=0.25)
    assert fitted.contours == baseline.contours
    assert fitted.diagnostics["softened_corners"] == 0
    assert fitted.diagnostics["softening_rejection"] == message


def test_unsafe_concave_rounding_is_restored_while_distant_corners_soften() -> None:
    white = Polygon([(2, 2), (8, 2), (8, 4), (4, 4), (4, 8), (2, 8)])
    orange = box(4.03, 4.03, 4.08, 4.08)
    original = source(white=[white], orange=[orange])
    fitted = fit_outlines(original, 0.2, corner_softening_mm=0.25)
    assert cast("int", fitted.diagnostics["softened_corners"]) > 0
    assert cast("int", fitted.diagnostics["softening_blocked"]) > 0
    assert fitted.proxy.white[0].disjoint(fitted.proxy.orange[0])
    assert fitted.proxy.white[0].exterior.distance(Point(4, 4)) < 1e-9


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan"), True])
def test_invalid_softening_budget_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="corner softening"):
        _ = fit_outlines(
            source(white=[box(3, 3, 13, 13)]), 0.2, corner_softening_mm=value
        )
