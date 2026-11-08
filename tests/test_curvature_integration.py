"""Boundary-aware refitting keeps counters and passes unchanged native checks."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from math import radians, tan
from typing import cast

import numpy as np
import pytest
from scipy.interpolate import BSpline
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

from panelchemy import (
    boundary_tapers,
    curvature,
    curvature_fitting,
    curved_occt,
    local_windows,
    occt,
)
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours
from panelchemy.fitting_feedback import Refinement
from panelchemy.outline_fitting import fit_outlines


@pytest.fixture
def smooth_curvature_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unsafe smooth inputs for tests of the later repair/rollback stage.

    Normal fitting now prevents these softening violations earlier. Those
    regressions must still exercise the optimizer, rather than silently pass
    because the new corner fallback made their inputs safe.
    """
    soften: Callable[
        ...,
        tuple[c.CurveContours, PanelContours, dict[str, object], list[fitting._Fit]],
    ] = fitting._soften_verified

    def unconstrained(
        *args: object, **kwargs: object
    ) -> tuple[c.CurveContours, PanelContours, dict[str, object], list[fitting._Fit]]:
        kwargs["material_taper_runs_mm"] = None
        return soften(*args, **kwargs)

    monkeypatch.setattr(fitting, "_soften_verified", unconstrained)

    def no_corner_candidates(*_args: object) -> list[tuple[fitting._Fit, str]]:
        return []

    monkeypatch.setattr(fitting, "_corner_curvature_candidates", no_corner_candidates)

    def no_corner_choices(*_args: object) -> list[list[float]]:
        return [[]]

    monkeypatch.setattr(fitting, "_curvature_corner_choices", no_corner_choices)

    def no_compact_candidate(*_args: object) -> tuple[None, dict[str, object]]:
        return None, {"status": "unrefined"}

    monkeypatch.setattr(fitting, "_compact_curvature_candidate", no_compact_candidate)
    monkeypatch.setattr(fitting, "_apex_curvature_candidate", no_compact_candidate)


def source_with_notched_counter() -> PanelContours:
    counter = Polygon(
        [(4, 4), (4.53, 4), (4.53, 4.17), (4.35, 4.17), (4.35, 4.35), (4, 4.35)]
    )
    orange = orient(Polygon(box(1, 1, 9, 9).exterior, [counter.exterior]))
    silhouette = box(0, 0, 10, 10)
    return PanelContours(silhouette, [orange], [], 0.1, (0, 0, 100, 100), 14)


def test_taper_keeps_detected_corner_when_rounding_would_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notch = Polygon([(1, 1), (7, 1), (7, 7), (4, 4), (1, 7)])
    source = PanelContours(box(0, 0, 8, 8), [], [notch], 0.1, (0, 0, 80, 80), 9)
    original = notch.wkb
    rounded = fit_outlines(
        source, 0.2, corner_softening_mm=0.25, span_simplification_mm=0.1
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("restoring a safe original corner should not need optimization")

    monkeypatch.setattr(fitting, "_curvature_candidate", unexpected)
    slope = tan(radians(10))
    runs = (0.0, slope, 2 * slope)
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.25,
        span_simplification_mm=0.1,
        material_taper_runs_mm=runs,
    )
    assert (4, 4) not in {
        s.control_points[0] for s in rounded.contours.white[0].exterior.spans
    }
    spans = fitted.contours.white[0].exterior.spans
    before, after = next(
        (a, b)
        for a, b in pairwise((*spans, spans[0]))
        if a.control_points[-1] == (4, 4)
    )
    u = [before.control_points[3][i] - before.control_points[2][i] for i in (0, 1)]
    v = [after.control_points[1][i] - after.control_points[0][i] for i in (0, 1)]
    assert u[0] * v[1] != u[1] * v[0]  # Actual C0 corner, not a tiny smooth bridge.
    assert fitted.diagnostics["curvature_blocked_corners"] == 1
    assert cast("int", fitted.diagnostics["softened_corners"]) > 0
    margin = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert margin["margin_status"] == "certified"
    boundary_tapers.require_curvature(fitted.contours, runs)
    assert source.white[0].wkb == original
    shape, _ = curved_occt.roof(fitted.contours.white[0], 2, 4, slope)
    assert occt.valid(shape)
    assert occt.volume(shape) > 0


@pytest.mark.usefixtures("smooth_curvature_input")
def test_whole_pinned_refits_preserve_touching_counter_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = (4.0, 4.0)
    orange = orient(
        Polygon(
            box(1, 1, 9, 9).exterior,
            [box(4, 4, 4.6, 4.6).exterior, box(3.4, 3.4, 4, 4).exterior],
        )
    )
    source = PanelContours(box(0, 0, 10, 10), [orange], [], 0.1, (0, 0, 100, 100), 12)
    original = orange.wkb
    repaired: list[int] = []
    offset = tan(radians(10))

    def whole_only(
        fit: fitting._Fit,
        index: int,
        limit: boundary_tapers.BoundaryTaper,
        tolerance: float,
        **_kwargs: object,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        # Exercise junction handling independently of local-window scheduling.
        assert not fit.chain.free_loop
        assert fit.chain.points[0] == fit.chain.points[-1] == anchor
        candidate, diagnostics = fitting._whole_curvature_refit(
            fit, index, limit, tolerance, 2.0
        )
        assert candidate is not None
        assert not fitting._source_failures(candidate, tolerance)
        assert all(
            curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
            for p in candidate.pieces
        )
        for previous, selected, pole, endpoint in (
            (fit.pieces[0].span, candidate.pieces[0].span, 1, 0),
            (fit.pieces[-1].span, candidate.pieces[-1].span, 2, 3),
        ):
            assert selected.control_points[endpoint] == anchor
            reference = previous.control_points[
                3 - endpoint if previous.is_line else pole
            ]
            inherited = [Fraction(reference[i]) - Fraction(anchor[i]) for i in (0, 1)]
            actual = [
                Fraction(selected.control_points[pole][i]) - Fraction(anchor[i])
                for i in (0, 1)
            ]
            assert inherited[0] * actual[1] == inherited[1] * actual[0]
            assert sum(a * b for a, b in zip(inherited, actual, strict=True)) > 0
        repaired.append(index)
        return candidate, diagnostics

    monkeypatch.setattr(fitting, "_curvature_candidate", whole_only)
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.05,
        material_taper_runs_mm=(0, offset, 2 * offset),
    )
    diagnostics = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert len(repaired) == diagnostics["refined_chains"] == 2
    assert diagnostics["margin_status"] == "certified"
    assert diagnostics["topology_rejected_chains"] == []
    selected = fitted.proxy.orange[0]
    assert selected.is_valid and len(selected.interiors) == 2
    assert (
        selected.interiors[0].intersection(selected.interiors[1]).equals(Point(anchor))
    )
    assert source.orange[0].wkb == original


def test_whole_pinned_refit_with_unavailable_tangent_does_not_start_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points: tuple[c.Point2, ...] = ((0, 0), (1, 0), (1, 1), (0, 0))
    spans = [c._line(a, b) for a, b in pairwise(points)]
    first = spans[0]
    spans[0] = replace(
        first,
        is_line=False,
        control_points=(
            first.control_points[0],
            first.control_points[0],
            first.control_points[2],
            first.control_points[3],
        ),
    )
    parameters = np.linspace(0, 1, len(points))
    fit = fitting._Fit(
        c._Chain(points, fixed=False),
        parameters,
        [],
        [],
        [c._Piece(0, i, i + 1, span) for i, span in enumerate(spans)],
        list(pairwise(parameters.tolist())),
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an unavailable pinned tangent must not reach the optimizer")

    monkeypatch.setattr(curvature_fitting, "refit", unexpected)
    candidate, diagnostics = fitting._whole_curvature_refit(
        fit, 0, boundary_tapers.BoundaryTaper(0, 1, 0.1, 1), 0.2, 1
    )
    assert candidate is None
    assert diagnostics["status"] == "unrefined"
    assert diagnostics["reason"] == "local repair has an unavailable endpoint tangent"


@pytest.mark.parametrize("integer_coordinates", [False, True])
@pytest.mark.parametrize(
    "topology_error",
    [None, "curve topology exceeds 1,000,000 candidate pairs", "unexpected invariant"],
)
def test_local_candidate_preserves_outside_geometry_and_original_corner_angles(
    monkeypatch: pytest.MonkeyPatch,
    integer_coordinates: bool,
    topology_error: str | None,
) -> None:
    # The horizontal repair meets diagonal outside edges at intentional corners.
    # Its inside tangent must survive without becoming collinear with those edges.
    points: tuple[c.Point2, ...] = (
        (0.0, -2.0),
        (1.0, -2.0),
        (2.0, 0.0),
        (3.0, 0.0),
        (4.0, 0.0),
        (5.0, 0.0),
        (6.0, 2.0),
        (7.0, 2.0),
    )
    if integer_coordinates:
        points = tuple((int(x), int(y)) for x, y in points)
    spans = [c._line(a, b) for a, b in pairwise(points)]
    times = np.linspace(0, 1, len(points))
    intervals = list(pairwise(times.tolist()))
    fit = fitting._Fit(
        c._Chain(points, fixed=False),
        times,
        [],
        [],
        [c._Piece(0, i, i + 1, span) for i, span in enumerate(spans)],
        intervals,
    )
    guide = local_windows.prepare(
        local_windows.Window(2, 3), points, times, spans, intervals
    )
    spline = BSpline([0, 0, 0, 0, 1, 1, 1, 1], [(2, 0), (3, 0), (4, 0), (5, 0)], 3)

    if topology_error is not None:

        def unavailable_topology(*_args: object, **_kwargs: object) -> None:
            raise ValueError(topology_error)

        monkeypatch.setattr(c, "_unsafe_pair", unavailable_topology)

    def search(
        *_args: object, **kwargs: object
    ) -> tuple[BSpline | None, dict[str, object]]:
        accept = cast("Callable[[BSpline], bool]", kwargs["accept"])
        approved = accept(spline)
        assert approved is (topology_error is None)
        return (spline if approved else None), {
            "status": "accepted" if approved else "unrefined"
        }

    monkeypatch.setattr(curvature_fitting, "refit", search)
    if topology_error == "unexpected invariant":
        with pytest.raises(ValueError, match=topology_error):
            _ = fitting._local_window_refit(
                fit, 0, guide, boundary_tapers.BoundaryTaper(0, 1, 0.1, 1), 0.2, 1
            )
        return
    result, _ = fitting._local_window_refit(
        fit, 0, guide, boundary_tapers.BoundaryTaper(0, 1, 0.1, 1), 0.2, 1
    )
    if topology_error is not None:
        assert result is None
        return
    assert result is not None
    assert result.chain is fit.chain
    assert result.parameters is times
    assert len(result.pieces) == 5
    assert all(
        result.pieces[i].span is spans[j] for i, j in ((0, 0), (1, 1), (3, 5), (4, 6))
    )
    replacement = result.pieces[2].span.control_points
    assert replacement == ((2, 0), (3, 0), (4, 0), (5, 0))
    assert not fitting._source_failures(result, 0.2)


@pytest.mark.usefixtures("smooth_curvature_input")
def test_final_refit_preserves_counter_and_produces_valid_native_taper() -> None:
    source = source_with_notched_counter()
    original = source.orange[0].wkb
    slope = tan(radians(10))
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.25,
        span_simplification_mm=0.1,
        material_taper_runs_mm=(0, slope, 2 * slope),
    )
    diagnostics = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert cast("int", diagnostics["refined_chains"]) >= 1
    assert not diagnostics["unrefined_chains"]
    assert diagnostics["margin_status"] == "certified"
    assert diagnostics["unresolved_boundaries"] == []
    assert len(fitted.contours.orange[0].interiors) == 1
    assert all(
        curvature.within_curvature(s, slope, -1)
        for s in fitted.contours.orange[0].interiors[0].spans
    )
    spans = fitted.contours.orange[0].interiors[0].spans
    for before, after in pairwise((*spans, spans[0])):
        assert before.control_points[-1] == after.control_points[0]
        u = [
            Fraction(before.control_points[3][i])
            - Fraction(before.control_points[2][i])
            for i in (0, 1)
        ]
        v = [
            Fraction(after.control_points[1][i]) - Fraction(after.control_points[0][i])
            for i in (0, 1)
        ]
        assert u[0] * v[1] == u[1] * v[0]
        assert u[0] * v[0] + u[1] * v[1] > 0
    assert source.orange[0].wkb == original
    roof, _ = curved_occt.roof(fitted.contours.orange[0], 2, 3, slope)
    assert occt.valid(roof)
    assert occt.volume(roof) > 0


@pytest.mark.parametrize("soften,simplify", [(0.0, 0.0), (0.25, 0.1)])
def test_zero_taper_keeps_ordinary_tracing_behavior(
    soften: float, simplify: float
) -> None:
    source = source_with_notched_counter()
    ordinary = fit_outlines(
        source, 0.2, corner_softening_mm=soften, span_simplification_mm=simplify
    )
    flat = fit_outlines(
        source,
        0.2,
        corner_softening_mm=soften,
        span_simplification_mm=simplify,
        material_taper_runs_mm=(0, 0, 0),
    )
    assert flat.contours == ordinary.contours
    assert "boundary_curvature" not in flat.diagnostics


@pytest.mark.usefixtures("smooth_curvature_input")
def test_unavailable_refit_preserves_the_certified_source_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(
        *_args: object, **_kwargs: object
    ) -> tuple[None, dict[str, object]]:
        return None, {"status": "time-budget"}

    monkeypatch.setattr(curvature_fitting, "refit", unavailable)
    source = source_with_notched_counter()
    ordinary = fit_outlines(source, 0.2, corner_softening_mm=0.25)
    fitted = fit_outlines(
        source,
        0.2,
        corner_softening_mm=0.25,
        material_taper_runs_mm=(0, 0.176, 0.353),
    )
    assert fitted.contours == ordinary.contours
    diagnostics = cast("dict[str, object]", fitted.diagnostics["boundary_curvature"])
    assert diagnostics["refined_chains"] == 0
    assert diagnostics["unrefined_chains"]
    assert diagnostics["margin_status"] == "unresolved"
    remaining = cast("list[dict[str, object]]", diagnostics["unresolved_boundaries"])
    assert [r["chain"] for r in remaining] == diagnostics["unrefined_chains"]
    assert all(r["spans"] for r in remaining)
    assert len(fitted.contours.orange[0].interiors) == 1


def test_public_curvature_fitting_normalizes_source_winding() -> None:
    source = source_with_notched_counter()
    reversed_source = replace(
        source,
        orange=[orient(source.orange[0], sign=-1)],
    )
    runs = (0.0, 0.176, 0.353)
    normal = fit_outlines(source, 0.2, material_taper_runs_mm=runs)
    reversed_fit = fit_outlines(reversed_source, 0.2, material_taper_runs_mm=runs)
    assert normal.contours == reversed_fit.contours


def test_shared_material_ownership_accepts_reversed_input_ring() -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [box(2, 2, 5, 8)],
        [box(5, 2, 8, 8)],
        0.1,
        (0, 0, 100, 100),
        12,
    )
    reversed_source = replace(source, white=[orient(source.white[0], sign=-1)])
    runs = (0.0, 0.1, 0.2)
    normal = fit_outlines(source, 0.05, material_taper_runs_mm=runs)
    reversed_fit = fit_outlines(reversed_source, 0.05, material_taper_runs_mm=runs)
    assert normal.contours == reversed_fit.contours


@pytest.mark.usefixtures("smooth_curvature_input")
def test_failed_canonicalization_cannot_reuse_previous_candidate_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = 0
    candidate = fitting._curvature_candidate

    def whole_only(
        fit: fitting._Fit,
        index: int,
        limit: boundary_tapers.BoundaryTaper,
        tolerance: float,
        **_kwargs: object,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        # This regression exercises the whole-chain candidate's callback state;
        # local seam conversion intentionally inserts a new spline knot.
        return candidate(fit, index, limit, tolerance, local=False)

    monkeypatch.setattr(fitting, "_curvature_candidate", whole_only)

    def search(*args: object, **kwargs: object) -> tuple[None, dict[str, object]]:
        nonlocal checked
        accept = cast("Callable[[BSpline], bool]", kwargs["accept"])
        refine = cast("Callable[[BSpline], Refinement]", kwargs["refine"])
        side = cast("int", args[3])
        coefficients = np.array([(0, 0), (0.3, 0), (0.7, 0), (1, 0)])
        a = BSpline([0, 0, 0, 0, 1, 1, 1, 1], coefficients, 3)
        b = BSpline([0, 0, 0, 0, 1, 1, 1, 1], coefficients.copy(), 3)
        k = 0.5522847498307936
        controls: c.Controls = (
            (0.01, 0),
            (0.01, k * 0.01),
            (k * 0.01, 0.01),
            (0, 0.01),
        )
        if side < 0:
            controls = cast("c.Controls", tuple(reversed(controls)))

        def convert(
            _fit: fitting._Fit, index: int, tolerance: float, spline: BSpline
        ) -> tuple[list[c._Piece], list[tuple[float, float]]]:
            if spline is b:
                raise ValueError("smooth outline chain cannot canonicalize candidate B")
            assert spline is a
            span = c.BezierSpan(controls, False, tolerance)
            return [c._Piece(index, 0, 1, span)], [(0.0, 1.0)]

        with monkeypatch.context() as context:
            context.setattr(fitting, "_spline_pieces", convert)
            assert not accept(a)
            assert refine(a).curve_parameters
            assert refine(b) == Refinement()  # Feedback belongs to one candidate.
            assert not accept(b)
            assert refine(b) == refine(a) == Refinement()
        checked += 1
        return None, {"status": "unrefined", "attempts": 1}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    slope = tan(radians(10))
    fitted = fit_outlines(
        source_with_notched_counter(),
        0.2,
        corner_softening_mm=0.25,
        material_taper_runs_mm=(0, slope, 2 * slope),
    )
    assert checked > 0
    assert len(fitted.contours.orange[0].interiors) == 1


def test_topology_rollback_keeps_an_independent_safe_refit() -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [box(1, 2, 2, 3), box(5, 2, 6, 3), box(6.2, 2, 7.2, 3)],
        [],
        0.1,
        (0, 0, 100, 100),
        16,
    )
    rings = [LineString(p.exterior.coords) for p in [source.silhouette, *source.orange]]
    chains, uses = c._graph(rings, split_loops=False)
    ranges = [range(i, i + 1) for i in range(len(rings))]
    fits: list[fitting._Fit] = []
    for index, chain in enumerate(chains):
        pieces = [
            c._Piece(index, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(chain.points))
        ]
        times = np.linspace(0, 1, len(chain.points))
        fits.append(
            fitting._Fit(
                chain,
                times,
                [],
                [],
                pieces,
                list(pairwise(times.tolist())),
            )
        )
    original_pieces = [p for f in fits for p in f.pieces]
    unsafe_pairs, cuts = c._unsafe_pair(original_pieces)
    assert not unsafe_pairs
    original_curves, original_proxy, _, _ = c._assemble(
        original_pieces, uses, ranges, source, 0.01, cuts
    )
    candidates = list(fits)
    safe, unsafe = uses[1][0][0], uses[2][0][0]
    for index, shift in ((safe, 0.1), (unsafe, 0.3)):
        pieces = [
            replace(
                p,
                span=c._line(
                    (p.span.control_points[0][0] + shift, p.span.control_points[0][1]),
                    (p.span.control_points[3][0] + shift, p.span.control_points[3][1]),
                ),
            )
            for p in fits[index].pieces
        ]
        candidates[index] = replace(fits[index], pieces=pieces)
    verified = fitting._verify_refits(
        fits,
        candidates,
        {safe, unsafe},
        uses,
        ranges,
        source,
        0.2,
        original_curves,
        original_proxy,
    )
    assert verified.accepted == {safe}
    assert verified.blocked == {unsafe}
    assert verified.proxy.orange[0].equals(
        affinity.translate(source.orange[0], xoff=0.1)
    )
    assert verified.proxy.orange[1].equals(source.orange[1])


def _touching_square_fits() -> tuple[
    PanelContours,
    list[fitting._Fit],
    list[list[tuple[int, bool]]],
    list[range],
    int,
]:
    pin = (4.13, 4.37)
    small = box(pin[0] - 0.18, pin[1] - 0.18, *pin)
    large = box(*pin, pin[0] + 1, pin[1] + 1)
    orange = orient(Polygon(box(1, 1, 9, 9).exterior, [small.exterior, large.exterior]))
    source = PanelContours(box(0, 0, 10, 10), [orange], [], 0.1, (0, 0, 100, 100), 16)
    rings = [
        LineString(ring.coords)
        for polygon in (source.silhouette, orange)
        for ring in (polygon.exterior, *polygon.interiors)
    ]
    chains, uses = c._graph(rings, split_loops=False)
    fits: list[fitting._Fit] = []
    for index, chain in enumerate(chains):
        times = np.linspace(0, 1, len(chain.points))
        pieces = [
            c._Piece(index, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(chain.points))
        ]
        fits.append(
            fitting._Fit(chain, times, [], [], pieces, list(pairwise(times.tolist())))
        )
    index = uses[2][0][0]
    assert fits[index].chain.points[0] == fits[index].chain.points[-1] == pin
    assert not fits[index].chain.free_loop
    return source, fits, uses, [range(1), range(1, 4)], index


def test_pinned_circle_preserves_off_grid_junction_and_complete_palette() -> None:
    source, fits, uses, ranges, index = _touching_square_fits()
    original = fits[index]
    pin = original.chain.points[0]
    assert any((Fraction(value) * 2**30).denominator != 1 for value in pin)
    offset = tan(radians(10))
    limit = boundary_tapers.boundaries(uses, ranges, 1, (0, offset, 2 * offset))[index]
    candidate = fitting._pinned_circle_candidate(original, index, limit, 0.2)
    assert candidate is not None
    assert candidate.chain is original.chain
    assert candidate.parameters is original.parameters
    assert not candidate.chain.free_loop
    assert candidate.pieces[0].span.control_points[0] == pin
    assert candidate.pieces[-1].span.control_points[-1] == pin
    assert not fitting._source_failures(candidate, 0.2)
    assert all(
        curvature.within_curvature(p.span, offset, limit.inward_sign)
        for p in candidate.pieces
    )
    before, after = candidate.pieces[-1].span, candidate.pieces[0].span
    u = [Fraction(pin[i]) - Fraction(before.control_points[2][i]) for i in (0, 1)]
    v = [Fraction(after.control_points[1][i]) - Fraction(pin[i]) for i in (0, 1)]
    assert u[0] * v[1] == u[1] * v[0]
    assert sum(a * b for a, b in zip(u, v, strict=True)) > 0
    pieces = [piece for fit in fits for piece in fit.pieces]
    unsafe, cuts = c._unsafe_pair(pieces)
    assert not unsafe
    curves, proxy, _, _ = c._assemble(pieces, uses, ranges, source, 0.2 / 64, cuts)
    proposals = list(fits)
    proposals[index] = candidate
    verified = fitting._verify_refits(
        fits, proposals, {index}, uses, ranges, source, 0.2, curves, proxy
    )
    assert verified.accepted == {index}
    assert not verified.blocked
    selected = verified.proxy.orange[0]
    assert selected.is_valid and len(selected.interiors) == 2
    assert selected.interiors[0].intersection(selected.interiors[1]).equals(Point(pin))


def test_pinned_circle_respects_a_smaller_source_error_budget() -> None:
    _source, fits, uses, ranges, index = _touching_square_fits()
    offset = tan(radians(10))
    limit = boundary_tapers.boundaries(uses, ranges, 1, (0, offset, 2 * offset))[index]
    assert fitting._pinned_circle_candidate(fits[index], index, limit, 0.05) is None


def test_pinned_circle_is_returned_after_preferred_searches_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, fits, uses, ranges, index = _touching_square_fits()
    offset = tan(radians(10))
    limit = boundary_tapers.boundaries(uses, ranges, 1, (0, offset, 2 * offset))[index]
    calls: list[str] = []

    def unavailable_local(*_args: object) -> tuple[None, dict[str, object]]:
        calls.append("local")
        return None, {"status": "unrefined"}

    def unavailable_whole(*args: object) -> tuple[None, dict[str, object]]:
        assert isinstance(args[-1], float)
        assert 0 < args[-1] <= curvature_fitting._SECONDS
        calls.append("whole")
        return None, {"status": "time-budget"}

    monkeypatch.setattr(fitting, "_local_curvature_refit", unavailable_local)
    monkeypatch.setattr(fitting, "_whole_curvature_refit", unavailable_whole)
    candidate, diagnostics = fitting._curvature_candidate(
        fits[index], index, limit, 0.2
    )
    assert candidate is not None
    assert not fitting._source_failures(candidate, 0.2)
    assert diagnostics["strategy"] == "pinned-circle"
    assert calls == ["local", "whole"]
