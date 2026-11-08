"""Corner-capable repairs retain curves and respect the full palette contract."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from math import radians, tan
from typing import Never, cast

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopoDS import TopoDS
from scipy.interpolate import BSpline
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient

from panelchemy import (
    boundary_tapers,
    curvature,
    curvature_fitting,
    curved_occt,
    local_windows,
    native_curves,
    occt,
)
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours

_TOLERANCE = 0.2
_SLOPE = tan(radians(10))
_RUNS = (0.0, _SLOPE, 2 * _SLOPE)


@dataclass
class Example:
    source: PanelContours
    fits: list[fitting._Fit]
    uses: list[list[tuple[int, bool]]]
    ranges: list[range]
    curves: c.CurveContours
    proxy: PanelContours
    index: int
    arc: c.BezierSpan

    @property
    def limit(self) -> boundary_tapers.BoundaryTaper:
        return boundary_tapers.boundaries(self.uses, self.ranges, 0, _RUNS)[self.index]


def example(shift: float = 0.0) -> Example:
    """A tiny U below a separate bridge, with a long curved outside boundary."""

    def point(x: float, y: float) -> c.Point2:
        return x + shift, y + shift

    k = 0.5522847498307936
    arc = c.BezierSpan(
        (point(4, 1), point(4 + k, 1), point(5, 2 - k), point(5, 2)), False
    )
    arc_points = tuple(
        cast(
            "c.Point2",
            tuple(
                (1 - t) ** 3 * arc.control_points[0][a]
                + 3 * (1 - t) ** 2 * t * arc.control_points[1][a]
                + 3 * (1 - t) * t**2 * arc.control_points[2][a]
                + t**3 * arc.control_points[3][a]
                for a in (0, 1)
            ),
        )
        for t in (i / 16 for i in range(17))
    )
    p, q = point(3, 5), point(3.125, 5)
    u = (p, point(3, 4.875), point(3.125, 4.875), q)
    panel = orient(
        Polygon([point(1, 1), *arc_points, point(5, 5), q, u[2], u[1], p, point(1, 5)])
    )
    bridge = orient(Polygon([p, q, point(3.125, 5.125), point(3, 5.125)]))
    silhouette = box(shift, shift, 8 + shift, 8 + shift)
    polygons = [silhouette, panel, bridge]
    rings = [LineString(polygon.exterior.coords) for polygon in polygons]
    source = PanelContours(
        silhouette,
        [],
        [panel, bridge],
        0.125,
        (0, 0, 64, 64),
        sum(len(r.coords) - 1 for r in rings),
    )
    chains, uses = c._graph(rings, split_loops=False)
    fits: list[fitting._Fit] = []
    index = -1
    for chain_index, chain in enumerate(chains):
        times = c._parameters(chain.points)
        parameters = np.asarray(times, dtype=np.float64)
        spans: list[c.BezierSpan] = []
        intervals: list[tuple[float, float]] = []
        if set(chain.points) == set(u) and len(chain.points) == 4:
            index = chain_index
            spans.append(c.BezierSpan(chain.points, False, _TOLERANCE))
            intervals.append((0, 1))
        else:
            at = 0
            while at < len(chain.points) - 1:
                selected = chain.points[at : at + len(arc_points)]
                if selected == arc_points or selected == tuple(reversed(arc_points)):
                    stop = at + len(arc_points) - 1
                    spans.append(
                        arc
                        if selected == arc_points
                        else replace(
                            arc,
                            control_points=cast(
                                "c.Controls", tuple(reversed(arc.control_points))
                            ),
                        )
                    )
                else:
                    stop = at + 1
                    spans.append(c._line(chain.points[at], chain.points[stop]))
                intervals.append((times[at], times[stop]))
                at = stop
        fits.append(
            fitting._Fit(
                chain,
                parameters,
                [],
                [],
                [c._Piece(chain_index, i, i + 1, span) for i, span in enumerate(spans)],
                intervals,
                fairing_mm=0,
            )
        )
    assert index >= 0 and not fits[index].chain.free_loop
    assert any(piece.span is arc for fit in fits for piece in fit.pieces)
    pieces = [piece for fit in fits for piece in fit.pieces]
    unsafe, cuts = c._unsafe_pair(pieces)
    assert not unsafe
    ranges = [range(1), range(1, 2), range(2, 3)]
    curves, proxy, _, proxy_unsafe = c._assemble(
        pieces, uses, ranges, source, _TOLERANCE / 64, cuts
    )
    assert not proxy_unsafe
    assert fitting._source_failures(fits[index], _TOLERANCE) == set()
    return Example(source, fits, uses, ranges, curves, proxy, index, arc)


def unexpected_smooth(*_args: object, **_kwargs: object) -> Never:
    pytest.fail(
        "a certified explicit-corner alternative should not need smooth optimization"
    )


def repaired(case: Example, monkeypatch: pytest.MonkeyPatch) -> c.CurveContours:
    monkeypatch.setattr(fitting, "_curvature_candidate", unexpected_smooth)
    curves, proxy, report = fitting._curvature_verified(
        case.fits,
        case.uses,
        case.ranges,
        case.source,
        _TOLERANCE,
        _RUNS,
        case.curves,
        case.proxy,
    )
    margin = cast("dict[str, object]", report["boundary_curvature"])
    assert margin["margin_status"] == "certified"
    assert all(polygon.is_valid for polygon in proxy.white)
    boundary_tapers.require_curvature(curves, _RUNS)
    return curves


@pytest.mark.parametrize("shift", [0.0, 1 / 7])
def test_missed_u_rejects_overlapping_chord_then_keeps_curves_and_graph_pins(
    monkeypatch: pytest.MonkeyPatch, shift: float
) -> None:
    case = example(shift)
    original = case.fits[case.index]
    assert original.corners == []
    assert not curvature.within_curvature(
        original.pieces[0].span, case.limit.offset_mm, case.limit.inward_sign
    )
    pool = fitting._corner_curvature_candidates(
        original, case.index, case.limit, _TOLERANCE, [0]
    )
    chord = next(
        fit for fit, _ in pool if len(fit.pieces) == 1 and fit.pieces[0].span.is_line
    )
    assert not fitting._source_failures(chord, _TOLERANCE)
    proposals = list(case.fits)
    proposals[case.index] = chord
    rejected = fitting._verify_refits(
        case.fits,
        proposals,
        {case.index},
        case.uses,
        case.ranges,
        case.source,
        _TOLERANCE,
        case.curves,
        case.proxy,
    )
    assert rejected.accepted == set() and rejected.blocked == {case.index}
    assert rejected.curves is case.curves
    snapshot = [polygon.wkb for polygon in case.source.white]
    selected = repaired(case, monkeypatch)
    selected_spans = [
        span for polygon in selected.white for span in polygon.exterior.spans
    ]
    assert any(span is case.arc for span in selected_spans)

    def controls(span: c.BezierSpan) -> tuple[c.Point2, ...]:
        return min(span.control_points, tuple(reversed(span.control_points)))

    untouched = {
        controls(piece.span)
        for i, fit in enumerate(case.fits)
        if i != case.index
        for piece in fit.pieces
    }
    all_selected = [
        span
        for polygon in (selected.silhouette, *selected.white)
        for span in polygon.exterior.spans
    ]
    assert untouched <= {controls(span) for span in all_selected}
    old_pins = original.chain.points[0], original.chain.points[-1]
    assert native_curves.protected_vertices(
        [selected.silhouette, *selected.white]
    ) == native_curves.protected_vertices([case.curves.silhouette, *case.curves.white])
    for a, b in pairwise(original.chain.points):
        assert any(
            span.is_line and {span.control_points[0], span.control_points[-1]} == {a, b}
            for span in selected_spans
        )
    assert all(pin in {s.control_points[0] for s in selected_spans} for pin in old_pins)
    assert snapshot == [polygon.wkb for polygon in case.source.white]


def test_topology_rejected_corner_pool_cannot_escape_as_a_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = example()
    original = case.fits[case.index]
    pool = fitting._corner_curvature_candidates(
        original, case.index, case.limit, _TOLERANCE, [0]
    )
    chord = next(
        fit for fit, _ in pool if len(fit.pieces) == 1 and fit.pieces[0].span.is_line
    )

    def only_chord(*_args: object) -> list[tuple[fitting._Fit, str]]:
        return [(chord, "chord")]

    calls = 0

    def unavailable_smooth(
        *_args: object, **_kwargs: object
    ) -> tuple[None, dict[str, object]]:
        nonlocal calls
        calls += 1
        return None, {"status": "unrefined"}

    monkeypatch.setattr(fitting, "_corner_curvature_candidates", only_chord)
    monkeypatch.setattr(fitting, "_curvature_candidate", unavailable_smooth)

    def no_independent_graph_repair(
        *_args: object,
    ) -> tuple[None, dict[str, object]]:
        return None, {"status": "unrefined"}

    # Isolate the rejected chord; a different valid graph-terminal repair may
    # otherwise succeed independently of this intentionally rejected proposal.
    monkeypatch.setattr(
        fitting, "_corner_run_curvature_candidate", no_independent_graph_repair
    )
    curves, proxy, report = fitting._curvature_verified(
        case.fits,
        case.uses,
        case.ranges,
        case.source,
        _TOLERANCE,
        _RUNS,
        case.curves,
        case.proxy,
    )
    assert curves is case.curves and proxy is case.proxy
    margin = cast("dict[str, object]", report["boundary_curvature"])
    assert margin["margin_status"] == "unresolved"
    assert margin["unrefined_chains"] == [case.index]
    assert margin["topology_rejected_chains"] == [case.index]
    assert calls == 1


def test_existing_corner_remains_a_real_g0_join() -> None:
    case = example()
    original = case.fits[case.index]
    p, a, b, q = original.chain.points
    corner = float(fitting._scalar(original.parameters, 1))
    fit = replace(
        original,
        corners=[corner],
        source_corners=(corner,),
        knots=[corner],
        pieces=[
            c._Piece(case.index, 0, 1, c._line(p, a)),
            c._Piece(case.index, 1, 2, c.BezierSpan((a, b, b, q), False)),
        ],
        intervals=[(0, corner), (corner, 1)],
    )
    assert not curvature.within_curvature(
        fit.pieces[1].span, case.limit.offset_mm, case.limit.inward_sign
    )
    pool = fitting._corner_curvature_candidates(
        fit, case.index, case.limit, _TOLERANCE, [1]
    )
    retained = next(candidate for candidate, _ in pool if len(candidate.pieces) == 2)
    before, after = (piece.span for piece in retained.pieces)
    assert before.is_line and after.is_line
    assert before.control_points[-1] == after.control_points[0] == a
    u = tuple(Fraction(a[i]) - Fraction(p[i]) for i in (0, 1))
    v = tuple(Fraction(q[i]) - Fraction(a[i]) for i in (0, 1))
    assert u[0] * v[1] != u[1] * v[0]
    assert corner in retained.corners
    assert fit.corners == [corner]


def test_local_corner_repair_keeps_outside_curves_and_controls() -> None:
    k = 0.5522847498307936
    arc = c.BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    arc_points = tuple(
        cast(
            "c.Point2",
            tuple(
                (1 - t) ** 3 * arc.control_points[0][a]
                + 3 * (1 - t) ** 2 * t * arc.control_points[1][a]
                + 3 * (1 - t) * t**2 * arc.control_points[2][a]
                + t**3 * arc.control_points[3][a]
                for a in (0, 1)
            ),
        )
        for t in (i / 16 for i in range(17))
    )
    p, a, b, q = (0.0, 1.0), (-0.125, 1.0), (-0.125, 1.125), (0.0, 1.125)
    end = (1.0, 1.125)
    points = (*arc_points, a, b, q, end)
    times = c._parameters(points)
    fit = fitting._Fit(
        c._Chain(points, False),
        np.asarray(times, dtype=np.float64),
        [],
        [times[16], times[19]],
        [
            c._Piece(0, 0, 1, arc),
            c._Piece(0, 1, 2, c.BezierSpan((p, a, b, q), False)),
            c._Piece(0, 2, 3, c._line(q, end)),
        ],
        [(0, times[16]), (times[16], times[19]), (times[19], 1)],
        fairing_mm=0,
        source_corners=(times[17], times[18]),
    )
    limit = boundary_tapers.BoundaryTaper(2, 0, 2 * _SLOPE, -1)
    assert not curvature.within_curvature(
        fit.pieces[1].span, limit.offset_mm, limit.inward_sign
    )
    guide = local_windows.prepare(
        local_windows.Window(1, 1),
        points,
        fit.parameters,
        [piece.span for piece in fit.pieces],
        fit.intervals,
    )
    candidate, _ = fitting._local_window_refit(fit, 0, guide, limit, _TOLERANCE, 2.0)
    assert candidate is not None
    assert candidate.pieces[0].span is arc
    assert candidate.pieces[-1].span is fit.pieces[-1].span
    assert candidate.intervals[0] is fit.intervals[0]
    assert candidate.intervals[-1] is fit.intervals[-1]
    assert not fitting._source_failures(candidate, _TOLERANCE)
    assert all(
        curvature.within_curvature(piece.span, limit.offset_mm, limit.inward_sign)
        for piece in candidate.pieces
    )
    assert len(candidate.corners) == 2


@pytest.mark.parametrize("explicit_corner", [False, True])
def test_free_loop_seam_is_g1_unless_explicitly_a_corner(
    monkeypatch: pytest.MonkeyPatch, explicit_corner: bool
) -> None:
    points: tuple[c.Point2, ...] = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    corners = [0.25, 0.5, 0.75]
    if explicit_corner:
        corners.insert(0, 0.0)
    fit = fitting._Fit(
        c._Chain(points, False, free_loop=True),
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        corners,
        [0.25, 0.5, 0.75],
        [],
        [],
        fairing_mm=0,
        source_corners=tuple(corners),
    )
    fit.pieces, fit.intervals = fitting._pieces(fit, 0, 1.0)

    def no_new_corners(*_args: object) -> list[int]:
        return []

    monkeypatch.setattr(fitting, "_corners", no_new_corners)
    pool = fitting._corner_curvature_candidates(
        fit, 0, boundary_tapers.BoundaryTaper(2, 0, 0.1, -1), 1.0, [0]
    )
    assert pool
    smooth: list[bool] = []
    for candidate, _ in pool:
        before = c._controls(candidate.pieces[-1].span)
        after = c._controls(candidate.pieces[0].span)
        u = (before[3][0] - before[2][0], before[3][1] - before[2][1])
        v = (after[1][0] - after[0][0], after[1][1] - after[0][1])
        smooth.append(u[0] * v[1] == u[1] * v[0] and u[0] * v[0] + u[1] * v[1] > 0)
        assert (0.0 in candidate.corners) == explicit_corner
    assert any(not value for value in smooth) if explicit_corner else all(smooth)


@pytest.mark.parametrize("gate", ["source", "curvature", "separation"])
def test_corner_candidates_cannot_bypass_existing_local_proofs(
    monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    case = example()
    if gate == "source":

        def failed_source(*_args: object) -> set[int]:
            return {0}

        monkeypatch.setattr(fitting, "_source_failures", failed_source)
    elif gate == "curvature":

        def failed_curvature(*_args: object) -> bool:
            return False

        monkeypatch.setattr(curvature, "within_curvature", failed_curvature)
    else:

        def failed_separation(
            *_args: object,
        ) -> tuple[list[tuple[int, int]], list[set[Fraction]]]:
            return [(0, 1)], []

        monkeypatch.setattr(c, "_unsafe_pair", failed_separation)
    assert (
        fitting._corner_curvature_candidates(
            case.fits[case.index], case.index, case.limit, _TOLERANCE, [0]
        )
        == []
    )


def test_added_source_corners_are_limited_to_bad_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points: tuple[c.Point2, ...] = ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2))
    parameters = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    active = parameters[4]
    fit = fitting._Fit(
        c._Chain(points, False),
        np.asarray(parameters, dtype=np.float64),
        [active],
        [],
        [
            c._Piece(0, 0, 1, c._line(points[0], points[1])),
            c._Piece(0, 1, 2, c._line(points[1], points[-1])),
        ],
        [(0, 0.3), (0.3, 1)],
        source_corners=parameters[1:-1],
    )

    def all_source_turns(*_args: object) -> list[int]:
        return [1, 2, 3, 4]

    monkeypatch.setattr(fitting, "_corners", all_source_turns)
    pool = fitting._corner_curvature_candidates(
        fit, 0, boundary_tapers.BoundaryTaper(0, 2, 0.1, 1), 100, [0]
    )
    assert pool
    allowed = {parameters[1], active}
    assert all(
        set(candidate.corners) <= allowed and active in candidate.corners
        for candidate, _ in pool
    )
    assert any(parameters[1] in candidate.corners for candidate, _ in pool)
    assert fit.corners == [active]


@pytest.mark.parametrize("expected", [True, False])
def test_inadmissible_corner_basis_keeps_other_safe_alternatives(
    monkeypatch: pytest.MonkeyPatch, expected: bool
) -> None:
    case = example()
    message = (
        "spline interval has no source forward direction"
        if expected
        else "unexpected corner fitting invariant"
    )

    def rejected_basis(*_args: object) -> Never:
        raise ValueError(message)

    monkeypatch.setattr(fitting, "_pieces", rejected_basis)
    if not expected:
        with pytest.raises(ValueError, match=message):
            _ = fitting._corner_curvature_candidates(
                case.fits[case.index], case.index, case.limit, _TOLERANCE, [0]
            )
        return
    pool = fitting._corner_curvature_candidates(
        case.fits[case.index], case.index, case.limit, _TOLERANCE, [0]
    )
    assert any(len(candidate.pieces) == 3 for candidate, _ in pool)


def test_corner_repair_keeps_a_mixed_native_curve_and_line_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = repaired(example(), monkeypatch)
    region = selected.white[0]
    assert any(span.is_line for span in region.exterior.spans)
    assert any(not span.is_line for span in region.exterior.spans)
    protected = native_curves.protected_vertices([selected.silhouette, *selected.white])
    shape, _ = curved_occt.roof(region, 2, 4, _SLOPE, protected_points=protected)
    assert occt.valid(shape) and occt.volume(shape) > 0
    assert any(
        BRepAdaptor_Curve(TopoDS.Edge(edge)).GetType()
        in (GeomAbs_BezierCurve, GeomAbs_BSplineCurve)
        for edge in curved_occt._entities(shape, TopAbs_EDGE)
    )


def test_active_corner_does_not_suppress_a_valid_uniform_local_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points: tuple[c.Point2, ...] = ((-1, 0), (0, 0), (1, 0), (1, 1), (1, 2))
    times = (0.0, 0.25, 0.5, 0.75, 1.0)
    fit = fitting._Fit(
        c._Chain(points, False),
        np.asarray(times, dtype=np.float64),
        [0.5],
        [0.5],
        [
            c._Piece(0, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        list(pairwise(times)),
        source_corners=(0.5,),
    )
    guide = local_windows.prepare(
        local_windows.Window(1, 2),
        points,
        fit.parameters,
        [piece.span for piece in fit.pieces],
        fit.intervals,
    )
    k = 0.5522847498307936 / 4
    rounding = c.BezierSpan(((0.75, 0), (0.75 + k, 0), (1, 0.25 - k), (1, 0.25)), False)
    first = c._line((0, 0), (0.75, 0))
    last = c._line((1, 0.25), (1, 1))
    poles = [
        *first.control_points,
        *rounding.control_points[1:],
        *last.control_points[1:],
    ]
    proposal: BSpline[np.float64] = BSpline(
        [0.0] * 4 + [0.375] * 3 + [0.625] * 3 + [1.0] * 4,
        np.asarray(poles, dtype=np.float64),
        3,
    )
    calls: list[str] = []

    def search(
        *_args: object, **kwargs: object
    ) -> tuple[BSpline[np.float64] | None, dict[str, object]]:
        if kwargs.get("corner_parameters") is not None:
            calls.append("corner")
            return None, {"status": "unrefined"}
        calls.append("uniform")
        assert kwargs["knot_breaks"] is None
        accept = cast("Callable[[BSpline[np.float64]], bool]", kwargs["accept"])
        assert accept(proposal)  # Real endpoint, source, radius and separation gates.
        return proposal, {"status": "accepted"}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    limit = boundary_tapers.BoundaryTaper(0, 2, 0.1, 1)
    candidate, _ = fitting._local_window_refit(fit, 0, guide, limit, _TOLERANCE, 1.0)
    assert calls == ["uniform"]
    assert candidate is not None
    assert candidate.pieces[0].span is fit.pieces[0].span
    assert candidate.pieces[-1].span is fit.pieces[-1].span
    assert candidate.intervals[0] is fit.intervals[0]
    assert candidate.intervals[-1] is fit.intervals[-1]
    assert candidate.pieces[1].span.control_points[0] == (0, 0)
    assert candidate.pieces[-2].span.control_points[-1] == (1, 1)
    assert candidate.pieces[1].span.control_points[1][1] == 0
    assert candidate.pieces[-2].span.control_points[2][0] == 1
    assert 0.5 not in candidate.corners
    assert fit.corners == [0.5]
    assert not fitting._source_failures(candidate, _TOLERANCE)
    assert all(
        curvature.within_curvature(piece.span, limit.offset_mm, limit.inward_sign)
        for piece in candidate.pieces
    )
