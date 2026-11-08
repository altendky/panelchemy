"""Exact cubic composites and topology-aware native edge grouping."""

from math import inf, nan
from typing import cast

import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int
from OCP.Geom import Geom_BSplineCurve
from OCP.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
from OCP.gp import gp_Pnt

from panelchemy import native_curves
from panelchemy.curves import BezierSpan, Controls, CurvePolygon, CurveRing, Point2
from panelchemy.native_curves import (
    composite_bspline,
    draft_runs,
    protected_vertices,
    smooth_runs,
)


def circle(radius: float = 5) -> CurveRing:
    k = 0.5522847498307936 * radius
    return CurveRing(
        tuple(
            BezierSpan(points, False)
            for points in (
                ((radius, 0.0), (radius, k), (k, radius), (0.0, radius)),
                ((0.0, radius), (-k, radius), (-radius, k), (-radius, 0.0)),
                ((-radius, 0.0), (-radius, -k), (-k, -radius), (0.0, -radius)),
                ((0.0, -radius), (k, -radius), (radius, -k), (radius, 0.0)),
            )
        )
    )


def line(a: Point2, b: Point2, *, analytic: bool = True) -> BezierSpan:
    return BezierSpan(
        (
            a,
            ((2 * a[0] + b[0]) / 3, (2 * a[1] + b[1]) / 3),
            ((a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3),
            b,
        ),
        analytic,
    )


def reversed_ring(ring: CurveRing) -> CurveRing:
    return CurveRing(
        tuple(
            BezierSpan(cast("Controls", tuple(reversed(s.control_points))), s.is_line)
            for s in reversed(ring.spans)
        )
    )


def run_key(run: tuple[BezierSpan, ...]):
    forward = tuple((s.is_line, s.control_points) for s in run)
    backward = tuple(
        (s.is_line, tuple(reversed(s.control_points))) for s in reversed(run)
    )
    return min(forward, backward)


def test_composite_native_polynomials_have_identical_bernstein_coefficients() -> None:
    spans = (
        BezierSpan(((0.1, 0.2), (0.3, 0.7), (1.2, 1.4), (2.3, 2.7)), False),
        BezierSpan(((2.3, 2.7), (2.9, 3.4), (3.7, 4.1), (5.2, 6.3)), False),
    )
    curve = composite_bspline(spans, 3.25)
    # Independent native decomposition exposes each polynomial's Bernstein
    # coefficients. Equality certifies whole intervals, not selected samples.
    decomposed = GeomConvert_BSplineCurveToBezierCurve(curve)
    assert decomposed.NbArcs() == 2
    for i, span in enumerate(spans, 1):
        arc = decomposed.Arc(i)
        assert arc.Degree() == 3
        assert tuple(arc.Pole(j).Coord() for j in range(1, 5)) == tuple(
            (*p, 3.25) for p in span.control_points
        )
    assert curve.FirstParameter() == 0
    assert curve.LastParameter() == 2
    assert curve.Degree() == 3
    assert not curve.IsRational()
    assert not curve.IsPeriodic()


def test_native_coefficient_readback_rejects_a_changed_constructor_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor = Geom_BSplineCurve

    def changed(
        poles: Array1_gp_Pnt,
        knots: Array1_double,
        multiplicities: Array1_int,
        degree: int,
        periodic: bool,
    ) -> Geom_BSplineCurve:
        curve = constructor(poles, knots, multiplicities, degree, periodic)
        point = curve.Pole(2)
        curve.SetPole(2, gp_Pnt(point.X() + 1e-6, point.Y(), point.Z()))
        return curve

    monkeypatch.setattr(native_curves, "Geom_BSplineCurve", changed)
    with pytest.raises(ValueError, match="changed its polynomial representation"):
        _ = composite_bspline(circle().spans, 4)


def test_smooth_closed_circle_can_be_one_exact_native_edge() -> None:
    ring = circle()
    runs = smooth_runs(ring, frozenset())
    assert len(runs) == 1 and len(runs[0]) == 4
    curve = composite_bspline(runs[0], 4)
    point = runs[0][0].control_points[0]
    vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point, 4)).Vertex()
    edge = BRepBuilderAPI_MakeEdge(curve, vertex, vertex, 0, 4)
    assert edge.IsDone()
    assert BRepCheck_Analyzer(edge.Edge()).IsValid()


def test_real_kinks_split_curved_runs() -> None:
    points = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    ring = CurveRing(
        tuple(
            line(p, points[(i + 1) % 4], analytic=False) for i, p in enumerate(points)
        )
    )
    assert [len(run) for run in smooth_runs(ring, frozenset())] == [1, 1, 1, 1]


def test_draft_runs_combine_small_spans_but_retain_quarter_turn_splits() -> None:
    def halves(span: BezierSpan) -> tuple[BezierSpan, BezierSpan]:
        a, b, c, d = span.control_points

        def mid(p: Point2, q: Point2) -> Point2:
            return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)

        ab, bc, cd = mid(a, b), mid(b, c), mid(c, d)
        abc, bcd = mid(ab, bc), mid(bc, cd)
        center = mid(abc, bcd)
        return (
            BezierSpan((a, ab, abc, center), False),
            BezierSpan((center, bcd, cd, d), False),
        )

    ring = CurveRing(tuple(part for span in circle().spans for part in halves(span)))
    runs = draft_runs(ring, frozenset())
    assert sorted(len(run) for run in runs) == [2, 2, 2, 2]
    assert {span for run in runs for span in run} == set(ring.spans)
    protected = frozenset({ring.spans[1].control_points[0]})
    assert any(
        run[0].control_points[0] in protected for run in draft_runs(ring, protected)
    )


def test_analytic_lines_remain_separate_even_at_tangent_joins() -> None:
    ring = circle()
    first, *rest = ring.spans
    a, b = first.control_points[0], first.control_points[-1]
    # The line remains an analytic primitive regardless of its rounded stored
    # degree-elevation controls; it cannot enter a cubic composite accidentally.
    selected = CurveRing((line(a, b), *rest))
    runs = smooth_runs(selected, frozenset())
    assert len(runs) == 2
    assert sum(len(run) for run in runs) == 4
    assert next(run for run in runs if run[0].is_line) == (selected.spans[0],)
    with pytest.raises(ValueError, match="curved spans"):
        _ = composite_bspline((selected.spans[0],), 0)


def test_source_array_rotation_and_reversal_preserve_geometric_runs() -> None:
    original = circle()
    protected = frozenset({(5.0, 0.0), (0.0, 5.0)})
    rotated = CurveRing((*original.spans[2:], *original.spans[:2]))
    expected = {run_key(r) for r in smooth_runs(original, protected)}
    assert len(expected) == 2
    assert {run_key(r) for r in smooth_runs(rotated, protected)} == expected
    assert {
        run_key(r) for r in smooth_runs(reversed_ring(original), protected)
    } == expected


def test_shared_ring_ownership_change_keeps_smooth_graph_junctions() -> None:
    inner, outer = circle(5), circle(10)
    shared = reversed_ring(CurveRing((inner.spans[0],))).spans[0]
    adjacent = CurveRing(
        (shared, line((5, 0), (10, 0)), outer.spans[0], line((0, 10), (0, 5)))
    )
    protected = protected_vertices((CurvePolygon(inner), CurvePolygon(adjacent)))
    assert {(5, 0), (0, 5)} <= protected
    assert len(smooth_runs(inner, protected)) == 2


def test_reverse_shared_closed_boundary_does_not_create_false_branches() -> None:
    inner, outer = circle(5), circle(10)
    protected = protected_vertices(
        (CurvePolygon(outer, (reversed_ring(inner),)), CurvePolygon(inner))
    )
    assert not protected
    a = smooth_runs(inner, protected)
    b = smooth_runs(reversed_ring(inner), protected)
    assert len(a) == len(b) == 1
    assert run_key(a[0]) == run_key(b[0])


def test_point_only_contact_is_an_explicit_native_vertex() -> None:
    def square(x: float, y: float) -> CurvePolygon:
        points = ((x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2))
        return CurvePolygon(
            CurveRing(tuple(line(p, points[(i + 1) % 4]) for i, p in enumerate(points)))
        )

    assert protected_vertices((square(0, 0), square(2, 2))) == frozenset({(2, 2)})


@pytest.mark.parametrize("z", [inf, nan])
def test_nonfinite_native_height_is_rejected(z: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = composite_bspline(circle().spans, z)


def test_empty_or_disconnected_curve_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _ = composite_bspline((), 0)
    with pytest.raises(ValueError, match="join exactly"):
        _ = composite_bspline((circle().spans[0], circle().spans[2]), 0)
    with pytest.raises(ValueError, match="closed connected"):
        _ = smooth_runs(CurveRing((circle().spans[0],)), frozenset())
