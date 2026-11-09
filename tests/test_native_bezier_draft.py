"""An exact Bernstein representation avoids a native one-edge draft failure."""

# pyright: reportPrivateUsage=false

from math import radians, tan
from typing import Never, cast

import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepFill import BRepFill_Draft
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int
from OCP.Geom import Geom_BezierCurve, Geom_BSplineCurve
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_CurveType
from OCP.gp import gp_Pnt
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS_Edge, TopoDS_Wire

from panelchemy import curved_occt as n
from panelchemy import occt
from panelchemy.curves import BezierSpan, CurveRing

SLOPE = tan(radians(10))
BOTTOM = 2 - max(1e-3, SLOPE * 2 * 0.01)


def generator(*, low: float = 0, high: float = 1, reverse: bool = False) -> TopoDS_Edge:
    # A shallow inflected cubic: native unit-domain BSpline draft fails on
    # affected kernels, while its identical native Bezier representation works.
    span = BezierSpan(((0, 0), (-0.1, -0.01), (-0.4, 0.05), (-0.6, 0.1)), False)
    wire = n._wire(CurveRing((span,)), 4, {}, draft_compatible=True)
    edge = BRepTools_WireExplorer(wire).Current()
    if (low, high) != (0, 1):
        edge = BRepBuilderAPI_MakeEdge(
            BRepAdaptor_Curve(edge).BSpline(), low, high
        ).Edge()
    if reverse:
        edge.Reverse()
    return edge


@pytest.mark.parametrize("reverse", [False, True])
def test_real_shallow_cubic_draft_preserves_exact_input_and_native_validity(
    reverse: bool,
) -> None:
    edge = generator(reverse=reverse)
    before = n._geometry_bytes(BRepAdaptor_Curve(edge).BSpline())
    diagnostics: dict[str, object] = {}
    drafts = n._independent_drafts(edge, BOTTOM, 4, SLOPE, True, diagnostics)
    assert len(drafts) == 1 and occt.valid(drafts[0].Shell())
    assert n._geometry_bytes(BRepAdaptor_Curve(edge).BSpline()) == before


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("interval", [(0.0, 1.0), (0.125, 0.875)])
def test_bernstein_coefficients_natural_vertices_trim_and_orientation_are_unchanged(
    reverse: bool, interval: tuple[float, float]
) -> None:
    edge = generator(low=interval[0], high=interval[1], reverse=reverse)
    original = BRepAdaptor_Curve(edge)
    before = n._geometry_bytes(original.BSpline())
    selected = n._bezier_generator(edge, original)
    assert selected is not None and occt.valid(selected)
    actual = BRepAdaptor_Curve(selected)
    assert actual.GetType() == GeomAbs_BezierCurve
    assert actual.Bezier().Degree() == 3 and not actual.Bezier().IsRational()
    assert [actual.Bezier().Pole(i).Coord() for i in range(1, 5)] == [
        original.BSpline().Pole(i).Coord() for i in range(1, 5)
    ]
    assert (actual.FirstParameter(), actual.LastParameter()) == interval
    assert selected.Orientation() == edge.Orientation()
    for cumulative in (False, True):
        assert TopExp.FirstVertex_s(selected, cumulative).IsSame(
            TopExp.FirstVertex_s(edge, cumulative)
        )
        assert TopExp.LastVertex_s(selected, cumulative).IsSame(
            TopExp.LastVertex_s(edge, cumulative)
        )
    assert n._geometry_bytes(original.BSpline()) == before


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("this representation fallback must not run")


def test_successful_initial_native_path_does_not_construct_an_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = cast("BRepFill_Draft", object())

    def verified(*_args: object) -> BRepFill_Draft:
        return token

    monkeypatch.setattr(n, "_verified_draft", verified)
    monkeypatch.setattr(n, "_bezier_generator", unexpected)
    diagnostics: dict[str, object] = {}
    assert n._independent_drafts(generator(), BOTTOM, 4, SLOPE, True, diagnostics) == [
        token
    ]
    assert diagnostics == {}


def test_known_failure_tries_identical_bezier_with_stock_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = generator(reverse=True)
    seen: list[GeomAbs_CurveType] = []
    token = cast("BRepFill_Draft", object())

    def verified(
        wire: TopoDS_Wire,
        z0: float,
        z1: float,
        slope: float,
        internal: bool,
        _diagnostics: dict[str, object] | None,
    ) -> BRepFill_Draft:
        part = BRepTools_WireExplorer(wire).Current()
        kind = BRepAdaptor_Curve(part).GetType()
        seen.append(kind)
        assert (z0, z1, slope, internal) == (BOTTOM, 4, SLOPE, True)
        assert _diagnostics is diagnostics
        assert part.Orientation() == edge.Orientation()
        if kind == GeomAbs_BSplineCurve:
            raise n._DraftArrayFailure("known native array failure")
        assert kind == GeomAbs_BezierCurve
        return token

    monkeypatch.setattr(n, "_verified_draft", verified)
    diagnostics: dict[str, object] = {}
    assert n._independent_drafts(edge, BOTTOM, 4, SLOPE, True, diagnostics) == [token]
    assert seen == [GeomAbs_BSplineCurve, GeomAbs_BezierCurve]
    assert diagnostics["bezier_draft_generators"] == 1


def unsupported(kind: str) -> TopoDS_Edge:
    poles = Array1_gp_Pnt(1, 7 if kind == "multiknot" else 4)
    for i in range(1, poles.Length() + 1):
        poles.SetValue(i, gp_Pnt(i, (i % 3) / 4, 4))
    knots = Array1_double(1, 3 if kind == "multiknot" else 2)
    multiplicities = Array1_int(1, knots.Length())
    values = (
        (0, 1, 2) if kind == "multiknot" else (2, 5) if kind == "nonunit" else (0, 1)
    )
    for i, value in enumerate(values, 1):
        knots.SetValue(i, value)
        multiplicities.SetValue(i, 3 if kind == "multiknot" and i == 2 else 4)
    if kind == "rational":
        weights = Array1_double(1, 4)
        for i in range(1, 5):
            weights.SetValue(i, 2 if i == 2 else 1)
        curve = Geom_BSplineCurve(poles, weights, knots, multiplicities, 3, False)
    else:
        curve = Geom_BSplineCurve(poles, knots, multiplicities, 3, False)
    return BRepBuilderAPI_MakeEdge(curve).Edge()


@pytest.mark.parametrize("kind", ["multiknot", "nonunit", "rational"])
def test_unsupported_basis_keeps_the_original_quarter_trim_fallback(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    edge = unsupported(kind)
    adapter = BRepAdaptor_Curve(edge)
    before = n._geometry_bytes(adapter.BSpline())
    assert n._bezier_generator(edge, adapter) is None
    monkeypatch.setattr(n, "Geom_BezierCurve", unexpected)
    intervals: list[tuple[float, float]] = []
    token = cast("BRepFill_Draft", object())
    full = adapter.FirstParameter(), adapter.LastParameter()

    def verified(wire: TopoDS_Wire, *_args: object) -> BRepFill_Draft:
        part = BRepAdaptor_Curve(BRepTools_WireExplorer(wire).Current())
        assert part.GetType() == GeomAbs_BSplineCurve
        assert n._geometry_bytes(part.BSpline()) == before
        interval = part.FirstParameter(), part.LastParameter()
        intervals.append(interval)
        if interval == full:
            raise n._DraftArrayFailure("known native array failure")
        return token

    monkeypatch.setattr(n, "_verified_draft", verified)
    assert n._independent_drafts(edge, BOTTOM, 4, SLOPE, True, {}) == [token] * 4
    assert intervals == [
        full,
        *[
            (
                full[0] + (full[1] - full[0]) * i / 4,
                full[0] + (full[1] - full[0]) * (i + 1) / 4,
            )
            for i in range(4)
        ],
    ]


def test_unrelated_bezier_failure_propagates_without_quarter_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds: list[GeomAbs_CurveType] = []

    def verified(wire: TopoDS_Wire, *_args: object) -> BRepFill_Draft:
        kind = BRepAdaptor_Curve(BRepTools_WireExplorer(wire).Current()).GetType()
        kinds.append(kind)
        if kind == GeomAbs_BSplineCurve:
            raise n._DraftArrayFailure("known native array failure")
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(n, "_verified_draft", verified)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        _ = n._independent_drafts(generator(), BOTTOM, 4, SLOPE, True, {})
    assert kinds == [GeomAbs_BSplineCurve, GeomAbs_BezierCurve]


def test_changed_bernstein_coefficients_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = generator()
    adapter = BRepAdaptor_Curve(edge)
    before = n._geometry_bytes(adapter.BSpline())

    def changed(poles: Array1_gp_Pnt) -> Geom_BezierCurve:
        curve = Geom_BezierCurve(poles)
        point = curve.Pole(2)
        point.SetX(point.X() + 2**-20)
        curve.SetPole(2, point)
        return curve

    monkeypatch.setattr(n, "Geom_BezierCurve", changed)
    with pytest.raises(occt.KernelError, match="changed cubic coefficients"):
        _ = n._bezier_generator(edge, adapter)
    assert n._geometry_bytes(adapter.BSpline()) == before
