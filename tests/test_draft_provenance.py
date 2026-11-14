"""Independent draft retries retain their actual successful native generators."""

# pyright: reportPrivateUsage=false

from typing import cast

import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepFill import BRepFill_Draft
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve
from OCP.gp import gp_Pln
from OCP.TopoDS import TopoDS_Edge, TopoDS_Wire

from panelchemy import curved_occt as n
from panelchemy import native_sweep
from panelchemy.curves import BezierSpan, CurveRing

type Draft = BRepFill_Draft | native_sweep.Draft


def generator() -> TopoDS_Edge:
    span = BezierSpan(((0, 0), (0.3, 0.01), (0.7, 0.2), (1, 0.3)), False)
    return BRepTools_WireExplorer(
        n._wire(CurveRing((span,)), 4, {}, draft_compatible=True)
    ).Current()


def owned_result(wire: TopoDS_Wire) -> native_sweep.Draft:
    edge = BRepTools_WireExplorer(wire).Current()
    face = BRepBuilderAPI_MakeFace(gp_Pln(), 0, 1, 0, 1).Face()
    # The retry controller only consumes the result and generator ownership.
    return native_sweep.Draft(None, ((edge, face),))


def assert_ownership(generators: list[tuple[TopoDS_Edge, Draft]]) -> None:
    for edge, draft in generators:
        assert len(list(draft.Generated(edge))) == 1


def test_original_success_appends_its_actual_generator_without_replacing_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = generator()

    def verified(
        wire: TopoDS_Wire, *_args: object, precise_draft: bool = False
    ) -> native_sweep.Draft:
        assert precise_draft
        return owned_result(wire)

    monkeypatch.setattr(n, "_verified_draft", verified)
    prefix = (edge, cast("BRepFill_Draft", object()))
    collected: list[tuple[TopoDS_Edge, Draft]] = [prefix]
    result = n._independent_drafts(
        edge, 2, 4, 0.1, False, {}, precise_draft=True, generators=collected
    )
    assert collected[0] is prefix
    assert len(collected) == 2
    assert collected[1][0].IsSame(edge)
    assert collected[1][1] is result[0]
    assert_ownership(collected[1:])


@pytest.mark.parametrize("failure", [n._DraftArrayFailure, n._DraftLocationFailure])
def test_equivalent_bezier_success_records_the_replacement_edge(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    edge = generator()
    edge.Reverse()

    def verified(
        wire: TopoDS_Wire, *_args: object, precise_draft: bool = False
    ) -> native_sweep.Draft:
        _ = precise_draft
        selected = BRepTools_WireExplorer(wire).Current()
        if BRepAdaptor_Curve(selected).GetType() == GeomAbs_BSplineCurve:
            raise failure("known native failure")
        return owned_result(wire)

    monkeypatch.setattr(n, "_verified_draft", verified)
    collected: list[tuple[TopoDS_Edge, Draft]] = []
    result = n._independent_drafts(edge, 2, 4, 0.1, False, {}, generators=collected)
    assert len(collected) == 1
    actual, draft = collected[0]
    assert not actual.IsSame(edge)
    assert BRepAdaptor_Curve(actual).GetType() == GeomAbs_BezierCurve
    assert actual.Orientation() == edge.Orientation()
    assert draft is result[0]
    assert list(draft.Generated(edge)) == []
    assert_ownership(collected)


@pytest.mark.parametrize("quarter_bezier", [False, True])
def test_quarter_retries_record_each_actual_successful_trim(
    monkeypatch: pytest.MonkeyPatch, quarter_bezier: bool
) -> None:
    edge = generator()
    edge.Reverse()

    def verified(
        wire: TopoDS_Wire, *_args: object, precise_draft: bool = False
    ) -> native_sweep.Draft:
        _ = precise_draft
        selected = BRepTools_WireExplorer(wire).Current()
        curve = BRepAdaptor_Curve(selected)
        interval = curve.FirstParameter(), curve.LastParameter()
        if interval == (0, 1) or (
            quarter_bezier
            and interval == (0.5, 0.75)
            and curve.GetType() == GeomAbs_BSplineCurve
        ):
            raise n._DraftArrayFailure("known native failure")
        return owned_result(wire)

    monkeypatch.setattr(n, "_verified_draft", verified)
    collected: list[tuple[TopoDS_Edge, Draft]] = []
    result = n._independent_drafts(edge, 2, 4, 0.1, False, {}, generators=collected)
    assert len(collected) == len(result) == 4
    for index, (actual, draft) in enumerate(collected):
        curve = BRepAdaptor_Curve(actual)
        assert (curve.FirstParameter(), curve.LastParameter()) == (
            index / 4,
            (index + 1) / 4,
        )
        assert curve.GetType() == (
            GeomAbs_BezierCurve
            if quarter_bezier and index == 2
            else GeomAbs_BSplineCurve
        )
        assert actual.Orientation() == edge.Orientation()
        assert draft is result[index]
    assert_ownership(collected)


@pytest.mark.parametrize("failure", [RuntimeError, n._DraftLocationFailure])
def test_later_quarter_failure_leaves_collector_unchanged(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    edge = generator()
    completed_quarters = 0

    def verified(
        wire: TopoDS_Wire, *_args: object, precise_draft: bool = False
    ) -> native_sweep.Draft:
        nonlocal completed_quarters
        _ = precise_draft
        curve = BRepAdaptor_Curve(BRepTools_WireExplorer(wire).Current())
        interval = curve.FirstParameter(), curve.LastParameter()
        if interval == (0, 1):
            raise n._DraftArrayFailure("known native failure")
        if interval == (0.25, 0.5):
            raise failure("second quarter failed")
        completed_quarters += 1
        return owned_result(wire)

    monkeypatch.setattr(n, "_verified_draft", verified)
    prefix = (edge, cast("BRepFill_Draft", object()))
    collected: list[tuple[TopoDS_Edge, Draft]] = [prefix]
    with pytest.raises(failure, match="second quarter failed"):
        _ = n._independent_drafts(edge, 2, 4, 0.1, False, {}, generators=collected)
    assert completed_quarters == 1
    assert len(collected) == 1 and collected[0] is prefix
