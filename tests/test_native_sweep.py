"""Explicit-accuracy sweeps retain native winding, ownership and roof guards."""

# pyright: reportPrivateUsage=false

from math import atan, cos, isnan, radians, tan
from typing import cast, override

import pytest
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepFill import BRepFill_Draft, BRepFill_Modified, BRepFill_Sweep
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.gp import gp_Dir, gp_Trsf, gp_Vec, gp_VectorWithNullMagnitude
from OCP.TopAbs import TopAbs_FACE
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Wire
from shapely.geometry import box

from panelchemy import curved_occt as n
from panelchemy import material_partition, native_sweep, occt
from panelchemy.curves import BezierSpan, CurvePolygon, CurveRing, polygon_lines
from panelchemy.export import _step

SLOPE = tan(radians(10))


def circle(radius: float) -> CurveRing:
    k = radius * 0.5522847498307936
    r = radius
    return CurveRing(
        tuple(
            BezierSpan(p, False)
            for p in (
                ((r, 0), (r, k), (k, r), (0, r)),
                ((0, r), (-k, r), (-r, k), (-r, 0)),
                ((-r, 0), (-r, -k), (-k, -r), (0, -r)),
                ((0, -r), (k, -r), (r, -k), (r, 0)),
            )
        )
    )


def shell_geometry(shape: TopoDS_Shape) -> object:
    return (
        occt.topology_counts(shape),
        n._bounds(shape),
        [
            (
                face.Orientation(),
                n._geometry_bytes(BRep_Tool.Surface_s(TopoDS.Face(face))),
            )
            for face in n._entities(shape, TopAbs_FACE)
        ],
    )


@pytest.mark.parametrize("reported_error", [1e-5, -1.0, float("nan"), float("inf")])
def test_reported_sweep_error_must_meet_requested_accuracy(
    monkeypatch: pytest.MonkeyPatch, reported_error: float
) -> None:
    class InaccurateSweep(BRepFill_Sweep):
        @override
        def ErrorOnSurface(self) -> float:
            return reported_error

    monkeypatch.setattr(native_sweep, "BRepFill_Sweep", InaccurateSweep)
    wire = n._wire(circle(4), 4, {}, draft_compatible=True)
    before = material_partition.shape_digest(wire)
    result = native_sweep.draft(
        wire, gp_Dir(0, 0, -1), atan(SLOPE), 2 / cos(atan(SLOPE)), False
    )
    assert not result.IsDone()
    assert result.error_mm is not None
    assert (
        isnan(result.error_mm)
        if isnan(reported_error)
        else result.error_mm == reported_error
    )
    assert material_partition.shape_digest(wire) == before


@pytest.mark.parametrize("kind", ["line", "bezier", "closed"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("located", [False, True])
def test_native_baseline_equivalence_and_original_generator_ownership(
    kind: str, reverse: bool, located: bool
) -> None:
    ring = (
        circle(4)
        if kind == "closed"
        else CurveRing(
            (
                BezierSpan(((0, 0), (0, 0), (3, 0), (3, 0)), True)
                if kind == "line"
                else BezierSpan(((0, 0), (1, 0.5), (2, 0.5), (3, 0)), False),
            )
        )
    )
    wire = n._wire(ring, 4, {}, draft_compatible=True)
    if reverse:
        wire.Reverse()
    if located:
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(7, -3, 0.5))
        wire = TopoDS.Wire(wire.Moved(TopLoc_Location(transform)))
    before = material_partition.shape_digest(wire)
    for internal in (False, True):
        stock = BRepFill_Draft(wire, gp_Dir(0, 0, -1), atan(SLOPE))
        stock.SetDraft(internal)
        stock.SetOptions(BRepFill_Modified, 1e-8)
        try:
            stock.Perform(2 / cos(atan(SLOPE)))
        except gp_VectorWithNullMagnitude:
            # Native Draft's winding test can be singular on a straight open
            # wire. Production builds line walls directly; preserve that native
            # failure rather than inventing a different winding heuristic.
            with pytest.raises(gp_VectorWithNullMagnitude):
                _ = native_sweep.draft(
                    wire,
                    gp_Dir(0, 0, -1),
                    atan(SLOPE),
                    2 / cos(atan(SLOPE)),
                    internal,
                    accuracy_mm=1e-4,
                )
            continue
        result = native_sweep.draft(
            wire,
            gp_Dir(0, 0, -1),
            atan(SLOPE),
            2 / cos(atan(SLOPE)),
            internal,
            accuracy_mm=1e-4,
        )
        assert result.IsDone() == stock.IsDone()
        assert shell_geometry(result.Shell()) == shell_geometry(stock.Shell())
        edges = BRepTools_WireExplorer(wire)
        while edges.More():
            edge = edges.Current()
            expected, actual = list(stock.Generated(edge)), result.Generated(edge)
            assert len(actual) == len(expected) == 1
            assert n._geometry_bytes(
                BRep_Tool.Surface_s(TopoDS.Face(actual[0]))
            ) == n._geometry_bytes(BRep_Tool.Surface_s(TopoDS.Face(expected[0])))
            edges.Next()
    assert material_partition.shape_digest(wire) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_precise_curved_roof_with_counter_keeps_source_and_step(reverse: bool) -> None:
    outer, hole = circle(4), circle(0.8)
    if reverse:
        outer, hole = n._oriented(outer, False), n._oriented(hole, False)
    region = CurvePolygon(outer, (hole,))
    diagnostics: dict[str, object] = {}
    shape = n._surface_region(
        region,
        2,
        4,
        SLOPE,
        consolidate_draft=True,
        precise_draft=True,
        diagnostics=diagnostics,
        protected_points=frozenset({(4, 0)}),
    )
    n._check_roof(shape, region, 2, 4)
    n._check_sections(shape, region, 2, 4, SLOPE)
    assert (
        cast("float", diagnostics["precise_sweep_error_mm"]) <= native_sweep.ACCURACY_MM
    )
    _, report = _step(shape)
    assert report["valid"] is True


def test_precise_section_failure_retains_old_preimprint_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = CurvePolygon(circle(4))
    shape, _ = n.roof(region, 2, 4, SLOPE)
    calls: list[tuple[bool, bool]] = []
    checked: list[tuple[bool, bool]] = []
    original_sections = n._check_sections

    def surface(*_args: object, **kwargs: object) -> TopoDS_Shape:
        attempt = bool(kwargs["preimprint"]), bool(kwargs["precise_draft"])
        calls.append(attempt)
        if len(calls) == 1:
            raise occt.KernelError("initial arrangement failure")
        return shape

    def sections(
        candidate: TopoDS_Shape,
        source: CurvePolygon,
        z0: float,
        z1: float,
        slope: float,
    ) -> None:
        original_sections(candidate, source, z0, z1, slope)
        checked.append(calls[-1])
        if calls[-1][1]:
            raise occt.KernelError("precise candidate failed its section")

    monkeypatch.setattr(n, "_surface_region", surface)
    monkeypatch.setattr(n, "_check_sections", sections)
    returned, method = n.roof(region, 2, 4, SLOPE)
    assert returned.IsSame(shape) and method == "surface-preimprint"
    assert calls == [(False, False), (False, True), (True, False)]
    assert checked == [(False, True), (True, False)]


def test_normal_success_does_not_invoke_precise_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> native_sweep.Draft:
        raise AssertionError("successful primary must not build a precise alternative")

    monkeypatch.setattr(native_sweep, "draft", unexpected)
    diagnostics: dict[str, object] = {}
    _, method = n.roof(
        polygon_lines(box(0, 0, 4, 4)), 2, 4, SLOPE, diagnostics=diagnostics
    )
    assert method == "surface"
    assert "precise_consolidated_retry" not in diagnostics


def test_precise_accuracy_survives_bezier_and_quarter_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = CurveRing((BezierSpan(((0, 0), (1, 0.5), (2, 0.5), (3, 0)), False),))
    wire = n._wire(ring, 4, {}, draft_compatible=True)
    edge = BRepTools_WireExplorer(wire).Current()
    seen: list[tuple[float, float]] = []
    token = native_sweep.Draft(None)

    def verified(
        selected: TopoDS_Wire, *_args: object, precise_draft: bool = False
    ) -> native_sweep.Draft:
        assert precise_draft
        adapter = BRepAdaptor_Curve(BRepTools_WireExplorer(selected).Current())
        interval = adapter.FirstParameter(), adapter.LastParameter()
        seen.append(interval)
        if interval == (0, 1):
            raise n._DraftArrayFailure("known native array failure")
        return token

    monkeypatch.setattr(n, "_verified_draft", verified)
    assert (
        n._independent_drafts(edge, 2, 4, SLOPE, False, {}, precise_draft=True)
        == [token] * 4
    )
    assert seen == [(0, 1), (0, 1), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]


@pytest.mark.parametrize("wrapped", [False, True])
def test_memory_failure_never_starts_another_construction(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    calls = 0

    def surface(*_args: object, **_kwargs: object) -> TopoDS_Shape:
        nonlocal calls
        calls += 1
        if wrapped:
            raise occt.KernelError("wrapped exhaustion") from MemoryError()
        raise MemoryError()

    monkeypatch.setattr(n, "_surface_region", surface)
    with pytest.raises((MemoryError, occt.KernelError)):
        _ = n.roof(polygon_lines(box(0, 0, 4, 4)), 2, 4, SLOPE)
    assert calls == 1
