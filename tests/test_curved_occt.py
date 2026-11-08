"""Native fitted profiles, regional draft solids and bounded closure cases."""

from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from math import pi, radians, tan
from typing import cast, override

import pytest
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_BooleanOperation
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeFace
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepFill import BRepFill_Draft
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeTorus
from OCP.BRepTools import BRepTools
from OCP.collections import List_TopoDS_Shape
from OCP.GeomAbs import GeomAbs_BSplineCurve
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.Standard import Standard_OutOfMemory, Standard_OutOfRange
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Wire,
)
from OCP.TopTools import TopTools_FormatVersion_VERSION_3
from shapely.geometry import Polygon, box

from panelchemy import curved_occt, occt
from panelchemy.curved_occt import (
    _bounds,  # pyright: ignore[reportPrivateUsage]
    _cell_interior,  # pyright: ignore[reportPrivateUsage]
    _cell_repair_state,  # pyright: ignore[reportPrivateUsage]
    _cell_repair_unchanged,  # pyright: ignore[reportPrivateUsage]
    _check_roof,  # pyright: ignore[reportPrivateUsage]
    _DraftArrayFailure,  # pyright: ignore[reportPrivateUsage]
    _flat_ring,  # pyright: ignore[reportPrivateUsage]
    _fuse_draft_cells,  # pyright: ignore[reportPrivateUsage]
    _geometry_bytes,  # pyright: ignore[reportPrivateUsage]
    _independent_drafts,  # pyright: ignore[reportPrivateUsage]
    _verified_draft,  # pyright: ignore[reportPrivateUsage]
    _wire,  # pyright: ignore[reportPrivateUsage]
)
from panelchemy.curves import (
    BezierSpan,
    Controls,
    CurvePolygon,
    CurveRing,
    _line,  # pyright: ignore[reportPrivateUsage]
    polygon_lines,
)
from panelchemy.export import _step  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("case", ["box", "thin", "counter", "torus"])
def test_cell_positive_volume_witness_needs_no_mass_integral(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    if case == "torus":
        solid = BRepPrimAPI_MakeTorus(2, 0.5).Shape()
    elif case == "counter":
        solid = occt.prism(
            Polygon(box(0, 0, 4, 4).exterior.coords, [box(1, 1, 3, 3).exterior.coords]),
            0,
            1,
        )
    else:
        solid = BRepPrimAPI_MakeBox(1, 1, 1e-6 if case == "thin" else 1).Shape()

    def unexpected(_shape: TopoDS_Shape) -> float:
        pytest.fail(
            "cell positivity must use boundedness and an actual interior, not accurate mass"
        )

    monkeypatch.setattr(occt, "volume", unexpected)
    point = _cell_interior(solid)
    assert BRepClass3d_SolidClassifier(solid, point, 1e-8).State() == TopAbs_IN


@pytest.mark.parametrize("case", ["reversed", "empty", "compound", "open", "null"])
def test_cell_witness_rejects_nonregular_or_unbounded_domains(case: str) -> None:
    box_shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    builder = BRep_Builder()
    if case == "null":
        solid = TopoDS_Shape()
    elif case == "reversed":
        solid = box_shape.Reversed()
    elif case == "compound":
        solid = TopoDS_Compound()
        builder.MakeCompound(solid)
        builder.Add(solid, box_shape)
    else:
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        if case == "open":
            shell = TopoDS_Shell()
            builder.MakeShell(shell)
            faces = TopExp_Explorer(box_shape, TopAbs_FACE)
            for _ in range(5):
                builder.Add(shell, faces.Current())
                faces.Next()
            builder.Add(solid, shell)
    with pytest.raises(occt.KernelError):
        _ = _cell_interior(solid)


@pytest.mark.parametrize("case", ["height", "footprint"])
def test_roof_rejects_geometric_mismatch_before_expensive_mass(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    solid = BRepPrimAPI_MakeBox(2, 2, 1).Shape()
    expected = polygon_lines(box(0, 0, 2, 2) if case == "height" else box(0, 0, 1, 1))

    def unexpected(_shape: TopoDS_Shape) -> float:
        pytest.fail("an already incorrect roof must fail before mass integration")

    monkeypatch.setattr(occt, "volume", unexpected)
    with pytest.raises(occt.KernelError, match=r"height|fitted top"):
        _check_roof(solid, expected, 0, 2 if case == "height" else 1)


def test_large_validation_ring_keeps_accuracy_with_a_scaled_resource_budget() -> None:
    count = 350
    spans = [
        BezierSpan(((i, 0), (i + 1 / 3, 1), (i + 2 / 3, 1), (i + 1, 0)), False)
        for i in range(count)
    ]
    spans.extend(
        (
            _line((count, 0), (count, -1)),
            _line((count, -1), (0, -1)),
            _line((0, -1), (0, 0)),
        )
    )
    points = _flat_ring(CurveRing(tuple(spans)), 1e-5)
    assert len(points) > 100_000
    assert Polygon(points).is_valid
    assert Polygon(points).area == pytest.approx(1.5 * count, abs=0.01)
    vertices = set(points)
    assert all(span.control_points[0] in vertices for span in spans)


def circle(radius: float) -> CurveRing:
    k = 0.5522847498307936
    coordinates = (
        ((radius, 0.0), (radius, k * radius), (k * radius, radius), (0.0, radius)),
        ((0.0, radius), (-k * radius, radius), (-radius, k * radius), (-radius, 0.0)),
        (
            (-radius, 0.0),
            (-radius, -k * radius),
            (-k * radius, -radius),
            (0.0, -radius),
        ),
        ((0.0, -radius), (k * radius, -radius), (radius, -k * radius), (radius, 0.0)),
    )
    return CurveRing(tuple(BezierSpan(points, False) for points in coordinates))


@pytest.mark.parametrize("reversed_edge", [False, True])
def test_failed_generator_uses_exact_native_parameter_trims(
    monkeypatch: pytest.MonkeyPatch, reversed_edge: bool
) -> None:
    span = circle(2).spans[0]
    wire = _wire(CurveRing((span,)), 3, {}, draft_compatible=True)
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    edge = TopoDS.Edge(explorer.Current())
    if reversed_edge:
        edge.Reverse()
    original_curve = _geometry_bytes(BRepAdaptor_Curve(edge).BSpline())
    seen: list[tuple[float, float]] = []

    def draft(
        wire: TopoDS_Wire,
        z0: float,
        z1: float,
        slope: float,
        internal: bool,
        diagnostics: dict[str, object] | None = None,
    ) -> BRepFill_Draft:
        explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        part = TopoDS.Edge(explorer.Current())
        curve = BRepAdaptor_Curve(part)
        assert part.Orientation() == edge.Orientation()
        assert _geometry_bytes(curve.BSpline()) == original_curve
        interval = curve.FirstParameter(), curve.LastParameter()
        seen.append(interval)
        if interval == (0, 1):
            raise _DraftArrayFailure("forced known full-generator failure")
        return _verified_draft(wire, z0, z1, slope, internal, diagnostics)

    monkeypatch.setattr(curved_occt, "_verified_draft", draft)
    diagnostics: dict[str, object] = {}
    parts = _independent_drafts(edge, 2, 3, 0.1, False, diagnostics)
    assert len(parts) == 4 and all(occt.valid(part.Shell()) for part in parts)
    assert seen == [(0, 1), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]
    assert diagnostics["subdivided_draft_generators"] == 1
    assert _geometry_bytes(BRepAdaptor_Curve(edge).BSpline()) == original_curve


def test_curve_face_consolidates_smooth_edges_and_preserves_counter_area() -> None:
    region = CurvePolygon(circle(5), (circle(1),))
    face = curved_occt.polygon_face(region, 4)
    assert occt.valid(face)
    assert occt.area(face) == pytest.approx(pi * (25 - 1), rel=0.0003)
    edges = TopExp_Explorer(face, TopAbs_EDGE)
    count = 0
    while edges.More():
        assert (
            BRepAdaptor_Curve(TopoDS.Edge(edges.Current())).GetType()
            == GeomAbs_BSplineCurve
        )
        count += 1
        edges.Next()
    assert count == 2


def test_curve_face_keeps_a_globally_protected_smooth_join() -> None:
    region = CurvePolygon(circle(5))
    # A junction belonging to a neighboring palette region is not visible in
    # this polygon alone, but must remain an explicit vertex in its CAD face.
    face = curved_occt.polygon_face(region, protected_points=frozenset({(0, 5)}))
    assert occt.valid(face)
    assert occt.topology_counts(face)["edges"] == 1
    assert occt.topology_counts(face)["vertices"] == 1
    edges = TopExp_Explorer(face, TopAbs_EDGE)
    curve = BRepAdaptor_Curve(TopoDS.Edge(edges.Current()))
    assert curve.Value(curve.FirstParameter()).Coord() == (0, 5, 0)


def test_line_face_rejects_disconnected_span_endpoints() -> None:
    region = polygon_lines(box(0, 0, 2, 2))
    first, *others = region.exterior.spans
    controls = first.control_points
    disconnected = replace(
        first, control_points=(controls[0], controls[1], controls[2], (1.0, 1.0))
    )
    malformed = CurvePolygon(CurveRing((disconnected, *others)))
    with pytest.raises(occt.KernelError, match="join exactly"):
        _ = curved_occt.polygon_face(malformed)


def test_zero_taper_extrudes_one_fitted_region_with_hole() -> None:
    region = CurvePolygon(circle(5), (circle(1),))
    shape, method = curved_occt.roof(region, 2, 4, 0)
    assert method == "prism"
    assert occt.volume(shape) == pytest.approx(
        2 * occt.area(curved_occt.polygon_face(region)), rel=1e-10
    )
    assert occt.topology_counts(shape)["solids"] == 1


def test_consolidated_curved_holes_keep_their_shared_contact_vertex() -> None:
    def moved_ring(x: float) -> CurveRing:
        return CurveRing(
            tuple(
                replace(
                    span,
                    control_points=cast(
                        "Controls", tuple((p[0] + x, p[1]) for p in span.control_points)
                    ),
                )
                for span in circle(1).spans
            )
        )

    region = CurvePolygon(circle(5), (moved_ring(-1), moved_ring(1)))
    face = curved_occt.polygon_face(region)
    assert occt.valid(face)
    assert occt.topology_counts(face)["edges"] == 3
    assert occt.topology_counts(face)["vertices"] == 2
    assert occt.area(face) == pytest.approx(pi * 23, rel=0.0003)


def test_native_curved_draft_preserves_top_and_expands_downward() -> None:
    region = CurvePolygon(circle(5), (circle(1),))
    shape, method = curved_occt.roof(region, 2, 4, 0.1, prefer_fast=False)
    assert method == "surface"
    expected = curved_occt.polygon_face(region, 4)
    actual = occt.cross_section(shape, 4)
    assert occt.valid(shape)
    assert occt.area(occt.cut(expected, actual)) < 1e-8
    assert occt.area(occt.cut(actual, expected)) < 1e-8
    assert occt.area(occt.cross_section(shape, 2)) > occt.area(actual)
    assert occt.topology_counts(shape)["faces"] < 30


def test_native_draft_corrects_global_reversal_for_exterior_and_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_flags: list[bool] = []

    class ReversedDraft(BRepFill_Draft):
        @override
        def SetDraft(self, IsInternal: bool = False) -> None:
            requested_flags.append(IsInternal)
            # Keep all generated geometry native; only force the first requested
            # sense to be wrong so correction cannot depend on a lucky profile.
            super().SetDraft(not IsInternal)

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", ReversedDraft)
    region = CurvePolygon(circle(2), (circle(1),))
    shape, method = curved_occt.roof(region, 2, 3, 0.1)
    assert method == "surface"
    assert requested_flags == [False, True, True, False]
    assert occt.valid(shape)
    expected = curved_occt.polygon_face(region, 3)
    top = occt.cross_section(shape, 3)
    assert occt.area(occt.cut(expected, top)) < 1e-8
    assert occt.area(occt.cut(top, expected)) < 1e-8
    bottom = occt.cross_section(shape, 2)
    for patch in (box(2.04, -0.01, 2.06, 0.01), box(0.94, -0.01, 0.96, 0.01)):
        assert occt.area(occt.common(top, occt.polygon_face(patch, 3))) < 1e-9
        assert occt.area(
            occt.common(bottom, occt.polygon_face(patch, 2))
        ) == pytest.approx(patch.area, abs=1e-10)


def test_curved_u_taper_expands_outer_and_concave_boundaries() -> None:
    outer = circle(2).spans[2:]
    inner = tuple(
        replace(
            span, control_points=cast("Controls", tuple(reversed(span.control_points)))
        )
        for span in reversed(circle(1).spans[2:])
    )
    region = CurvePolygon(
        CurveRing(
            (
                *outer,
                BezierSpan(((2, 0), (2, 0), (1, 0), (1, 0)), True),
                *inner,
                BezierSpan(((-1, 0), (-1, 0), (-2, 0), (-2, 0)), True),
            )
        )
    )
    shape, _ = curved_occt.roof(region, 2, 3, 0.1)
    assert occt.valid(shape)
    expected = curved_occt.polygon_face(region, 3)
    top = occt.cross_section(shape, 3)
    assert occt.area(occt.cut(expected, top)) < 1e-8
    assert occt.area(occt.cut(top, expected)) < 1e-8
    bottom = occt.cross_section(shape, 2)
    for patch in (box(-0.01, -2.06, 0.01, -2.04), box(-0.01, -0.96, 0.01, -0.94)):
        assert occt.area(occt.common(top, occt.polygon_face(patch, 3))) < 1e-9
        assert occt.area(
            occt.common(bottom, occt.polygon_face(patch, 2))
        ) == pytest.approx(patch.area, abs=1e-10)


def test_wrong_draft_direction_retry_uses_one_fresh_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected: list[object] = []

    def reversed_direction(*args: object, **kwargs: object) -> bool:
        inspected.append(args[0] if args else kwargs["draft"])
        return False

    monkeypatch.setattr(curved_occt, "_draft_direction", reversed_direction)
    with pytest.raises(occt.KernelError):
        _ = curved_occt._unjoined_sheets(circle(2), 2, 3, 0.1, False)  # pyright: ignore[reportPrivateUsage]
    assert len(inspected) == 2
    assert inspected[0] is not inspected[1]


@pytest.mark.parametrize("semicircle", [False, True])
def test_independent_drafts_preserve_curved_counter_and_analytic_line(
    monkeypatch: pytest.MonkeyPatch, semicircle: bool
) -> None:
    counter = circle(0.5)
    if semicircle:
        a, b = (0.5, 0.0), (-0.5, 0.0)
        counter = CurveRing((*counter.spans[2:], BezierSpan((a, a, b, b), True)))
    region = CurvePolygon(circle(2), (counter,))
    protected = frozenset({(0.0, 2.0), (0.5, 0.0)})
    original = [
        s.control_points for r in (region.exterior, *region.interiors) for s in r.spans
    ]
    attempted: list[int] = []

    def build(wire: TopoDS_Wire, direction: gp_Dir, angle: float) -> BRepFill_Draft:
        count = occt.topology_counts(wire)["edges"]
        attempted.append(count)
        if count > 1:
            raise Standard_OutOfRange("NCollection_Array1::Value")
        explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        assert (
            BRepAdaptor_Curve(TopoDS.Edge(explorer.Current())).GetType()
            == GeomAbs_BSplineCurve
        )  # Analytic lines must use their exact planar wall constructor.
        return BRepFill_Draft(wire, direction, angle)

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", build)
    diagnostics: dict[str, object] = {"per_edge_draft_rings": 37}
    shape, method = curved_occt.roof(
        region, 2, 3, 0.1, protected_points=protected, diagnostics=diagnostics
    )
    assert method == "surface" and occt.valid(shape)
    assert diagnostics["per_edge_draft_rings"] == 2
    assert sum(count > 1 for count in attempted) == 4  # Both flags for both rings.
    assert [
        s.control_points for r in (region.exterior, *region.interiors) for s in r.spans
    ] == original
    expected = curved_occt.polygon_face(region, 3, protected_points=protected)
    top = occt.cross_section(shape, 3)
    assert occt.area(occt.cut(expected, top)) < 1e-8
    assert occt.area(occt.cut(top, expected)) < 1e-8
    assert occt.area(occt.cross_section(shape, 2)) > occt.area(top)


@pytest.mark.parametrize(
    "error_type", [MemoryError, Standard_OutOfMemory, Standard_OutOfRange]
)
def test_unrelated_draft_failure_does_not_try_another_flag_or_independent_edges(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    calls = 0

    def fail(*_args: object, **_kwargs: object) -> BRepFill_Draft:
        nonlocal calls
        calls += 1
        raise error_type("unrelated native failure")

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", fail)
    diagnostics: dict[str, object] = {}
    with pytest.raises(error_type):
        _ = curved_occt._unjoined_sheets(  # pyright: ignore[reportPrivateUsage]
            circle(2), 2, 3, 0.1, False, diagnostics=diagnostics
        )
    assert calls == 1 and "per_edge_draft_rings" not in diagnostics


def test_ambiguous_draft_direction_does_not_try_independent_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def ambiguous(*_args: object, **_kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        raise occt.KernelError("native draft has inconsistent taper directions")

    monkeypatch.setattr(curved_occt, "_draft_direction", ambiguous)
    with pytest.raises(occt.KernelError, match="inconsistent taper directions"):
        _ = curved_occt._unjoined_sheets(circle(2), 2, 3, 0.1, False)  # pyright: ignore[reportPrivateUsage]
    assert calls == 1


def test_one_array_failure_and_unverified_sense_do_not_trigger_independent_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def build(wire: TopoDS_Wire, direction: gp_Dir, angle: float) -> BRepFill_Draft:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Standard_OutOfRange("NCollection_Array1::Value")
        return BRepFill_Draft(wire, direction, angle)

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", build)

    def unverified(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr(curved_occt, "_draft_direction", unverified)
    with pytest.raises(occt.KernelError, match="could not verify"):
        _ = curved_occt._unjoined_sheets(circle(2), 2, 3, 0.1, False)  # pyright: ignore[reportPrivateUsage]
    assert calls == 2


def test_array_exception_wrapping_memory_failure_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*_args: object, **_kwargs: object) -> BRepFill_Draft:
        nonlocal calls
        calls += 1
        raise Standard_OutOfRange("NCollection_Array1::Value") from MemoryError(
            "native"
        )

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", fail)
    with pytest.raises(Standard_OutOfRange):
        _ = curved_occt._unjoined_sheets(circle(2), 2, 3, 0.1, False)  # pyright: ignore[reportPrivateUsage]
    assert calls == 1


def test_repeated_native_array_failure_identifies_the_independent_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDraft(BRepFill_Draft):
        @override
        def Perform(self, *_args: object, **_kwargs: object) -> None:
            raise Standard_OutOfRange("NCollection_Array1::Value")

    monkeypatch.setattr(curved_occt, "BRepFill_Draft", BrokenDraft)
    diagnostics: dict[str, object] = {}
    with pytest.raises(
        occt.KernelError, match="independent draft generator 0"
    ) as caught:
        _ = curved_occt._unjoined_sheets(  # pyright: ignore[reportPrivateUsage]
            circle(2), 2, 3, 0.1, False, diagnostics=diagnostics
        )
    assert diagnostics["per_edge_draft_rings"] == 1
    assert caught.value.__cause__ is not None
    assert isinstance(caught.value.__cause__.__cause__, Standard_OutOfRange)


def test_volume_sweep_closes_counter_without_losing_its_top() -> None:
    polygon = Polygon(
        box(0, 0, 4, 4).exterior.coords, [box(1.9, 1.9, 2.1, 2.1).exterior.coords]
    )
    region = polygon_lines(polygon)
    shape, method = curved_occt.roof(region, 2, 4, 0.2, prefer_fast=False)
    assert method == "surface"
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(15.96, abs=1e-8)
    expected_bottom = 16 + 16 * 0.4 + pi * 0.4**2
    assert occt.area(occt.cross_section(shape, 2)) == pytest.approx(
        expected_bottom, abs=1e-8
    )
    outer_volume = 16 * 2 + 16 * 0.2 * 2**2 / 2 + pi * 0.2**2 * 2**3 / 3
    # The square counter is a small pyramid, ending at z=3.5.
    assert occt.volume(shape) == pytest.approx(outer_volume - 0.04 * 0.5 / 3, rel=1e-8)


def test_volume_sweep_handles_a_closing_narrow_notch() -> None:
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (2.25, 4), (2.25, 1), (1.75, 1), (1.75, 4), (0, 4)]
    )
    shape, _ = curved_occt.roof(polygon_lines(polygon), 2, 4, 0.25, prefer_fast=False)
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(
        polygon.area, abs=1e-8
    )
    assert occt.area(occt.cross_section(shape, 2)) > polygon.area


@pytest.mark.parametrize(
    "polygon",
    [
        Polygon(
            box(0, 0, 5, 5).exterior.coords,
            [box(1, 1, 2, 2).exterior.coords, box(2, 2, 3, 3).exterior.coords],
        ),
        Polygon(
            [(0, 1), (1, 1), (1, 0), (3, 0), (3, 3), (0, 3)],
            [box(1, 1, 2, 2).exterior.coords],
        ),
    ],
)
def test_point_touching_counter_boundaries_remain_explicit(polygon: Polygon) -> None:
    region = polygon_lines(polygon)
    face = curved_occt.polygon_face(region, 4)
    assert occt.valid(face)
    assert occt.area(face) == pytest.approx(polygon.area, abs=1e-9)
    shape, _ = curved_occt.roof(region, 2, 4, 0.25, prefer_fast=False)
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(
        polygon.area, abs=1e-8
    )


@pytest.mark.parametrize("consolidate_draft", [False, True])
def test_curved_counter_closes_below_its_preserved_top(consolidate_draft: bool) -> None:
    region = CurvePolygon(circle(5), (circle(0.1),))
    diagnostics: dict[str, object] = {}
    shape, method = curved_occt.roof(
        region, 2, 4, 0.2, diagnostics=diagnostics, consolidate_draft=consolidate_draft
    )
    assert method == "surface"
    assert diagnostics["repaired_draft_cells"] == 3
    assert occt.valid(shape)
    expected = curved_occt.polygon_face(region, 4)
    top = occt.cross_section(shape, 4)
    assert occt.area(occt.cut(expected, top)) < 1e-8
    assert occt.area(occt.cut(top, expected)) < 1e-8
    patch = box(-0.01, -0.01, 0.01, 0.01)
    upper = occt.common(occt.cross_section(shape, 3.75), occt.polygon_face(patch, 3.75))
    lower = occt.common(occt.cross_section(shape, 3), occt.polygon_face(patch, 3))
    assert occt.area(upper) == pytest.approx(0, abs=1e-9)
    assert occt.area(lower) == pytest.approx(patch.area, abs=1e-9)


def test_curved_counter_fold_resolves_inside_material() -> None:
    # The inward normal sweep of this five-cubic counter folds near its lower
    # closure. That fold is inside filled material, not an exterior cavity wall.
    controls = (
        (
            (-0.1980198019801982, -0.0396039603960503),
            (-0.12266853544861078, 0.1015370674431324),
            (-0.03714631870388985, 0.24463452585041523),
            (0.07352493144571781, 0.3374022636562586),
        ),
        (
            (0.07352493144571781, 0.3374022636562586),
            (0.18419618159532547, 0.43017000146210194),
            (0.32011975068598986, 0.47267014905810356),
            (0.41060772631317377, 0.4047221504151821),
        ),
        (
            (0.41060772631317377, 0.4047221504151821),
            (0.46234303060919046, 0.3658737801015377),
            (0.4992249682545662, 0.2909869933500886),
            (0.5336637627333403, 0.2202794821932912),
        ),
        (
            (0.5336637627333403, 0.2202794821932912),
            (0.45435767993330956, 0.18453670293092728),
            (0.37644844595342875, 0.14880672842264175),
            (0.3020905600860715, 0.11716489121317863),
        ),
        (
            (0.3020905600860715, 0.11716489121317863),
            (0.12860754691064358, 0.04334190394729376),
            (-0.025341734290122986, -0.008139144629240036),
            (-0.1980198019801982, -0.0396039603960503),
        ),
    )
    counter = CurveRing(tuple(BezierSpan(points, False) for points in controls))
    region = CurvePolygon(polygon_lines(box(-1, -1, 1, 1)).exterior, (counter,))
    shape, _ = curved_occt.roof(region, 2, 3, tan(radians(10)))
    assert occt.valid(shape)
    assert occt.topology_counts(shape)["solids"] == 1
    expected_top = curved_occt.polygon_face(region, 3)
    top = occt.cross_section(shape, 3)
    assert occt.area(occt.cut(expected_top, top)) < 1e-8
    assert occt.area(occt.cut(top, expected_top)) < 1e-8
    patch = box(0.309, 0.269, 0.311, 0.271)
    assert occt.area(occt.common(top, occt.polygon_face(patch, 3))) < 1e-9
    lower = occt.cross_section(shape, 2.05)
    assert occt.area(
        occt.common(lower, occt.polygon_face(patch, 2.05))
    ) == pytest.approx(patch.area, abs=1e-9)


def test_cell_repair_rejects_a_compound_before_single_point_classification() -> None:
    shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    state = _cell_repair_state(shape)
    context = ShapeBuild_ReShape()
    assert _cell_repair_unchanged(state, shape, context)
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, shape)
    # Even a one-member compound must not be interpreted as one native cell.
    assert not _cell_repair_unchanged(state, compound, context)


@pytest.mark.parametrize("changed", ["surface", "curve", "vertex_tolerance"])
def test_cell_repair_rejects_geometry_or_precision_changes(changed: str) -> None:
    shape = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    state = _cell_repair_state(shape)
    if changed == "surface":
        # Native handles and topology remain identical, so handle identity alone
        # would miss this in-place support mutation.
        state.faces[0].surface.Translate(gp_Vec(0.01, 0, 0))
    elif changed == "curve":
        state.edges[0].curve.Translate(gp_Vec(0.01, 0, 0))
    else:
        vertex = state.vertices[0]
        BRep_Builder().UpdateVertex(vertex.vertex, vertex.tolerance * 2)
    assert not _cell_repair_unchanged(state, shape, ShapeBuild_ReShape())


def test_convex_rounding_and_concave_intersection_have_expected_volume() -> None:
    polygon = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)])
    shape, _ = curved_occt.roof(polygon_lines(polygon), 2, 4, 0.2)
    # Five convex quarter-circles add round corners; the one concave corner
    # trims the two planar walls at their intersection, removing a square.
    expected = 7 * 2 + 16 * 0.2 * 2**2 / 2 + (5 * pi / 4 - 1) * 0.2**2 * 2**3 / 3
    assert occt.volume(shape) == pytest.approx(expected, rel=1e-9)


def _physical_brep(shape: TopoDS_Shape) -> bytes:
    stream = BytesIO()
    BRepTools.Write_s(shape, stream, False, False, TopTools_FormatVersion_VERSION_3)
    return stream.getvalue()


@pytest.fixture
def sibling_cells() -> list[TopoDS_Shape]:
    """Three adjacent slabs sharing native faces, with no first/last adjacency."""
    arguments, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
    _ = arguments.Append(BRepPrimAPI_MakeBox(3, 1, 1).Shape())
    for x in (1, 2):
        plane = gp_Pln(gp_Pnt(x, 0, 0), gp_Dir(1, 0, 0))
        _ = tools.Append(BRepBuilderAPI_MakeFace(plane, -1, 2, -1, 2).Face())
    splitter = BOPAlgo_Splitter()
    splitter.SetArguments(arguments)
    splitter.SetTools(tools)
    splitter.SetNonDestructive(True)
    splitter.Perform()
    assert not splitter.HasErrors()
    solids = TopExp_Explorer(splitter.Shape(), TopAbs_SOLID)
    cells: list[TopoDS_Shape] = []
    while solids.More():
        cells.append(solids.Current())
        solids.Next()
    assert len(cells) == 3
    return sorted(cells, key=lambda cell: _bounds(cell)[0][0])


@pytest.fixture
def failed_ordinary_fuse(monkeypatch: pytest.MonkeyPatch) -> occt.KernelError:
    error = occt.KernelError("forced ordinary fuse failure")

    def unavailable(_shapes: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        raise error

    monkeypatch.setattr(occt, "fuse", unavailable)
    return error


@pytest.mark.usefixtures("failed_ordinary_fuse")
@pytest.mark.parametrize("count", [2, 3])
def test_shared_sibling_cells_glue_after_ordinary_union_fails(
    sibling_cells: list[TopoDS_Shape],
    count: int,
) -> None:
    cells = sibling_cells[:count]
    before = [_physical_brep(cell) for cell in cells]
    diagnostics: dict[str, object] = {}
    result = _fuse_draft_cells(cells, diagnostics)
    assert occt.valid(result)
    assert occt.topology_counts(result)["solids"] == 1
    assert occt.volume(result) == pytest.approx(count, abs=1e-10)
    assert occt.area(result) == pytest.approx(4 * count + 2, abs=1e-10)
    assert occt.area(occt.cross_section(result, 0.5)) == pytest.approx(count, abs=1e-10)
    assert [_physical_brep(cell) for cell in cells] == before
    assert diagnostics["glued_draft_cells"] == count


@pytest.mark.usefixtures("failed_ordinary_fuse")
@pytest.mark.parametrize(
    "kind", ["copied", "independent", "overlap", "duplicate", "disconnected"]
)
def test_partition_glue_rejects_cells_without_a_disjoint_sibling_partition(
    sibling_cells: list[TopoDS_Shape],
    kind: str,
) -> None:
    if kind == "copied":
        cells = [
            sibling_cells[0],
            BRepBuilderAPI_Copy(sibling_cells[1], True, False).Shape(),
        ]
    elif kind == "independent":
        cells = [BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 1, 1, 1).Shape() for x in (0, 1)]
    elif kind == "overlap":
        cells = [
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 1, 1, 1).Shape() for x in (0, 0.5)
        ]
    elif kind == "duplicate":
        cells = [sibling_cells[0], sibling_cells[0]]
    else:
        cells = [sibling_cells[0], sibling_cells[2]]
    before = [_physical_brep(cell) for cell in cells]
    diagnostics: dict[str, object] = {}
    with pytest.raises(occt.KernelError):
        _ = _fuse_draft_cells(cells, diagnostics)
    assert "glued_draft_cells" not in diagnostics
    assert [_physical_brep(cell) for cell in cells] == before


@pytest.mark.parametrize("stage", ["ordinary-failure", "ordinary-success", "glue"])
def test_partition_union_rejects_input_precision_mutation(
    monkeypatch: pytest.MonkeyPatch,
    sibling_cells: list[TopoDS_Shape],
    stage: str,
) -> None:
    original_fuse = occt.fuse
    original_boolean = occt._boolean  # pyright: ignore[reportPrivateUsage]
    vertex = _cell_repair_state(sibling_cells[0]).vertices[0]
    mutations = 0

    def mutate() -> None:
        nonlocal mutations
        mutations += 1
        BRep_Builder().UpdateVertex(vertex.vertex, max(1e-6, 2 * vertex.tolerance))

    def ordinary(shapes: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        if stage.startswith("ordinary"):
            mutate()
        if stage == "ordinary-success":
            return original_fuse(shapes)
        raise occt.KernelError("forced ordinary fuse failure")

    def after_boolean(
        operation: BRepAlgoAPI_BooleanOperation,
        arguments: Sequence[TopoDS_Shape],
        tools: Sequence[TopoDS_Shape],
        name: str,
    ) -> TopoDS_Shape:
        result = original_boolean(operation, arguments, tools, name)
        if stage == "glue" and name == "native partition glue":
            mutate()
        return result

    monkeypatch.setattr(occt, "fuse", ordinary)
    monkeypatch.setattr(occt, "_boolean", after_boolean)
    with pytest.raises(occt.KernelError):
        _ = _fuse_draft_cells(sibling_cells)
    assert mutations == 1


@pytest.mark.parametrize("error_type", [MemoryError, Standard_OutOfMemory])
@pytest.mark.parametrize("wrapped", [False, True])
def test_partition_memory_failure_does_not_start_glue(
    monkeypatch: pytest.MonkeyPatch,
    sibling_cells: list[TopoDS_Shape],
    error_type: type[Exception],
    wrapped: bool,
) -> None:
    cause = error_type("native memory exhausted")
    error = occt.KernelError("native fuse failed") if wrapped else cause
    if wrapped:
        error.__cause__ = cause

    def unavailable(_shapes: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        raise error

    def unexpected() -> None:
        pytest.fail("memory failure must not start another Boolean")

    monkeypatch.setattr(occt, "fuse", unavailable)
    monkeypatch.setattr(curved_occt, "BRepAlgoAPI_Fuse", unexpected)
    with pytest.raises(type(error)) as failed:
        _ = _fuse_draft_cells(sibling_cells)
    assert failed.value is error


@pytest.mark.parametrize("first_failure", ["construction", "top-check"])
def test_curved_roof_retries_preimprint_and_keeps_final_checks(
    monkeypatch: pytest.MonkeyPatch, first_failure: str
) -> None:
    region = CurvePolygon(circle(2), (circle(0.5),))
    solid = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    calls: list[bool] = []
    checks: list[str] = []

    def surface(
        *_args: object, preimprint: bool = False, **_kwargs: object
    ) -> TopoDS_Shape:
        calls.append(preimprint)
        if not preimprint and first_failure == "construction":
            raise occt.KernelError("default curved construction failed")
        return solid

    def top_check(*_args: object) -> None:
        checks.append("top")
        if len(calls) == 1 and first_failure == "top-check":
            raise occt.KernelError("default curved top changed")

    def section_check(*_args: object) -> None:
        checks.append("sections")

    monkeypatch.setattr(curved_occt, "_surface_region", surface)
    monkeypatch.setattr(curved_occt, "_check_roof", top_check)
    monkeypatch.setattr(curved_occt, "_check_sections", section_check)
    diagnostics: dict[str, object] = {}
    result, method = curved_occt.roof(region, 2, 3, 0.1, diagnostics=diagnostics)
    assert result.IsSame(solid) and method == "surface-preimprint"
    assert calls == [False, True]
    assert checks[-2:] == ["top", "sections"]
    assert isinstance(diagnostics["surface_failure"], str)


def test_curved_roof_preserves_both_native_failure_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = occt.KernelError("default curved construction failed")
    retry = occt.KernelError("pair imprint failed")

    def surface(
        *_args: object, preimprint: bool = False, **_kwargs: object
    ) -> TopoDS_Shape:
        raise retry if preimprint else original

    monkeypatch.setattr(curved_occt, "_surface_region", surface)
    diagnostics: dict[str, object] = {}
    with pytest.raises(occt.KernelError) as failed:
        _ = curved_occt.roof(
            CurvePolygon(circle(2)), 2, 3, 0.1, diagnostics=diagnostics
        )
    assert str(original) in str(failed.value) and str(retry) in str(failed.value)
    assert failed.value.__cause__ is retry
    assert diagnostics["surface_failure"] == str(original)
    assert diagnostics["preimprint_failure"] == str(retry)


@pytest.mark.parametrize("failed_check", ["_check_roof", "_check_sections"])
def test_curved_preimprint_result_must_pass_every_final_check(
    monkeypatch: pytest.MonkeyPatch, failed_check: str
) -> None:
    calls: list[bool] = []
    solid = BRepPrimAPI_MakeBox(1, 1, 1).Shape()

    def surface(
        *_args: object, preimprint: bool = False, **_kwargs: object
    ) -> TopoDS_Shape:
        calls.append(preimprint)
        if not preimprint:
            raise occt.KernelError("default curved construction failed")
        return solid

    def accepted(*_args: object) -> None:
        pass

    def rejected(*_args: object) -> None:
        raise occt.KernelError("fallback physical check failed")

    monkeypatch.setattr(curved_occt, "_surface_region", surface)
    monkeypatch.setattr(curved_occt, "_check_roof", accepted)
    monkeypatch.setattr(curved_occt, "_check_sections", accepted)
    monkeypatch.setattr(curved_occt, failed_check, rejected)
    diagnostics: dict[str, object] = {}
    with pytest.raises(occt.KernelError, match="fallback physical check failed"):
        _ = curved_occt.roof(
            CurvePolygon(circle(2)), 2, 3, 0.1, diagnostics=diagnostics
        )
    assert calls == [False, True]
    assert diagnostics["preimprint_failure"] == "fallback physical check failed"


@pytest.mark.parametrize("error_type", [MemoryError, Standard_OutOfMemory])
@pytest.mark.parametrize("stage", ["default", "preimprint"])
def test_curved_roof_propagates_wrapped_memory_failures_without_retry(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception], stage: str
) -> None:
    error = occt.KernelError("native memory failure")
    wrapped = occt.KernelError("native Boolean failed")
    wrapped.__cause__ = error_type("native memory exhausted")
    error.__cause__ = wrapped
    calls: list[bool] = []

    def surface(
        *_args: object, preimprint: bool = False, **_kwargs: object
    ) -> TopoDS_Shape:
        calls.append(preimprint)
        if stage == "preimprint" and not preimprint:
            raise occt.KernelError("default curved construction failed")
        raise error

    monkeypatch.setattr(curved_occt, "_surface_region", surface)
    with pytest.raises(occt.KernelError) as failed:
        _ = curved_occt.roof(CurvePolygon(circle(2)), 2, 3, 0.1)
    assert failed.value is error
    assert calls == ([False] if stage == "default" else [False, True])


def test_native_pair_preimprint_retains_a_notched_polygon_top_and_expansion() -> None:
    # Repeated narrow notches expose an intersection missing from the first
    # native batch arrangement. Pair imprints retain every selected line.
    polygon = Polygon(
        [
            (67.00848656294201, 165.66478076379067),
            (65.24045261669023, 166.018387553041),
            (65.94766619519095, 166.018387553041),
            (65.94766619519095, 166.1951909476662),
            (68.06930693069307, 166.1951909476662),
            (62.94200848656294, 166.72560113154174),
            (62.94200848656294, 166.54879773691655),
            (63.118811881188115, 166.54879773691655),
            (63.118811881188115, 166.37199434229137),
            (64.002828854314, 166.37199434229137),
            (62.765205091937766, 165.84158415841583),
        ]
    )
    slope = tan(radians(10))
    diagnostics: dict[str, object] = {}
    shape, method = curved_occt.roof(
        polygon_lines(polygon), 2, 4, slope, diagnostics=diagnostics
    )
    assert method == "surface-preimprint"
    assert isinstance(diagnostics["surface_failure"], str)
    assert cast("int", diagnostics["preimprint_pairs"]) > 0
    assert diagnostics["preimprint_pairs_processed"] == diagnostics["preimprint_pairs"]
    assert cast("int", diagnostics["preimprint_output_faces"]) > 0
    assert occt.valid(shape)
    assert occt.topology_counts(shape)["solids"] == 1
    low, high = curved_occt._bounds(shape)  # pyright: ignore[reportPrivateUsage]
    assert low[2] == pytest.approx(2, abs=1e-7)
    assert high[2] == pytest.approx(4, abs=1e-7)
    top = occt.cross_section(shape, 4)
    expected = occt.polygon_face(polygon, 4)
    assert occt.area(occt.cut(expected, top)) < 1e-8
    assert occt.area(occt.cut(top, expected)) < 1e-8
    assert occt.area(occt.cross_section(shape, 2)) == pytest.approx(
        polygon.buffer(2 * slope, quad_segs=2048).area, abs=2e-6
    )
    _, step = _step(shape)
    assert step["valid"] is True


def test_pair_preimprint_does_not_accept_an_unresolved_native_cell() -> None:
    polygon = Polygon(
        [
            (0, 0),
            (-1.77, 0.35),
            (-1.06, 0.35),
            (-1.06, 0.53),
            (1.06, 0.53),
            (-4.07, 1.06),
            (-4.07, 0.88),
            (-3.89, 0.88),
            (-3.89, 0.71),
            (-3.01, 0.71),
            (-4.24, 0.18),
        ]
    )
    diagnostics: dict[str, object] = {}
    with pytest.raises(occt.KernelError):
        _ = curved_occt.roof(
            polygon_lines(polygon), 2, 4, tan(radians(10)), diagnostics=diagnostics
        )
    assert isinstance(diagnostics["surface_failure"], str)
    assert isinstance(diagnostics["preimprint_failure"], str)


@pytest.mark.parametrize("error_type", [MemoryError, Standard_OutOfMemory])
def test_memory_failure_does_not_start_another_native_construction(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    calls = 0

    def out_of_memory(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise error_type("native construction memory exhausted")

    monkeypatch.setattr(curved_occt, "_surface_region", out_of_memory)
    with pytest.raises(error_type):
        _ = curved_occt.roof(polygon_lines(box(0, 0, 1, 1)), 2, 4, 0.2)
    assert calls == 1


def test_material_box_bound_covers_the_whole_height() -> None:
    source = curved_occt.polygon_face(CurvePolygon(circle(2), (circle(0.1),)), 4)
    inside = occt.prism(box(-0.01, -0.01, 0.01, 0.01), 2, 2.1)
    crosses_cavity = occt.prism(box(-0.01, -0.01, 0.01, 0.01), 2, 3.9)
    outside = occt.prism(box(3, 0, 3.1, 0.1), 2, 2.1)
    assert curved_occt._material_box_clearance(inside, source, 4, 0.2) > 0  # pyright: ignore[reportPrivateUsage]
    assert curved_occt._material_box_clearance(crosses_cavity, source, 4, 0.2) < 0  # pyright: ignore[reportPrivateUsage]
    assert curved_occt._material_box_clearance(outside, source, 4, 0.2) < 0  # pyright: ignore[reportPrivateUsage]


def test_separate_volume_tapers_merge_below_preserved_tops() -> None:
    left, left_method = curved_occt.roof(polygon_lines(box(0, 0, 4, 4)), 2, 4, 0.25)
    right, right_method = curved_occt.roof(
        polygon_lines(box(4.2, 0, 8.2, 4)), 2, 4, 0.25
    )
    assert left_method == right_method == "surface"
    joined = occt.fuse([left, right])
    assert occt.valid(joined)
    assert occt.topology_counts(joined)["solids"] == 1
    assert occt.area(occt.cross_section(joined, 4)) == pytest.approx(32, abs=1e-8)
    bridge = box(4.075, 1, 4.125, 3)
    assert occt.area(
        occt.common(occt.cross_section(joined, 4), occt.polygon_face(bridge, 4))
    ) == pytest.approx(0, abs=1e-9)
    assert occt.area(
        occt.common(occt.cross_section(joined, 2), occt.polygon_face(bridge, 2))
    ) == pytest.approx(bridge.area, abs=1e-9)


def test_line_spans_use_endpoints_instead_of_control_polygon() -> None:
    original = polygon_lines(box(0, 0, 4, 4))
    changed = CurveRing(
        tuple(
            replace(
                span,
                control_points=(
                    span.control_points[0],
                    (1e6, 1e6),
                    (-1e6, -1e6),
                    span.control_points[3],
                ),
            )
            for span in original.exterior.spans
        )
    )
    face = curved_occt.polygon_face(CurvePolygon(changed))
    assert occt.area(face) == pytest.approx(16, abs=1e-9)


@pytest.mark.parametrize(
    "z0,z1,slope", [(2, 2, 0.1), (4, 2, 0.1), (2, 4, -0.1), (2, 4, float("inf"))]
)
def test_invalid_parameters_fail_before_native_sweep(
    z0: float, z1: float, slope: float
) -> None:
    with pytest.raises(occt.KernelError):
        _ = curved_occt.roof(CurvePolygon(circle(5)), z0, z1, slope)
