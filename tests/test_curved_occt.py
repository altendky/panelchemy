"""Native fitted profiles, regional draft solids and bounded closure cases."""

from dataclasses import replace
from math import pi, radians, tan
from typing import cast

import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GeomAbs import GeomAbs_BSplineCurve
from OCP.gp import gp_Vec
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.Standard import Standard_OutOfMemory
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound
from shapely.geometry import Polygon, box

from panelchemy import curved_occt, occt
from panelchemy.curved_occt import (
    _cell_repair_state,  # pyright: ignore[reportPrivateUsage]
    _cell_repair_unchanged,  # pyright: ignore[reportPrivateUsage]
)
from panelchemy.curves import (
    BezierSpan,
    Controls,
    CurvePolygon,
    CurveRing,
    polygon_lines,
)
from panelchemy.export import _step  # pyright: ignore[reportPrivateUsage]


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
