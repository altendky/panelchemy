"""Exact roof geometry, plateau retention, and material-section primitives."""

from collections.abc import Sequence
from math import cos, pi, sin
from pathlib import Path

import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
)
from OCP.BRepTools import BRepTools, BRepTools_ReShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_INTERNAL, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shape, TopoDS_Wire
from shapely import union_all
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box

from panelchemy import occt
from panelchemy.occt import (
    _annotation_boundary_unchanged,  # pyright: ignore[reportPrivateUsage]
    _bounds,  # pyright: ignore[reportPrivateUsage]
    _is_open_internal_wire,  # pyright: ignore[reportPrivateUsage]
    _open_internal_wires,  # pyright: ignore[reportPrivateUsage]
    _without_open_internal_wires,  # pyright: ignore[reportPrivateUsage]
)


def test_annotation_cleanup_keeps_counter_without_redundant_mass_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon(
        box(0, 0, 4, 4).exterior.coords, [box(1, 1, 3, 3).exterior.coords]
    )
    solid = occt.prism(polygon, 0, 2)
    faces = TopExp_Explorer(solid, TopAbs_FACE)
    face = TopoDS_Face()
    while faces.More():
        face = TopoDS.Face(faces.Current())
        if _bounds(face)[2] > 1.999:
            break
        faces.Next()
    assert faces.More()
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(0.2, 0.2, 2), gp_Pnt(0.8, 0.2, 2)).Edge()
    edge.Orientation(TopAbs_INTERNAL)
    annotation = TopoDS_Wire()
    builder = BRep_Builder()
    builder.MakeWire(annotation)
    builder.Add(annotation, edge)
    face.Free(True)
    builder.Add(face, annotation)
    assert occt.valid(solid)
    assert _open_internal_wires(solid)
    native_volume = occt.volume

    def unexpected_volume(_shape: TopoDS_Shape) -> float:
        pytest.fail("exact oriented boundary identity already proves mass preservation")

    with monkeypatch.context() as patched:
        patched.setattr(occt, "volume", unexpected_volume)
        cleaned = _without_open_internal_wires(solid)
    assert occt.valid(cleaned)
    assert not _open_internal_wires(cleaned)
    assert native_volume(cleaned) == pytest.approx(24, abs=1e-10)
    top = occt.cross_section(cleaned, 2)
    assert occt.area(top) == pytest.approx(polygon.area, abs=1e-10)
    counter = occt.polygon_face(box(1, 1, 3, 3), 2)
    assert occt.area(occt.common(top, counter)) == pytest.approx(0, abs=1e-10)


def test_closed_internal_loop_is_retained_even_with_unset_closed_flag() -> None:
    face = occt.polygon_face(box(0, 0, 4, 4))
    boundary = BRepBuilderAPI_MakePolygon()
    for x, y in ((1, 1), (2, 1), (2, 2), (1, 2)):
        boundary.Add(gp_Pnt(x, y, 0))
    boundary.Close()
    loop = TopoDS_Wire()
    builder = BRep_Builder()
    builder.MakeWire(loop)
    edges = TopExp_Explorer(boundary.Wire(), TopAbs_EDGE)
    while edges.More():
        edge = edges.Current()
        edge.Orientation(TopAbs_INTERNAL)
        builder.Add(loop, edge)
        edges.Next()
    assert not loop.Closed()
    builder.Add(face, loop)
    assert not _is_open_internal_wire(loop)
    assert not _open_internal_wires(face)
    assert _without_open_internal_wires(face).IsEqual(face)


@pytest.mark.parametrize("links", [((0, 1), (1, 2), (1, 3)), ((0, 1), (2, 3))])
def test_annotation_cleanup_accepts_internal_trees(
    links: tuple[tuple[int, int], ...],
) -> None:
    vertices = [
        BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, 0)).Vertex()
        for x, y in ((0.5, 0.5), (1.5, 0.5), (1, 1.5), (2, 2))
    ]
    builder = BRep_Builder()
    annotation = TopoDS_Wire()
    builder.MakeWire(annotation)
    for start, end in links:
        edge = BRepBuilderAPI_MakeEdge(vertices[start], vertices[end]).Edge()
        edge.Orientation(TopAbs_INTERNAL)
        builder.Add(annotation, edge)
    face = occt.polygon_face(box(0, 0, 4, 4))
    builder.Add(face, annotation)
    assert _is_open_internal_wire(annotation)
    cleaned = _without_open_internal_wires(face)
    assert occt.valid(cleaned)
    assert not _open_internal_wires(cleaned)
    assert occt.area(cleaned) == pytest.approx(16, abs=1e-10)


@pytest.mark.parametrize(
    "links",
    [
        ((0, 1), (1, 2), (2, 0), (2, 3)),  # Loop with an open tail.
        ((0, 1), (1, 2), (2, 0), (3, 4)),  # Disconnected loop and chain.
        ((0, 1), (0, 1)),  # Two distinct edges close a multigraph cycle.
    ],
)
def test_internal_annotation_with_any_cycle_is_retained(
    links: tuple[tuple[int, int], ...],
) -> None:
    vertices = [
        BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, 0)).Vertex()
        for x, y in ((0.5, 0.5), (1.5, 0.5), (1, 1.5), (2, 2), (2.5, 2.5))
    ]
    builder = BRep_Builder()
    annotation = TopoDS_Wire()
    builder.MakeWire(annotation)
    for start, end in links:
        edge = BRepBuilderAPI_MakeEdge(vertices[start], vertices[end]).Edge()
        edge.Orientation(TopAbs_INTERNAL)
        builder.Add(annotation, edge)
    assert not _is_open_internal_wire(annotation)


def test_annotation_certificate_rejects_removing_a_counter_boundary() -> None:
    face = occt.polygon_face(
        Polygon(box(0, 0, 4, 4).exterior.coords, [box(1, 1, 3, 3).exterior.coords])
    )
    outer = BRepTools.OuterWire_s(face)
    wires = TopExp_Explorer(face, TopAbs_WIRE)
    counter = TopoDS_Wire()
    while wires.More():
        counter = TopoDS.Wire(wires.Current())
        if not counter.IsSame(outer):
            break
        wires.Next()
    assert wires.More()
    reshape = BRepTools_ReShape()
    reshape.Remove(counter)
    filled = reshape.Apply(face)
    assert occt.valid(filled)
    assert not _annotation_boundary_unchanged(face, filled, [counter])


def test_annotation_certificate_rejects_global_wire_removal_outside_a_face() -> None:
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(1, 1, 0), gp_Pnt(2, 1, 0)).Edge()
    edge.Orientation(TopAbs_INTERNAL)
    builder = BRep_Builder()
    annotation = TopoDS_Wire()
    builder.MakeWire(annotation)
    builder.Add(annotation, edge)
    face = occt.polygon_face(box(0, 0, 4, 4))
    builder.Add(face, annotation)
    assembly = TopoDS_Compound()
    builder.MakeCompound(assembly)
    builder.Add(assembly, face)
    # ReShape removal works by identity, including this separate wire use.
    builder.Add(assembly, annotation)
    reshape = BRepTools_ReShape()
    reshape.Remove(annotation)
    cleaned = reshape.Apply(assembly)
    assert not _annotation_boundary_unchanged(assembly, cleaned, [annotation])


def independent_wires_face(
    rings: Sequence[Sequence[occt.Point2]], z: float
) -> TopoDS_Face:
    """Exercise the checked fallback with intentionally unshared contact vertices."""
    wires: list[TopoDS_Wire] = []
    for ring in rings:
        builder = BRepBuilderAPI_MakePolygon()
        for x, y in ring:
            builder.Add(gp_Pnt(x, y, z))
        builder.Close()
        wires.append(builder.Wire())
    face = BRepBuilderAPI_MakeFace(wires[0])
    for wire in wires[1:]:
        face.Add(wire)
    return face.Face()


def test_square_roof_matches_integrated_round_offset_area() -> None:
    polygon = box(0, 0, 4, 4)
    shape, method = occt.roof(polygon, 2, 4, 0.25)
    expected_volume = 16 * 2 + 16 * 0.25 * 2**2 / 2 + pi * 0.25**2 * 2**3 / 3
    assert method == "evolved"
    assert occt.valid(shape)
    assert occt.volume(shape) == pytest.approx(expected_volume, rel=1e-8)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(16)
    assert occt.area(occt.cross_section(shape, 3)) == pytest.approx(
        16 + 16 * 0.25 + pi * 0.25**2
    )


@pytest.mark.parametrize("width", [0.5, 4.0])
def test_counter_retains_its_top_when_lower_hole_closes(width: float) -> None:
    polygon = Polygon(
        box(0, 0, 10, 10).exterior.coords,
        [
            box(
                5 - width / 2, 5 - width / 2, 5 + width / 2, 5 + width / 2
            ).exterior.coords
        ],
    )
    shape, _ = occt.roof(polygon, 2, 4, 0.25)
    assert occt.valid(shape)
    top = occt.cross_section(shape, 4)
    assert occt.area(top) == pytest.approx(100 - width**2)
    assert occt.area(occt.cut(top, occt.polygon_face(polygon, 4))) < 1e-8
    if width == 0.5:
        outer_volume = 100 * 2 + 40 * 0.25 * 2**2 / 2 + pi * 0.25**2 * 2**3 / 3
        assert occt.volume(shape) == pytest.approx(outer_volume - 1 / 12, rel=1e-8)
        assert occt.area(occt.cross_section(shape, 2.5)) == pytest.approx(
            100 + 40 * 0.375 + pi * 0.375**2
        )


def test_narrow_notch_can_close_below_an_unchanged_plateau() -> None:
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (2.25, 4), (2.25, 1), (1.75, 1), (1.75, 4), (0, 4)]
    )
    shape, _ = occt.roof(polygon, 2, 4, 0.25)
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(14.5)
    assert occt.area(occt.cross_section(shape, 3)) > 14.5


def test_thin_stroke_top_is_preserved_by_the_regional_construction() -> None:
    polygon = box(0, 0, 0.01, 4)
    shape, method = occt.roof(polygon, 2, 4, 0.2, prefer_fast=False)
    assert method in {"draft", "surface", "evolved-volume"}
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(0.04, rel=1e-8)
    assert occt.area(occt.cross_section(shape, 2)) == pytest.approx(
        0.04 + 8.02 * 0.4 + pi * 0.4**2
    )


def test_evolved_roof_handles_translated_profiles_and_many_edges() -> None:
    polygon = Polygon(
        [(5 * cos(index * pi / 32), 5 * sin(index * pi / 32)) for index in range(64)]
    )
    reference, _ = occt.roof(polygon, 0.6, 2.3, 0.1)
    moved, method = occt.roof(translate(polygon, xoff=50, yoff=-70), 0.6, 2.3, 0.1)
    assert method == "evolved"
    assert occt.volume(moved) == pytest.approx(occt.volume(reference), rel=1e-8)
    assert occt.area(occt.cross_section(moved, 2.3)) == pytest.approx(polygon.area)


@pytest.mark.parametrize("replacement", [box(0, 0, 10, 10), box(0, 0, 2, 2)])
def test_valid_but_changed_fast_top_is_rejected(
    monkeypatch: pytest.MonkeyPatch, replacement: Polygon
) -> None:
    original = Polygon(
        box(0, 0, 10, 10).exterior.coords, [box(3, 3, 7, 7).exterior.coords]
    )

    def wrong_top(
        _polygon: Polygon, z0: float, z1: float, _run_per_height: float
    ) -> TopoDS_Shape:
        return occt.prism(replacement, z0, z1)

    monkeypatch.setattr(occt, "_evolved_roof", wrong_top)
    shape, method = occt.roof(original, 2, 4, 0.25)
    assert method in {"draft", "surface"}
    assert occt.valid(shape)
    assert occt.area(occt.cross_section(shape, 4)) == pytest.approx(original.area)


def test_zero_taper_and_disjoint_regions_have_registered_prism_sections() -> None:
    first, method = occt.roof(box(0, 0, 2, 3), 2, 4, 0)
    assert method == "prism"
    assert occt.volume(first) == pytest.approx(12)
    assert occt.area(occt.cross_section(first, 2)) == pytest.approx(6)
    assert occt.area(occt.cross_section(first, 4)) == pytest.approx(6)
    second = occt.prism(MultiPolygon([box(0, 0, 2, 3), box(5, 0, 6, 1)]), 2, 4)
    assert occt.valid(second)
    assert occt.volume(second) == pytest.approx(14)
    assert occt.area(occt.cross_section(second, 3)) == pytest.approx(7)


def test_boolean_material_bands_have_disjoint_interiors_and_matching_total_volume() -> (
    None
):
    shape, _ = occt.roof(box(0, 0, 4, 4), 2, 4, 0.25)
    orange = occt.common(shape, occt.prism(box(-1, -1, 5, 5), 2, 3))
    white = occt.common(shape, occt.prism(box(-1, -1, 5, 5), 3, 4))
    assert occt.volume(occt.common(orange, white)) == pytest.approx(0, abs=1e-10)
    assert occt.volume(orange) + occt.volume(white) == pytest.approx(occt.volume(shape))
    assert occt.area(occt.cross_section(orange, 3)) == pytest.approx(
        occt.area(occt.cross_section(white, 3))
    )


def test_empty_regions_remain_explicitly_empty() -> None:
    empty = occt.prism(Polygon(), 0, 2)
    solid = occt.prism(box(0, 0, 2, 2), 0, 2)
    assert occt.valid(empty)
    assert occt.volume(empty) == 0
    assert occt.area(occt.cross_section(empty, 1)) == 0
    assert occt.volume(occt.common(empty, solid)) == 0
    assert occt.volume(occt.cut(solid, empty)) == pytest.approx(8)
    assert occt.volume(occt.fuse([empty, solid])) == pytest.approx(8)


@pytest.mark.parametrize(
    ("z0", "z1", "run"),
    [(2.0, 2.0, 0.1), (3.0, 2.0, 0.1), (2.0, 4.0, -0.1), (2.0, 4.0, float("nan"))],
)
def test_invalid_taper_parameters_fail_before_native_construction(
    z0: float, z1: float, run: float
) -> None:
    with pytest.raises(occt.KernelError):
        _ = occt.roof(box(0, 0, 2, 2), z0, z1, run)


def test_brep_round_trip_preserves_the_checked_solid(tmp_path: Path) -> None:
    shape, _ = occt.roof(box(0, 0, 4, 4), 2, 4, 0.25)
    output = tmp_path / "roof.brep"
    occt.write_brep(shape, output)
    restored = TopoDS_Shape()
    assert BRepTools.Read_s(restored, str(output), BRep_Builder())
    assert occt.valid(restored)
    assert occt.volume(restored) == pytest.approx(occt.volume(shape))


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
        Polygon(box(0, 0, 5, 5).exterior.coords, [box(1, 1, 2, 2).exterior.coords]),
    ],
    ids=["touching-holes", "hole-touching-outer", "disjoint-hole"],
)
def test_exact_face_and_roof_preserve_point_touching_counter_arrangements(
    polygon: Polygon,
) -> None:
    assert polygon.is_valid
    face = occt.polygon_face(polygon, 4)
    assert occt.valid(face)
    assert occt.area(face) == pytest.approx(polygon.area, abs=1e-10)
    prism = occt.prism(polygon, 2, 4)
    assert occt.valid(prism)
    assert occt.volume(prism) == pytest.approx(polygon.area * 2, abs=1e-10)
    # A simple cell-face union is independent of the touching-ring constructor.
    reference = occt.fuse(
        [
            occt.polygon_face(box(x, y, x + 1, y + 1), 4)
            for x in range(5)
            for y in range(5)
            if polygon.covers(box(x, y, x + 1, y + 1))
        ]
    )
    assert occt.area(occt.cut(face, reference)) < 1e-10
    assert occt.area(occt.cut(reference, face)) < 1e-10
    roof, _ = occt.roof(polygon, 2, 4, 0.25)
    assert occt.valid(roof)
    top = occt.cross_section(roof, 4)
    assert occt.area(occt.cut(top, reference)) < 1e-10
    assert occt.area(occt.cut(reference, top)) < 1e-10


def test_counter_arrangement_rejects_a_valid_face_that_loses_the_holes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon(
        box(0, 0, 5, 5).exterior.coords,
        [box(1, 1, 2, 2).exterior.coords, box(2, 2, 3, 3).exterior.coords],
    )
    wrong_face = occt.polygon_face(box(0, 0, 5, 5))

    def lose_counters(*_args: object) -> TopoDS_Shape:
        return wrong_face

    monkeypatch.setattr(occt, "_shared_polygon_face", independent_wires_face)
    monkeypatch.setattr(occt, "_boolean", lose_counters)
    with pytest.raises(occt.KernelError, match="changed the exact source area"):
        _ = occt.polygon_face(polygon)


def test_counter_arrangement_rejects_multiple_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon(
        box(0, 0, 5, 5).exterior.coords,
        [box(1, 1, 2, 2).exterior.coords, box(2, 2, 3, 3).exterior.coords],
    )
    wrong_faces = occt.fuse(
        [occt.polygon_face(box(0, 0, 1, 1)), occt.polygon_face(box(2, 2, 3, 3))]
    )

    def multiple_faces(*_args: object) -> TopoDS_Shape:
        return wrong_faces

    monkeypatch.setattr(occt, "_shared_polygon_face", independent_wires_face)
    monkeypatch.setattr(occt, "_boolean", multiple_faces)
    with pytest.raises(occt.KernelError, match="one valid face"):
        _ = occt.polygon_face(polygon)


def test_direct_contact_builder_failure_uses_checked_counter_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon(
        box(0, 0, 5, 5).exterior.coords, [box(1, 1, 2, 2).exterior.coords]
    )

    def unsupported_multiple_wires(
        rings: Sequence[Sequence[occt.Point2]], z: float
    ) -> TopoDS_Face:
        if len(rings) > 1:
            raise RuntimeError("unsupported direct contact arrangement")
        return independent_wires_face(rings, z)

    monkeypatch.setattr(occt, "_shared_polygon_face", unsupported_multiple_wires)
    face = occt.polygon_face(polygon)
    assert occt.valid(face)
    assert occt.area(face) == pytest.approx(24)
    solid = occt.prism(polygon, 0, 2)
    assert occt.valid(solid)
    assert occt.volume(solid) == pytest.approx(48)


def test_fast_roof_with_correct_top_but_no_expansion_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = box(0, 0, 4, 4)

    def vertical_only(
        region: Polygon, z0: float, z1: float, _slope: float
    ) -> TopoDS_Shape:
        return occt.prism(region, z0, z1)

    monkeypatch.setattr(occt, "_evolved_roof", vertical_only)
    shape, method = occt.roof(polygon, 2, 4, 0.25)
    assert method in {"draft", "surface"}
    assert occt.area(occt.cross_section(shape, 2)) > polygon.area


def test_critical_narrow_mouth_retains_a_supported_bottom_section() -> None:
    geometry = union_all(
        [
            box(0, 0, 2, 4),
            box(0, 0, 5, 1),
            box(3, 3, 5, 4),
            box(4, 0, 5, 4),
        ]
    )
    assert isinstance(geometry, Polygon)
    shape, _ = occt.roof(geometry, 2, 4, 0.25)
    bottom = occt.cross_section(shape, 2)
    assert occt.area(bottom) == pytest.approx(
        geometry.buffer(0.5, quad_segs=64).area, abs=0.001
    )
    assert occt.area(occt.cut(occt.polygon_face(geometry, 2), bottom)) == pytest.approx(
        0, abs=1e-9
    )


@pytest.mark.parametrize(
    "polygon",
    [
        Polygon(box(0, 0, 5, 5).exterior.coords, [box(1, 1, 2, 2).exterior.coords]),
        Polygon(
            box(0, 0, 5, 5).exterior.coords,
            [box(1, 1, 2, 2).exterior.coords, box(2, 2, 3, 3).exterior.coords],
        ),
        Polygon(
            [(0, 1), (1, 1), (1, 0), (3, 0), (3, 3), (0, 3)],
            [box(1, 1, 2, 2).exterior.coords],
        ),
        Polygon(box(0, 0, 4, 4).exterior.coords, [[(2, 4), (1, 2), (3, 2)]]),
        Polygon(
            box(0, 0, 6, 6).exterior.coords,
            [box(1, 1, 3, 3).exterior.coords, [(3, 2), (4, 1), (4, 3)]],
        ),
    ],
    ids=[
        "ordinary",
        "hole-hole-point",
        "hole-outer-corner",
        "hole-outer-edge",
        "hole-hole-edge",
    ],
)
def test_shared_contact_faces_and_prisms_do_not_require_planar_booleans(
    polygon: Polygon, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert polygon.is_valid

    def unexpected_boolean(*_args: object) -> TopoDS_Shape:
        raise AssertionError(
            "shared contact topology must not require a planar Boolean"
        )

    monkeypatch.setattr(occt, "_boolean", unexpected_boolean)
    face = occt.polygon_face(polygon, 1.25)
    assert occt.valid(face)
    assert occt.area(face) == pytest.approx(polygon.area, abs=1e-10)
    solid = occt.prism(polygon, 1.25, 3.25)
    assert occt.valid(solid)
    assert occt.volume(solid) == pytest.approx(polygon.area * 2, abs=1e-10)


@pytest.mark.parametrize(
    "polygon",
    [
        Polygon(box(0, 0, 4, 4).exterior.coords, [[(2, 4), (1, 2), (3, 2)]]),
        Polygon(
            box(0, 0, 6, 6).exterior.coords,
            [box(1, 1, 3, 3).exterior.coords, [(3, 2), (4, 1), (4, 3)]],
        ),
    ],
    ids=["hole-outer-edge", "hole-hole-edge"],
)
def test_mid_edge_noding_preserves_roof_top_against_independent_native_cut(
    polygon: Polygon,
) -> None:
    reference: TopoDS_Shape = occt.polygon_face(Polygon(polygon.exterior.coords), 4)
    for hole in polygon.interiors:
        reference = occt.cut(reference, occt.polygon_face(Polygon(hole.coords), 4))
    face = occt.polygon_face(polygon, 4)
    assert occt.area(occt.cut(face, reference)) < 1e-10
    assert occt.area(occt.cut(reference, face)) < 1e-10
    solid, _ = occt.roof(polygon, 2, 4, 0.25)
    assert occt.valid(solid)
    top = occt.cross_section(solid, 4)
    assert occt.area(occt.cut(top, reference)) < 1e-10
    assert occt.area(occt.cut(reference, top)) < 1e-10
