"""STEP preparation preserves point-touching holes using exact cap partitions."""

from collections.abc import Callable
from io import BytesIO
from math import pi
from typing import cast

import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepFeat import BRepFeat_SplitShape
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Precision import Precision
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Vertex,
    TopoDS_Wire,
)

from panelchemy import material_partition, occt, step_topology
from panelchemy.export import (
    _STEP_LOCK,  # pyright: ignore[reportPrivateUsage]
    _step,  # pyright: ignore[reportPrivateUsage]
)
from panelchemy.step_topology import normalize_step_topology

type Point = tuple[float, float, float]


def _touching_holes() -> TopoDS_Shell:
    """Two square hole loops and four adjacent tapered planar wall faces.

    Share native endpoints/edges explicitly. This small open shell exposes the
    STEP writer's incomplete common-vertex split without an artwork fixture.
    """
    vertices: dict[Point, TopoDS_Vertex] = {}
    edges: dict[tuple[Point, Point], TopoDS_Edge] = {}

    def vertex(point: Point) -> TopoDS_Vertex:
        if point not in vertices:
            vertices[point] = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
        return vertices[point]

    def wire(points: list[Point]) -> TopoDS_Wire:
        maker = BRepBuilderAPI_MakeWire()
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            if (b, a) in edges:
                edge = TopoDS.Edge(edges[b, a].Reversed())
            else:
                if (a, b) not in edges:
                    edges[a, b] = BRepBuilderAPI_MakeEdge(vertex(a), vertex(b)).Edge()
                edge = edges[a, b]
            maker.Add(edge)
        assert maker.IsDone()
        return maker.Wire()

    cap = BRepBuilderAPI_MakeFace(
        wire([(-4, -4, 1), (4, -4, 1), (4, 4, 1), (-4, 4, 1)]), True
    )
    cap.Add(wire([(-2, -2, 1), (-2, 0, 1), (0, 0, 1), (0, -2, 1)]))
    cap.Add(wire([(0, 0, 1), (0, 2, 1), (2, 2, 1), (2, 0, 1)]))
    walls: list[list[Point]] = [
        [(-2, 0, 1), (-2, -0.2, 0), (-0.2, -0.2, 0), (0, 0, 1)],
        [(0, 0, 1), (-0.2, -0.2, 0), (-0.2, -2, 0), (0, -2, 1)],
        [(2, 0, 1), (2, 0.2, 0), (0.2, 0.2, 0), (0, 0, 1)],
        [(0, 0, 1), (0.2, 0.2, 0), (0.2, 2, 0), (0, 2, 1)],
    ]
    result = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(result)
    for points in walls:
        builder.Add(result, BRepBuilderAPI_MakeFace(wire(points), True).Face())
    builder.Add(result, cap.Face())
    assert occt.valid(result)
    return result


def _raw_roundtrip(shape: TopoDS_Shape) -> TopoDS_Shape:
    with _STEP_LOCK:
        writer = STEPControl_Writer()
        previous = Interface_Static.CVal_s("write.step.unit")
        assert Interface_Static.SetCVal_s("write.step.unit", "MM")
        try:
            transfer = cast("Callable[[TopoDS_Shape, object], object]", writer.Transfer)
            assert transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
            stream = BytesIO()
            assert writer.WriteStream(stream) == IFSelect_RetDone
        finally:
            assert Interface_Static.SetCVal_s("write.step.unit", previous)
    reader = STEPControl_Reader()
    read = cast("Callable[[str, BytesIO], object]", reader.ReadStream)
    assert read("touching-holes.step", BytesIO(stream.getvalue())) == IFSelect_RetDone
    reader.SetSystemLengthUnit(1)
    assert reader.TransferRoots() > 0
    return reader.OneShape()


def test_cap_partition_prevents_incomplete_step_vertex_split() -> None:
    source = _touching_holes()
    before = material_partition.shape_digest(source)
    broken = _raw_roundtrip(source)
    assert not occt.valid(broken)
    assert occt.topology_counts(broken)["vertices"] == 18
    normalized, report = normalize_step_topology(source)
    assert occt.valid(normalized)
    assert report["touching_vertices_before"] == 1
    assert report["touching_vertices_after"] == 0
    assert occt.area(normalized) == pytest.approx(occt.area(source), abs=1e-10)
    assert occt.valid(_raw_roundtrip(normalized))
    assert material_partition.shape_digest(source) == before
    # The ordinary fused serializer must also use the general preparation path.
    data, validation = _step(source)
    assert data.startswith(b"ISO-10303-21;")
    assert validation["valid"] is True
    assert material_partition.shape_digest(source) == before


def test_contact_partition_uses_face_coordinates_after_rotation_and_translation() -> (
    None
):
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), pi / 7)
    rotation.SetTranslationPart(gp_Vec(123, -47, 8))
    source = BRepBuilderAPI_Transform(_touching_holes(), rotation, True).Shape()
    before = material_partition.shape_digest(source)
    normalized, report = normalize_step_topology(source)
    assert report["touching_vertices_after"] == 0
    assert occt.valid(_raw_roundtrip(normalized))
    assert occt.area(normalized) == pytest.approx(occt.area(source), abs=1e-10)
    assert material_partition.shape_digest(source) == before


class _IdentityHistory:
    """Truthful feature history for the unchanged copied boundary graph."""

    def Modified(self, shape: TopoDS_Shape) -> list[TopoDS_Shape]:
        return [shape]


@pytest.mark.parametrize("extra_kind", ["edge", "vertex"])
def test_partition_certificate_rejects_loose_geometry(extra_kind: str) -> None:
    source = _touching_holes()
    copier = BRepBuilderAPI_Copy(source, True, False)
    copied = copier.Shape()
    history = cast("BRepFeat_SplitShape", cast("object", _IdentityHistory()))
    bound = Precision.Confusion_s() * 2
    # Establish that identity history itself is acceptable; the extra geometry
    # is the only difference in the rejected candidate below.
    verify = step_topology._verify  # pyright: ignore[reportPrivateUsage]
    verify(source, copied, history, copier, bound)
    extra = (
        BRepBuilderAPI_MakeEdge(gp_Pnt(-3, -3, 0.5), gp_Pnt(-2.5, -3, 0.5)).Edge()
        if extra_kind == "edge"
        else BRepBuilderAPI_MakeVertex(gp_Pnt(-3, -3, 0.5)).Vertex()
    )
    candidate = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(candidate)
    builder.Add(candidate, copied)
    builder.Add(candidate, extra)
    assert occt.valid(candidate)
    assert occt.area(candidate) == occt.area(copied)
    assert occt.volume(candidate) == occt.volume(copied)
    with pytest.raises(occt.KernelError, match="loose"):
        verify(source, candidate, history, copier, bound)


def test_failed_candidate_cannot_mutate_intermediate_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _touching_holes()
    translation = gp_Trsf()
    translation.SetTranslation(gp_Vec(12, 0, 0))
    second = BRepBuilderAPI_Transform(_touching_holes(), translation, True).Shape()
    source = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(source)
    builder.Add(source, first)
    builder.Add(source, second)
    before = material_partition.shape_digest(source)
    partition = step_topology._partition  # pyright: ignore[reportPrivateUsage]
    directions = step_topology._directions  # pyright: ignore[reportPrivateUsage]
    attempts = 0
    direction_calls = 0
    mutation_observed = False

    def repeated_directions(face: object, vertex: object) -> list[float]:
        nonlocal direction_calls
        direction_calls += 1
        selected = directions(cast("TopoDS_Face", face), cast("TopoDS_Vertex", vertex))
        # A second candidate must be available after the injected failure so
        # silently swallowing mutation could actually proceed to another try.
        return selected if direction_calls == 1 else [selected[0], selected[0]]

    def fail_after_first_split(
        working: TopoDS_Shape, contact: object, angle: float, bound: float
    ) -> tuple[TopoDS_Shape, dict[str, object]]:
        nonlocal attempts, mutation_observed
        attempts += 1
        if attempts == 2:
            assert not working.IsSame(source)
            initial = material_partition.shape_digest(working)
            vertex = TopoDS.Vertex(TopExp_Explorer(working, TopAbs_VERTEX).Current())
            builder.UpdateVertex(
                vertex,
                BRep_Tool.Tolerance_s(vertex) + Precision.Confusion_s() / 4,
            )
            mutation_observed = material_partition.shape_digest(working) != initial
            raise occt.KernelError("injected failed native candidate")
        return partition(
            working,
            cast("step_topology._Contact", contact),  # pyright: ignore[reportPrivateUsage]
            angle,
            bound,
        )

    monkeypatch.setattr(step_topology, "_directions", repeated_directions)
    monkeypatch.setattr(step_topology, "_partition", fail_after_first_split)
    with pytest.raises(occt.KernelError, match="mutated"):
        _ = normalize_step_topology(source)
    assert mutation_observed
    assert attempts == 2
    assert material_partition.shape_digest(source) == before


def test_open_wire_detector_locates_step_duplicate_vertex() -> None:
    broken = _raw_roundtrip(_touching_holes())
    assert not occt.valid(broken)
    assert step_topology.open_wire_contact_points(broken) == ((0.0, 0.0, 1.0),)
