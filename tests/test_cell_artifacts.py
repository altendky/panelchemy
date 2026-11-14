"""Auxiliary native topology may disappear; physical boundaries must survive."""

# Synthetic mutations exercise the private acceptance boundary deliberately.
# pyright: reportPrivateUsage=false

from collections.abc import Callable
from io import BytesIO
from math import pi
from typing import Literal

import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCP.BRepTools import BRepTools
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
from OCP.gp import gp_Pnt, gp_Vec
from OCP.ShapeFix import ShapeFix_ShapeTolerance
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_INTERNAL,
    TopAbs_SHELL,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape, TopoDS_Wire

from panelchemy import curved_occt as n
from panelchemy import occt


def _bytes(shape: TopoDS_Shape) -> bytes:
    stream = BytesIO()
    BRepTools.Write_s(shape, stream)
    return stream.getvalue()


def _wire(edge: TopoDS_Shape) -> TopoDS_Wire:
    wire = TopoDS_Wire()
    builder = BRep_Builder()
    builder.MakeWire(wire)
    builder.Add(wire, edge)
    return wire


def _box_with_crossing_internals(
    internal_on: Literal["wire", "edge"] = "wire",
) -> TopoDS_Shape:
    shape = BRepPrimAPI_MakeBox(2, 2, 2).Shape()
    face = next(
        TopoDS.Face(f)
        for f in n._entities(shape, TopAbs_FACE)
        if BRepAdaptor_Surface(TopoDS.Face(f)).Plane().Location().Z() == 0
        and abs(BRepAdaptor_Surface(TopoDS.Face(f)).Plane().Axis().Direction().Z()) == 1
    )
    builder = BRep_Builder()
    face.Free(True)
    for a, b in (
        ((0.25, 0.25, 0), (1.75, 1.75, 0)),
        ((0.25, 1.75, 0), (1.75, 0.25, 0)),
    ):
        edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*a), gp_Pnt(*b)).Edge()
        assert edge.Orientation() == TopAbs_FORWARD
        if internal_on == "wire":
            # F/R descendants inherit an INTERNAL parent's auxiliary status.
            wire = _wire(edge).Oriented(TopAbs_INTERNAL)
        else:
            # A F/R wire with no physical edge descendants is auxiliary too.
            wire = _wire(edge.Oriented(TopAbs_INTERNAL))
        builder.Add(face, wire)
    return shape


@pytest.mark.parametrize("internal_on", ["wire", "edge"])
def test_crossing_internal_wires_clean_without_changing_source(
    internal_on: Literal["wire", "edge"],
) -> None:
    source = _box_with_crossing_internals(internal_on)
    before = _bytes(source)
    assert not occt.valid(source)

    result = n._repair_cell(source)

    assert occt.valid(result)
    assert occt.volume(result) == pytest.approx(8)
    assert len(n._entities(result, TopAbs_FACE)) == 6
    assert len(n._entities(result, TopAbs_EDGE)) == 12
    assert _bytes(source) == before
    assert not occt.valid(source)


def test_empty_forward_wires_are_absent_from_physical_topology() -> None:
    shape = _box_with_crossing_internals("edge")
    identities = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    before = n._physical_topology(shape, identities)
    # Native removal prunes the now-empty F/R wires as well as their INTERNAL
    # edges. Those containers never contributed a physical boundary.
    assert len(n._entities(shape, TopAbs_WIRE)) == 8

    BRepTools.RemoveInternals_s(shape, False)

    assert len(n._entities(shape, TopAbs_WIRE)) == 6
    assert n._physical_topology(shape, identities) == before
    assert occt.valid(shape)


def test_internal_ancestor_excludes_forward_descendants() -> None:
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(3, 3, 3), gp_Pnt(4, 4, 4)).Edge()
    wire = _wire(edge)
    auxiliary = TopoDS_Compound()
    shape = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(auxiliary)
    builder.MakeCompound(shape)
    builder.Add(auxiliary, wire)
    builder.Add(shape, auxiliary.Oriented(TopAbs_INTERNAL))
    builder.Add(shape, BRepPrimAPI_MakeBox(1, 1, 1).Shape())
    identities = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()

    _ = n._physical_topology(shape, identities)
    state = n._cell_repair_state(shape, physical=identities)

    assert wire.Orientation() == TopAbs_FORWARD
    assert not identities.Contains(edge)
    assert not identities.Contains(wire)
    assert len(state.faces) == 6
    assert len(state.edges) == 12
    assert len(state.vertices) == 8


def test_internal_first_identity_still_protects_its_physical_use() -> None:
    box = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    edge = TopoDS.Edge(n._entities(box, TopAbs_EDGE)[0])
    shape = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(shape)
    builder.Add(shape, _wire(edge).Oriented(TopAbs_INTERNAL))
    builder.Add(shape, box)
    first = next(e for e in n._entities(shape, TopAbs_EDGE) if e.IsSame(edge))
    assert first.Orientation() == TopAbs_INTERNAL
    identities = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    _ = n._physical_topology(shape, identities)
    state = n._cell_repair_state(shape, physical=identities)
    assert identities.Contains(edge)
    assert len(state.edges) == 12
    assert n._physical_geometry_unchanged(state)

    builder.UpdateEdge(edge, BRep_Tool.Tolerance_s(edge) * 2)

    assert not n._physical_geometry_unchanged(state)


def test_physical_sphere_pole_edges_survive_cleanup() -> None:
    source = BRepPrimAPI_MakeSphere(1).Shape()
    before = _bytes(source)
    degenerate = sum(
        BRep_Tool.Degenerated_s(TopoDS.Edge(e))
        for e in n._entities(source, TopAbs_EDGE)
    )
    assert degenerate > 0

    result = n._remove_internal_cell_artifacts(source)

    assert result is not None and occt.valid(result)
    assert (
        sum(
            BRep_Tool.Degenerated_s(TopoDS.Edge(e))
            for e in n._entities(result, TopAbs_EDGE)
        )
        == degenerate
    )
    assert occt.volume(result) == pytest.approx(4 * pi / 3)
    assert _bytes(source) == before


def _corrupt_physical_boundary(shape: TopoDS_Shape, change: str) -> None:
    face = TopoDS.Face(n._entities(shape, TopAbs_FACE)[0])
    wire = TopoDS.Wire(n._entities(face, TopAbs_WIRE)[0])
    edge = TopoDS.Edge(n._entities(wire, TopAbs_EDGE)[0])
    vertex = TopoDS.Vertex(n._entities(edge, TopAbs_VERTEX)[0])
    builder = BRep_Builder()
    if change in ("delete_edge", "reverse_edge"):
        wire.Free(True)
        builder.Remove(wire, edge)
        if change == "reverse_edge":
            builder.Add(wire, edge.Reversed())
    elif change == "range":
        first, last = BRep_Tool.Range_s(edge)
        builder.Range(edge, first, first + (last - first) * 0.9)
    elif change.startswith("tolerance_"):
        factor = 2 if change.endswith("up") else 0.5
        ShapeFix_ShapeTolerance().SetTolerance(
            edge, BRep_Tool.Tolerance_s(edge) * factor, TopAbs_EDGE
        )
    elif change == "vertex_point":
        point = BRep_Tool.Pnt_s(vertex)
        builder.UpdateVertex(
            vertex,
            gp_Pnt(point.X() + 0.01, point.Y(), point.Z()),
            BRep_Tool.Tolerance_s(vertex),
        )
    elif change == "replace_vertex":
        replacement = BRepBuilderAPI_MakeVertex(BRep_Tool.Pnt_s(vertex)).Vertex()
        edge.Free(True)
        builder.Remove(edge, vertex)
        builder.Add(edge, replacement.Oriented(vertex.Orientation()))
    elif change == "curve":
        curve = BRep_Tool.Curve_s(edge, 0.0, 0.0)
        curve.Translate(gp_Vec(0.01, 0, 0))
    elif change == "surface":
        BRep_Tool.Surface_s(face).Translate(gp_Vec(0.01, 0, 0))
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "delete_edge",
        "reverse_edge",
        "range",
        "tolerance_up",
        "tolerance_down",
        "vertex_point",
        "replace_vertex",
        "curve",
        "surface",
    ],
)
def test_cleanup_rejects_physical_mutation_even_if_native_validity_passes(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    source = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    for entity in n._entities(source, TopAbs_EDGE):
        ShapeFix_ShapeTolerance().SetTolerance(entity, 1e-4, TopAbs_EDGE)
    before = _bytes(source)
    native_remove: Callable[[TopoDS_Shape, bool], None] = BRepTools.RemoveInternals_s
    called = 0

    def faulty_remove(shape: TopoDS_Shape, force: bool) -> None:
        nonlocal called
        called += 1
        assert force is False
        native_remove(shape, force)
        _corrupt_physical_boundary(shape, change)

    monkeypatch.setattr(BRepTools, "RemoveInternals_s", faulty_remove)

    # Validity cannot certify unchanged geometry. Force that independent gate
    # to pass so each deliberate mutation must fail the physical invariant.
    def permissive_valid(_shape: TopoDS_Shape) -> bool:
        return True

    monkeypatch.setattr(occt, "valid", permissive_valid)

    assert n._remove_internal_cell_artifacts(source) is None
    assert called == 1
    assert _bytes(source) == before


def test_auxiliary_cleanup_cannot_accept_an_open_physical_shell() -> None:
    shape = _box_with_crossing_internals()
    shell = n._entities(shape, TopAbs_SHELL)[0]
    shell.Free(True)
    BRep_Builder().Remove(shell, n._entities(shell, TopAbs_FACE)[0])
    before = _bytes(shape)

    assert n._remove_internal_cell_artifacts(shape) is None
    assert _bytes(shape) == before
