"""Exact planar boundary copies for validation, independent of solid tolerances."""

from math import isfinite
from typing import cast

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
from OCP.Geom import Geom_Plane
from OCP.Geom2d import Geom2d_Curve
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GeomAPI import GeomAPI
from OCP.gp import gp_Pnt
from OCP.Precision import Precision
from OCP.Standard import Standard_OutOfMemory
from OCP.TopAbs import (
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Vertex,
    TopoDS_Wire,
)

from panelchemy import occt


def _children(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    iterator = TopoDS_Iterator(shape)
    result: list[TopoDS_Shape] = []
    while iterator.More():
        result.append(iterator.Value())
        iterator.Next()
    return result


def _copy_face(source: TopoDS_Face) -> TopoDS_Face:
    canonical = TopoDS.Face(source.Oriented(TopAbs_FORWARD))
    plane = BRepAdaptor_Surface(canonical).Plane()
    precision = Precision.Confusion_s()
    builder = BRep_Builder()
    face = TopoDS_Face()
    builder.MakeFace(face, Geom_Plane(plane), precision)
    vertices = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(canonical, TopAbs_VERTEX, vertices)
    copied_vertices: dict[int, TopoDS_Vertex] = {}

    def vertex(original: TopoDS_Vertex, point: gp_Pnt) -> TopoDS_Vertex:
        if original.IsNull() or not all(isfinite(value) for value in point.Coord()):
            raise occt.KernelError("invalid planar boundary endpoint")
        index = vertices.FindIndex(original)
        if not index:
            raise occt.KernelError("planar boundary endpoint lacks a source vertex")
        if index not in copied_vertices:
            copied_vertices[index] = BRepBuilderAPI_MakeVertex(point).Vertex()
        selected = copied_vertices[index]
        if BRep_Tool.Pnt_s(selected).Distance(point) > precision:
            raise occt.KernelError(
                "planar boundary has an endpoint gap; no repair allowed"
            )
        return selected

    for selected in _children(canonical):
        if selected.ShapeType() != TopAbs_WIRE:
            raise occt.KernelError("planar boundary face contains a non-wire child")
        source_wire = TopoDS.Wire(selected)
        wire = TopoDS_Wire()
        builder.MakeWire(wire)
        for child in _children(source_wire.Oriented(TopAbs_FORWARD)):
            edge = TopoDS.Edge(child)
            natural = TopoDS.Edge(edge.Oriented(TopAbs_FORWARD))
            if BRep_Tool.Degenerated_s(natural):
                raise occt.KernelError("degenerate planar boundary edge")
            low, high = BRep_Tool.Range_s(natural, canonical)
            if not isfinite(low) or not isfinite(high) or high <= low:
                raise occt.KernelError("invalid planar boundary parameter range")
            # Native null handles become None despite the narrower stub type.
            pcurve = cast(
                "Geom2d_Curve | None",
                BRep_Tool.CurveOnSurface_s(natural, canonical, 0, 0),
            )
            if pcurve is None:
                raise occt.KernelError("planar boundary edge has no pcurve")
            curve = GeomAPI.To3d_s(pcurve, plane)
            first = vertex(TopExp.FirstVertex_s(natural), curve.Value(low))
            last = vertex(TopExp.LastVertex_s(natural), curve.Value(high))
            made = BRepBuilderAPI_MakeEdge(curve, first, last, low, high)
            if not made.IsDone():
                raise occt.KernelError(
                    "native planar boundary edge construction failed"
                )
            copied = made.Edge()
            builder.UpdateEdge(
                copied, cast("Geom2d_Curve", pcurve.Copy()), face, precision
            )
            builder.Range(copied, face, low, high)
            if BRep_Tool.Range_s(copied) != (low, high) or BRep_Tool.Range_s(
                copied, face
            ) != (low, high):
                raise occt.KernelError("planar boundary copy changed a parameter range")
            copied.Orientation(edge.Orientation())
            builder.Add(wire, copied)
        wire.Orientation(source_wire.Orientation())
        wire.Closed(source_wire.Closed())
        builder.Add(face, wire)
    face.Orientation(source.Orientation())
    if [wire.Orientation() for wire in _children(face.Oriented(TopAbs_FORWARD))] != [
        wire.Orientation() for wire in _children(canonical)
    ]:
        raise occt.KernelError("planar boundary copy changed wire orientations")
    if not occt.valid(face):
        raise occt.KernelError("planar boundary copy is invalid; no repair allowed")
    before, after = occt.area(source), occt.area(face)
    if (
        not isfinite(before)
        or not isfinite(after)
        or min(before, after) <= 0
        or abs(after - before) > max(1e-10, abs(before) * 1e-8)
    ):
        raise occt.KernelError("planar boundary copy changed its enclosed area")
    return face


def top_boundary(shape: TopoDS_Shape, z: float) -> TopoDS_Compound:
    """Copy physical horizontal boundary faces at ``z`` for strict comparisons.

    This is not an interior cross-section. Source validity and requested height
    remain the caller's checks. Exact planar trims are lifted onto their support
    with fresh vertices, retaining their topology and ordinary native precision.
    Inflated solid vertex tolerances cannot erase tiny caps during validation.
    These copies never replace geometry in the model or exporter.
    """
    if shape.IsNull() or not isfinite(z):
        raise occt.KernelError("planar boundary needs native CAD and a finite height")
    try:
        faces = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
        TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
        result = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(result)
        for index in range(1, faces.Extent() + 1):
            face = TopoDS.Face(faces.FindKey(index))
            if face.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
                continue
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() != GeomAbs_Plane:
                continue
            plane = surface.Plane()
            direction = plane.Axis().Direction()
            if (
                abs(direction.X()) < 1e-12
                and abs(direction.Y()) < 1e-12
                and abs(abs(direction.Z()) - 1) < 1e-12
                and abs(plane.Location().Z() - z) < 1e-9
            ):
                builder.Add(result, _copy_face(face))
        return result
    except (MemoryError, Standard_OutOfMemory, occt.KernelError):
        raise
    except Exception as error:
        raise occt.KernelError(
            f"native planar boundary copy failed: {error}"
        ) from error
