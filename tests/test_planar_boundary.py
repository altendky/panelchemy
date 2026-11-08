"""Physical top trims survive inflated solid tolerances without CAD repair."""

from collections.abc import Sequence
from math import inf, nan, pi
from typing import cast

import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.Geom import Geom_Circle
from OCP.Geom2d import Geom2d_Curve
from OCP.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec, gp_Vec2d
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_EXTERNAL,
    TopAbs_FACE,
    TopAbs_INTERNAL,
    TopAbs_Orientation,
    TopAbs_ShapeEnum,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Face, TopoDS_Shape
from shapely.geometry import Polygon, box

from panelchemy import occt
from panelchemy.material_partition import shape_digest
from panelchemy.planar_boundary import top_boundary


def entities(shape: TopoDS_Shape, kind: TopAbs_ShapeEnum) -> list[TopoDS_Shape]:
    explorer = TopExp_Explorer(shape, kind)
    result: list[TopoDS_Shape] = []
    while explorer.More():
        result.append(explorer.Current())
        explorer.Next()
    return result


def compound(shapes: Sequence[TopoDS_Shape]) -> TopoDS_Compound:
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def counter_face() -> TopoDS_Face:
    return occt.polygon_face(
        Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 4, 4).exterior.coords]),
        4,
    )


@pytest.mark.parametrize("placement", ["original", "reversed", "located"])
def test_preserves_counter_orientation_and_location(placement: str) -> None:
    source = counter_face()
    if placement == "reversed":
        source = TopoDS.Face(source.Reversed())
    elif placement == "located":
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(13, 17, 0))
        source = TopoDS.Face(source.Moved(TopLoc_Location(transform)))
    before = shape_digest(source)
    actual = top_boundary(source, 4)
    assert occt.valid(actual)
    assert occt.area(actual) == pytest.approx(96, abs=1e-10)
    faces = entities(actual, TopAbs_FACE)
    assert len(faces) == 1
    assert faces[0].Orientation() == source.Orientation()
    assert len(entities(actual, TopAbs_WIRE)) == 2
    assert occt.area(occt.cut(source, actual)) == 0
    assert occt.area(occt.cut(actual, source)) == 0
    assert shape_digest(source) == before


def test_periodic_curve_and_nonzero_trim_range_survive() -> None:
    circle = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 4), gp_Dir(0, 0, 1)), 2)
    edge = BRepBuilderAPI_MakeEdge(circle, 0.25, 2 * pi + 0.25).Edge()
    source = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, 4), gp_Dir(0, 0, 1)),
        BRepBuilderAPI_MakeWire(edge).Wire(),
        True,
    ).Face()
    before = shape_digest(source)
    actual = top_boundary(source, 4)
    face = TopoDS.Face(entities(actual, TopAbs_FACE)[0])
    copied = TopoDS.Edge(entities(face, TopAbs_EDGE)[0])
    assert BRep_Tool.Range_s(copied) == BRep_Tool.Range_s(edge, source)
    assert BRep_Tool.Range_s(copied, face) == BRep_Tool.Range_s(edge, source)
    assert occt.area(actual) == pytest.approx(4 * pi, abs=1e-10)
    assert shape_digest(source) == before


def test_inflated_vertex_tolerances_do_not_erase_a_tiny_cap() -> None:
    polygon = box(0, 0, 0.02, 0.05)
    source = occt.prism(polygon, 2, 4)
    builder = BRep_Builder()
    for vertex in entities(source, TopAbs_VERTEX):
        builder.UpdateVertex(TopoDS.Vertex(vertex), 0.023)
    before = shape_digest(source)
    actual = top_boundary(source, 4)
    expected = occt.polygon_face(polygon, 4)
    assert occt.valid(actual)
    assert occt.area(actual) == pytest.approx(0.001, abs=1e-12)
    assert occt.area(occt.cut(expected, actual)) == 0
    assert occt.area(occt.cut(actual, expected)) == 0
    assert all(
        BRep_Tool.Tolerance_s(TopoDS.Vertex(v)) <= 1e-7
        for v in entities(actual, TopAbs_VERTEX)
    )
    assert shape_digest(source) == before


def test_omitting_a_tiny_cap_fails_the_strict_area_comparison() -> None:
    large = box(0, 0, 1, 1)
    tiny = box(2, 0, 2.02, 0.05)
    expected = compound([occt.polygon_face(p, 4) for p in (large, tiny)])
    source = occt.prism(large, 2, 4)
    before = shape_digest(source)
    actual = top_boundary(source, 4)
    missing = occt.area(occt.cut(expected, actual))
    assert missing == pytest.approx(tiny.area, abs=1e-12)
    assert missing > (large.area + tiny.area) * 1e-8
    assert shape_digest(source) == before


def test_rejects_a_real_pcurve_gap_instead_of_healing_it() -> None:
    source = TopoDS.Face(BRepBuilderAPI_Copy(counter_face(), True, False).Shape())
    edge = TopoDS.Edge(entities(source, TopAbs_EDGE)[0])
    pcurve = cast("Geom2d_Curve", BRep_Tool.CurveOnSurface_s(edge, source, 0, 0).Copy())
    pcurve.Translate(gp_Vec2d(0.01, 0))
    builder = BRep_Builder()
    builder.UpdateEdge(edge, pcurve, source, 0.02)
    for vertex in entities(source, TopAbs_VERTEX):
        builder.UpdateVertex(TopoDS.Vertex(vertex), 0.02)
    before = shape_digest(source)
    with pytest.raises(occt.KernelError, match="endpoint gap"):
        _ = top_boundary(source, 4)
    assert shape_digest(source) == before


@pytest.mark.parametrize("orientation", [TopAbs_INTERNAL, TopAbs_EXTERNAL])
def test_auxiliary_faces_cannot_authorize_a_missing_cap(
    orientation: TopAbs_Orientation,
) -> None:
    source = counter_face()
    source.Orientation(orientation)
    holder = compound([source])
    before = shape_digest(source)
    holder_before = shape_digest(holder)
    actual = top_boundary(holder, 4)
    assert not entities(actual, TopAbs_FACE)
    assert occt.area(actual) == 0
    assert shape_digest(source) == before
    assert shape_digest(holder) == holder_before


def test_only_horizontal_boundary_faces_at_the_requested_height_are_selected() -> None:
    source = occt.prism(box(0, 0, 1, 1), 2, 4)
    assert occt.area(top_boundary(source, 3)) == 0
    assert occt.area(occt.cross_section(source, 3)) == pytest.approx(1)
    tilted = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, 4), gp_Dir(1e-7, 0, 1)), 0, 1, 0, 1
    ).Face()
    assert occt.area(top_boundary(tilted, 4)) == 0


@pytest.mark.parametrize("height", [inf, -inf, nan])
def test_rejects_nonfinite_height(height: float) -> None:
    with pytest.raises(occt.KernelError, match="finite height"):
        _ = top_boundary(counter_face(), height)
