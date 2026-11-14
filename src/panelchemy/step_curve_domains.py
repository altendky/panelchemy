"""Retain finite BRep trims when STEP omits their parameter intervals."""

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
from OCP.Geom import Geom_BSplineCurve
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from panelchemy import occt


def prepare_curve_domains(
    source: TopoDS_Shape,
) -> tuple[TopoDS_Shape, dict[str, object]]:
    """Exactly segment nonperiodic spline bases on a private graph copy.

    STEP stores edge vertices but omits the BRep parameter range. A loose native
    vertex can otherwise extend the imported trim along a larger spline basis.
    Knot insertion restricts that basis to the existing trim; vertices, pcurves,
    parameters, placements and precision metadata are retained.
    """
    copied = BRepBuilderAPI_Copy(source, True, False).Shape()
    edges = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(copied, TopAbs_EDGE, edges)
    builder = BRep_Builder()
    segmented = periodic = 0
    for index in range(1, edges.Extent() + 1):
        edge = TopoDS.Edge(edges.FindKey(index))
        if BRep_Tool.Degenerated_s(edge):
            continue
        location = TopLoc_Location()
        curve = BRep_Tool.Curve_s(edge, location, 0.0, 0.0)
        if not isinstance(curve, Geom_BSplineCurve):
            continue
        if curve.IsPeriodic():
            periodic += 1
            continue
        adaptor = BRepAdaptor_Curve(edge)
        first, last = adaptor.FirstParameter(), adaptor.LastParameter()
        if curve.FirstParameter() == first and curve.LastParameter() == last:
            continue
        trimmed = curve.Copy()
        assert isinstance(trimmed, Geom_BSplineCurve)
        trimmed.Segment(first, last, 0.0)
        builder.UpdateEdge(edge, trimmed, location, BRep_Tool.Tolerance_s(edge))
        builder.Range(edge, first, last, True)
        segmented += 1
    if segmented and not BRepCheck_Analyzer(copied, True, False, True).IsValid():
        raise occt.KernelError("STEP curve-domain preparation invalidated native CAD")
    return copied, {
        "method": "exact-bspline-basis-segmentation",
        "segmented_edges": segmented,
        "skipped_periodic_edges": periodic,
    }
