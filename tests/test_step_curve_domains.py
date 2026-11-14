"""STEP retains bounded spline geometry without changing loose native vertices."""

# pyright: reportPrivateUsage=false

import pytest
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeEdge
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
from OCP.Geom import Geom_Circle, Geom_TrimmedCurve
from OCP.GeomConvert import GeomConvert
from OCP.gp import gp_Ax2, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_ShapeEnum, TopAbs_VERTEX
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from shapely.geometry import box

from panelchemy import material_partition, occt, step_curve_domains, step_export
from panelchemy.export import ExportError
from panelchemy.separation import SeparationConfig


def entities(shape: TopoDS_Shape, kind: TopAbs_ShapeEnum) -> list[TopoDS_Shape]:
    values = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, kind, values)
    return [values.FindKey(i) for i in range(1, values.Extent() + 1)]


def long_basis_box(*, loose_vertex: bool = False) -> TopoDS_Shape:
    source = occt.prism(box(0, 0, 2, 2), 2, 3)
    builder = BRep_Builder()
    displaced = False
    for item in entities(source, TopAbs_EDGE):
        edge = TopoDS.Edge(item)
        adaptor = BRepAdaptor_Curve(edge)
        first, last = adaptor.FirstParameter(), adaptor.LastParameter()
        location = TopLoc_Location()
        curve = BRep_Tool.Curve_s(edge, location, 0.0, 0.0)
        longer = GeomConvert.CurveToBSplineCurve_s(
            Geom_TrimmedCurve(curve, first - 0.1, last + 0.1)
        )
        builder.UpdateEdge(edge, longer, location, BRep_Tool.Tolerance_s(edge))
        builder.Range(edge, first, last, True)
        p, q = adaptor.Value(first), adaptor.Value(last)
        if loose_vertex and not displaced and abs(p.Z() - q.Z()) > 0.9:
            vertex = (
                TopExp.FirstVertex_s(edge, False)
                if p.Z() < q.Z()
                else TopExp.LastVertex_s(edge, False)
            )
            point = BRep_Tool.Pnt_s(vertex)
            builder.UpdateVertex(
                vertex, gp_Pnt(point.X(), point.Y(), point.Z() - 2e-6), 3e-6
            )
            displaced = True
    assert BRepCheck_Analyzer(source, True, False, True).IsValid()
    return source


def native_metadata(shape: TopoDS_Shape) -> object:
    vertices = [
        (
            BRep_Tool.Pnt_s(TopoDS.Vertex(v)).Coord(),
            BRep_Tool.Tolerance_s(TopoDS.Vertex(v)),
        )
        for v in entities(shape, TopAbs_VERTEX)
    ]
    edges = [
        (
            e.Orientation(),
            BRep_Tool.Range_s(TopoDS.Edge(e)),
            BRep_Tool.Tolerance_s(TopoDS.Edge(e)),
            BRep_Tool.SameParameter_s(TopoDS.Edge(e)),
            BRep_Tool.SameRange_s(TopoDS.Edge(e)),
        )
        for e in entities(shape, TopAbs_EDGE)
    ]
    pcurves: list[object] = []
    for item in entities(shape, TopAbs_FACE):
        face = TopoDS.Face(item)
        for other in entities(face, TopAbs_EDGE):
            edge = TopoDS.Edge(other)
            first, last = BRep_Tool.Range_s(edge, face)
            curve = BRep_Tool.CurveOnSurface_s(edge, face, 0.0, 0.0)
            pcurves.append(
                (
                    first,
                    last,
                    [
                        curve.Value(first + (last - first) * t).Coord()
                        for t in (0, 0.5, 1)
                    ],
                    BRep_Tool.Tolerance_s(face),
                )
            )
    return vertices, edges, pcurves


@pytest.mark.parametrize("located", [False, True])
@pytest.mark.parametrize("reversed_shape", [False, True])
def test_exact_domain_restriction_preserves_native_graph_and_source(
    located: bool,
    reversed_shape: bool,
) -> None:
    source = long_basis_box(loose_vertex=True)
    if reversed_shape:
        source.Reverse()
    if located:
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(10, -7, 0.5))
        source = source.Moved(TopLoc_Location(transform))
    before = material_partition.shape_digest(source)
    metadata = native_metadata(source)
    copied, diagnostics = step_curve_domains.prepare_curve_domains(source)
    assert not copied.IsSame(source)
    assert native_metadata(copied) == metadata
    assert copied.Location().IsEqual(source.Location())
    assert BRepCheck_Analyzer(copied, True, False, True).IsValid()
    assert diagnostics["segmented_edges"] == 12
    for old_item, new_item in zip(
        entities(source, TopAbs_EDGE), entities(copied, TopAbs_EDGE), strict=True
    ):
        old, new = (
            BRepAdaptor_Curve(TopoDS.Edge(old_item)),
            BRepAdaptor_Curve(TopoDS.Edge(new_item)),
        )
        first, last = old.FirstParameter(), old.LastParameter()
        spline = new.BSpline()
        assert (spline.FirstParameter(), spline.LastParameter()) == (first, last)
        for t in (0, 0.25, 0.5, 0.75, 1):
            parameter = first + t * (last - first)
            assert old.Value(parameter).Distance(new.Value(parameter)) < 1e-12
    assert material_partition.shape_digest(source) == before


def test_real_solid_raw_step_loses_trim_but_prepared_step_retains_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = long_basis_box(loose_vertex=True)
    before = material_partition.shape_digest(source)

    def unprepared(shape: TopoDS_Shape) -> tuple[TopoDS_Shape, dict[str, object]]:
        return BRepBuilderAPI_Copy(shape, True, False).Shape(), {}

    with monkeypatch.context() as context:
        context.setattr(
            step_curve_domains,
            "prepare_curve_domains",
            unprepared,
        )
        with pytest.raises(ExportError, match="registration or dimensions"):
            _ = step_export._multipart({"orange": source}, SeparationConfig().palette)
    _, report = step_export._multipart({"orange": source}, SeparationConfig().palette)
    assert report["valid"] is True
    assert report["curve_domain_preparation"] == {
        "orange": {
            "method": "exact-bspline-basis-segmentation",
            "segmented_edges": 12,
            "skipped_periodic_edges": 0,
        }
    }
    assert material_partition.shape_digest(source) == before


def test_periodic_and_analytic_curves_are_not_rewritten() -> None:
    periodic = GeomConvert.CurveToBSplineCurve_s(Geom_Circle(gp_Ax2(), 1))
    assert periodic.IsPeriodic()
    source = BRepBuilderAPI_MakeEdge(periodic).Edge()
    copied, report = step_curve_domains.prepare_curve_domains(source)
    assert report["segmented_edges"] == 0
    assert report["skipped_periodic_edges"] == 1
    assert material_partition.shape_digest(copied) == material_partition.shape_digest(
        source
    )
    _, report = step_curve_domains.prepare_curve_domains(
        occt.prism(box(0, 0, 1, 1), 0, 1)
    )
    assert report["segmented_edges"] == 0


def test_invalid_candidate_is_rejected_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = long_basis_box()
    before = material_partition.shape_digest(source)

    class Reject:
        def __init__(self, *_: object) -> None:
            pass

        def IsValid(self) -> bool:
            return False

    monkeypatch.setattr(step_curve_domains, "BRepCheck_Analyzer", Reject)
    with pytest.raises(occt.KernelError, match="invalidated native CAD"):
        _ = step_curve_domains.prepare_curve_domains(source)
    assert material_partition.shape_digest(source) == before


def test_readback_keeps_original_material_as_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = long_basis_box()

    def read(
        _: bytes, materials: dict[str, TopoDS_Shape], palette: object
    ) -> dict[str, object]:
        assert materials["orange"].IsSame(source)
        assert palette == SeparationConfig().palette
        return {"valid": True}

    monkeypatch.setattr(step_export, "_read_assembly", read)
    _, report = step_export._multipart_once(
        {"orange": source}, SeparationConfig().palette
    )
    assert report["valid"] is True
