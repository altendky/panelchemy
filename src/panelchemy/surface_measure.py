"""Expose exact extrusion polynomial intervals to native area quadrature.

OCCT's surface integrator ignores basis-curve knots on linear extrusions. A
verified, disposable tensor-product representation exposes those same knots;
the source shape and its exported geometry are never modified.
"""

from fractions import Fraction
from io import BytesIO
from math import fsum, isfinite

from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.collections import Array1_double, Array1_int, Array2_gp_Pnt
from OCP.Geom import (
    Geom_BSplineCurve,
    Geom_BSplineSurface,
    Geom_SurfaceOfLinearExtrusion,
)
from OCP.Geom2d import Geom2d_Curve
from OCP.GeomTools import GeomTools
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Iterator, TopoDS_Shape


class AreaIntegrationError(ValueError):
    """An exact measurement representation could not be certified."""


def _native_area(shape: TopoDS_Shape) -> float:
    properties = GProp_GProps()
    error = BRepGProp.SurfaceProperties_s(shape, properties, Eps=1e-11)
    result = properties.Mass()
    if not isfinite(result) or not isfinite(error):
        raise AreaIntegrationError("OpenCascade returned a non-finite area")
    return result


def _curve_bytes(curve: Geom2d_Curve) -> bytes:
    stream = BytesIO()
    GeomTools.Write_s(curve, stream)
    return stream.getvalue()


def _tensor_support(
    source: TopoDS_Face,
    extrusion: Geom_SurfaceOfLinearExtrusion,
    basis: Geom_BSplineCurve,
) -> Geom_BSplineSurface:
    """Encode the original full U basis, even when its face is trimmed inside it."""
    if basis.IsRational() or basis.IsPeriodic():
        raise AreaIntegrationError("uncertified extrusion area basis")
    _, _, v0, v1 = BRepTools.UVBounds_s(source)
    if not isfinite(v0) or not isfinite(v1) or v0 >= v1:
        raise AreaIntegrationError("extrusion area needs a finite nonempty V domain")
    poles = Array2_gp_Pnt(1, basis.NbPoles(), 1, 2)
    direction = extrusion.Direction().Coord()
    for i in range(1, basis.NbPoles() + 1):
        coordinates = basis.Pole(i).Coord()
        for j, v in enumerate((v0, v1), 1):
            exact = tuple(
                Fraction(coordinates[k]) + Fraction(v) * Fraction(direction[k])
                for k in range(3)
            )
            rounded = tuple(float(value) for value in exact)
            if any(
                Fraction(value) != target
                for value, target in zip(rounded, exact, strict=True)
            ):
                raise AreaIntegrationError("extrusion area poles need inexact rounding")
            poles.SetValue(i, j, gp_Pnt(*rounded))
    u_knots, u_mults = Array1_double(1, basis.NbKnots()), Array1_int(1, basis.NbKnots())
    for i in range(1, basis.NbKnots() + 1):
        u_knots.SetValue(i, basis.Knot(i))
        u_mults.SetValue(i, basis.Multiplicity(i))
    v_knots, v_mults = Array1_double(1, 2), Array1_int(1, 2)
    for j, v in enumerate((v0, v1), 1):
        v_knots.SetValue(j, v)
        v_mults.SetValue(j, 2)
    return Geom_BSplineSurface(
        poles, u_knots, v_knots, u_mults, v_mults, basis.Degree(), 1
    )


def _certify_support(
    source: TopoDS_Face,
    result: TopoDS_Face,
    extrusion: Geom_SurfaceOfLinearExtrusion,
    basis: Geom_BSplineCurve,
) -> None:
    source_location, result_location = TopLoc_Location(), TopLoc_Location()
    _ = BRep_Tool.Surface_s(source, source_location)
    surface = BRep_Tool.Surface_s(result, result_location)
    if (
        not isinstance(surface, Geom_BSplineSurface)
        or not source_location.IsEqual(result_location)
        or basis.IsRational()
        or basis.IsPeriodic()
        or surface.IsURational()
        or surface.IsVRational()
        or surface.IsUPeriodic()
        or surface.IsVPeriodic()
        or surface.UDegree() != basis.Degree()
        or surface.VDegree() != 1
        or surface.NbUPoles() != basis.NbPoles()
        or surface.NbVPoles() != 2
        or surface.NbUKnots() != basis.NbKnots()
        or surface.NbVKnots() != 2
        or any(surface.VMultiplicity(j) != 2 for j in (1, 2))
        or any(
            surface.Weight(i, j) != 1
            for i in range(1, surface.NbUPoles() + 1)
            for j in (1, 2)
        )
    ):
        raise AreaIntegrationError("uncertified extrusion area support")
    u0, u1, v0, v1 = BRepTools.UVBounds_s(source)
    if not (
        surface.UKnot(1) <= u0 <= u1 <= surface.UKnot(surface.NbUKnots())
        and surface.VKnot(1) <= v0 <= v1 <= surface.VKnot(2)
    ):
        raise AreaIntegrationError("extrusion area support excludes the trim domain")
    for i in range(1, basis.NbKnots() + 1):
        if surface.UKnot(i) != basis.Knot(i) or surface.UMultiplicity(
            i
        ) != basis.Multiplicity(i):
            raise AreaIntegrationError("extrusion area conversion changed U knots")
    # Exact rational arithmetic certifies polynomial coefficients, not samples
    # or a tolerance-based geometry comparison. Unsupported rounded conversions
    # fail explicitly rather than quietly accepting an altered domain.
    direction = extrusion.Direction().Coord()
    for i in range(1, basis.NbPoles() + 1):
        pole = basis.Pole(i).Coord()
        for j in (1, 2):
            actual = surface.Pole(i, j).Coord()
            v = Fraction(surface.VKnot(j))
            if any(
                Fraction(actual[k]) != Fraction(pole[k]) + v * Fraction(direction[k])
                for k in range(3)
            ):
                raise AreaIntegrationError("extrusion area conversion changed poles")


def _certify_topology(
    source: TopoDS_Shape,
    result: TopoDS_Shape,
    converter: BRepBuilderAPI_Copy,
) -> None:
    old_children, new_children = TopoDS_Iterator(source), TopoDS_Iterator(result)
    while old_children.More() and new_children.More():
        old, new = old_children.Value(), new_children.Value()
        mapped = converter.ModifiedShape(old)
        if (
            not mapped.IsSame(new)
            or old.Orientation() != new.Orientation()
            or old.ShapeType() != new.ShapeType()
        ):
            raise AreaIntegrationError(
                "extrusion area conversion changed trim topology"
            )
        _certify_topology(old, new, converter)
        old_children.Next()
        new_children.Next()
    if old_children.More() or new_children.More():
        raise AreaIntegrationError(
            "extrusion area conversion changed trim multiplicity"
        )


def _measurement_face(
    face: TopoDS_Face,
    extrusion: Geom_SurfaceOfLinearExtrusion,
    basis: Geom_BSplineCurve,
) -> TopoDS_Face:
    # Seam selection depends on face orientation. Normalize both faces before
    # copying their two oriented seam representations.
    source = TopoDS.Face(face.Oriented(TopAbs_FORWARD))
    converter = BRepBuilderAPI_Copy(source, True, False)
    result = TopoDS.Face(converter.Shape().Oriented(TopAbs_FORWARD))
    if result.IsPartner(source):
        raise AreaIntegrationError("extrusion area adapter did not copy the face")
    _certify_topology(source, result, converter)
    builder = BRep_Builder()
    location = TopLoc_Location()
    _ = BRep_Tool.Surface_s(source, location)
    builder.UpdateFace(
        result,
        _tensor_support(source, extrusion, basis),
        location,
        BRep_Tool.Tolerance_s(source),
    )
    _certify_support(source, result, extrusion, basis)
    seen: list[TopoDS_Edge] = []
    edges = TopExp_Explorer(source, TopAbs_EDGE)
    while edges.More():
        edge = TopoDS.Edge(edges.Current())
        edges.Next()
        if any(edge.IsSame(previous) for previous in seen):
            continue
        seen.append(edge)
        mapped = TopoDS.Edge(converter.ModifiedShape(edge))
        if mapped.IsPartner(edge):
            raise AreaIntegrationError("extrusion area adapter did not copy an edge")
        forward = TopoDS.Edge(edge.Oriented(TopAbs_FORWARD))
        curve = BRep_Tool.CurveOnSurface_s(forward, source, 0.0, 0.0)
        if BRep_Tool.IsClosed_s(edge, source):
            reverse = TopoDS.Edge(edge.Oriented(TopAbs_REVERSED))
            other = BRep_Tool.CurveOnSurface_s(reverse, source, 0.0, 0.0)
            builder.UpdateEdge(
                mapped, curve, other, result, BRep_Tool.Tolerance_s(edge)
            )
        else:
            builder.UpdateEdge(mapped, curve, result, BRep_Tool.Tolerance_s(edge))
        builder.Range(mapped, result, *BRep_Tool.Range_s(edge, source))
    edges = TopExp_Explorer(source, TopAbs_EDGE)
    while edges.More():
        edge = TopoDS.Edge(edges.Current())
        mapped = TopoDS.Edge(converter.ModifiedShape(edge).Oriented(edge.Orientation()))
        if (
            BRep_Tool.IsClosed_s(edge, source) != BRep_Tool.IsClosed_s(mapped, result)
            or BRep_Tool.Range_s(edge, source) != BRep_Tool.Range_s(mapped, result)
            or _curve_bytes(BRep_Tool.CurveOnSurface_s(edge, source, 0.0, 0.0))
            != _curve_bytes(BRep_Tool.CurveOnSurface_s(mapped, result, 0.0, 0.0))
        ):
            raise AreaIntegrationError("extrusion area adapter changed a trim curve")
        edges.Next()
    return result


def surface_area(shape: TopoDS_Shape) -> float:
    """Preserve native face-use multiplicity with certified extrusion corrections."""
    corrections: list[float] = []
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face(faces.Current())
        faces.Next()
        location = TopLoc_Location()
        surface = BRep_Tool.Surface_s(face, location)
        if not isinstance(surface, Geom_SurfaceOfLinearExtrusion):
            continue
        basis = surface.BasisCurve()
        if not isinstance(basis, Geom_BSplineCurve) or basis.NbKnots() <= 2:
            continue
        measured = _measurement_face(face, surface, basis)
        corrections.extend((-_native_area(face), _native_area(measured)))
    return fsum((_native_area(shape), *corrections))
