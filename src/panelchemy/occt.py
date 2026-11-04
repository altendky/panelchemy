"""OpenCascade solids with preserved plateaus and outward distance tapers."""

from collections.abc import Iterable, Sequence
from math import atan2, hypot, isfinite, pi, sin
from pathlib import Path

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_BooleanOperation,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
)
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeEvolved
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakePrism
from OCP.BRepTools import BRepTools
from OCP.collections import List_TopoDS_Shape
from OCP.GeomAbs import GeomAbs_Arc
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Wire,
)
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

type Point2 = tuple[float, float]


class KernelError(ValueError):
    """The requested construction did not produce verified kernel geometry."""


def _empty() -> TopoDS_Shape:
    result = TopoDS_Compound()
    BRep_Builder().MakeCompound(result)
    return result


def _is_empty(shape: TopoDS_Shape) -> bool:
    return shape.IsNull() or (
        shape.ShapeType() == TopAbs_COMPOUND and not TopoDS_Iterator(shape).More()
    )


def valid(shape: TopoDS_Shape) -> bool:
    """An empty compound is valid; an uninitialized shape is not."""
    return not shape.IsNull() and BRepCheck_Analyzer(shape).IsValid()


def volume(shape: TopoDS_Shape) -> float:
    """Compute signed closed-solid volume from the exact surfaces."""
    if _is_empty(shape):
        return 0.0
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, OnlyClosed=True)
    result = properties.Mass()
    if not isfinite(result):
        raise KernelError("OpenCascade returned a non-finite volume")
    return result


def area(shape: TopoDS_Shape) -> float:
    """Compute surface area, including planar faces returned by cross_section."""
    if _is_empty(shape):
        return 0.0
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, properties)
    result = properties.Mass()
    if not isfinite(result):
        raise KernelError("OpenCascade returned a non-finite area")
    return result


def _normalized(polygon: Polygon) -> Polygon:
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise KernelError("a non-empty valid polygon with positive area is required")
    if polygon.has_z:
        raise KernelError("construction polygons must use two-dimensional coordinates")
    return orient(polygon, sign=1.0)


def _points(coordinates: Iterable[Sequence[float]]) -> list[Point2]:
    """Remove only exact repetitions and straight-line intermediate vertices."""
    points = [(float(point[0]), float(point[1])) for point in coordinates]
    if points and points[0] == points[-1]:
        _ = points.pop()
    points = [
        point
        for index, point in enumerate(points)
        if index == 0 or point != points[index - 1]
    ]
    changed = True
    while changed and len(points) > 3:
        changed = False
        keep: list[Point2] = []
        for index, point in enumerate(points):
            previous, following = points[index - 1], points[(index + 1) % len(points)]
            dx1, dy1 = point[0] - previous[0], point[1] - previous[1]
            dx2, dy2 = following[0] - point[0], following[1] - point[1]
            if dx1 * dy2 == dy1 * dx2 and dx1 * dx2 + dy1 * dy2 > 0:
                changed = True
            else:
                keep.append(point)
        points = keep
    if len(points) < 3 or any(
        not isfinite(value) for point in points for value in point
    ):
        raise KernelError("polygon rings require at least three finite vertices")
    return points


def _rings(polygon: Polygon) -> list[list[Point2]]:
    return [
        _points(polygon.exterior.coords),
        *[_points(ring.coords) for ring in polygon.interiors],
    ]


def _wire(points: Sequence[gp_Pnt]) -> TopoDS_Wire:
    builder = BRepBuilderAPI_MakePolygon()
    for point in points:
        builder.Add(point)
    builder.Close()
    if not builder.IsDone():
        raise KernelError("OpenCascade could not construct a closed wire")
    return builder.Wire()


def _face(points: Sequence[gp_Pnt]) -> TopoDS_Face:
    builder = BRepBuilderAPI_MakeFace(_wire(points))
    if not builder.IsDone():
        raise KernelError("OpenCascade could not construct a planar face")
    return builder.Face()


def polygon_face(polygon: Polygon, z: float = 0) -> TopoDS_Face:
    """Construct the exact face, including point-touching counter boundaries.

    Independently built wires do not share vertices at tangencies, so a valid
    planar cell union can initially fail the kernel's wire-intersection check.
    A native planar Boolean arrangement shares that topology without moving
    edges or removing any counter. It must still produce exactly one face.
    """
    if not isfinite(z):
        raise KernelError("face height must be finite")
    polygon = _normalized(polygon)
    rings = _rings(polygon)
    try:
        builder = BRepBuilderAPI_MakeFace(_wire([gp_Pnt(x, y, z) for x, y in rings[0]]))
        for ring in rings[1:]:
            builder.Add(_wire([gp_Pnt(x, y, z) for x, y in ring]))
        result = builder.Face()
    except Exception as error:
        raise KernelError("OpenCascade could not construct the polygon face") from error
    if not valid(result):
        if not polygon.interiors:
            raise KernelError("OpenCascade constructed an invalid polygon face")
        exterior = polygon_face(Polygon(polygon.exterior.coords), z)
        holes = [polygon_face(Polygon(ring.coords), z) for ring in polygon.interiors]
        arranged = _boolean(
            BRepAlgoAPI_Cut(), [exterior], holes, "planar counter arrangement"
        )
        faces = TopExp_Explorer(arranged, TopAbs_FACE)
        if not faces.More():
            raise KernelError("OpenCascade counter arrangement produced no face")
        result = TopoDS.Face(faces.Current())
        faces.Next()
        if faces.More() or not valid(result):
            raise KernelError(
                "OpenCascade counter arrangement must produce one valid face"
            )
    if abs(area(result) - polygon.area) > max(1e-10, polygon.area * 1e-10):
        raise KernelError("OpenCascade polygon face changed the exact source area")
    return result


def _shape_list(shapes: Sequence[TopoDS_Shape]) -> List_TopoDS_Shape:
    result = List_TopoDS_Shape()
    for shape in shapes:
        _ = result.Append(shape)
    return result


def _boolean(
    operation: BRepAlgoAPI_BooleanOperation,
    arguments: Sequence[TopoDS_Shape],
    tools: Sequence[TopoDS_Shape],
    name: str,
) -> TopoDS_Shape:
    try:
        operation.SetArguments(_shape_list(arguments))
        operation.SetTools(_shape_list(tools))
        operation.SetNonDestructive(True)
        operation.Build()
        if not operation.IsDone():
            raise KernelError(f"OpenCascade {name} did not complete")
        result = operation.Shape()
    except Exception as error:
        raise KernelError(f"OpenCascade {name} failed") from error
    if not valid(result):
        raise KernelError(f"OpenCascade {name} returned invalid geometry")
    return result


def fuse(shapes: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
    """Fuse all non-empty arguments in one non-destructive Boolean operation."""
    present = [shape for shape in shapes if not _is_empty(shape)]
    if not present:
        return _empty()
    if len(present) == 1:
        return present[0]
    return _boolean(BRepAlgoAPI_Fuse(), present[:1], present[1:], "union")


def common(left: TopoDS_Shape, right: TopoDS_Shape) -> TopoDS_Shape:
    """Return the exact intersection, including face/solid intersections."""
    if _is_empty(left) or _is_empty(right):
        return _empty()
    return _boolean(BRepAlgoAPI_Common(), [left], [right], "intersection")


def cut(left: TopoDS_Shape, right: TopoDS_Shape) -> TopoDS_Shape:
    """Return the exact difference without modifying either input."""
    if _is_empty(left):
        return _empty()
    if _is_empty(right):
        return left
    return _boolean(BRepAlgoAPI_Cut(), [left], [right], "difference")


def prism(region: Polygon | MultiPolygon, z0: float, z1: float) -> TopoDS_Shape:
    """Extrude planar regions; empty regions yield an empty compound."""
    if not isfinite(z0) or not isfinite(z1) or z1 <= z0:
        raise KernelError("prism heights must be finite and strictly increasing")
    if region.is_empty:
        return _empty()
    polygons = [region] if isinstance(region, Polygon) else list(region.geoms)
    parts: list[TopoDS_Shape] = []
    for polygon in polygons:
        try:
            result = BRepPrimAPI_MakePrism(
                polygon_face(polygon, z0), gp_Vec(0, 0, z1 - z0)
            ).Shape()
        except Exception as error:
            raise KernelError("OpenCascade could not extrude the polygon") from error
        if not valid(result) or volume(result) <= 0:
            raise KernelError("OpenCascade extrusion is not a valid positive solid")
        parts.append(result)
    return fuse(parts)


def _bounds(shape: TopoDS_Shape) -> tuple[float, float, float, float, float, float]:
    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds, False)
    if bounds.IsVoid() or bounds.IsWhole():
        raise KernelError("geometry has no finite bounding box")
    low, high = bounds.CornerMin(), bounds.CornerMax()
    return low.X(), low.Y(), low.Z(), high.X(), high.Y(), high.Z()


def cross_section(shape: TopoDS_Shape, z: float) -> TopoDS_Shape:
    """Intersect the solid with a broad bounded XY face at the exact height."""
    if not isfinite(z):
        raise KernelError("cross-section height must be finite")
    if _is_empty(shape):
        return _empty()
    xmin, ymin, zmin, xmax, ymax, zmax = _bounds(shape)
    if z < zmin or z > zmax:
        return _empty()
    margin = max(xmax - xmin, ymax - ymin, 1.0) * 0.01
    plane = polygon_face(
        box(xmin - margin, ymin - margin, xmax + margin, ymax + margin), z
    )
    return common(shape, plane)


def _check_roof(shape: TopoDS_Shape, polygon: Polygon, z0: float, z1: float) -> None:
    if not valid(shape) or volume(shape) <= 0:
        raise KernelError("taper construction is not a valid positive solid")
    _, _, low, _, _, high = _bounds(shape)
    if abs(low - z0) > 1e-5 or abs(high - z1) > 1e-5:
        raise KernelError("taper construction changed the requested height range")
    expected = polygon_face(polygon, z1)
    actual = cross_section(shape, z1)
    tolerance = max(1e-12, polygon.area * 1e-8)
    missing = area(cut(expected, actual))
    extra = area(cut(actual, expected))
    if missing > tolerance or extra > tolerance:
        raise KernelError(
            f"taper construction changed the top footprint (missing {missing:.9g}, extra {extra:.9g} square millimeters)"
        )


def _evolved_roof(
    polygon: Polygon, z0: float, z1: float, run_per_height: float
) -> TopoDS_Shape:
    profile = BRepBuilderAPI_MakePolygon(
        gp_Pnt(0, -run_per_height * (z1 - z0), z0), gp_Pnt(0, 0, z1)
    ).Wire()
    operation = BRepOffsetAPI_MakeEvolved(
        polygon_face(polygon),
        profile,
        GeomAbs_Arc,
        True,
        True,
        False,
        1e-7,
        False,
        False,
    )
    if not operation.IsDone():
        raise KernelError("OpenCascade evolved taper did not complete")
    return operation.Shape()


def _check_fast_sections(
    shape: TopoDS_Shape, polygon: Polygon, z0: float, z1: float, slope: float
) -> None:
    """Reject unreliable critical caps, missing core, and unsupported sections."""
    previous: TopoDS_Shape | None = None
    previous_z = z0
    vertices = sum(len(ring) for ring in _rings(polygon))
    angle_step = 2 * pi / (4 * 64)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        z = z0 + (z1 - z0) * fraction
        actual = cross_section(shape, z)
        radius = slope * (z1 - z)
        expected_area = polygon.buffer(radius, quad_segs=64).area
        # Polygonal circle chords underestimate round joins. Sum a full-disk
        # error per source vertex as a conservative numerical comparison budget.
        arc_error = vertices * pi * radius**2 * (1 - sin(angle_step) / angle_step)
        tolerance = max(1e-9, polygon.area * 1e-8)
        if abs(area(actual) - expected_area) > arc_error + tolerance:
            raise KernelError(
                "evolved taper does not match an expected distance section"
            )
        if area(cut(polygon_face(polygon, z), actual)) > tolerance:
            raise KernelError("evolved taper lost source material below its plateau")
        if previous is not None:
            transform = gp_Trsf()
            transform.SetTranslation(gp_Vec(0, 0, previous_z - z))
            lowered = BRepBuilderAPI_Transform(actual, transform, True).Shape()
            if area(cut(lowered, previous)) > tolerance:
                raise KernelError("evolved taper contains an unsupported section")
        previous, previous_z = actual, z


def _exact_roof(
    polygon: Polygon, z0: float, z1: float, run_per_height: float
) -> TopoDS_Shape:
    height = z1 - z0
    run = run_per_height * height
    parts = [prism(polygon, z0, z1)]
    for ring in _rings(polygon):
        for index, (x, y) in enumerate(ring):
            following = ring[(index + 1) % len(ring)]
            dx, dy = following[0] - x, following[1] - y
            length = hypot(dx, dy)
            nx, ny = dy / length, -dx / length
            triangle = _face(
                [
                    gp_Pnt(x, y, z0),
                    gp_Pnt(x + nx * run, y + ny * run, z0),
                    gp_Pnt(x, y, z1),
                ]
            )
            parts.append(BRepPrimAPI_MakePrism(triangle, gp_Vec(dx, dy, 0)).Shape())
            previous = ring[index - 1]
            px, py = x - previous[0], y - previous[1]
            previous_length = hypot(px, py)
            turn = px * dy - py * dx
            if turn > 0:
                angle = atan2(turn, px * dx + py * dy)
                axes = gp_Ax2(
                    gp_Pnt(x, y, z0),
                    gp_Dir(0, 0, 1),
                    gp_Dir(py / previous_length, -px / previous_length, 0),
                )
                parts.append(BRepPrimAPI_MakeCone(axes, run, 0, height, angle).Shape())
    return fuse(parts)


def roof(
    polygon: Polygon,
    z0: float,
    z1: float,
    run_per_height: float,
    prefer_fast: bool = True,
) -> tuple[TopoDS_Shape, str]:
    """Preserve the top polygon and expand down with a Euclidean-distance taper.

    Cross-sections at z are the polygon buffered by run_per_height * (z1-z).
    The evolved sweep is fast but can fail at offset topology changes. The exact
    fallback unions outward edge wedges and convex corner cone sectors. Both
    paths verify the original top footprint, including counters, before return.
    """
    if not isfinite(z0) or not isfinite(z1) or z1 <= z0:
        raise KernelError("roof heights must be finite and strictly increasing")
    if not isfinite(run_per_height) or run_per_height < 0:
        raise KernelError("taper run per height must be finite and non-negative")
    if not isfinite(run_per_height * (z1 - z0)):
        raise KernelError("total taper run must be finite")
    polygon = _normalized(polygon)
    if run_per_height == 0:
        result = prism(polygon, z0, z1)
        _check_roof(result, polygon, z0, z1)
        return result, "prism"
    if prefer_fast:
        try:
            result = _evolved_roof(polygon, z0, z1, run_per_height)
            _check_roof(result, polygon, z0, z1)
            _check_fast_sections(result, polygon, z0, z1, run_per_height)
        except Exception:
            # The returned method records the fallback; no feature is removed.
            pass
        else:
            return result, "evolved"
    try:
        result = _exact_roof(polygon, z0, z1, run_per_height)
        _check_roof(result, polygon, z0, z1)
    except Exception as error:
        raise KernelError(
            "OpenCascade could not construct a verified outward taper"
        ) from error
    return result, "exact"


def write_brep(shape: TopoDS_Shape, path: Path) -> None:
    """Write verified exact geometry without substituting a triangulated model."""
    if not valid(shape):
        raise KernelError("cannot write an invalid BRep")
    try:
        written = BRepTools.Write_s(shape, str(path))
    except Exception as error:
        raise KernelError(f"OpenCascade could not write BRep to {path}") from error
    if not written:
        raise KernelError(f"OpenCascade could not write BRep to {path}")
