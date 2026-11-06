"""Regional native-curve solids, without per-edge closed-solid construction."""

# The legacy polygon adapter dispatches here only inside occt.roof, after both
# modules initialize; the shared low-level helpers do not form a runtime cycle.
# pyright: reportImportCycles=false

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from math import atan, atan2, cos, fsum, hypot, isfinite, pi, sqrt
from typing import cast

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder, BOPAlgo_MakerVolume
from OCP.BOPTools import BOPTools_AlgoTools3D
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepFill import BRepFill_Draft, BRepFill_Modified
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
from OCP.collections import (
    Array1_gp_Pnt,
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    List_TopoDS_Shape,
)
from OCP.Geom import Geom_BezierCurve, Geom_ConicalSurface, Geom_Curve, Geom_Surface
from OCP.GeomConvert import GeomConvert
from OCP.GeomTools import GeomTools
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IntTools import IntTools_Context
from OCP.ShapeBuild import ShapeBuild_ReShape
from OCP.ShapeFix import ShapeFix_Shape
from OCP.Standard import Standard_OutOfMemory
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_IN,
    TopAbs_REVERSED,
    TopAbs_ShapeEnum,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Vertex,
    TopoDS_Wire,
)
from shapely import STRtree
from shapely.geometry import Polygon, box

from panelchemy import occt, spatial_roof
from panelchemy.curves import BezierSpan, CurvePolygon, CurveRing
from panelchemy.curves import polygon_lines as polygon_lines

_SPATIAL_EDGE_THRESHOLD = 2048


def _ring_area(ring: CurveRing) -> float:
    """Three-point Gaussian integration is exact for cubic signed area."""
    values: list[float] = []
    for span in ring.spans:
        p = span.control_points
        if span.is_line:
            values.append((p[0][0] * p[3][1] - p[0][1] * p[3][0]) / 2)
            continue
        for t, weight in (
            (0.5 * (1 - sqrt(3 / 5)), 5 / 9),
            (0.5, 8 / 9),
            (0.5 * (1 + sqrt(3 / 5)), 5 / 9),
        ):
            s = 1 - t
            point = tuple(
                s**3 * p[0][axis]
                + 3 * s * s * t * p[1][axis]
                + 3 * s * t * t * p[2][axis]
                + t**3 * p[3][axis]
                for axis in (0, 1)
            )
            derivative = tuple(
                3 * s * s * (p[1][axis] - p[0][axis])
                + 6 * s * t * (p[2][axis] - p[1][axis])
                + 3 * t * t * (p[3][axis] - p[2][axis])
                for axis in (0, 1)
            )
            values.append(
                weight * (point[0] * derivative[1] - point[1] * derivative[0]) / 4
            )
    return fsum(values)


def _oriented(ring: CurveRing, positive: bool) -> CurveRing:
    if not ring.spans:
        raise occt.KernelError("a fitted boundary requires at least one span")
    for index, span in enumerate(ring.spans):
        if not all(isfinite(value) for point in span.control_points for value in point):
            raise occt.KernelError("fitted curve control points must be finite")
        if (
            span.control_points[-1]
            != ring.spans[(index + 1) % len(ring.spans)].control_points[0]
        ):
            raise occt.KernelError("fitted boundary spans must join exactly and close")
    area = _ring_area(ring)
    if area == 0 or not isfinite(area):
        raise occt.KernelError("fitted boundary has no positive enclosed area")
    if (area > 0) == positive:
        return ring
    return CurveRing(
        tuple(
            BezierSpan(
                (
                    span.control_points[3],
                    span.control_points[2],
                    span.control_points[1],
                    span.control_points[0],
                ),
                span.is_line,
                span.certified_error_mm,
            )
            for span in reversed(ring.spans)
        )
    )


def _wire(
    ring: CurveRing,
    z: float,
    vertices: dict[tuple[float, float], TopoDS_Vertex],
    *,
    draft_compatible: bool = False,
) -> TopoDS_Wire:
    builder = BRepBuilderAPI_MakeWire()
    for span in ring.spans:
        points = span.control_points
        for point in (points[0], points[-1]):
            if point not in vertices:
                vertices[point] = BRepBuilderAPI_MakeVertex(gp_Pnt(*point, z)).Vertex()
        first, last = vertices[points[0]], vertices[points[-1]]
        if span.is_line:
            edge = BRepBuilderAPI_MakeEdge(first, last)
        else:
            poles = Array1_gp_Pnt(1, 4)
            for index, point in enumerate(points, 1):
                poles.SetValue(index, gp_Pnt(*point, z))
            curve = Geom_BezierCurve(poles)
            # MakeDraft's corner machinery rejects some native Bezier/line
            # joins. Exact B-spline conversion preserves the same polynomial,
            # parameters and endpoints while avoiding that native failure.
            if draft_compatible:
                edge = BRepBuilderAPI_MakeEdge(
                    GeomConvert.CurveToBSplineCurve_s(curve), first, last, 0.0, 1.0
                )
            else:
                edge = BRepBuilderAPI_MakeEdge(curve, first, last, 0.0, 1.0)
        if not edge.IsDone():
            raise occt.KernelError(
                "OpenCascade could not construct a fitted curve edge"
            )
        builder.Add(edge.Edge())
    if not builder.IsDone():
        raise occt.KernelError("OpenCascade could not connect the fitted boundary")
    return builder.Wire()


def polygon_face(region: CurvePolygon, z: float = 0) -> TopoDS_Face:
    """Construct a native planar cubic/line face, including every counter."""
    if not isfinite(z):
        raise occt.KernelError("curve face height must be finite")
    exterior = _oriented(region.exterior, True)
    holes = tuple(_oriented(ring, False) for ring in region.interiors)
    if all(span.is_line for ring in (exterior, *holes) for span in ring.spans):
        # The polygon adapter nodes mid-edge counter contacts before native
        # construction. This delegation retains exactly the same segments.
        polygon = Polygon(
            [span.control_points[0] for span in exterior.spans],
            [[span.control_points[0] for span in ring.spans] for ring in holes],
        )
        return occt.polygon_face(polygon, z)
    vertices: dict[tuple[float, float], TopoDS_Vertex] = {}
    builder = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), _wire(exterior, z, vertices), True
    )
    for ring in holes:
        builder.Add(_wire(ring, z, vertices))
    if not builder.IsDone():
        raise occt.KernelError(
            "OpenCascade could not construct the fitted regional face"
        )
    face = builder.Face()
    expected = _ring_area(exterior) + sum(_ring_area(ring) for ring in holes)
    if not occt.valid(face) or abs(occt.area(face) - expected) > max(
        1e-9, abs(expected) * 1e-8
    ):
        raise occt.KernelError(
            "OpenCascade fitted face is invalid or changed its enclosed area"
        )
    return face


def _check_roof(
    shape: TopoDS_Shape, region: CurvePolygon, z0: float, z1: float
) -> None:
    if not occt.valid(shape) or occt.volume(shape) <= 0:
        raise occt.KernelError("regional curve taper is not a valid positive solid")
    low, high = _bounds(shape)
    if abs(low[2] - z0) > 1e-6 or abs(high[2] - z1) > 1e-6:
        raise occt.KernelError("regional taper changed its requested height range")
    expected = polygon_face(region, z1)
    actual = occt.cross_section(shape, z1)
    tolerance = max(1e-10, occt.area(expected) * 1e-8)
    missing = occt.area(occt.cut(expected, actual))
    extra = occt.area(occt.cut(actual, expected))
    if missing > tolerance or extra > tolerance:
        raise occt.KernelError(
            f"regional taper changed its fitted top (missing {missing:.9g}, extra {extra:.9g} square millimeters)"
        )


def _bounds(
    shape: TopoDS_Shape,
    *,
    include_tolerance: bool = False,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds, False, include_tolerance)
    if bounds.IsVoid() or bounds.IsWhole():
        raise occt.KernelError("regional curve shape has no finite bounds")
    low, high = bounds.CornerMin(), bounds.CornerMax()
    return ((low.X(), low.Y(), low.Z()), (high.X(), high.Y(), high.Z()))


def _flat_ring(ring: CurveRing, tolerance: float) -> list[tuple[float, float]]:
    """Certified chords for validation only; these never construct native CAD."""
    points: list[tuple[float, float]] = []
    pending = [(span.control_points, span.is_line, 0) for span in reversed(ring.spans)]
    while pending:
        p, line, depth = pending.pop()
        dx, dy = p[3][0] - p[0][0], p[3][1] - p[0][1]
        length = hypot(dx, dy)
        flat = length > 0 and all(
            abs(dx * (q[1] - p[0][1]) - dy * (q[0] - p[0][0])) / length <= tolerance
            and 0 <= (q[0] - p[0][0]) * dx + (q[1] - p[0][1]) * dy <= length**2
            for q in (p[1], p[2])
        )
        if line or flat:
            points.append(p[0])
            continue
        if depth >= 30 or len(points) + len(pending) > 100_000:
            raise occt.KernelError(
                "curve validation subdivision exceeded its resource limit"
            )
        a = ((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2)
        b = ((p[1][0] + p[2][0]) / 2, (p[1][1] + p[2][1]) / 2)
        c = ((p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2)
        d = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        e = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
        middle = ((d[0] + e[0]) / 2, (d[1] + e[1]) / 2)
        pending.append(((middle, e, c, p[3]), False, depth + 1))
        pending.append(((p[0], a, d, middle), False, depth + 1))
    return points


def _check_sections(
    shape: TopoDS_Shape, region: CurvePolygon, z0: float, z1: float, slope: float
) -> None:
    flattening = 1e-5
    proxy = Polygon(
        _flat_ring(region.exterior, flattening),
        [_flat_ring(ring, flattening) for ring in region.interiors],
    )
    if not proxy.is_valid:
        raise occt.KernelError("fitted curve validation proxy has invalid topology")
    previous: TopoDS_Shape | None = None
    previous_z = z0
    area_tolerance = max(1e-9, proxy.area * 1e-8)
    # Include half-steps: a small counter can change entirely between the old
    # quarter-height planes while those five planes all look correct.
    for fraction in (i / 8 for i in range(9)):
        z = z0 + (z1 - z0) * fraction
        actual = occt.cross_section(shape, z)
        radius = slope * (z1 - z)
        if radius > 0:
            lower = proxy.buffer(max(0.0, radius - flattening), quad_segs=128).area
            upper = proxy.buffer(
                (radius + flattening) / cos(pi / (4 * 128)), quad_segs=128
            ).area
            if (
                not lower - area_tolerance
                <= occt.area(actual)
                <= upper + area_tolerance
            ):
                raise occt.KernelError(
                    "native taper sweep does not match the expected distance section"
                )
        if occt.area(occt.cut(polygon_face(region, z), actual)) > area_tolerance:
            raise occt.KernelError(
                "native taper sweep lost fitted source material below its top"
            )
        if previous is not None:
            transform = gp_Trsf()
            transform.SetTranslation(gp_Vec(0, 0, previous_z - z))
            lowered = BRepBuilderAPI_Transform(actual, transform, True).Shape()
            if occt.area(occt.cut(lowered, previous)) > area_tolerance:
                raise occt.KernelError(
                    "native taper sweep contains an unsupported cross-section"
                )
        previous, previous_z = actual, z


def _interior_point(solid: TopoDS_Shape) -> gp_Pnt:
    """Accept a point only after the native solid classifier proves it is IN."""
    properties = GProp_GProps()
    _ = BRepGProp.VolumeProperties_s(solid, properties, 1e-11)
    center = properties.CentreOfMass()
    classifier = BRepClass3d_SolidClassifier(solid, center, 1e-8)
    if classifier.State() == TopAbs_IN:
        return center
    context = IntTools_Context()
    faces = TopExp_Explorer(solid, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face(faces.Current())
        point, uv = gp_Pnt(), gp_Pnt2d()
        if BOPTools_AlgoTools3D.PointInFace_s(face, point, uv, context) == 0:
            surface = BRepAdaptor_Surface(face)
            du, dv = gp_Vec(), gp_Vec()
            surface.D1(uv.X(), uv.Y(), point, du, dv)
            normal = du.Crossed(dv)
            if normal.Magnitude() > 0:
                normal.Normalize()
                if face.Orientation() == TopAbs_REVERSED:
                    normal.Reverse()
                for displacement in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                    candidate = point.Translated(normal.Multiplied(-displacement))
                    classifier.Perform(candidate, 1e-8)
                    if classifier.State() == TopAbs_IN:
                        return candidate
        faces.Next()
    raise occt.KernelError(
        "could not prove an interior point for a regional draft cell"
    )


def _line_wall_sheets(
    material: CurveRing, z0: float, z1: float, slope: float
) -> list[TopoDS_Shape]:
    """Build exact planar wall faces with material on each edge's left."""
    run = slope * (z1 - z0)
    sheets: list[TopoDS_Shape] = []
    for span in material.spans:
        (x, y), (x1, y1) = span.control_points[0], span.control_points[3]
        dx, dy = x1 - x, y1 - y
        length = hypot(dx, dy)
        if length == 0:
            raise occt.KernelError("a straight fitted boundary has a zero-length span")
        nx, ny = dy / length, -dx / length
        boundary = BRepBuilderAPI_MakePolygon()
        for point in (
            gp_Pnt(x, y, z1),
            gp_Pnt(x1, y1, z1),
            gp_Pnt(x1 + nx * run, y1 + ny * run, z0),
            gp_Pnt(x + nx * run, y + ny * run, z0),
        ):
            boundary.Add(point)
        boundary.Close()
        face = BRepBuilderAPI_MakeFace(boundary.Wire(), True)
        if not face.IsDone() or not occt.valid(face.Face()):
            raise occt.KernelError("native planar taper wall is invalid")
        sheets.append(face.Face())
    return sheets


def _unjoined_sheets(
    ring: CurveRing, z0: float, z1: float, slope: float, internal: bool
) -> list[TopoDS_Shape]:
    """Sweep native walls; round only convex material corners separately."""
    angle_tolerance = 1e-8
    material = _oriented(ring, not internal)
    sheets: list[TopoDS_Shape]
    if all(span.is_line for span in ring.spans):
        sheets = _line_wall_sheets(material, z0, z1, slope)
    else:
        # The public MakeDraft wrapper maps Transformed back to Right. The
        # direct API preserves the genuinely unjoined curved wall surfaces.
        draft = BRepFill_Draft(
            _wire(_oriented(ring, True), z1, {}, draft_compatible=True),
            gp_Dir(0, 0, -1),
            atan(slope),
        )
        draft.SetDraft(internal)
        draft.SetOptions(BRepFill_Modified, angle_tolerance)
        draft.Perform((z1 - z0) / cos(atan(slope)))
        if not draft.IsDone():
            raise occt.KernelError("native unjoined draft sweep did not complete")
        sheets = []
        faces = TopExp_Explorer(draft.Shell(), TopAbs_FACE)
        while faces.More():
            if not occt.valid(faces.Current()):
                raise occt.KernelError("native unjoined draft returned an invalid wall")
            sheets.append(faces.Current())
            faces.Next()
    for index, span in enumerate(material.spans):
        previous = material.spans[index - 1]
        a, b = previous.control_points, span.control_points
        ux = a[3][0] - a[0 if previous.is_line else 2][0]
        uy = a[3][1] - a[0 if previous.is_line else 2][1]
        vx = b[3 if span.is_line else 1][0] - b[0][0]
        vy = b[3 if span.is_line else 1][1] - b[0][1]
        ul, vl = hypot(ux, uy), hypot(vx, vy)
        if ul == 0 or vl == 0:
            raise occt.KernelError("native taper requires nonstationary edge ends")
        turn = atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        if turn <= angle_tolerance:
            continue
        x, y = b[0]
        axes = gp_Ax3(gp_Pnt(x, y, z0), gp_Dir(0, 0, 1), gp_Dir(uy / ul, -ux / ul, 0))
        cone = Geom_ConicalSurface(axes, -atan(slope), slope * (z1 - z0))
        sector = BRepBuilderAPI_MakeFace(
            cone, 0.0, turn, 0.0, (z1 - z0) / cos(atan(slope)), 1e-7
        )
        if not sector.IsDone() or not occt.valid(sector.Face()):
            raise occt.KernelError("native convex taper corner is invalid")
        sheets.append(sector.Face())
    return sheets


def _entities(shape: TopoDS_Shape, kind: TopAbs_ShapeEnum) -> list[TopoDS_Shape]:
    mapping = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, kind, mapping)
    return [mapping.FindKey(index) for index in range(1, mapping.Extent() + 1)]


def _geometry_bytes(geometry: Geom_Curve | Geom_Surface) -> bytes:
    stream = BytesIO()
    GeomTools.Write_s(geometry, stream)
    return stream.getvalue()


def _real_edges(shape: TopoDS_Shape) -> list[TopoDS_Edge]:
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edges: list[TopoDS_Edge] = []
    while explorer.More():
        edge = TopoDS.Edge(explorer.Current())
        if not BRep_Tool.Degenerated_s(edge):
            edges.append(edge)
        explorer.Next()
    return edges


@dataclass(frozen=True)
class _FaceRepairState:
    face: TopoDS_Face
    surface: Geom_Surface
    geometry: bytes
    location: TopLoc_Location
    tolerance: float
    natural_restriction: bool
    edges: tuple[TopoDS_Edge, ...]


@dataclass(frozen=True)
class _EdgeRepairState:
    edge: TopoDS_Edge
    curve: Geom_Curve
    geometry: bytes
    location: TopLoc_Location
    parameters: tuple[float, float]
    tolerance: float


@dataclass(frozen=True)
class _VertexRepairState:
    vertex: TopoDS_Vertex
    point: tuple[float, float, float]
    tolerance: float


@dataclass(frozen=True)
class _CellRepairState:
    faces: tuple[_FaceRepairState, ...]
    edges: tuple[_EdgeRepairState, ...]
    vertices: tuple[_VertexRepairState, ...]
    degenerate_edges: tuple[tuple[TopoDS_Edge, float], ...]


def _cell_repair_state(shape: TopoDS_Shape) -> _CellRepairState:
    faces: list[_FaceRepairState] = []
    for entity in _entities(shape, TopAbs_FACE):
        face = TopoDS.Face(entity)
        location = TopLoc_Location()
        surface = BRep_Tool.Surface_s(face, location)
        faces.append(
            _FaceRepairState(
                face,
                surface,
                _geometry_bytes(surface),
                location,
                BRep_Tool.Tolerance_s(face),
                BRep_Tool.NaturalRestriction_s(face),
                tuple(_real_edges(face)),
            )
        )
    edges: list[_EdgeRepairState] = []
    degenerate_edges: list[tuple[TopoDS_Edge, float]] = []
    for entity in _entities(shape, TopAbs_EDGE):
        edge = TopoDS.Edge(entity)
        if BRep_Tool.Degenerated_s(edge):
            degenerate_edges.append((edge, BRep_Tool.Tolerance_s(edge)))
            continue
        location = TopLoc_Location()
        # The native API permits null handles for edges without a 3D curve;
        # its generated Python stub does not express that nullable return.
        curve = cast("Geom_Curve | None", BRep_Tool.Curve_s(edge, location, 0.0, 0.0))
        if curve is None:
            raise occt.KernelError(
                "cannot certify a native cell edge without a 3D curve"
            )
        edges.append(
            _EdgeRepairState(
                edge,
                curve,
                _geometry_bytes(curve),
                location,
                BRep_Tool.Range_s(edge),
                BRep_Tool.Tolerance_s(edge),
            )
        )
    vertices = tuple(
        _VertexRepairState(
            TopoDS.Vertex(vertex),
            BRep_Tool.Pnt_s(TopoDS.Vertex(vertex)).Coord(),
            BRep_Tool.Tolerance_s(TopoDS.Vertex(vertex)),
        )
        for vertex in _entities(shape, TopAbs_VERTEX)
    )
    return _CellRepairState(
        tuple(faces), tuple(edges), vertices, tuple(degenerate_edges)
    )


def _same_occurrences(
    expected: Sequence[TopoDS_Shape], actual: Sequence[TopoDS_Shape]
) -> bool:
    remaining = list(actual)
    for shape in expected:
        index = next(
            (i for i, candidate in enumerate(remaining) if shape.IsEqual(candidate)),
            None,
        )
        if index is None:
            return False
        _ = remaining.pop(index)
    return not remaining


def _cell_repair_unchanged(
    before: _CellRepairState, after: TopoDS_Shape, context: ShapeBuild_ReShape
) -> bool:
    """Permit singular UV completion, preserving every real boundary exactly."""
    if after.ShapeType() != TopAbs_SOLID:
        return False
    faces = _entities(after, TopAbs_FACE)
    mapped_faces: list[TopoDS_Shape] = []
    for state in before.faces:
        mapped = context.Apply(state.face)
        if mapped.IsNull() or mapped.ShapeType() != TopAbs_FACE:
            return False
        face = TopoDS.Face(mapped)
        location = TopLoc_Location()
        surface = BRep_Tool.Surface_s(face, location)
        if (
            surface is not state.surface
            or _geometry_bytes(surface) != state.geometry
            or not location.IsEqual(state.location)
            or face.Orientation() != state.face.Orientation()
            or BRep_Tool.Tolerance_s(face) > state.tolerance
            or BRep_Tool.NaturalRestriction_s(face) != state.natural_restriction
            or any(
                context.Apply(edge).Orientation() != edge.Orientation()
                for edge in state.edges
            )
            or not _same_occurrences(
                [context.Apply(edge) for edge in state.edges],
                list(_real_edges(face)),
            )
        ):
            return False
        mapped_faces.append(face)
    if not _same_occurrences(mapped_faces, faces):
        return False
    mapped_edges: list[TopoDS_Shape] = []
    for state in before.edges:
        mapped = context.Apply(state.edge)
        if mapped.IsNull() or mapped.ShapeType() != TopAbs_EDGE:
            return False
        edge = TopoDS.Edge(mapped)
        location = TopLoc_Location()
        curve = BRep_Tool.Curve_s(edge, location, 0.0, 0.0)
        if (
            BRep_Tool.Degenerated_s(edge)
            or curve is not state.curve
            or _geometry_bytes(curve) != state.geometry
            or not location.IsEqual(state.location)
            or edge.Orientation() != state.edge.Orientation()
            or BRep_Tool.Range_s(edge) != state.parameters
            or BRep_Tool.Tolerance_s(edge) > state.tolerance
        ):
            return False
        mapped_edges.append(edge)
    if not _same_occurrences(
        mapped_edges,
        [
            edge
            for edge in _entities(after, TopAbs_EDGE)
            if not BRep_Tool.Degenerated_s(TopoDS.Edge(edge))
        ],
    ):
        return False
    mapped_degenerate: list[TopoDS_Shape] = []
    for old_edge, tolerance in before.degenerate_edges:
        mapped = context.Apply(old_edge)
        if (
            mapped.IsNull()
            or mapped.ShapeType() != TopAbs_EDGE
            or not BRep_Tool.Degenerated_s(TopoDS.Edge(mapped))
            or BRep_Tool.Tolerance_s(TopoDS.Edge(mapped)) > tolerance
        ):
            return False
        mapped_degenerate.append(mapped)
    after_edges = _entities(after, TopAbs_EDGE)
    if any(
        not any(edge.IsSame(candidate) for candidate in after_edges)
        for edge in mapped_degenerate
    ):
        return False
    for entity in after_edges:
        edge = TopoDS.Edge(entity)
        if not BRep_Tool.Degenerated_s(edge):
            continue
        if (
            cast(
                "Geom_Curve | None",
                BRep_Tool.Curve_s(edge, TopLoc_Location(), 0.0, 0.0),
            )
            is not None
        ):
            return False
        if (
            not any(edge.IsSame(mapped) for mapped in mapped_degenerate)
            and BRep_Tool.Tolerance_s(edge) > 1e-7
        ):
            return False
    mapped_vertices: list[TopoDS_Shape] = []
    for state in before.vertices:
        mapped = context.Apply(state.vertex)
        if mapped.IsNull() or mapped.ShapeType() != TopAbs_VERTEX:
            return False
        vertex = TopoDS.Vertex(mapped)
        if (
            BRep_Tool.Pnt_s(vertex).Coord() != state.point
            or BRep_Tool.Tolerance_s(vertex) > state.tolerance
        ):
            return False
        mapped_vertices.append(vertex)
    # New singular-edge vertices must coincide with an existing native vertex;
    # they may not introduce a new physical point or increase its precision.
    for entity in _entities(after, TopAbs_VERTEX):
        if any(entity.IsSame(vertex) for vertex in mapped_vertices):
            continue
        vertex = TopoDS.Vertex(entity)
        if not any(
            BRep_Tool.Pnt_s(vertex).Coord() == state.point
            and BRep_Tool.Tolerance_s(vertex) <= state.tolerance
            for state in before.vertices
        ):
            return False
    return True


def _repair_cell(cell: TopoDS_Shape) -> TopoDS_Shape:
    """Complete singular UV boundaries without changing physical geometry."""
    if occt.valid(cell):
        return cell
    before = _bounds(cell)
    copied = BRepBuilderAPI_Copy(cell, True, False).Shape()
    state = _cell_repair_state(copied)
    repair = ShapeFix_Shape(copied)
    repair.SetPrecision(1e-7)
    repair.SetMinTolerance(1e-7)
    repair.SetMaxTolerance(1e-7)
    _ = repair.Perform()
    result = repair.Shape()
    if (
        not occt.valid(result)
        or not _cell_repair_unchanged(state, result, repair.Context())
        or any(
            abs(a - b) > 1e-7
            for old, new in zip(before, _bounds(result), strict=True)
            for a, b in zip(old, new, strict=True)
        )
    ):
        raise occt.KernelError(
            "native draft cell repair failed or changed physical geometry or precision"
        )
    return result


def _material_box_clearance(
    shape: TopoDS_Shape, source: TopoDS_Face, z1: float, slope: float
) -> float:
    """Lower bound on material clearance throughout a shape's entire box.

    Distance to the filled top is 1-Lipschitz in XY. Its value at the box center
    plus the XY half-diagonal bounds every point; the highest Z has least taper
    expansion. A positive bound certifies volume, not merely sampled sections.
    """
    low, high = _bounds(shape, include_tolerance=True)
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, z1)
    ).Vertex()
    distance = BRepExtrema_DistShapeShape(vertex, source)
    _ = distance.Perform()
    if not distance.IsDone():
        raise occt.KernelError("could not bound a draft partition's material distance")
    radius = hypot(high[0] - low[0], high[1] - low[1]) / 2
    return slope * (z1 - high[2]) - distance.Value() - radius


def _resolve_draft_cells(
    raw: TopoDS_Shape,
    source: TopoDS_Face,
    z1: float,
    slope: float,
    diagnostics: dict[str, object] | None,
) -> list[TopoDS_Shape]:
    """Resolve certified internal partitions before validating material cells."""
    originals = _entities(raw, TopAbs_SOLID)
    cells: list[TopoDS_Shape | None] = list(originals)
    unresolved: dict[int, occt.KernelError] = {}
    repaired = merged = 0
    for index, cell in enumerate(originals):
        try:
            cells[index] = _repair_cell(cell)
        except occt.KernelError as error:
            unresolved[index] = error
        else:
            repaired += not occt.valid(cell)
    # Shared original face identities survive as adjacency evidence even when
    # successful UV-only repair copied the neighboring cell's native topology.
    faces = [_entities(cell, TopAbs_FACE) for cell in originals]
    for index in tuple(unresolved):
        if index not in unresolved:
            continue
        bad_faces = [face for face in faces[index] if not occt.valid(face)]
        if not bad_faces or any(
            _material_box_clearance(face, source, z1, slope) <= 1e-7
            for face in bad_faces
        ):
            continue
        neighbors: set[int] = set()
        for face in bad_faces:
            adjacent = {
                other
                for other, cell in enumerate(cells)
                if other != index
                and cell is not None
                and any(
                    face.IsSame(candidate)
                    and face.Orientation() != candidate.Orientation()
                    for candidate in faces[other]
                )
                and _material_box_clearance(cell, source, z1, slope) > 1e-7
            }
            if not adjacent:
                break
            neighbors.update(adjacent)
        else:
            current = cells[index]
            assert current is not None
            arguments = [current]
            arguments.extend(
                cell
                for other in sorted(neighbors)
                if (cell := cells[other]) is not None
            )
            try:
                union = occt.fuse(arguments)
            except occt.KernelError:
                # No invalid result or omitted cell is accepted. The original
                # repair failure remains actionable if exact union cannot work.
                continue
            solids = _entities(union, TopAbs_SOLID)
            if len(solids) != 1 or occt.topology_counts(
                solids[0]
            ) != occt.topology_counts(union):
                continue
            cells[index] = solids[0]
            for other in neighbors:
                cells[other] = None
                _ = unresolved.pop(other, None)
            _ = unresolved.pop(index)
            merged += len(neighbors)
    if diagnostics is not None:
        diagnostics.update(
            draft_cells=len(originals),
            repaired_draft_cells=repaired,
            merged_material_partitions=merged,
        )
    if unresolved:
        raise next(iter(unresolved.values()))
    return [cell for cell in cells if cell is not None]


def _preimprint_sheets(
    sheets: Sequence[TopoDS_Shape], diagnostics: dict[str, object] | None
) -> list[TopoDS_Shape]:
    """Imprint overlapping native generator pairs without altering their surfaces."""
    pair_limit, image_limit = 16384, 32768
    if diagnostics is not None:
        diagnostics.update(
            preimprint_input_faces=len(sheets),
            preimprint_pair_limit=pair_limit,
            preimprint_image_limit=image_limit,
        )
    if len(sheets) > 8192:
        raise occt.KernelError("native pair preimprint exceeds 8,192 input faces")
    bounds = [_bounds(sheet, include_tolerance=True) for sheet in sheets]
    boxes = [box(low[0], low[1], high[0], high[1]) for low, high in bounds]
    tree = STRtree(boxes)
    pairs: list[tuple[int, int]] = []
    for i, footprint in enumerate(boxes):
        candidates = tree.query(footprint)
        for offset in range(len(candidates)):
            j = int(cast("int", candidates[offset]))
            if (
                i < j
                and bounds[i][0][2] <= bounds[j][1][2]
                and bounds[j][0][2] <= bounds[i][1][2]
            ):
                pairs.append((i, j))
                if len(pairs) > pair_limit:
                    if diagnostics is not None:
                        diagnostics["preimprint_pairs"] = len(pairs)
                    raise occt.KernelError(
                        "native pair preimprint exceeds 16,384 pairs"
                    )
    pairs.sort()

    def unique(shapes: Sequence[TopoDS_Shape]) -> list[TopoDS_Shape]:
        indexed = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
        for shape in shapes:
            _ = indexed.Add(shape)
        return [indexed.FindKey(i) for i in range(1, indexed.Extent() + 1)]

    groups = [[sheet] for sheet in sheets]
    total_images = len(sheets)
    if diagnostics is not None:
        diagnostics.update(
            preimprint_input_faces=len(sheets),
            preimprint_pairs=len(pairs),
            preimprint_pairs_processed=0,
        )
    for processed, (i, j) in enumerate(pairs, 1):
        arguments = unique([*groups[i], *groups[j]])
        if len(arguments) > 512:
            raise occt.KernelError(
                "native pair preimprint exceeds 512 pair input faces"
            )
        if len(arguments) > 1:
            shapes = List_TopoDS_Shape()
            for shape in arguments:
                _ = shapes.Append(shape)
            builder = BOPAlgo_Builder()
            builder.SetArguments(shapes)
            builder.SetNonDestructive(True)
            builder.SetToFillHistory(True)
            builder.Perform()
            if builder.HasErrors():
                raise occt.KernelError("native pair preimprint intersection failed")
            replacements: list[list[TopoDS_Shape]] = []
            for index in (i, j):
                replacement: list[TopoDS_Shape] = []
                for original in groups[index]:
                    images = (
                        list(builder.Images().Find(original))
                        if builder.Images().IsBound(original)
                        else list(builder.Modified(original))
                    )
                    if not images and not builder.IsDeleted(original):
                        images = [original]
                    for image in images:
                        replacement.extend(_entities(image, TopAbs_FACE))
                replacement = unique(replacement)
                if not replacement:
                    raise occt.KernelError("native pair preimprint lost a generator")
                replacements.append(replacement)
            total_images += (
                sum(len(group) for group in replacements)
                - len(groups[i])
                - len(groups[j])
            )
            groups[i], groups[j] = replacements
            if total_images > image_limit:
                raise occt.KernelError(
                    "native pair preimprint exceeds 32,768 face images"
                )
        if diagnostics is not None:
            diagnostics["preimprint_pairs_processed"] = processed
            diagnostics["preimprint_generator_images"] = total_images
    result = unique([shape for group in groups for shape in group])
    if diagnostics is not None:
        diagnostics["preimprint_output_faces"] = len(result)
    return result


def _surface_region(
    region: CurvePolygon,
    z0: float,
    z1: float,
    slope: float,
    *,
    diagnostics: dict[str, object] | None = None,
    preimprint: bool = False,
) -> TopoDS_Shape:
    """Intersect native walls, repair singular boundaries, select material cells."""
    source = polygon_face(region, z1)
    low, high = _bounds(source)
    run = slope * (z1 - z0)
    margin = max(1e-3, run * 0.01)
    slab = BRepPrimAPI_MakeBox(
        gp_Pnt(low[0] - run - margin, low[1] - run - margin, z0),
        high[0] - low[0] + 2 * (run + margin),
        high[1] - low[1] + 2 * (run + margin),
        z1 - z0,
    ).Shape()
    arguments = List_TopoDS_Shape()
    _ = arguments.Append(slab)
    sheets: list[TopoDS_Shape] = []
    for ring, internal in (
        (region.exterior, False),
        *((ring, True) for ring in region.interiors),
    ):
        sheets.extend(_unjoined_sheets(ring, z0 - margin, z1, slope, internal))
    if preimprint:
        sheets = _preimprint_sheets(sheets, diagnostics)
    for sheet in sheets:
        _ = arguments.Append(sheet)
    maker = BOPAlgo_MakerVolume()
    maker.SetArguments(arguments)
    maker.SetIntersect(True)
    maker.SetAvoidInternalShapes(True)
    maker.SetNonDestructive(True)
    maker.Perform()
    if maker.HasErrors():
        raise occt.KernelError("native regional surface arrangement failed")
    selected: list[TopoDS_Shape] = []
    for cell in _resolve_draft_cells(maker.Shape(), source, z1, slope, diagnostics):
        if not occt.valid(cell) or occt.volume(cell) <= 0:
            raise occt.KernelError(
                "native regional draft produced invalid cells; cannot certify this surface arrangement"
            )
        point = _interior_point(cell)
        projection = BRepBuilderAPI_MakeVertex(
            gp_Pnt(point.X(), point.Y(), z1)
        ).Vertex()
        distance = BRepExtrema_DistShapeShape(projection, source)
        _ = distance.Perform()
        if not distance.IsDone():
            raise occt.KernelError(
                "native regional draft cell distance could not be classified"
            )
        if distance.Value() <= slope * (z1 - point.Z()):
            selected.append(cell)
    if diagnostics is not None:
        diagnostics["selected_draft_cells"] = len(selected)
    if not selected:
        raise occt.KernelError("native regional draft did not enclose source material")
    return occt.fuse(selected)


def roof(
    region: CurvePolygon,
    z0: float,
    z1: float,
    run_per_height: float,
    prefer_fast: bool = True,
    *,
    diagnostics: dict[str, object] | None = None,
) -> tuple[TopoDS_Shape, str]:
    """Build a regional solid from intersecting native taper boundary surfaces.

    ``prefer_fast`` remains accepted for the polygon adapter's existing callers;
    positive fitted tapers use the same native wall arrangement in either mode.
    """
    _ = prefer_fast
    if (
        not isfinite(z0)
        or not isfinite(z1)
        or z1 <= z0
        or not isfinite(z1 - z0)
        or not isfinite(run_per_height)
        or run_per_height < 0
        or not isfinite(run_per_height * (z1 - z0))
    ):
        raise occt.KernelError(
            "regional taper needs increasing finite heights and nonnegative finite slope"
        )
    native_edges = sum(len(ring.spans) for ring in (region.exterior, *region.interiors))
    if diagnostics is not None:
        diagnostics.update(
            input_native_edges=native_edges,
            counter_count=len(region.interiors),
            taper_run_per_height=run_per_height,
        )
    if run_per_height == 0:
        shape = BRepPrimAPI_MakePrism(
            polygon_face(region, z0), gp_Vec(0, 0, z1 - z0)
        ).Shape()
        _check_roof(shape, region, z0, z1)
        return shape, "prism"
    if native_edges > _SPATIAL_EDGE_THRESHOLD and all(
        span.is_line
        for ring in (region.exterior, *region.interiors)
        for span in ring.spans
    ):
        exterior = _oriented(region.exterior, True)
        interiors = [_oriented(ring, False) for ring in region.interiors]
        source = Polygon(
            [span.control_points[0] for span in exterior.spans],
            [[span.control_points[0] for span in ring.spans] for ring in interiors],
        )

        def build_piece(
            piece: Polygon, piece_diagnostics: dict[str, object]
        ) -> tuple[TopoDS_Shape, str]:
            return roof(
                polygon_lines(piece),
                z0,
                z1,
                run_per_height,
                diagnostics=piece_diagnostics,
            )

        def check_global_piece(shape: TopoDS_Shape, piece: Polygon) -> None:
            selected = polygon_lines(piece)
            _check_roof(shape, selected, z0, z1)
            _check_sections(shape, selected, z0, z1, run_per_height)

        try:
            shape = spatial_roof.build_roof(
                source,
                run_per_height * (z1 - z0),
                build_piece,
                check_global_piece,
                diagnostics=diagnostics,
            )
            _check_roof(shape, region, z0, z1)
            _check_sections(shape, region, z0, z1, run_per_height)
            return shape, "spatial-decomposition"
        except (MemoryError, Standard_OutOfMemory):
            raise
        except Exception as error:
            if diagnostics is not None:
                diagnostics["spatial_failure"] = str(error)
            raise occt.KernelError(
                f"OpenCascade could not construct a verified spatial regional taper: {error}"
            ) from error
    try:
        shape = _surface_region(region, z0, z1, run_per_height, diagnostics=diagnostics)
        _check_roof(shape, region, z0, z1)
        _check_sections(shape, region, z0, z1, run_per_height)
        return shape, "surface"
    except (MemoryError, Standard_OutOfMemory):
        raise
    except Exception as error:
        if diagnostics is not None:
            diagnostics["surface_failure"] = str(error)
        if all(
            span.is_line
            for ring in (region.exterior, *region.interiors)
            for span in ring.spans
        ):
            try:
                shape = _surface_region(
                    region,
                    z0,
                    z1,
                    run_per_height,
                    diagnostics=diagnostics,
                    preimprint=True,
                )
                _check_roof(shape, region, z0, z1)
                _check_sections(shape, region, z0, z1, run_per_height)
                return shape, "surface-preimprint"
            except (MemoryError, Standard_OutOfMemory):
                raise
            except Exception as preimprint_error:
                if diagnostics is not None:
                    diagnostics["preimprint_failure"] = str(preimprint_error)
                raise occt.KernelError(
                    f"OpenCascade could not construct a verified native regional taper: {error}; native pair preimprint: {preimprint_error}"
                ) from preimprint_error
        raise occt.KernelError(
            f"OpenCascade could not construct a verified native regional taper: {error}"
        ) from error
