"""OpenCascade solids with preserved plateaus and outward distance tapers."""

from collections.abc import Iterable, Sequence
from math import isfinite, pi, sin
from pathlib import Path
from typing import cast

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_BooleanOperation,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
)
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
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeEvolved
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.BRepTools import BRepTools, BRepTools_ReShape
from OCP.collections import (
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    List_TopoDS_Shape,
)
from OCP.GeomAbs import GeomAbs_Arc
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_INTERNAL,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import (
    TopoDS,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shape,
    TopoDS_Vertex,
    TopoDS_Wire,
)
from shapely import get_coordinates
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree

from panelchemy.surface_measure import AreaIntegrationError, surface_area

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


def topology_counts(shape: TopoDS_Shape) -> dict[str, int]:
    """Count unique native topology, rather than repeated face-edge uses."""
    counts: dict[str, int] = {}
    for name, kind in (
        ("solids", TopAbs_SOLID),
        ("faces", TopAbs_FACE),
        ("edges", TopAbs_EDGE),
        ("vertices", TopAbs_VERTEX),
    ):
        shapes = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
        TopExp.MapShapes_s(shape, kind, shapes)
        counts[name] = shapes.Extent()
    return counts


def volume(shape: TopoDS_Shape) -> float:
    """Compute signed closed-solid volume from the exact surfaces."""
    if _is_empty(shape):
        return 0.0
    # OnlyClosed excludes faces and open shells. GK reports -1 when there is
    # no eligible domain at all, whereas the previous overload returned zero.
    shells = TopExp_Explorer(shape, TopAbs_SHELL)
    while shells.More():
        if BRep_Tool.IsClosed_s(shells.Current()):
            break
        shells.Next()
    else:
        return 0.0
    properties = GProp_GProps()
    # Consolidated curves can contain several polynomial intervals inside one
    # native face. The older Gauss integrator can miss their contribution and
    # report a near-zero error despite a percent-level mass error. Kronrod
    # integration with span handling preserves the signed closed-solid mass.
    error = BRepGProp.VolumePropertiesGK_s(
        shape, properties, Eps=1e-11, OnlyClosed=True, IsUseSpan=True
    )
    result = properties.Mass()
    if not isfinite(result) or not isfinite(error):
        raise KernelError("OpenCascade returned a non-finite volume")
    if error < 0:
        raise KernelError("OpenCascade could not integrate the closed-solid volume")
    return result


def area(shape: TopoDS_Shape) -> float:
    """Compute surface area, including planar faces returned by cross_section."""
    if _is_empty(shape):
        return 0.0
    try:
        return surface_area(shape)
    except AreaIntegrationError as error:
        raise KernelError(str(error)) from error


def _is_open_internal_wire(wire: TopoDS_Wire) -> bool:
    """Recognize an acyclic INTERNAL annotation graph, independent of its flag.

    Native wire closure checks ignore INTERNAL edges. Check their vertex graph
    instead, so a closed INTERNAL loop with an unset Closed flag is retained.
    Branches and disconnected trees cannot enclose a counter and are allowed.
    """
    if wire.Closed():
        return False
    vertices: list[TopoDS_Vertex] = []
    parents: list[int] = []
    edges = TopExp_Explorer(wire, TopAbs_EDGE)
    while edges.More():
        edge = TopoDS.Edge(edges.Current())
        if edge.Orientation() != TopAbs_INTERNAL:
            return False
        ends: list[int] = []
        for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
            if vertex.IsNull():
                return False
            index = next(
                (i for i, existing in enumerate(vertices) if vertex.IsSame(existing)),
                len(vertices),
            )
            if index == len(vertices):
                vertices.append(vertex)
                parents.append(index)
            ends.append(index)
        roots: list[int] = []
        for index in ends:
            while parents[index] != index:
                index = parents[index]
            roots.append(index)
        # This also rejects self-loops, parallel-edge cycles and repeated uses
        # of an edge. One cyclic component retains the entire wire.
        if roots[0] == roots[1]:
            return False
        parents[roots[0]] = roots[1]
        edges.Next()
    return bool(vertices)


def _open_internal_wires(shape: TopoDS_Shape) -> list[TopoDS_Wire]:
    wires: list[TopoDS_Wire] = []
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        face_wires = TopExp_Explorer(faces.Current(), TopAbs_WIRE)
        while face_wires.More():
            wire = TopoDS.Wire(face_wires.Current())
            if _is_open_internal_wire(wire):
                wires.append(wire)
            face_wires.Next()
        faces.Next()
    return wires


def _annotation_boundary_unchanged(
    before: TopoDS_Shape,
    after: TopoDS_Shape,
    removed: Sequence[TopoDS_Wire],
) -> bool:
    """Certify exact physical boundary identity after annotation removal.

    ReShape can replace containing faces, shells and solids. Their ordered
    membership, locations and orientations must remain; only recognized open
    INTERNAL face-wire occurrences may disappear. Retained wire identities keep
    all native edges and pcurves, and face supports remain the same native object.
    """
    if before.IsEqual(after):
        return True
    if (
        before.ShapeType() != after.ShapeType()
        or before.Orientation() != after.Orientation()
        or not before.Location().IsEqual(after.Location())
        or before.Closed() != after.Closed()
        or before.ShapeType()
        not in (
            TopAbs_COMPOUND,
            TopAbs_COMPSOLID,
            TopAbs_SOLID,
            TopAbs_SHELL,
            TopAbs_FACE,
        )
    ):
        return False
    is_face = before.ShapeType() == TopAbs_FACE
    if is_face:
        old, new = TopoDS.Face(before), TopoDS.Face(after)
        old_location, new_location = TopLoc_Location(), TopLoc_Location()
        old_surface = BRep_Tool.Surface_s(old, old_location)
        new_surface = BRep_Tool.Surface_s(new, new_location)
        if (
            old_surface is not new_surface
            or not old_location.IsEqual(new_location)
            or BRep_Tool.Tolerance_s(old) != BRep_Tool.Tolerance_s(new)
            or BRep_Tool.NaturalRestriction_s(old)
            != BRep_Tool.NaturalRestriction_s(new)
        ):
            return False
    old_children, new_children = TopoDS_Iterator(before), TopoDS_Iterator(after)
    while old_children.More():
        child = old_children.Value()
        if child.ShapeType() == TopAbs_WIRE and any(
            child.IsSame(wire) for wire in removed
        ):
            # A globally shared wire may only be removed where it is the same
            # recognized face annotation, never from another topology context.
            if not is_face or not _is_open_internal_wire(TopoDS.Wire(child)):
                return False
        else:
            if not new_children.More() or not _annotation_boundary_unchanged(
                child, new_children.Value(), removed
            ):
                return False
            new_children.Next()
        old_children.Next()
    return not new_children.More()


def _without_open_internal_wires(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Remove non-boundary Boolean annotations that the native mesher rejects.

    Closed loops and every boundary edge remain. These open INTERNAL wires
    delimit no surface area; removing them does not simplify artwork or impose
    a minimum feature size. Verify exact boundary identity on the working copy,
    plus native validity and area. Exact oriented physical boundary identity
    already preserves signed volume; no numerical mass comparison is needed.
    """
    if not _open_internal_wires(shape):
        return shape
    copied = BRepBuilderAPI_Copy(shape, True, False).Shape()
    reshape = BRepTools_ReShape()
    removed = _open_internal_wires(copied)
    for wire in removed:
        reshape.Remove(wire)
    cleaned = reshape.Apply(copied)
    if not valid(cleaned):
        raise KernelError(
            "removing internal annotation wires produced invalid geometry"
        )
    boundary_unchanged = _annotation_boundary_unchanged(copied, cleaned, removed)
    if not boundary_unchanged:
        raise KernelError("removing internal annotation wires changed CAD boundaries")
    before, after = area(shape), area(cleaned)
    if abs(before - after) > max(1e-10, abs(before) * 1e-10):
        raise KernelError("removing internal annotation wires changed CAD geometry")
    return cleaned


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


def _shared_polygon_face(rings: Sequence[Sequence[Point2]], z: float) -> TopoDS_Face:
    """Reuse identical vertices across rings instead of rediscovering contacts."""
    vertices: dict[Point2, TopoDS_Vertex] = {}
    wires: list[TopoDS_Wire] = []
    for ring in rings:
        native_vertices: list[TopoDS_Vertex] = []
        for point in ring:
            if point not in vertices:
                vertices[point] = BRepBuilderAPI_MakeVertex(gp_Pnt(*point, z)).Vertex()
            native_vertices.append(vertices[point])
        edges: list[TopoDS_Shape] = []
        for index, vertex in enumerate(native_vertices):
            edge = BRepBuilderAPI_MakeEdge(
                vertex, native_vertices[(index + 1) % len(native_vertices)]
            )
            if not edge.IsDone():
                raise KernelError("OpenCascade could not construct a source edge")
            edges.append(edge.Edge())
        wire = BRepBuilderAPI_MakeWire()
        wire.Add(_shape_list(edges))
        if not wire.IsDone():
            raise KernelError("OpenCascade could not connect the source edges")
        selected = wire.Wire()
        explorer = TopExp_Explorer(selected, TopAbs_EDGE)
        edge_count = 0
        while explorer.More():
            edge_count += 1
            explorer.Next()
        if edge_count != len(edges):
            raise KernelError("OpenCascade wire construction lost a source edge")
        wires.append(selected)
    builder = BRepBuilderAPI_MakeFace(wires[0])
    for wire in wires[1:]:
        builder.Add(wire)
    if not builder.IsDone():
        raise KernelError("OpenCascade could not construct the polygon face")
    return builder.Face()


def _contact_noded_rings(rings: Sequence[Sequence[Point2]]) -> list[list[Point2]]:
    """Insert only exact point contacts into straight segments, without moving them."""
    boundaries = [LineString([*ring, ring[0]]) for ring in rings]
    tree = STRtree(boundaries)
    contacts: list[set[Point2]] = [set() for _ in rings]
    for index, boundary in enumerate(boundaries):
        neighbors = cast(
            "list[int]", tree.query(boundary, predicate="intersects").tolist()
        )
        for neighbor in neighbors:
            if neighbor <= index:
                continue
            intersection = boundary.intersection(boundaries[neighbor])
            if intersection.geom_type not in {"Point", "MultiPoint"}:
                raise KernelError(
                    "polygon rings share more than isolated source contacts"
                )
            coordinates = cast(
                "list[list[float]]", get_coordinates(intersection).tolist()
            )
            for coordinate in coordinates:
                point = (coordinate[0], coordinate[1])
                contacts[index].add(point)
                contacts[neighbor].add(point)
    noded: list[list[Point2]] = []
    for ring, boundary, points in zip(rings, boundaries, contacts, strict=True):
        missing = points - set(ring)
        if not missing:
            noded.append(list(ring))
            continue
        segments = [
            LineString([point, ring[(index + 1) % len(ring)]])
            for index, point in enumerate(ring)
        ]
        segment_tree = STRtree(segments)
        insertions: dict[int, list[tuple[float, Point2]]] = {}
        for point in sorted(missing):
            contact = Point(point)
            candidates = cast("list[int]", segment_tree.query(contact).tolist())
            found = False
            for candidate in candidates:
                segment = segments[candidate]
                if segment.covers(contact):
                    insertions.setdefault(candidate, []).append(
                        (segment.project(contact), point)
                    )
                    found = True
            if not found:
                raise KernelError(
                    "a source contact cannot be noded without moving its boundary"
                )
        selected: list[Point2] = []
        for index, point in enumerate(ring):
            selected.append(point)
            selected.extend(point for _, point in sorted(insertions.get(index, [])))
        if not LineString([*selected, selected[0]]).equals(boundary):
            raise KernelError("contact noding changed the exact source boundary")
        noded.append(selected)
    return noded


def polygon_face(polygon: Polygon, z: float = 0) -> TopoDS_Face:
    """Construct the exact face, including point-touching counter boundaries.

    Ring wires share native vertices at identical source points. Contacts within
    an edge are noded without changing its geometry. A checked native planar
    Boolean remains a fallback when the direct arrangement is unsupported.
    """
    if not isfinite(z):
        raise KernelError("face height must be finite")
    polygon = _normalized(polygon)
    rings = _rings(polygon)
    try:
        result = _shared_polygon_face(rings, z)
    except Exception as error:
        if not polygon.interiors:
            raise KernelError(
                "OpenCascade could not construct the polygon face"
            ) from error
        # A direct multi-wire arrangement may be unsupported even though its
        # exterior and counter faces can be built independently for the cut.
        result = TopoDS_Face()
        result_valid = False
    else:
        result_valid = valid(result)
    if not result_valid and polygon.interiors:
        try:
            noded = _contact_noded_rings(rings)
            if noded != rings:
                result = _shared_polygon_face(noded, z)
                result_valid = valid(result)
        except Exception:
            # The unchanged exterior-minus-counters operation below is checked
            # independently; no failed candidate is accepted as a replacement.
            pass
    if not result_valid:
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
        # A Boolean can reconnect UV wires incorrectly even for valid inputs.
        # Repair only a single solid with an exact physical-boundary certificate;
        # planar errors, extra fragments and broader geometry changes still fail.
        solids = TopExp_Explorer(result, TopAbs_SOLID)
        if not solids.More():
            raise KernelError(f"OpenCascade {name} returned invalid geometry")
        solid = solids.Current()
        solids.Next()
        if solids.More() or topology_counts(solid) != topology_counts(result):
            raise KernelError(f"OpenCascade {name} returned invalid geometry")
        from panelchemy.curved_occt import _repair_cell  # pyright: ignore[reportPrivateUsage]

        try:
            result = _repair_cell(solid)
        except KernelError as error:
            raise KernelError(
                f"OpenCascade {name} returned invalid geometry; {error}"
            ) from error
    return _without_open_internal_wires(result)


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


def roof(
    polygon: Polygon,
    z0: float,
    z1: float,
    run_per_height: float,
    prefer_fast: bool = True,
) -> tuple[TopoDS_Shape, str]:
    """Preserve the top polygon and expand down with a Euclidean-distance taper.

    Cross-sections at z are the polygon buffered by run_per_height * (z1-z).
    The evolved sweep is fast but can fail at offset topology changes. The
    regional builder supplies a native draft-surface alternative, with no
    per-edge closed-solid union. Both preserve the original top and counters.
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
    from panelchemy.curved_occt import roof as regional_roof
    from panelchemy.curves import polygon_lines

    try:
        result, method = regional_roof(
            polygon_lines(polygon), z0, z1, run_per_height, prefer_fast=False
        )
    except Exception as error:
        raise KernelError(
            "OpenCascade could not construct a verified outward taper"
        ) from error
    return result, method


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
