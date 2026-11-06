"""Partition planar point contacts into STEP-compatible face boundaries.

Two face wires may share a vertex in a valid native BRep. STEP's automatic
normalization can detach adjacent walls when splitting that vertex. Splitting
the planar face in its original shell instead retains the physical boundary.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
from math import atan2, cos, hypot, isclose, isfinite, pi, sin, tau
from typing import cast

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy, BRepBuilderAPI_MakeFace
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepFeat import BRepFeat_SplitShape
from OCP.BRepTools import BRepTools, BRepTools_ReShape
from OCP.collections import (
    HSequence_TopoDS_Shape,
    IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
)
from OCP.Geom import Geom_Curve, Geom_Surface
from OCP.Geom2d import Geom2d_Curve
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GeomTools import GeomTools
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec
from OCP.Precision import Precision
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.Standard import Standard_OutOfMemory
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_IN,
    TopAbs_REVERSED,
    TopAbs_ShapeEnum,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Vertex

from panelchemy import material_partition, occt

_MAX_SPLITS = 128
_MAX_DIRECTIONS = 32
_MAX_SECTION_EDGES = 16_384

type Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class _Contact:
    face_index: int
    vertex: TopoDS_Vertex
    wire_count: int


def _map(
    shape: TopoDS_Shape, kind: TopAbs_ShapeEnum
) -> IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher:
    result = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, kind, result)
    return result


def _contacts(shape: TopoDS_Shape) -> list[_Contact]:
    faces = _map(shape, TopAbs_FACE)
    result: list[_Contact] = []
    for index in range(1, faces.Extent() + 1):
        face = TopoDS.Face(faces.FindKey(index))
        wires, vertices = _map(face, TopAbs_WIRE), _map(face, TopAbs_VERTEX)
        outer = BRepTools.OuterWire_s(face)
        uses: Counter[int] = Counter()
        for wire_index in range(1, wires.Extent() + 1):
            if wires.FindKey(wire_index).IsSame(outer):
                continue
            members = _map(wires.FindKey(wire_index), TopAbs_VERTEX)
            uses.update(
                vertices.FindIndex(members.FindKey(i))
                for i in range(1, members.Extent() + 1)
            )
        result.extend(
            _Contact(index, TopoDS.Vertex(vertices.FindKey(vertex)), count)
            for vertex, count in sorted(uses.items())
            if count > 1
        )
    return result


def _selected_contacts(
    contacts: list[_Contact], points: Sequence[Point3] | None
) -> list[_Contact]:
    if points is None:
        return contacts
    return [
        contact
        for contact in contacts
        if any(
            BRep_Tool.Pnt_s(contact.vertex).IsEqual(
                gp_Pnt(*point), BRep_Tool.Tolerance_s(contact.vertex)
            )
            for point in points
        )
    ]


def _remaining_contacts(
    contacts: list[_Contact], originals: Sequence[tuple[Point3, float]] | None
) -> list[_Contact]:
    if originals is None:
        return contacts
    return [
        contact
        for contact in contacts
        if any(
            BRep_Tool.Pnt_s(contact.vertex).IsEqual(gp_Pnt(*point), tolerance)
            for point, tolerance in originals
        )
    ]


def open_wire_contact_points(
    shape: TopoDS_Shape, checker: BRepCheck_Analyzer | None = None
) -> tuple[Point3, ...]:
    """Locate coincident, topologically distinct open-wire endpoints.

    Only wires belonging to invalid faces are examined. A split common vertex
    leaves two odd-degree endpoints at the same physical point; ordinary edge
    splits retain even degree and do not trigger this diagnosis. Returned
    coordinates come directly from existing vertices, without averaging.
    """
    if shape.IsNull():
        return ()
    if checker is None:
        checker = BRepCheck_Analyzer(shape)
    found: list[tuple[Point3, float]] = []
    faces = _map(shape, TopAbs_FACE)
    for i in range(1, faces.Extent() + 1):
        face = faces.FindKey(i)
        if checker.IsValid(face):
            continue
        wires = _map(face, TopAbs_WIRE)
        for j in range(1, wires.Extent() + 1):
            vertices = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
            degree: Counter[int] = Counter()
            edges = TopExp_Explorer(wires.FindKey(j), TopAbs_EDGE)
            while edges.More():
                edge = TopoDS.Edge(edges.Current())
                for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
                    if not vertex.IsNull():
                        degree[vertices.Add(vertex)] += 1
                edges.Next()
            odd = [
                TopoDS.Vertex(vertices.FindKey(index))
                for index, count in degree.items()
                if count % 2
            ]
            for index, first in enumerate(odd):
                p = BRep_Tool.Pnt_s(first)
                tolerance = BRep_Tool.Tolerance_s(first)
                for second in odd[index + 1 :]:
                    other_tolerance = BRep_Tool.Tolerance_s(second)
                    if p.IsEqual(
                        BRep_Tool.Pnt_s(second), max(tolerance, other_tolerance)
                    ):
                        found.extend(
                            (
                                (p.Coord(), tolerance),
                                (BRep_Tool.Pnt_s(second).Coord(), other_tolerance),
                            )
                        )
    # Prefer the most precise existing endpoint when reports from adjacent
    # invalid wall faces describe the same physical contact.
    unique: list[tuple[Point3, float]] = []
    for point, tolerance in sorted(found, key=lambda item: (item[1], item[0])):
        if not any(
            gp_Pnt(*point).IsEqual(gp_Pnt(*other), max(tolerance, other_tolerance))
            for other, other_tolerance in unique
        ):
            unique.append((point, tolerance))
    return tuple(point for point, _ in unique)


def _boundary_ownership(shape: TopoDS_Shape) -> None:
    """Reject loose edges/vertices outside the certified face boundary graph."""
    faces, edges, vertices = (
        _map(shape, TopAbs_FACE),
        _map(shape, TopAbs_EDGE),
        _map(shape, TopAbs_VERTEX),
    )
    face_edges: set[int] = set()
    for i in range(1, faces.Extent() + 1):
        members = _map(faces.FindKey(i), TopAbs_EDGE)
        face_edges.update(
            edges.FindIndex(members.FindKey(j)) for j in range(1, members.Extent() + 1)
        )
    edge_vertices: set[int] = set()
    for i in range(1, edges.Extent() + 1):
        members = _map(edges.FindKey(i), TopAbs_VERTEX)
        edge_vertices.update(
            vertices.FindIndex(members.FindKey(j))
            for j in range(1, members.Extent() + 1)
        )
    if len(face_edges) != edges.Extent() or len(edge_vertices) != vertices.Extent():
        raise occt.KernelError("STEP topology contains loose edges or vertices")


def _weight(contacts: list[_Contact]) -> int:
    # A contact of three wires may require two cuts. Its unresolved pair count
    # must decrease even when one remaining touching vertex survives a cut.
    return sum(c.wire_count * (c.wire_count - 1) // 2 for c in contacts)


def _bounds(shape: TopoDS_Shape) -> tuple[tuple[float, ...], tuple[float, ...]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    if box.IsVoid():
        raise occt.KernelError("STEP topology needs a nonempty native boundary")
    return box.CornerMin().Coord(), box.CornerMax().Coord()


def _directions(face: TopoDS_Face, vertex: TopoDS_Vertex) -> list[float]:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Plane:
        raise occt.KernelError(
            "STEP point-contact normalization currently supports planar faces only"
        )
    position = surface.Plane().Position()
    x, y = gp_Vec(position.XDirection()), gp_Vec(position.YDirection())
    point = BRep_Tool.Pnt_s(vertex)
    angles: list[float] = []
    scales: list[float] = []
    edges = _map(face, TopAbs_EDGE)
    for i in range(1, edges.Extent() + 1):
        edge = TopoDS.Edge(edges.FindKey(i))
        first, last = TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)
        if not (first.IsSame(vertex) or last.IsSame(vertex)):
            continue
        if BRep_Tool.Degenerated_s(edge):
            raise occt.KernelError("STEP contact has an unsupported degenerate tangent")
        curve = BRepAdaptor_Curve(edge)
        for end, parameter, sign in (
            (first, curve.FirstParameter(), 1),
            (last, curve.LastParameter(), -1),
        ):
            if not end.IsSame(vertex):
                continue
            p, tangent = gp_Pnt(), gp_Vec()
            curve.D1(parameter, p, tangent)
            u, v = sign * tangent.Dot(x), sign * tangent.Dot(y)
            if not isfinite(u) or not isfinite(v) or hypot(u, v) == 0:
                raise occt.KernelError("STEP contact tangent is undefined")
            angles.append(atan2(v, u) % tau)
        parameters = [(curve.FirstParameter() + curve.LastParameter()) / 2]
        if not first.IsSame(vertex):
            parameters.append(curve.FirstParameter())
        if not last.IsSame(vertex):
            parameters.append(curve.LastParameter())
        for parameter in parameters:
            distance = point.Distance(curve.Value(parameter))
            if distance > 0:
                scales.append(distance)
    angles = sorted({*angles, *((angle + pi) % tau for angle in angles)})
    if len(angles) < 2 or not scales:
        raise occt.KernelError("STEP contact has no usable incident tangent sectors")
    if len(angles) > _MAX_DIRECTIONS:
        raise occt.KernelError("STEP contact exceeds the tangent-direction bound")
    tolerance = max(
        Precision.Confusion_s(),
        BRep_Tool.Tolerance_s(face),
        BRep_Tool.Tolerance_s(vertex),
    )
    # Probe progressively closer to the common point, always outside the
    # declared native uncertainty. ON/UNKNOWN classification is never accepted.
    radius = min(scales) / 32
    gaps = sorted(
        (
            ((b - a) % tau, (a + ((b - a) % tau) / 2) % tau)
            for a, b in zip(angles, [*angles[1:], angles[0]], strict=True)
        ),
        reverse=True,
    )
    candidates: list[float] = []
    for _, middle in gaps:
        for angle in (middle, (middle + pi) % tau):
            angle %= pi  # The two antipodal rays describe one cutting plane.
            if angle in candidates:
                continue
            direction = x.Multiplied(cos(angle)).Added(y.Multiplied(sin(angle)))
            for reduction in range(6):
                step = radius / 4.0**reduction
                if step / 4 <= tolerance * 8:
                    break
                if all(
                    BRepClass_FaceClassifier(
                        face, point.Translated(direction.Multiplied(offset)), tolerance
                    ).State()
                    == TopAbs_IN
                    for offset in (step, -step, step / 4, -step / 4)
                ):
                    candidates.append(angle)
                    break
    if not candidates:
        raise occt.KernelError(
            "STEP contact has no certified opposite material sectors"
        )
    return candidates[:_MAX_DIRECTIONS]


def _images(
    old: TopoDS_Shape,
    feature: BRepFeat_SplitShape,
    result: IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    kind: TopAbs_ShapeEnum,
) -> list[TopoDS_Shape]:
    images = list(feature.Modified(old))
    if not images and result.Contains(old):
        images = [old]
    if not images or any(
        image.ShapeType() != kind or not result.Contains(image) for image in images
    ):
        raise occt.KernelError("STEP face partition has incomplete native history")
    return images


def _geometry_bytes(
    geometry: Geom_Curve | Geom_Surface | Geom2d_Curve | None,
) -> bytes | None:
    if geometry is None:
        return None
    stream = BytesIO()
    GeomTools.Write_s(geometry, stream)
    return stream.getvalue()


def _edge_images(
    source: TopoDS_Shape,
    result: TopoDS_Shape,
    feature: BRepFeat_SplitShape,
    copier: BRepBuilderAPI_Copy,
) -> tuple[
    dict[int, list[TopoDS_Edge]], IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
]:
    old_edges, new_edges = _map(source, TopAbs_EDGE), _map(result, TopAbs_EDGE)
    image_map: dict[int, list[TopoDS_Edge]] = {}
    owners: dict[int, int] = {}
    for index in range(1, old_edges.Extent() + 1):
        old = TopoDS.Edge(old_edges.FindKey(index))
        images = [
            TopoDS.Edge(image)
            for image in _images(
                copier.ModifiedShape(old), feature, new_edges, TopAbs_EDGE
            )
        ]
        location = TopLoc_Location()
        curve = cast("Geom_Curve | None", BRep_Tool.Curve_s(old, location, 0.0, 0.0))
        geometry = _geometry_bytes(curve)
        intervals: list[tuple[float, float]] = []
        for edge in images:
            new_location = TopLoc_Location()
            if (
                _geometry_bytes(BRep_Tool.Curve_s(edge, new_location, 0.0, 0.0))
                != geometry
                or not new_location.IsEqual(location)
                or BRep_Tool.Tolerance_s(edge) != BRep_Tool.Tolerance_s(old)
                or BRep_Tool.Degenerated_s(edge) != BRep_Tool.Degenerated_s(old)
            ):
                raise occt.KernelError("STEP face partition changed an original edge")
            intervals.append(BRep_Tool.Range_s(edge))
            member = new_edges.FindIndex(edge)
            if member in owners:
                raise occt.KernelError(
                    "STEP face partition merged original edge images"
                )
            owners[member] = index
        intervals.sort()
        start, end = BRep_Tool.Range_s(old)
        if (
            intervals[0][0] != start
            or intervals[-1][1] != end
            or any(a[1] != b[0] for a, b in pairwise(intervals))
        ):
            raise occt.KernelError(
                "STEP face partition changed an original curve range"
            )
        image_map[index] = images
    return image_map, new_edges


def _uses(
    face: TopoDS_Face,
    edges: IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
) -> tuple[Counter[int], Counter[tuple[int, int]]]:
    chain: Counter[int] = Counter()
    other: Counter[tuple[int, int]] = Counter()
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        index = edges.FindIndex(edge)
        if not index:
            raise occt.KernelError("STEP face boundary has an untracked edge")
        if edge.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
            chain[index] += 1 if edge.Orientation() == TopAbs_FORWARD else -1
        else:
            other[index, int(edge.Orientation())] += 1
        explorer.Next()
    return chain, other


def _face_history(
    source: TopoDS_Shape,
    result: TopoDS_Shape,
    feature: BRepFeat_SplitShape,
    copier: BRepBuilderAPI_Copy,
    image_map: dict[int, list[TopoDS_Edge]],
    new_edges: IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
) -> None:
    old_faces, new_faces = _map(source, TopAbs_FACE), _map(result, TopAbs_FACE)
    old_edges = _map(source, TopAbs_EDGE)
    owned_faces: set[int] = set()
    for index in range(1, old_faces.Extent() + 1):
        old = TopoDS.Face(old_faces.FindKey(index))
        images = [
            TopoDS.Face(image)
            for image in _images(
                copier.ModifiedShape(old), feature, new_faces, TopAbs_FACE
            )
        ]
        location = TopLoc_Location()
        surface = BRep_Tool.Surface_s(old, location)
        geometry = _geometry_bytes(surface)
        planar = BRepAdaptor_Surface(old).GetType() == GeomAbs_Plane
        for face in images:
            new_location = TopLoc_Location()
            if (
                _geometry_bytes(BRep_Tool.Surface_s(face, new_location)) != geometry
                or not new_location.IsEqual(location)
                or face.Orientation() != old.Orientation()
                or BRep_Tool.Tolerance_s(face) != BRep_Tool.Tolerance_s(old)
                or BRep_Tool.NaturalRestriction_s(face)
                != BRep_Tool.NaturalRestriction_s(old)
            ):
                raise occt.KernelError(
                    "STEP face partition changed a supporting surface"
                )
            member = new_faces.FindIndex(face)
            if member in owned_faces:
                raise occt.KernelError(
                    "STEP face partition merged original face images"
                )
            owned_faces.add(member)
        if len(images) == 1 and images[0].IsEqual(old):
            continue
        expected: Counter[int] = Counter()
        expected_other: Counter[tuple[int, int]] = Counter()
        explorer = TopExp_Explorer(old, TopAbs_EDGE)
        while explorer.More():
            edge = TopoDS.Edge(explorer.Current())
            pieces = image_map[old_edges.FindIndex(edge)]
            pcurve_ranges: list[tuple[float, float]] = []
            for piece in pieces:
                member = new_edges.FindIndex(piece)
                if edge.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
                    expected[member] += (
                        1 if edge.Orientation() == TopAbs_FORWARD else -1
                    )
                else:
                    expected_other[member, int(edge.Orientation())] += 1
                if not planar:
                    # A periodic curved support needs its original UV curves,
                    # not just a coincident 3D boundary. Only planar faces gain
                    # new interior seams in this operation.
                    if len(images) != 1:
                        raise occt.KernelError(
                            "STEP partition subdivided a curved face"
                        )
                    oriented = TopoDS.Edge(piece.Oriented(edge.Orientation()))
                    if _geometry_bytes(
                        BRep_Tool.CurveOnSurface_s(oriented, images[0], 0.0, 0.0)
                    ) != _geometry_bytes(
                        BRep_Tool.CurveOnSurface_s(edge, old, 0.0, 0.0)
                    ):
                        raise occt.KernelError("STEP partition changed a native pcurve")
                    pcurve_ranges.append(BRep_Tool.Range_s(oriented, images[0]))
            if not planar:
                pcurve_ranges.sort()
                first, last = BRep_Tool.Range_s(edge, old)
                if (
                    pcurve_ranges[0][0] != first
                    or pcurve_ranges[-1][1] != last
                    or any(a[1] != b[0] for a, b in pairwise(pcurve_ranges))
                ):
                    raise occt.KernelError(
                        "STEP partition changed a native pcurve range"
                    )
            explorer.Next()
        actual: Counter[int] = Counter()
        actual_other: Counter[tuple[int, int]] = Counter()
        for face in images:
            chain, other = _uses(face, new_edges)
            actual.update(chain)
            actual_other.update(other)
        # Counter equality treats missing and zero counts equally. Opposite
        # uses of every newly introduced seam therefore cancel exactly. On an
        # unchanged plane, equality of these oriented compact boundary chains
        # certifies equality of the face domains, including every counter.
        if actual != expected or actual_other != expected_other:
            raise occt.KernelError("STEP face partition changed an oriented boundary")
        if not planar and any(member not in expected for member in actual):
            raise occt.KernelError("STEP partition added a curved-face seam")
        if not isclose(
            sum(occt.area(face) for face in images),
            occt.area(old),
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise occt.KernelError("STEP face partition changed a native face area")
    if len(owned_faces) != new_faces.Extent():
        raise occt.KernelError("STEP face partition introduced an untracked face")


def _vertex_records(shape: TopoDS_Shape) -> set[tuple[float, float, float, float]]:
    vertices = _map(shape, TopAbs_VERTEX)
    result: set[tuple[float, float, float, float]] = set()
    for i in range(1, vertices.Extent() + 1):
        vertex = TopoDS.Vertex(vertices.FindKey(i))
        point = BRep_Tool.Pnt_s(vertex)
        result.add((*point.Coord(), BRep_Tool.Tolerance_s(vertex)))
    return result


def _max_tolerance(shape: TopoDS_Shape) -> float:
    return max(
        BRep_Tool.Tolerance_s(entity)
        for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX)
        for members in [_map(shape, kind)]
        for i in range(1, members.Extent() + 1)
        for entity in [
            TopoDS.Face(members.FindKey(i))
            if kind == TopAbs_FACE
            else TopoDS.Edge(members.FindKey(i))
            if kind == TopAbs_EDGE
            else TopoDS.Vertex(members.FindKey(i))
        ]
    )


def _verify(
    source: TopoDS_Shape,
    result: TopoDS_Shape,
    feature: BRepFeat_SplitShape,
    copier: BRepBuilderAPI_Copy,
    seam_tolerance_bound: float,
) -> None:
    if not occt.valid(result):
        raise occt.KernelError("STEP face partition produced invalid native topology")
    _boundary_ownership(source)
    _boundary_ownership(result)
    if any(
        _map(source, kind).Extent() != _map(result, kind).Extent()
        for kind in (TopAbs_SOLID, TopAbs_SHELL)
    ):
        raise occt.KernelError("STEP face partition changed solid or shell membership")
    image_map, new_edges = _edge_images(source, result, feature, copier)
    _face_history(source, result, feature, copier, image_map, new_edges)
    old_vertices, new_vertices = (
        _map(source, TopAbs_VERTEX),
        _map(result, TopAbs_VERTEX),
    )
    for i in range(1, old_vertices.Extent() + 1):
        old = TopoDS.Vertex(old_vertices.FindKey(i))
        for image in _images(
            copier.ModifiedShape(old), feature, new_vertices, TopAbs_VERTEX
        ):
            vertex = TopoDS.Vertex(image)
            if BRep_Tool.Pnt_s(vertex).Coord() != BRep_Tool.Pnt_s(
                old
            ).Coord() or BRep_Tool.Tolerance_s(vertex) != BRep_Tool.Tolerance_s(old):
                raise occt.KernelError(
                    "STEP partition changed an original vertex image"
                )
    if not _vertex_records(source) <= _vertex_records(result):
        raise occt.KernelError("STEP face partition changed original vertex data")
    if _max_tolerance(result) > seam_tolerance_bound:
        raise occt.KernelError("STEP new seam exceeds its native tolerance bound")
    if any(
        abs(a - b) > Precision.Confusion_s()
        for first, second in zip(_bounds(source), _bounds(result), strict=True)
        for a, b in zip(first, second, strict=True)
    ):
        raise occt.KernelError("STEP face partition changed native bounds")
    for measure in (occt.area, occt.volume):
        if not isclose(measure(source), measure(result), rel_tol=1e-8, abs_tol=1e-7):
            raise occt.KernelError("STEP face partition changed native area or volume")


def _partition(
    source: TopoDS_Shape,
    contact: _Contact,
    angle: float,
    seam_tolerance_bound: float,
) -> tuple[TopoDS_Shape, dict[str, object]]:
    # Every attempt gets its own graph: failed section/split candidates cannot
    # contaminate either the caller's source or an already accepted partition.
    source_before = material_partition.shape_digest(source)
    copier = BRepBuilderAPI_Copy(source, True, False)
    base = copier.Shape()
    if not occt.valid(base):
        raise occt.KernelError("STEP face partition copy is invalid")
    source_contacts = _contacts(base)
    current = next(
        (
            item
            for item in source_contacts
            if item.face_index == contact.face_index
            and BRep_Tool.Pnt_s(item.vertex).IsEqual(
                BRep_Tool.Pnt_s(contact.vertex), 0.0
            )
        ),
        None,
    )
    if current is None:
        raise occt.KernelError("STEP copy lost a native point contact")
    face = TopoDS.Face(_map(base, TopAbs_FACE).FindKey(current.face_index))
    position = BRepAdaptor_Surface(face).Plane().Position()
    direction = (
        gp_Vec(position.XDirection())
        .Multiplied(cos(angle))
        .Added(gp_Vec(position.YDirection()).Multiplied(sin(angle)))
    )
    normal = gp_Vec(position.Direction()).Crossed(direction)
    point = BRep_Tool.Pnt_s(current.vertex)
    low, high = _bounds(face)
    extent = max(1.0, 2 * hypot(*(b - a for a, b in zip(low, high, strict=True))))
    tool = BRepBuilderAPI_MakeFace(
        gp_Pln(point, gp_Dir(normal)), -extent, extent, -extent, extent
    ).Face()
    section = BRepAlgoAPI_Section(face, tool, False)
    section.SetNonDestructive(True)
    section.SetRunParallel(False)
    section.ComputePCurveOn1(True)
    section.Approximation(False)
    section.Build()
    if not section.IsDone():
        raise occt.KernelError("STEP contact section failed")
    seam_shape = section.Shape()
    original_vertices = _map(face, TopAbs_VERTEX)
    section_vertices = _map(seam_shape, TopAbs_VERTEX)
    reshape = BRepTools_ReShape()
    for i in range(1, section_vertices.Extent() + 1):
        vertex = TopoDS.Vertex(section_vertices.FindKey(i))
        seam_point = BRep_Tool.Pnt_s(vertex)
        matches = [
            TopoDS.Vertex(original_vertices.FindKey(j))
            for j in range(1, original_vertices.Extent() + 1)
            if BRep_Tool.Pnt_s(TopoDS.Vertex(original_vertices.FindKey(j))).IsEqual(
                seam_point,
                BRep_Tool.Tolerance_s(TopoDS.Vertex(original_vertices.FindKey(j))),
            )
        ]
        if len(matches) > 1:
            raise occt.KernelError("STEP seam vertex has ambiguous native endpoints")
        if matches and not matches[0].IsSame(vertex):
            reshape.Replace(
                vertex.Oriented(TopAbs_FORWARD), matches[0].Oriented(TopAbs_FORWARD)
            )
    seam_shape = reshape.Apply(seam_shape)
    seams = _map(seam_shape, TopAbs_EDGE)
    if not 0 < seams.Extent() <= _MAX_SECTION_EDGES:
        raise occt.KernelError("STEP contact section exceeds its edge bound")
    feature = BRepFeat_SplitShape(base)
    sequence = HSequence_TopoDS_Shape()
    for index in range(1, seams.Extent() + 1):
        sequence.Append(seams.FindKey(index))
    wires = ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(sequence, 0.0, True)
    for index in range(1, wires.Length() + 1):
        feature.Add(TopoDS.Wire(wires.Value(index)), face)
    feature.Build()
    if not feature.IsDone():
        raise occt.KernelError("STEP contact face partition failed")
    result = feature.Shape()
    if material_partition.shape_digest(source) != source_before:
        raise occt.KernelError("STEP contact partition changed its immutable source")
    _verify(source, result, feature, copier, seam_tolerance_bound)
    if material_partition.shape_digest(source) != source_before:
        raise occt.KernelError("STEP contact audit changed its immutable source")
    after_contacts = _contacts(result)
    if _weight(after_contacts) >= _weight(source_contacts):
        raise occt.KernelError("STEP contact partition did not reduce point contacts")
    return result, {
        "face_index": current.face_index,
        "contact_point_mm": list(point.Coord()),
        "direction_radians": angle,
        "section_edges": seams.Extent(),
        "touching_vertices_before": len(source_contacts),
        "touching_vertices_after": len(after_contacts),
        "geometry_certificate": "native-support-range-and-oriented-boundary-history",
        "source_max_tolerance_mm": _max_tolerance(source),
        "new_seam_tolerance_bound_mm": seam_tolerance_bound,
    }


def normalize_step_topology(
    shape: TopoDS_Shape,
    *,
    contact_points: Sequence[Point3] | None = None,
) -> tuple[TopoDS_Shape, dict[str, object]]:
    """Partition contacts between planar inner wires, or explicitly fail.

    Original coordinates, tolerances and physical boundaries are preserved.
    New internal seams carry bounded native construction precision. Contacts
    involving the outer wire remain subject to the caller's STEP roundtrip
    validation. Supplied contact points restrict repair to diagnosed failures;
    unrelated inner-wire contacts remain unchanged. With no point filter all
    inner-wire contacts are considered. Unselected inputs take the no-copy path.
    """
    if not occt.valid(shape):
        raise occt.KernelError("STEP topology normalization needs valid native CAD")
    _boundary_ownership(shape)
    if contact_points is not None:
        contact_points = tuple(contact_points)
        if any(not all(isfinite(value) for value in p) for p in contact_points):
            raise occt.KernelError("STEP contact selection needs finite coordinates")
    contacts = _contacts(shape)
    selected = _selected_contacts(contacts, contact_points)
    originals = (
        None
        if contact_points is None
        else tuple(
            (BRep_Tool.Pnt_s(item.vertex).Coord(), BRep_Tool.Tolerance_s(item.vertex))
            for item in selected
        )
    )
    report: dict[str, object] = {
        "method": "planar-contact-face-partition",
        "contact_scope": "shared vertices between distinct inner face wires",
        "touching_vertices_before": len(contacts),
        "touching_vertices_after": len(contacts),
        "selected_vertices_before": len(selected),
        "selected_vertices_after": len(selected),
        "splits": [],
        "source_unchanged": True,
    }
    if not selected:
        return shape, report
    original = material_partition.shape_digest(shape)
    seam_tolerance_bound = _max_tolerance(shape) + Precision.Confusion_s()
    working = shape
    splits: list[dict[str, object]] = []
    try:
        while selected:
            if len(splits) >= _MAX_SPLITS:
                raise occt.KernelError(
                    "STEP contact normalization exceeds its split bound"
                )
            contact = selected[0]
            face = TopoDS.Face(_map(working, TopAbs_FACE).FindKey(contact.face_index))
            failures: list[str] = []
            for angle in _directions(face, contact.vertex):
                before_candidate = material_partition.shape_digest(working)
                try:
                    candidate, diagnostics = _partition(
                        working, contact, angle, seam_tolerance_bound
                    )
                    remaining = _contacts(candidate)
                    remaining_selected = _remaining_contacts(remaining, originals)
                    if _weight(remaining_selected) >= _weight(selected):
                        raise occt.KernelError(
                            "STEP partition did not reduce selected point contacts"
                        )
                except (MemoryError, Standard_OutOfMemory):
                    raise
                except Exception as error:
                    # OCCT exposes individual kernel exception classes without
                    # a shared Python Standard_Failure base in this binding.
                    failures.append(f"{type(error).__name__}: {error}")
                    continue
                finally:
                    # A failed native attempt must not become the next
                    # candidate's reference graph. This includes exceptions
                    # raised before _partition reaches its own source audit.
                    if material_partition.shape_digest(working) != before_candidate:
                        raise occt.KernelError(
                            "STEP candidate mutated its immutable working source"
                        )
                working = candidate
                splits.append(diagnostics)
                contacts, selected = remaining, remaining_selected
                break
            else:
                raise occt.KernelError(
                    "no certified native STEP contact partition: " + "; ".join(failures)
                )
    finally:
        if material_partition.shape_digest(shape) != original:
            raise occt.KernelError("STEP normalization changed its source CAD")
    report.update(
        touching_vertices_after=len(contacts),
        selected_vertices_after=0,
        splits=splits,
    )
    return working, report
