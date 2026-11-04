"""Registered, validated mesh and exact-CAD exports from unchanged relief CAD.

STL files retain their common modeling coordinates.
"""

from __future__ import annotations

import struct
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from math import fsum, hypot, isfinite, sqrt, ulp
from threading import Lock
from typing import Protocol, Self, cast

from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.collections import List_TopoDS_Shape, Sequence_TCollection_AsciiString
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Poly import Poly_Triangulation
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Iterator, TopoDS_Shape
from shapely import STRtree, union_all
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy import occt
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours

type Coordinate = tuple[float, float, float]
type Triangle = tuple[int, int, int]
type RationalCoordinate = tuple[Fraction, Fraction, Fraction]
_NAMES = ("black", "orange", "white")
_STEP_LOCK = Lock()


class ExportError(ValueError):
    """An export could not preserve a valid, registered printable model."""


class _Relief(Protocol):
    """Fields needed for export, independent of the model's file adapter."""

    profile: ConstructionProfile
    contours: PanelContours
    solid: TopoDS_Shape
    materials: dict[str, TopoDS_Shape]


@dataclass
class _CopiedRelief:
    profile: ConstructionProfile
    contours: PanelContours
    solid: TopoDS_Shape
    materials: dict[str, TopoDS_Shape]


def _positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        finite = isfinite(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not finite or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ExportConfig:
    linear_deflection_mm: float = 0.02
    angular_deflection_rad: float = 0.2
    max_triangles: int = 2_000_000

    def __post_init__(self) -> None:
        for name, value in (
            ("linear_deflection_mm", self.linear_deflection_mm),
            ("angular_deflection_rad", self.angular_deflection_rad),
        ):
            _positive_number(value, name)
        if self.angular_deflection_rad >= 3.141592653589793:
            raise ValueError("angular_deflection_rad must be less than pi")
        if type(self.max_triangles) is not int or self.max_triangles < 1:
            raise ValueError("max_triangles must be a positive integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        unknown = set(data) - {
            "linear_deflection_mm",
            "angular_deflection_rad",
            "max_triangles",
        }
        if unknown:
            raise ValueError(f"unknown export settings: {', '.join(sorted(unknown))}")
        linear = data.get("linear_deflection_mm", cls().linear_deflection_mm)
        angular = data.get("angular_deflection_rad", cls().angular_deflection_rad)
        maximum = data.get("max_triangles", cls().max_triangles)
        if (
            not isinstance(linear, (int, float))
            or not isinstance(angular, (int, float))
            or type(maximum) is not int
        ):
            raise ValueError(
                "export settings require numeric deflections and an integer max_triangles"
            )
        return cls(linear, angular, maximum)


@dataclass
class ExportBundle:
    files: dict[str, bytes]
    validation: dict[str, object]


@dataclass
class _Mesh:
    points: list[Coordinate]
    triangles: list[Triangle]


def _subtract(a: Coordinate, b: Coordinate) -> Coordinate:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Coordinate, b: Coordinate) -> float:
    return fsum(x * y for x, y in zip(a, b, strict=True))


def _cross(a: Coordinate, b: Coordinate) -> Coordinate:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Coordinate) -> float:
    return hypot(*a)


def _rational(point: Coordinate) -> RationalCoordinate:
    return (Fraction(point[0]), Fraction(point[1]), Fraction(point[2]))


def _exact_triangle_cross(
    a: RationalCoordinate, b: RationalCoordinate, c: RationalCoordinate
) -> RationalCoordinate:
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _exact_face_cross(
    triangle: Triangle, rational: Sequence[RationalCoordinate]
) -> RationalCoordinate:
    return _exact_triangle_cross(
        rational[triangle[0]], rational[triangle[1]], rational[triangle[2]]
    )


def _triangle_cross(points: Sequence[Coordinate]) -> Coordinate:
    """Keep exact positive faces when floating products cancel to zero.

    This fallback belongs to triangle area/normal predicates, never the generic
    vector cross product used by the intersection candidate loops.
    """
    raw = _cross(_subtract(points[1], points[0]), _subtract(points[2], points[0]))
    if raw != (0.0, 0.0, 0.0):
        return raw
    exact = _exact_triangle_cross(
        _rational(points[0]), _rational(points[1]), _rational(points[2])
    )
    raw = (float(exact[0]), float(exact[1]), float(exact[2]))
    if raw == (0.0, 0.0, 0.0) and any(exact):
        raise ExportError(
            "a positive triangle cross product cannot be represented numerically"
        )
    return raw


def _normal(points: Sequence[Coordinate]) -> Coordinate:
    raw = _triangle_cross(points)
    length = _length(raw)
    if length == 0:
        raise ExportError("mesh contains a zero-area triangle")
    return (raw[0] / length, raw[1] / length, raw[2] / length)


def _edge_distance(point: Coordinate, a: Coordinate, b: Coordinate) -> float:
    direction = _subtract(b, a)
    fraction = max(
        0.0, min(1.0, _dot(_subtract(point, a), direction) / _dot(direction, direction))
    )
    nearest = (
        a[0] + fraction * direction[0],
        a[1] + fraction * direction[1],
        a[2] + fraction * direction[2],
    )
    return _length(_subtract(point, nearest))


def _plane_chart(
    target: Sequence[Coordinate],
) -> tuple[Coordinate, Callable[[Coordinate], tuple[float, float]]]:
    normal = _normal(target)
    edge = max(
        range(3),
        key=lambda index: _length(_subtract(target[(index + 1) % 3], target[index])),
    )
    direction = _subtract(target[(edge + 1) % 3], target[edge])
    length = _length(direction)
    u = (direction[0] / length, direction[1] / length, direction[2] / length)
    v = _cross(normal, u)

    def projected(point: Coordinate) -> tuple[float, float]:
        relative = _subtract(point, target[0])
        return (_dot(relative, u), _dot(relative, v))

    return normal, projected


def _whole_boundary_contact(
    source: Sequence[Coordinate], target: Sequence[Coordinate], tolerance: float
) -> bool:
    """Bound the entire source triangle, including chords between its vertices.

    Orthogonal projection gives an isometric target-plane chart. Reserving the
    maximum normal displacement from the Euclidean error budget makes coverage
    by the buffered boundary a conservative 3D test, including the interior.
    """
    normal, projected = _plane_chart(target)
    plane_error = max(
        abs(_dot(_subtract(point, target[0]), normal)) for point in source
    )
    if plane_error > tolerance:
        return False

    target_points = [projected(point) for point in target]
    boundary = LineString([*target_points, target_points[0]])
    source_footprint = MultiPoint([projected(point) for point in source]).convex_hull
    radius = sqrt(max(0.0, tolerance**2 - plane_error**2))
    allowed = boundary.buffer(radius) if radius > 0 else boundary
    return source_footprint.difference(allowed).is_empty


def _bounds(shape: TopoDS_Shape) -> tuple[Coordinate, Coordinate]:
    bounds = Bnd_Box()
    # Membership/registration checks need geometric extents. Add_s includes
    # edge tolerances, which can exceed the crossing allowance after Booleans
    # even when the actual geometry remains exactly on its material planes.
    BRepBndLib.AddOptimal_s(shape, bounds, False, False)
    if bounds.IsVoid():
        raise ExportError("cannot export an empty shape")
    low, high = bounds.CornerMin(), bounds.CornerMax()
    return (low.X(), low.Y(), low.Z()), (high.X(), high.Y(), high.Z())


def _compact(points: Sequence[Coordinate], triangles: Sequence[Triangle]) -> _Mesh:
    used = sorted({index for triangle in triangles for index in triangle})
    remap = {old: new for new, old in enumerate(used)}
    return _Mesh(
        [points[i] for i in used],
        [(remap[a], remap[b], remap[c]) for a, b, c in triangles],
    )


def _segment_parameter(
    point: RationalCoordinate, start: RationalCoordinate, end: RationalCoordinate
) -> Fraction | None:
    if any(_exact_triangle_cross(start, end, point)):
        return None
    axis = next((axis for axis in range(3) if start[axis] != end[axis]), None)
    if axis is None:
        raise ExportError("a conformance edge has coincident endpoints")
    parameter = (point[axis] - start[axis]) / (end[axis] - start[axis])
    return parameter if 0 < parameter < 1 else None


def _conform_meshes(
    meshes: dict[str, _Mesh], max_triangles: int
) -> tuple[dict[str, _Mesh], dict[str, object]]:
    """Partition positive triangles at exact-zero bridge middle vertices.

    All arithmetic deciding collinearity and segment membership is exact on the
    stored binary floats. An AABB tree only finds candidates. Every split uses
    an existing global coordinate; positive surfaces and winding are preserved.
    Shared caps follow the same global edge order even with opposite winding.
    """
    if not meshes:
        raise ExportError("conformance requires nonempty material meshes")
    input_count = sum(len(mesh.triangles) for mesh in meshes.values())
    if input_count > max_triangles:
        raise ExportError(f"mesh conformance exceeds max_triangles={max_triangles:,}")
    for mesh in meshes.values():
        if (
            not mesh.points
            or not mesh.triangles
            or not all(isfinite(value) for point in mesh.points for value in point)
        ):
            raise ExportError(
                "mesh conformance requires nonempty finite coordinates and triangles"
            )
        if len(set(mesh.points)) != len(mesh.points):
            raise ExportError(
                "mesh conformance cannot merge coincident distinct vertices"
            )
        if any(
            len(set(triangle)) != 3
            or any(index < 0 or index >= len(mesh.points) for index in triangle)
            for triangle in mesh.triangles
        ):
            raise ExportError(
                "mesh conformance requires distinct valid triangle indices"
            )
    points = sorted({point for mesh in meshes.values() for point in mesh.points})
    global_ids = {point: index for index, point in enumerate(points)}
    rational = [_rational(point) for point in points]
    positive: dict[str, list[Triangle]] = {}
    removed: Counter[str] = Counter()
    middle_vertices: set[int] = set()
    for name, mesh in meshes.items():
        mapping = [global_ids[point] for point in mesh.points]
        positive[name] = []
        for a, b, c in mesh.triangles:
            triangle = (mapping[a], mapping[b], mapping[c])
            if any(_exact_face_cross(triangle, rational)):
                positive[name].append(triangle)
            else:
                left, middle, right = sorted(triangle)
                if (
                    _segment_parameter(
                        rational[middle], rational[left], rational[right]
                    )
                    is None
                ):
                    raise ExportError(
                        "an exact-zero triangle has no unique strict middle vertex"
                    )
                middle_vertices.add(middle)
                removed[name] += 1
    report: dict[str, object] = {
        "input_triangles": input_count,
        "removed_exact_zero_triangles": sum(removed.values()),
        "strict_middle_vertices": len(middle_vertices),
        "global_coordinates": len(points),
        "coordinate_movement_mm": 0.0,
    }
    if not middle_vertices:
        report.update(
            output_triangles=input_count,
            added_positive_triangles=0,
            edge_candidate_checks=0,
            materials={
                name: {
                    "removed_exact_zero_triangles": 0,
                    "added_positive_triangles": 0,
                    "added_local_coordinates": 0,
                }
                for name in meshes
            },
        )
        return dict(meshes), report
    edges = sorted(
        {
            (min(a, b), max(a, b))
            for triangles in positive.values()
            for triangle in triangles
            for a, b in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        }
    )
    if not edges:
        raise ExportError("exact-zero conformance would discard every positive face")
    edge_boxes = [
        box(
            min(points[a][0], points[b][0]),
            min(points[a][1], points[b][1]),
            max(points[a][0], points[b][0]),
            max(points[a][1], points[b][1]),
        )
        for a, b in edges
    ]
    tree = STRtree(edge_boxes)
    insertions: dict[tuple[int, int], list[tuple[Fraction, int]]] = defaultdict(list)
    checked = 0
    for middle in sorted(middle_vertices):
        point = points[middle]
        candidates = tree.query(Point(point[0], point[1]))
        for candidate in range(len(candidates)):
            checked += 1
            if checked > 5_000_000:
                raise ExportError(
                    "exact-zero conformance exceeds 5,000,000 edge candidates; review tessellation settings"
                )
            edge = edges[int(cast("int", candidates[candidate]))]
            a, b = edge
            if middle in edge or not min(points[a][2], points[b][2]) <= point[2] <= max(
                points[a][2], points[b][2]
            ):
                continue
            parameter = _segment_parameter(rational[middle], rational[a], rational[b])
            if parameter is not None:
                insertions[edge].append((parameter, middle))
    chains = {
        edge: [index for _, index in sorted(values)]
        for edge, values in insertions.items()
    }
    output: dict[str, _Mesh] = {}
    total = 0
    material_reports: dict[str, object] = {}
    for name, mesh in meshes.items():
        local_points = list(mesh.points)
        local_ids = {point: index for index, point in enumerate(local_points)}
        output_triangles: list[Triangle] = []
        added = 0
        for triangle in positive[name]:
            pieces = [triangle]
            boundary_edges = sorted(
                (min(a, b), max(a, b))
                for a, b in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                )
            )
            for edge in boundary_edges:
                if edge not in chains:
                    continue
                children: list[Triangle] = []
                found = 0
                for piece in pieces:
                    for position in range(3):
                        a, b, c = (
                            piece[position],
                            piece[(position + 1) % 3],
                            piece[(position + 2) % 3],
                        )
                        if (min(a, b), max(a, b)) == edge:
                            chain = [edge[0], *chains[edge], edge[1]]
                            if a != edge[0]:
                                chain.reverse()
                            if total + len(children) + len(chain) - 1 > max_triangles:
                                raise ExportError(
                                    f"conforming subdivision exceeds max_triangles={max_triangles:,}; review tessellation settings"
                                )
                            children.extend(
                                (chain[index], chain[index + 1], c)
                                for index in range(len(chain) - 1)
                            )
                            found += 1
                            break
                    else:
                        children.append(piece)
                if found != 1:
                    raise ExportError(
                        "conforming subdivision could not identify one original boundary edge"
                    )
                pieces = children
            added += len(pieces) - 1
            total += len(pieces)
            if total > max_triangles:
                raise ExportError(
                    f"conforming subdivision exceeds max_triangles={max_triangles:,}; review tessellation settings"
                )
            for piece in pieces:
                if len(pieces) > 1 and not any(_exact_face_cross(piece, rational)):
                    raise ExportError(
                        "conforming subdivision created an exact-zero child triangle"
                    )
                local: list[int] = []
                for index in piece:
                    coordinate = points[index]
                    if coordinate not in local_ids:
                        local_ids[coordinate] = len(local_points)
                        local_points.append(coordinate)
                    local.append(local_ids[coordinate])
                output_triangles.append((local[0], local[1], local[2]))
        if len({index for triangle in output_triangles for index in triangle}) != len(
            local_points
        ):
            raise ExportError(
                "conforming exact-zero cells would discard a material vertex"
            )
        output[name] = _Mesh(local_points, output_triangles)
        material_reports[name] = {
            "removed_exact_zero_triangles": removed[name],
            "added_positive_triangles": added,
            "added_local_coordinates": len(local_points) - len(mesh.points),
        }
    report.update(
        output_triangles=total,
        added_positive_triangles=total - input_count + sum(removed.values()),
        edge_candidate_checks=checked,
        materials=material_reports,
    )
    return output, report


def _partition(
    model: _Relief, config: ExportConfig
) -> tuple[dict[str, _Mesh], dict[str, object]]:
    if (
        not model.materials
        or set(model.materials) - set(_NAMES)
        or "black" not in model.materials
    ):
        raise ExportError("expected nonempty black/orange/white material solids")
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for name in _NAMES:
        if name in model.materials:
            _ = builder.Add(compound, model.materials[name])
    copied = BRepBuilderAPI_Copy(compound, True, False).Shape()
    arguments = List_TopoDS_Shape()
    children = TopoDS_Iterator(copied)
    while children.More():
        _ = arguments.Append(children.Value())
        children.Next()
    if len(model.materials) == 1:
        partition = copied
    else:
        operation = BOPAlgo_Builder()
        operation.SetArguments(arguments)
        operation.SetNonDestructive(True)
        operation.Perform()
        if operation.HasErrors():
            raise ExportError("OCCT could not create shared material interfaces")
        partition = operation.Shape()
    if not occt.valid(partition):
        raise ExportError("shared material partition is invalid")
    mesher = BRepMesh_IncrementalMesh(
        partition,
        config.linear_deflection_mm,
        False,
        config.angular_deflection_rad,
        False,
    )
    if not mesher.IsDone() or mesher.GetStatusFlags() != 0:
        raise ExportError(f"OCCT meshing failed (status {mesher.GetStatusFlags()})")
    bands = {
        "black": (0.0, model.profile.black_top),
        "orange": (model.profile.black_top, model.profile.orange_top),
        "white": (model.profile.orange_top, model.profile.white_top),
    }
    points: list[Coordinate] = []
    ids: dict[Coordinate, int] = {}
    faces: dict[str, list[Triangle]] = {
        name: [] for name in _NAMES if name in model.materials
    }
    cells: Counter[str] = Counter()
    discarded = 0
    raw_nodes = 0
    rounding = 0.0
    total_triangles = 0
    solids = TopExp_Explorer(partition, TopAbs_SOLID)
    while solids.More():
        solid = solids.Current()
        low, high = _bounds(solid)
        middle = (low[2] + high[2]) / 2
        matches = [
            name for name, (bottom, top) in bands.items() if bottom < middle < top
        ]
        if len(matches) != 1 or matches[0] not in faces:
            raise ExportError("partition cell cannot be assigned to a material band")
        name = matches[0]
        bottom, top = bands[name]
        if low[2] < bottom - 1e-6 or high[2] > top + 1e-6:
            raise ExportError("partition cell crosses a material boundary")
        cells[name] += 1
        surfaces = TopExp_Explorer(solid, TopAbs_FACE)
        while surfaces.More():
            face = TopoDS.Face(surfaces.Current())
            location = TopLoc_Location()
            triangulation = cast("object", BRep_Tool.Triangulation_s(face, location))
            if triangulation is None:
                raise ExportError("OCCT left an unmeshed face")
            triangulation = cast("Poly_Triangulation", triangulation)
            total_triangles += triangulation.NbTriangles()
            if total_triangles > config.max_triangles:
                raise ExportError(
                    f"mesh exceeds max_triangles={config.max_triangles:,}; review the tessellation settings"
                )
            mapping: list[int] = []
            for node in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node).Transformed(location.Transformation())
                raw = (point.X(), point.Y(), point.Z())
                coordinate = (round(raw[0], 12), round(raw[1], 12), round(raw[2], 12))
                rounding = max(
                    rounding,
                    sqrt(
                        sum((a - b) ** 2 for a, b in zip(raw, coordinate, strict=True))
                    ),
                )
                raw_nodes += 1
                if coordinate not in ids:
                    ids[coordinate] = len(points)
                    points.append(coordinate)
                mapping.append(ids[coordinate])
            for index in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(index)
                a, b, c = (mapping[tri.Value(k) - 1] for k in (1, 2, 3))
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                if len({a, b, c}) != 3:
                    discarded += 1
                else:
                    faces[name].append((a, b, c))
            surfaces.Next()
        solids.Next()
    if any(not triangles for triangles in faces.values()):
        raise ExportError("meshing omitted a material volume")
    if sum(occt.volume(shape) for shape in model.materials.values()) <= 0:
        raise ExportError("material CAD has no positive volume")
    return {name: _compact(points, triangles) for name, triangles in faces.items()}, {
        "partition_cells": dict(cells),
        "native_triangle_count": total_triangles,
        "coordinate_decimal_places": 12,
        "maximum_coordinate_rounding_mm": rounding,
        "face_node_weld_count": raw_nodes - len(points),
        "collapsed_native_triangles_discarded": discarded,
    }


def _triangle_intersections(
    mesh: _Mesh, uncertainty_mm: float
) -> tuple[int, int, float]:
    """Use geometric boundary predicates for numerical cone-seam slivers.

    A near-collinear triangle may touch another boundary within the recorded
    coordinate uncertainty. Its unstable normal must not invent penetration.
    All robust interior intersections fail; no triangle or vertex is removed.
    """
    triangles = [
        [mesh.points[index] for index in triangle] for triangle in mesh.triangles
    ]
    low = [
        tuple(min(point[axis] for point in triangle) for axis in range(3))
        for triangle in triangles
    ]
    high = [
        tuple(max(point[axis] for point in triangle) for axis in range(3))
        for triangle in triangles
    ]
    scale = max(1.0, max(abs(value) for point in mesh.points for value in point))
    epsilon = uncertainty_mm + ulp(scale) * 16
    boxes = [
        box(a[0] - epsilon, a[1] - epsilon, b[0] + epsilon, b[1] + epsilon)
        for a, b in zip(low, high, strict=True)
    ]
    tree = STRtree(boxes)
    normals = [_normal(triangle) for triangle in triangles]
    skinny: list[bool] = []
    for triangle in triangles:
        longest = max(
            _length(_subtract(triangle[(i + 1) % 3], triangle[i])) for i in range(3)
        )
        cross = _cross(
            _subtract(triangle[1], triangle[0]), _subtract(triangle[2], triangle[0])
        )
        skinny.append(_length(cross) / longest <= epsilon * 4)
    # OCCT can bridge two subdivisions of the same cone generator with a
    # numerical sliver. Discover that topological evidence before permitting
    # unindexed boundary contact between the otherwise nonadjacent triangles.
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    vertex_faces: dict[int, set[int]] = defaultdict(set)
    for face, triangle in enumerate(mesh.triangles):
        for vertex in triangle:
            vertex_faces[vertex].add(face)
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_faces[(min(a, b), max(a, b))].append(face)
    neighbors: list[set[int]] = [set() for _ in triangles]
    for adjacent in edge_faces.values():
        for face in adjacent:
            neighbors[face].update(set(adjacent) - {face})
    seam_groups: dict[int, set[int]] = defaultdict(set)
    seam_segments: dict[int, list[tuple[Coordinate, Coordinate]]] = {}
    remaining = {i for i, is_skinny in enumerate(skinny) if is_skinny}
    group = 0
    while remaining:
        pending = [next(iter(remaining))]
        members: set[int] = set()
        boundary_faces: set[int] = set()
        while pending:
            face = pending.pop()
            if face not in remaining:
                continue
            remaining.remove(face)
            members.add(face)
            boundary_faces.update(neighbors[face])
            for vertex in mesh.triangles[face]:
                boundary_faces.update(vertex_faces[vertex])
            pending.extend(neighbors[face] & remaining)
        for face in members | boundary_faces:
            seam_groups[face].add(group)
        segments: list[tuple[Coordinate, Coordinate]] = []
        for face in members:
            seam_triangle = triangles[face]
            edge = max(
                range(3),
                key=lambda edge: _length(
                    _subtract(seam_triangle[(edge + 1) % 3], seam_triangle[edge])
                ),
            )
            segments.append((seam_triangle[edge], seam_triangle[(edge + 1) % 3]))
        seam_segments[group] = segments
        group += 1
    checked = 0
    numeric_contacts: set[tuple[int, int]] = set()
    for i, bounds in enumerate(boxes):
        candidates = tree.query(bounds)
        for candidate in range(len(candidates)):
            j = int(cast("int", candidates[candidate]))
            if (
                j <= i
                or high[i][2] < low[j][2] - epsilon
                or high[j][2] < low[i][2] - epsilon
            ):
                continue
            checked += 1
            if checked > 5_000_000:
                raise ExportError(
                    "self-intersection validation exceeds 5,000,000 candidate pairs; review tessellation settings"
                )
            shared = set(mesh.triangles[i]) & set(mesh.triangles[j])
            if len(shared) == 3:
                raise ExportError("mesh contains duplicate triangles")
            a, b = triangles[i], triangles[j]
            seam_related = bool(seam_groups[i] & seam_groups[j])
            seam_contact = False
            for index, source, target in ((i, a, b), (j, b, a)):
                if (
                    seam_related
                    and skinny[index]
                    and _whole_boundary_contact(source, target, epsilon * 4)
                ):
                    seam_contact = True
                    break
            if seam_contact:
                numeric_contacts.add((i, j))
                continue
            na, nb = normals[i], normals[j]
            da = [_dot(_subtract(point, b[0]), nb) for point in a]
            db = [_dot(_subtract(point, a[0]), na) for point in b]
            if (
                all(d > epsilon for d in da)
                or all(d < -epsilon for d in da)
                or all(d > epsilon for d in db)
                or all(d < -epsilon for d in db)
            ):
                continue
            if (
                max(abs(d) for d in da) <= epsilon
                and max(abs(d) for d in db) <= epsilon
            ):
                _, projected = _plane_chart(a)
                projected_a = Polygon([projected(point) for point in a])
                projected_b = Polygon([projected(point) for point in b])
                # Reserve the normal offset so a chart-space tolerance does
                # not quietly enlarge the reported Euclidean contact bound.
                radius = sqrt((epsilon * 4) ** 2 - max(abs(d) for d in db) ** 2)
                overlap = projected_a.intersection(projected_b)
                if overlap.is_empty:
                    continue
                shared_points = [projected(mesh.points[k]) for k in sorted(shared)]
                allowed = (
                    LineString(shared_points)
                    if len(shared_points) == 2
                    else Point(shared_points[0])
                    if shared_points
                    else Point()
                )
                if seam_related:
                    boundary_contact = projected_a.boundary.buffer(radius).intersection(
                        projected_b.boundary.buffer(radius)
                    )
                    seam_contact_region = union_all(
                        [
                            LineString([projected(point) for point in segment]).buffer(
                                radius
                            )
                            for group in seam_groups[i] & seam_groups[j]
                            for segment in seam_segments[group]
                        ]
                    )
                    if overlap.difference(
                        boundary_contact.intersection(seam_contact_region)
                    ).is_empty:
                        numeric_contacts.add((i, j))
                        continue
                if not overlap.difference(allowed.buffer(radius)).is_empty:
                    raise ExportError("mesh has intersecting or overlapping triangles")
                continue
            if len(shared) == 2:
                continue  # Distinct planes meet only on their common indexed edge.
            for source, target, distances, normal, target_index in (
                (a, b, da, nb, j),
                (b, a, db, na, i),
            ):
                for edge in range(3):
                    p, q = source[edge], source[(edge + 1) % 3]
                    dp, dq = distances[edge], distances[(edge + 1) % 3]
                    hits: list[Coordinate] = []
                    if abs(dp) <= epsilon:
                        hits.append(p)
                    if dp * dq < 0:
                        fraction = dp / (dp - dq)
                        hits.append(
                            (
                                p[0] + (q[0] - p[0]) * fraction,
                                p[1] + (q[1] - p[1]) * fraction,
                                p[2] + (q[2] - p[2]) * fraction,
                            )
                        )
                    for hit in hits:
                        inside = all(
                            min(point[axis] for point in target) - epsilon * 4
                            <= hit[axis]
                            <= max(point[axis] for point in target) + epsilon * 4
                            for axis in range(3)
                        ) and all(
                            _dot(
                                _cross(
                                    _subtract(target[(k + 1) % 3], target[k]),
                                    _subtract(hit, target[k]),
                                ),
                                normal,
                            )
                            >= -epsilon
                            * _length(_subtract(target[(k + 1) % 3], target[k]))
                            for k in range(3)
                        )
                        if (
                            skinny[target_index]
                            and min(
                                _edge_distance(hit, target[k], target[(k + 1) % 3])
                                for k in range(3)
                            )
                            > epsilon * 4
                        ):
                            inside = False
                        if inside and not any(
                            _length(_subtract(hit, mesh.points[k])) <= epsilon * 4
                            for k in shared
                        ):
                            if (
                                seam_related
                                and any(
                                    _edge_distance(hit, start, end) <= epsilon * 4
                                    for group in seam_groups[i] & seam_groups[j]
                                    for start, end in seam_segments[group]
                                )
                                and all(
                                    min(
                                        _edge_distance(
                                            hit, triangle[k], triangle[(k + 1) % 3]
                                        )
                                        for k in range(3)
                                    )
                                    <= epsilon * 4
                                    for triangle in (a, b)
                                )
                            ):
                                numeric_contacts.add((i, j))
                                continue
                            raise ExportError("mesh has intersecting triangles")
    return checked, len(numeric_contacts), epsilon * 4


def _validate_mesh(mesh: _Mesh, *, uncertainty_mm: float = 0.0) -> dict[str, object]:
    if not mesh.triangles or not mesh.points:
        raise ExportError("mesh is empty")
    points = mesh.points
    if not all(isfinite(value) for point in points for value in point) or len(
        set(points)
    ) != len(points):
        raise ExportError("mesh has nonfinite or coincident distinct vertices")
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    links: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    adjacency: list[set[int]] = [set() for _ in mesh.triangles]
    directed: Counter[tuple[int, int]] = Counter()
    for index, triangle in enumerate(mesh.triangles):
        if len(set(triangle)) != 3 or any(i < 0 or i >= len(points) for i in triangle):
            raise ExportError("mesh contains a collapsed triangle or invalid index")
        a, b, c = triangle
        if _length(_triangle_cross([points[a], points[b], points[c]])) == 0:
            raise ExportError("mesh contains a zero-area triangle")
        for x, y in ((a, b), (b, c), (c, a)):
            edges[(min(x, y), max(x, y))].append(index)
            directed[(x, y)] += 1
        for center, x, y in ((a, b, c), (b, c, a), (c, a, b)):
            links[center][x].add(y)
            links[center][y].add(x)
    if len(links) != len(points):
        raise ExportError("mesh contains unused vertices")
    for (a, b), incident in edges.items():
        if len(incident) != 2 or directed[(a, b)] != 1 or directed[(b, a)] != 1:
            raise ExportError(
                "mesh is open, inconsistently oriented, or has a nonmanifold edge"
            )
        x, y = incident
        adjacency[x].add(y)
        adjacency[y].add(x)
    for link in links.values():
        if any(len(neighbors) != 2 for neighbors in link.values()):
            raise ExportError("mesh has a nonmanifold vertex link")
        visited: set[int] = set()
        pending = [next(iter(link))]
        while pending:
            current = pending.pop()
            if current not in visited:
                visited.add(current)
                pending.extend(link[current] - visited)
        if len(visited) != len(link):
            raise ExportError("mesh has a pinched nonmanifold vertex")
    pending_faces = set(range(len(mesh.triangles)))
    volumes: list[float] = []
    while pending_faces:
        component: list[int] = []
        pending = [next(iter(pending_faces))]
        while pending:
            face = pending.pop()
            if face in pending_faces:
                pending_faces.remove(face)
                component.append(face)
                pending.extend(adjacency[face] & pending_faces)
        origin = points[mesh.triangles[component[0]][0]]
        signed = (
            fsum(
                _dot(
                    _subtract(points[a], origin),
                    _cross(_subtract(points[b], origin), _subtract(points[c], origin)),
                )
                for a, b, c in (mesh.triangles[i] for i in component)
            )
            / 6
        )
        if signed <= 0:
            raise ExportError("mesh shell has nonpositive signed volume")
        volumes.append(signed)
    pairs, numeric_contacts, tolerance = _triangle_intersections(mesh, uncertainty_mm)
    return {
        "vertices": len(points),
        "triangles": len(mesh.triangles),
        "closed_shells": len(volumes),
        "volume_mm3": sum(volumes),
        "bounds_mm": [
            [min(point[axis] for point in points) for axis in range(3)],
            [max(point[axis] for point in points) for axis in range(3)],
        ],
        "self_intersection_candidate_pairs_checked": pairs,
        "numerical_boundary_contact_pairs": numeric_contacts,
        "intersection_coordinate_uncertainty_mm": uncertainty_mm,
        "intersection_boundary_tolerance_mm": tolerance,
        "nonmanifold_edges": 0,
        "nonmanifold_vertices": 0,
    }


def _key(mesh: _Mesh, triangle: Triangle) -> tuple[Coordinate, Coordinate, Coordinate]:
    return cast(
        "tuple[Coordinate, Coordinate, Coordinate]",
        tuple(sorted(mesh.points[index] for index in triangle)),
    )


def _interfaces(meshes: dict[str, _Mesh], model: _Relief) -> dict[str, object]:
    result: dict[str, object] = {}
    for lower, upper, height in (
        ("black", "orange", model.profile.black_top),
        ("orange", "white", model.profile.orange_top),
    ):
        if upper not in meshes:
            continue
        if lower not in meshes:
            raise ExportError("an upper material lacks its supporting material")

        def cap(
            mesh: _Mesh, height: float
        ) -> dict[tuple[Coordinate, Coordinate, Coordinate], float]:
            output: dict[tuple[Coordinate, Coordinate, Coordinate], float] = {}
            for triangle in mesh.triangles:
                coords = [mesh.points[i] for i in triangle]
                # STL rounds the band plane too; use the same float32 image of H.
                if all(
                    p[2]
                    in (
                        height,
                        round(height, 12),
                        _float32((0, 0, round(height, 12)))[2],
                    )
                    for p in coords
                ):
                    a, b, c = coords
                    output[_key(mesh, triangle)] = _cross(
                        _subtract(b, a), _subtract(c, a)
                    )[2]
            return output

        top, bottom = cap(meshes[lower], height), cap(meshes[upper], height)
        if (
            not bottom
            or set(bottom) - set(top)
            or any(bottom[key] >= 0 or top[key] <= 0 for key in bottom)
        ):
            raise ExportError(
                f"serialized {lower}/{upper} material interfaces do not match with opposite winding"
            )
        result[f"{lower}/{upper}"] = {
            "shared_triangles": len(bottom),
            "missing_shared_triangles": 0,
        }
    return result


def _combined(meshes: dict[str, _Mesh]) -> _Mesh:
    points: list[Coordinate] = []
    ids: dict[Coordinate, int] = {}
    faces: dict[tuple[int, int, int], list[Triangle]] = defaultdict(list)
    for mesh in meshes.values():
        mapping: list[int] = []
        for point in mesh.points:
            if point not in ids:
                ids[point] = len(points)
                points.append(point)
            mapping.append(ids[point])
        for a, b, c in mesh.triangles:
            triangle = (mapping[a], mapping[b], mapping[c])
            key = cast("Triangle", tuple(sorted(triangle)))
            faces[key].append(triangle)
    boundary: list[Triangle] = []
    for candidates in faces.values():
        if len(candidates) == 1:
            boundary.append(candidates[0])
        elif len(candidates) == 2:
            a, b = candidates
            if b not in ((a[0], a[2], a[1]), (a[2], a[1], a[0]), (a[1], a[0], a[2])):
                raise ExportError("shared material triangles have inconsistent winding")
        else:
            raise ExportError("unexpected material triangle multiplicity")
    return _compact(points, boundary)


def _stl(mesh: _Mesh) -> bytes:
    output = bytearray(b"Panelchemy registered millimeter mesh".ljust(80, b"\0"))
    output.extend(struct.pack("<I", len(mesh.triangles)))
    for triangle in mesh.triangles:
        vertices = [mesh.points[index] for index in triangle]
        normal = _normal(vertices)
        try:
            output.extend(
                struct.pack(
                    "<12fH",
                    *normal,
                    *(value for point in vertices for value in point),
                    0,
                )
            )
        except (OverflowError, struct.error) as error:
            raise ExportError(
                "STL float32 cannot represent the model coordinates"
            ) from error
    return bytes(output)


def _read_stl(data: bytes) -> _Mesh:
    if len(data) < 84:
        raise ExportError("truncated binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise ExportError("binary STL triangle count does not match its bytes")
    points: list[Coordinate] = []
    ids: dict[Coordinate, int] = {}
    triangles: list[Triangle] = []
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        triangle: list[int] = []
        for offset in (3, 6, 9):
            point = (values[offset], values[offset + 1], values[offset + 2])
            if point not in ids:
                ids[point] = len(points)
                points.append(point)
            triangle.append(ids[point])
        triangles.append((triangle[0], triangle[1], triangle[2]))
    return _Mesh(points, triangles)


def _step(shape: TopoDS_Shape) -> tuple[bytes, dict[str, object]]:
    copied = BRepBuilderAPI_Copy(shape, True, False).Shape()
    stream = BytesIO()
    # This binding does not expose DESTEP_Parameters. OCCT's transfer consults
    # the global output unit despite StepModel.SetWriteLengthUnit. Serialize
    # Panelchemy writers and restore the external setting even after failure.
    # Other code mutating OCCT globals must coordinate with this module's calls.
    with _STEP_LOCK:
        writer = STEPControl_Writer()
        previous_unit = Interface_Static.CVal_s("write.step.unit")
        if not Interface_Static.SetCVal_s("write.step.unit", "MM"):
            raise ExportError("OCCT could not select millimeters for STEP")
        try:
            writer.Model().SetLocalLengthUnit(1.0)
            transfer = cast("Callable[[TopoDS_Shape, object], object]", writer.Transfer)
            if transfer(copied, STEPControl_AsIs) != IFSelect_RetDone:
                raise ExportError("STEP transfer failed")
            if writer.WriteStream(stream) != IFSelect_RetDone:
                raise ExportError("STEP serialization failed")
        finally:
            if not Interface_Static.SetCVal_s("write.step.unit", previous_unit):
                raise ExportError(
                    "OCCT could not restore the previous STEP output unit"
                )
    data = stream.getvalue()
    reader = STEPControl_Reader()
    if (
        cast("Callable[[str, BytesIO], object]", reader.ReadStream)(
            "relief.step", BytesIO(data)
        )
        != IFSelect_RetDone
    ):
        raise ExportError("serialized STEP cannot be read back")
    lengths = Sequence_TCollection_AsciiString()
    angles = Sequence_TCollection_AsciiString()
    solid_angles = Sequence_TCollection_AsciiString()
    reader.FileUnits(lengths, angles, solid_angles)
    length_units = [
        lengths.Value(index).ToCString() for index in range(1, lengths.Length() + 1)
    ]
    if not length_units or any(
        unit.lower() not in {"millimetre", "millimeter", "mm"} for unit in length_units
    ):
        raise ExportError("serialized STEP length contexts are not millimeters")
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() < 1:
        raise ExportError("serialized STEP cannot transfer its CAD roots")
    roundtrip = reader.OneShape()
    original, restored = occt.volume(shape), occt.volume(roundtrip)
    if not occt.valid(roundtrip) or abs(original - restored) > max(
        1e-7, original * 1e-8
    ):
        raise ExportError("STEP roundtrip changed or invalidated the CAD solid")
    before, after = _bounds(shape), _bounds(roundtrip)
    if (
        max(
            abs(a - b)
            for original_corner, restored_corner in zip(before, after, strict=True)
            for a, b in zip(original_corner, restored_corner, strict=True)
        )
        > 1e-6
    ):
        raise ExportError("STEP roundtrip changed model registration or dimensions")
    return data, {
        "valid": True,
        "volume_mm3": restored,
        "volume_difference_mm3": abs(original - restored),
        "units": "millimeter",
        "serialized_length_units": length_units,
        "timestamp_may_vary": True,
        "unit_configuration": "Scoped global write.step.unit=MM, restored under a Panelchemy writer lock; unrelated OCCT global mutations require external coordination",
    }


def _float32(point: Coordinate) -> Coordinate:
    if not all(isfinite(value) for value in point):
        raise ExportError("STL float32 requires finite model coordinates")
    try:
        converted = struct.unpack("<3f", struct.pack("<3f", *point))
    except (OverflowError, struct.error) as error:
        raise ExportError(
            "STL float32 cannot represent the model coordinates"
        ) from error
    if not all(isfinite(value) for value in converted):
        raise ExportError("STL float32 cannot represent finite model coordinates")
    return converted


def _quantize_meshes(
    meshes: dict[str, _Mesh], budget: float
) -> tuple[dict[str, _Mesh], dict[str, object]]:
    """Measure global native-to-float32 displacement before any conformance."""
    conversions = {
        point: _float32(point) for mesh in meshes.values() for point in mesh.points
    }
    if len(set(conversions.values())) != len(conversions):
        raise ExportError(
            "STL float32 collapses distinct feature vertices; the model cannot be represented at this registration"
        )
    error = max(
        (
            _length(_subtract(point, converted))
            for point, converted in conversions.items()
        ),
        default=0.0,
    )
    if error > budget:
        raise ExportError("STL float32 coordinate error exceeds linear_deflection_mm")
    converted_meshes = {
        name: _Mesh([conversions[point] for point in mesh.points], list(mesh.triangles))
        for name, mesh in meshes.items()
    }
    return converted_meshes, {
        "maximum_coordinate_error_mm": error,
        "global_coordinates": len(conversions),
        "coordinate_collapses": 0,
    }


def _check_float32(original: _Mesh, serialized: _Mesh, budget: float) -> float:
    converted = [_float32(point) for point in original.points]
    if len(set(converted)) != len(original.points):
        raise ExportError(
            "STL float32 collapses distinct feature vertices; the model cannot be represented at this registration"
        )
    if len(serialized.points) != len(original.points) or len(
        serialized.triangles
    ) != len(original.triangles):
        raise ExportError("STL serialization changed the mesh topology")
    expected_triangles = [
        tuple(converted[index] for index in triangle) for triangle in original.triangles
    ]
    actual_triangles = [
        tuple(serialized.points[index] for index in triangle)
        for triangle in serialized.triangles
    ]
    if actual_triangles != expected_triangles:
        raise ExportError("STL serialization changed oriented triangle coordinates")
    error = max(
        sqrt(sum((a - b) ** 2 for a, b in zip(point, rounded, strict=True)))
        for point, rounded in zip(original.points, converted, strict=True)
    )
    if not isfinite(error) or error > budget:
        raise ExportError("STL float32 coordinate error exceeds linear_deflection_mm")
    return error


def _footprints(
    meshes: dict[str, _Mesh], model: _Relief, uncertainty_mm: float
) -> dict[str, object]:
    """Check planar source tops and the vertical black wall after serialization."""
    scale = max(1.0, model.profile.long_edge_mm)
    tolerance = uncertainty_mm * 4 + ulp(scale) * 64

    def cap(name: str, height: float) -> BaseGeometry:
        if name not in meshes:
            return Polygon()
        mesh = meshes[name]
        polygons: list[Polygon] = []
        heights = (height, round(height, 12), _float32((0, 0, round(height, 12)))[2])
        for triangle in mesh.triangles:
            points = [mesh.points[index] for index in triangle]
            if all(point[2] in heights for point in points):
                polygons.append(Polygon([(point[0], point[1]) for point in points]))
        return union_all(polygons)

    checks: list[tuple[str, BaseGeometry, BaseGeometry]] = [
        ("silhouette_bottom", model.contours.silhouette, cap("black", 0.0))
    ]
    if "white" in meshes:
        checks.append(
            (
                "white_top",
                union_all(model.contours.white),
                cap("white", model.profile.white_top),
            )
        )

    def topology(geometry: BaseGeometry) -> tuple[int, int]:
        if geometry.is_empty:
            return (0, 0)
        polygons = (
            [geometry]
            if isinstance(geometry, Polygon)
            else list(geometry.geoms)
            if isinstance(geometry, MultiPolygon)
            else []
        )
        if not polygons:
            raise ExportError("mesh cap contains a non-polygonal footprint")
        return (len(polygons), sum(len(polygon.interiors) for polygon in polygons))

    result: dict[str, object] = {}
    for name, expected, actual in checks:
        error = expected.symmetric_difference(actual).area
        budget = expected.length * tolerance * 2 + tolerance**2 * 4
        if (
            topology(expected) != topology(actual)
            or error > budget
            or expected.hausdorff_distance(actual) > tolerance * 2
        ):
            raise ExportError(
                f"serialized {name} footprint exceeds its coordinate error budget"
            )
        result[name] = {
            "symmetric_difference_mm2": error,
            "area_error_budget_mm2": budget,
            "boundary_tolerance_mm": tolerance * 2,
        }
    if "orange" in meshes:
        source = union_all([*model.contours.orange, *model.contours.white])
        top = cap("orange", model.profile.orange_top)
        if not source.difference(top.buffer(tolerance)).is_empty:
            raise ExportError(
                "serialized orange top no longer covers the source feature tops"
            )
        result["orange_source_top_coverage"] = True
    black = meshes["black"]
    boundary = model.contours.silhouette.boundary
    for triangle in black.triangles:
        points = [black.points[index] for index in triangle]
        if len({point[2] for point in points}) > 1 and any(
            boundary.distance(Point(point[0], point[1])) > tolerance for point in points
        ):
            raise ExportError(
                "serialized black outer wall is not vertical on the silhouette boundary"
            )
    result["vertical_outer_wall"] = True
    return result


def _registered_bounds(
    mesh: _Mesh, cad: TopoDS_Shape, coordinate_error: float, config: ExportConfig
) -> None:
    low, high = _bounds(cad)
    for corner, measured in (
        (low, [min(point[axis] for point in mesh.points) for axis in range(3)]),
        (high, [max(point[axis] for point in mesh.points) for axis in range(3)]),
    ):
        for axis in range(3):
            tolerance = (
                1e-6 + coordinate_error
                if axis == 2
                else config.linear_deflection_mm + coordinate_error + 1e-6
            )
            if abs(corner[axis] - measured[axis]) > tolerance:
                raise ExportError(
                    "serialized mesh changed CAD registration or dimensions"
                )


def _export_model(
    model: _Relief,
    config: ExportConfig | None = None,
) -> ExportBundle:
    """Return bytes only after CAD, mesh, interface and serialized checks pass.

    No minimum-feature cleanup is performed. Repeated-coordinate cone tips and
    exact-zero bridges receive reported conformance without moving coordinates.
    Distinct-coordinate float32 merges or any remaining mesh defect reject the
    complete export.
    """
    config = config or ExportConfig()
    # Some kernel queries populate geometry caches. Keep even those queries off
    # the caller's shapes; exact CAD snapshots must survive exporting unchanged.
    model = _CopiedRelief(
        model.profile,
        model.contours,
        BRepBuilderAPI_Copy(model.solid, True, False).Shape(),
        {
            name: BRepBuilderAPI_Copy(shape, True, False).Shape()
            for name, shape in model.materials.items()
        },
    )
    if not occt.valid(model.solid) or occt.volume(model.solid) <= 0:
        raise ExportError("source relief CAD must be valid with positive volume")
    try:
        meshes, native = _partition(model, config)
    except RuntimeError as error:
        raise ExportError("OCCT material partition or tessellation failed") from error
    meshes, native_conformance = _conform_meshes(meshes, config.max_triangles)
    native["native_conformance"] = native_conformance
    interfaces = _interfaces(meshes, model)
    native_meshes = {**meshes, "relief": _combined(meshes)}
    rounding_error = cast("float", native["maximum_coordinate_rounding_mm"])
    for mesh in native_meshes.values():
        _ = _validate_mesh(mesh, uncertainty_mm=rounding_error)
    quantized, quantization = _quantize_meshes(meshes, config.linear_deflection_mm)
    final_parts, float_conformance = _conform_meshes(quantized, config.max_triangles)
    native["float32_quantization"] = quantization
    native["float32_conformance"] = float_conformance
    float_error = cast("float", quantization["maximum_coordinate_error_mm"])
    _ = _interfaces(final_parts, model)
    expected_meshes = {**final_parts, "relief": _combined(final_parts)}
    files: dict[str, bytes] = {}
    reports: dict[str, object] = {}
    serialized_meshes: dict[str, _Mesh] = {}
    for name, mesh in expected_meshes.items():
        data = _stl(mesh)
        roundtrip = _read_stl(data)
        # Expected meshes already contain the globally conformed float32
        # coordinates. Compare the actual bytes to that final connectivity.
        _ = _check_float32(mesh, roundtrip, config.linear_deflection_mm)
        report = _validate_mesh(roundtrip, uncertainty_mm=rounding_error + float_error)
        if name == "relief" and report["closed_shells"] != 1:
            raise ExportError("combined STL must remain one connected closed shell")
        cad = model.solid if name == "relief" else model.materials[name]
        _registered_bounds(roundtrip, cad, float_error, config)
        expected = occt.volume(cad)
        error = abs(cast("float", report["volume_mm3"]) - expected)
        volume_budget = max(
            1e-7, occt.area(cad) * (config.linear_deflection_mm + float_error) * 2
        )
        if error > volume_budget:
            raise ExportError(
                f"{name} mesh/CAD volume error exceeds the tessellation budget"
            )
        report.update(
            cad_volume_mm3=expected,
            volume_error_mm3=error,
            volume_error_budget_mm3=volume_budget,
        )
        reports[name] = report
        files[f"{name}.stl"] = data
        if name != "relief":
            serialized_meshes[name] = roundtrip
    stl_interfaces = _interfaces(serialized_meshes, model)
    footprints = _footprints(meshes, model, rounding_error)
    serialized_footprints = _footprints(
        serialized_meshes, model, rounding_error + float_error
    )
    try:
        files["relief.step"], step_report = _step(model.solid)
    except RuntimeError as error:
        raise ExportError(
            "OCCT STEP serialization or roundtrip validation failed"
        ) from error
    return ExportBundle(
        files,
        {
            "mesh_config": {
                "linear_deflection_mm": config.linear_deflection_mm,
                "angular_deflection_rad": config.angular_deflection_rad,
                "max_triangles": config.max_triangles,
            },
            "numerical_processing": native,
            "materials": list(meshes),
            "stl": reports,
            "interfaces": interfaces,
            "serialized_stl_interfaces": stl_interfaces,
            "footprints": footprints,
            "serialized_stl_footprints": serialized_footprints,
            "maximum_stl_float32_error_mm": float_error,
            "step": step_report,
        },
    )


def export_model(model: _Relief, config: ExportConfig | None = None) -> ExportBundle:
    """Export unchanged CAD to registered STEP/STLs, or fail before returning bytes."""
    try:
        return _export_model(model, config)
    except RuntimeError as error:
        raise ExportError("OCCT export or geometry validation failed") from error
