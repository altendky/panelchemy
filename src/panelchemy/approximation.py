"""Optional, distance-bounded contour approximation with pinned contacts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, nextafter
from typing import cast

from shapely import get_coordinates, union_all
from shapely.errors import GEOSException
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import substring
from shapely.strtree import STRtree

from panelchemy.contours import PanelContours
from panelchemy.separation import Palette, SeparationConfig

type Point2 = tuple[float, float]
type Relationships = dict[tuple[int, int], str]


@dataclass(frozen=True)
class ApproximationResult:
    contours: PanelContours
    diagnostics: dict[str, object]
    comparison_svg: str


def _vertices(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def _relationships(geometries: Sequence[BaseGeometry]) -> Relationships:
    tree = STRtree(geometries)
    relationships: Relationships = {}
    for index, geometry in enumerate(geometries):
        neighbors = cast(
            "list[int]", tree.query(geometry, predicate="intersects").tolist()
        )
        for neighbor in neighbors:
            if neighbor > index:
                relationships[index, neighbor] = geometry.relate(geometries[neighbor])
    return relationships


def _changed_pairs(
    original: Mapping[tuple[int, int], str], selected: Mapping[tuple[int, int], str]
) -> set[tuple[int, int]]:
    return {
        pair
        for pair in original.keys() | selected.keys()
        if original.get(pair) != selected.get(pair)
    }


def _contact_anchors(
    rings: Sequence[LinearRing], relationships: Relationships
) -> list[set[Point2]]:
    anchors: list[set[Point2]] = [set() for _ in rings]
    for left, right in relationships:
        coordinates = cast(
            "list[list[float]]",
            get_coordinates(rings[left].intersection(rings[right])).tolist(),
        )
        for coordinate in coordinates:
            point = (coordinate[0], coordinate[1])
            anchors[left].add(point)
            anchors[right].add(point)
    return anchors


def _simplify_ring(
    ring: LinearRing, tolerance: float, anchors: set[Point2]
) -> LinearRing:
    if not anchors:
        return LinearRing(ring.simplify(tolerance, preserve_topology=True).coords)
    path = LineString(ring.coords)
    ordered = sorted((path.project(Point(point)), point) for point in anchors)
    coordinates: list[Point2] = []
    for index, (start, start_point) in enumerate(ordered):
        stop, stop_point = ordered[(index + 1) % len(ordered)]
        if stop <= start:
            chunk = list(substring(path, start, path.length).coords)
            chunk.extend(list(substring(path, 0, stop).coords)[1:])
        else:
            chunk = list(substring(path, start, stop).coords)
        # Reuse the exact intersection coordinates on both neighboring rings.
        chunk[0], chunk[-1] = start_point, stop_point
        segment = LineString(chunk).simplify(tolerance, preserve_topology=False)
        coordinates.extend((float(x), float(y)) for x, y in list(segment.coords)[:-1])
    if coordinates:
        coordinates.append(coordinates[0])
    return LinearRing(coordinates)


def boundary_within_tolerance(
    original: BaseGeometry, selected: BaseGeometry, tolerance: float
) -> bool:
    """Certify whole segments, rather than sampling a discrete Hausdorff distance.

    Positive round buffers use inscribed arc chords, so containment is a
    conservative distance check. A radius just below the requested tolerance
    avoids increasing the user's budget through a fixed numerical allowance.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("boundary tolerance must be finite and nonnegative")
    if original.equals(selected):
        return True
    radius = nextafter(tolerance, 0.0)
    if radius <= 0:
        return False
    return (
        original.difference(selected.buffer(radius, quad_segs=16)).is_empty
        and selected.difference(original.buffer(radius, quad_segs=16)).is_empty
    )


def _comparison_svg(
    original: PanelContours,
    selected: PanelContours,
    tolerance: float,
    palette: Palette,
) -> str:
    left, bottom, right, top = original.silhouette.bounds
    width, height = right - left, top - bottom
    gap = max(width, height) * 0.05
    heading = gap * 1.5
    line_width = max(width, height) / 500
    colors = ["#" + "".join(f"{channel:02x}" for channel in color) for color in palette]

    def paths(polygons: Sequence[Polygon], fill: str, stroke: str = "none") -> str:
        result: list[str] = []
        for polygon in polygons:
            commands: list[str] = []
            for ring in [polygon.exterior, *polygon.interiors]:
                commands.append(
                    "M "
                    + " L ".join(f"{x:.12g},{y:.12g}" for x, y in ring.coords)
                    + " Z"
                )
            result.append(
                f'<path fill="{fill}" stroke="{stroke}" stroke-width="{line_width:.12g}" fill-rule="evenodd" d="{" ".join(commands)}"/>'
            )
        return "".join(result)

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {3 * width + 4 * gap:.12g} {height + heading + gap:.12g}">',
        '<rect width="100%" height="100%" fill="#ededed"/>',
    ]
    for index, (title, contours) in enumerate(
        [("Original contours", original), ("Selected approximation", selected)]
    ):
        offset = gap + index * (width + gap)
        content.append(
            f'<text x="{offset:.12g}" y="{gap:.12g}" font-size="{gap / 3:.12g}" font-family="sans-serif">{title}</text>'
        )
        content.append(
            f'<g transform="translate({offset - left:.12g},{heading + top:.12g}) scale(1,-1)">'
        )
        content.extend(
            [
                paths([contours.silhouette], colors[0]),
                paths(contours.orange, colors[1]),
                paths(contours.white, colors[2]),
                "</g>",
            ]
        )
    offset = gap + 2 * (width + gap)
    content.extend(
        [
            f'<text x="{offset:.12g}" y="{gap:.12g}" font-size="{gap / 3:.12g}" font-family="sans-serif">Outline overlay: original blue, selected orange; budget {tolerance:g} mm</text>',
            f'<g transform="translate({offset - left:.12g},{heading + top:.12g}) scale(1,-1)">',
            paths([*original.orange, *original.white], "none", "#1166cc"),
            paths([*selected.orange, *selected.white], "none", "#e07700"),
            "</g></svg>",
        ]
    )
    return "".join(content)


def approximate(
    contours: PanelContours,
    tolerance_mm: float,
    *,
    palette: Palette | None = None,
) -> ApproximationResult:
    """Approximate plateaus without dropping islands, counters, or contacts.

    The silhouette stays exact. Rejected ring proposals explicitly retain their
    original geometry and report why, rather than invoking feature cleanup.
    """
    raw_tolerance = cast("object", tolerance_mm)
    if isinstance(raw_tolerance, bool) or not isinstance(raw_tolerance, (int, float)):
        raise ValueError("contour tolerance must be finite and nonnegative")
    try:
        tolerance = float(raw_tolerance)
    except OverflowError as error:
        raise ValueError("contour tolerance must be finite and nonnegative") from error
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("contour tolerance must be finite and nonnegative")
    colors = SeparationConfig().palette if palette is None else palette
    _ = SeparationConfig(palette=colors)
    originals = [*contours.orange, *contours.white]
    rings: list[LinearRing] = []
    ranges: list[range] = []
    owners: list[tuple[int, int]] = []
    for component, polygon in enumerate(originals):
        start = len(rings)
        for ring_index, ring in enumerate([polygon.exterior, *polygon.interiors]):
            rings.append(ring)
            owners.append((component, ring_index))
        ranges.append(range(start, len(rings)))
    selected_rings = list(rings)
    rejected: dict[int, str] = {}
    polygons = list(originals)
    if tolerance > 0 and originals:
        if not contours.silhouette.is_valid or any(
            not polygon.is_valid
            or polygon.is_empty
            or polygon.has_z
            or not contours.silhouette.covers(polygon)
            for polygon in originals
        ):
            raise ValueError("approximation requires valid contained source polygons")
        source_relationships = _relationships(originals)
        if any(relation[0] != "F" for relation in source_relationships.values()):
            raise ValueError("source plateau components must have disjoint interiors")
        ring_relationships = _relationships(rings)
        anchors = _contact_anchors(rings, ring_relationships)
        for index, (ring, pinned) in enumerate(zip(rings, anchors, strict=True)):
            try:
                candidate = _simplify_ring(ring, tolerance, pinned)
                if not candidate.is_ring or Polygon(candidate).area <= 0:
                    raise ValueError("ring would collapse or intersect itself")
                if not boundary_within_tolerance(ring, candidate, tolerance):
                    raise ValueError("continuous boundary deviation exceeds the budget")
            except (ValueError, GEOSException) as error:
                rejected[index] = str(error)
            else:
                selected_rings[index] = ring if candidate.equals(ring) else candidate

        def restore(index: int, reason: str) -> bool:
            if selected_rings[index].equals(rings[index]):
                return False
            selected_rings[index] = rings[index]
            _ = rejected.setdefault(index, reason)
            return True

        # Each retry restores at least one changed ring, so this terminates in
        # at most the finite number of proposed ring changes.
        for _ in range(len(rings) + 1):
            changed = _changed_pairs(ring_relationships, _relationships(selected_rings))
            if changed:
                restored = False
                for index in {index for pair in changed for index in pair}:
                    restored |= restore(
                        index, "ring contact relationships would change"
                    )
                if not restored:
                    raise ValueError("could not preserve original ring relationships")
                continue
            polygons = [
                Polygon(
                    selected_rings[indices.start].coords,
                    [selected_rings[index].coords for index in list(indices)[1:]],
                )
                for indices in ranges
            ]
            restored = False
            for polygon, source, indices in zip(
                polygons, originals, ranges, strict=True
            ):
                if (
                    not polygon.is_valid
                    or polygon.is_empty
                    or len(polygon.interiors) != len(source.interiors)
                ):
                    for index in indices:
                        restored |= restore(
                            index, "polygon or counter validity would change"
                        )
                elif not contours.silhouette.covers(polygon):
                    restored |= restore(
                        indices.start, "plateau would leave the silhouette"
                    )
            if restored:
                continue
            changed = _changed_pairs(source_relationships, _relationships(polygons))
            if not changed:
                break
            for component in {index for pair in changed for index in pair}:
                for index in ranges[component]:
                    restored |= restore(
                        index, "plateau contact relationships would change"
                    )
            if not restored:
                raise ValueError("could not preserve original plateau relationships")
        else:
            raise ValueError("contour validation did not converge to original topology")
        polygons = [
            source if selected.equals(source) else orient(selected, sign=1.0)
            for source, selected in zip(originals, polygons, strict=True)
        ]
    split = len(contours.orange)
    vertex_count = _vertices(contours.silhouette) + sum(_vertices(p) for p in polygons)
    selected = (
        contours
        if tolerance == 0
        else PanelContours(
            silhouette=contours.silhouette,
            orange=polygons[:split],
            white=polygons[split:],
            mm_per_pixel=contours.mm_per_pixel,
            source_bounds_px=contours.source_bounds_px,
            vertex_count=vertex_count,
        )
    )
    changed_ring_count = sum(
        not original.equals(candidate)
        for original, candidate in zip(rings, selected_rings, strict=True)
    )
    fallback_rings: list[dict[str, object]] = []
    for index, reason in sorted(rejected.items()):
        component, ring_index = owners[index]
        color = "orange" if component < split else "white"
        fallback_rings.append(
            {
                "color": color,
                "component": component if color == "orange" else component - split,
                "ring": "exterior" if ring_index == 0 else f"counter-{ring_index - 1}",
                "reason": reason,
            }
        )
    color_changes: dict[str, object] = {}
    for color, source_parts, selected_parts in (
        ("orange", contours.orange, selected.orange),
        ("white", contours.white, selected.white),
    ):
        source_region = union_all(source_parts)
        selected_region = union_all(selected_parts)
        color_changes[color] = {
            "source_area_mm2": source_region.area,
            "selected_area_mm2": selected_region.area,
            "symmetric_difference_area_mm2": source_region.symmetric_difference(
                selected_region
            ).area,
            "source_perimeter_mm": source_region.length,
            "selected_perimeter_mm": selected_region.length,
        }
    observed_changes: list[dict[str, object]] = []
    for index, (original, candidate) in enumerate(
        zip(rings, selected_rings, strict=True)
    ):
        if original.equals(candidate):
            continue
        component, ring_index = owners[index]
        color = "orange" if component < split else "white"
        observed_changes.append(
            {
                "color": color,
                "component": component if color == "orange" else component - split,
                "ring": "exterior" if ring_index == 0 else f"counter-{ring_index - 1}",
                "observed_discrete_hausdorff_mm": original.hausdorff_distance(
                    candidate
                ),
            }
        )
    diagnostics: dict[str, object] = {
        "method": "exact" if tolerance == 0 else "pinned-douglas-peucker",
        "tolerance_mm": tolerance,
        "silhouette_preserved": True,
        "source_vertex_count": contours.vertex_count,
        "output_vertex_count": selected.vertex_count,
        "source_components": {"orange": split, "white": len(contours.white)},
        "output_components": {
            "orange": len(selected.orange),
            "white": len(selected.white),
        },
        "counter_count": sum(len(p.interiors) for p in originals),
        "changed_ring_count": changed_ring_count,
        "fallback_ring_count": len(fallback_rings),
        "fallback_rings": fallback_rings,
        "color_changes": color_changes,
        "observed_ring_changes": observed_changes,
        "observed_discrete_distances_are_certificates": False,
        "certified_boundary_deviation_upper_bound_mm": tolerance
        if changed_ring_count
        else 0.0,
        "deviation_validation": "mutual containment in inscribed round boundary buffers",
        "boundary_buffer_arc_segments_per_quadrant": 16,
        "relationships_preserved": True,
    }
    return ApproximationResult(
        selected, diagnostics, _comparison_svg(contours, selected, tolerance, colors)
    )
