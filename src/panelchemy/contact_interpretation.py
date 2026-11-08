"""Interpret tiny same-color corner contacts before extracting shared chains.

This is an explicit approximation step, separate from exact raster vectorization.
Its caller must certify the final fitted boundaries against the original source.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

from shapely import STRtree, union_all
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from panelchemy.contours import PanelContours

type Point2 = tuple[float, float]
type SourceGroups = tuple[tuple[int, ...], ...]
type Color = Literal["orange", "white"]


@dataclass(frozen=True)
class PointContact:
    """An interpreted attachment, indexed against the unchanged source parts."""

    color: Color
    tip: int
    parent: int
    point: Point2
    apex: Point2
    members: tuple[int, ...] = ()
    points: tuple[Point2, ...] = ()


@dataclass(frozen=True)
class ContactInterpretation:
    """Interpreted geometry and the original components represented by each part."""

    contours: PanelContours
    orange_sources: SourceGroups
    white_sources: SourceGroups
    tip_corners: tuple[Point2, ...]
    diagnostics: dict[str, object]
    contacts: tuple[PointContact, ...] = ()


@dataclass(frozen=True)
class _Join:
    child: int
    parent: int
    point: Point2
    apex: Point2
    bridge: BaseGeometry
    members: tuple[int, ...]
    points: tuple[Point2, ...]
    counters: tuple[Polygon, ...]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _hits(tree: STRtree, polygon: BaseGeometry) -> list[int]:
    return cast("list[int]", tree.query(polygon, predicate="intersects").tolist())


def _vertices(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def _extent(polygon: BaseGeometry) -> float:
    x0, y0, x1, y1 = polygon.bounds
    return max(x1 - x0, y1 - y0)


def _points(contact: BaseGeometry) -> tuple[Point2, ...] | None:
    if isinstance(contact, Point):
        return ((float(contact.x), float(contact.y)),)
    if isinstance(contact, MultiPoint):
        return tuple(sorted((float(p.x), float(p.y)) for p in contact.geoms))
    return None


def _holes(
    parent: Polygon, merged: Polygon
) -> tuple[list[Polygon], list[Polygon]] | None:
    """Keep every old hole once; distinguish newly enclosed contact counters."""
    holes = [Polygon(ring) for ring in merged.interiors]
    owned: list[int] = []
    for original in parent.interiors:
        matches = [
            i
            for i, hole in enumerate(holes)
            if Polygon(original).relate_pattern(hole, "T********")
        ]
        if len(matches) != 1 or matches[0] in owned:
            return None
        owned.append(matches[0])
    return [holes[i] for i in owned], [
        hole for i, hole in enumerate(holes) if i not in owned
    ]


def _clear(
    addition: Polygon,
    allowed: set[int],
    tree: STRtree,
    other: STRtree,
    silhouette: Polygon,
) -> bool:
    return (
        addition.is_valid
        and addition.area > 0
        and silhouette.covers(addition)
        and not addition.intersects(silhouette.boundary)
        and not _hits(other, addition)
        and all(i in allowed for i in _hits(tree, addition))
    )


def _color(
    polygons: list[Polygon],
    foreign: list[Polygon],
    silhouette: Polygon,
    scale: float,
    half_size: float,
    selected_tips: frozenset[int] | None,
) -> tuple[list[Polygon], SourceGroups, list[_Join], Counter[str]]:
    tree, other = STRtree(polygons), STRtree(foreign)
    skipped: Counter[str] = Counter()
    small = {i for i, polygon in enumerate(polygons) if _extent(polygon) <= scale}
    neighbors = {i: set(_hits(tree, polygons[i])) - {i} for i in small}
    visited: set[int] = set()
    clusters: list[tuple[int, ...]] = []
    for first in sorted(small):
        if first in visited:
            continue
        cluster, pending = set[int](), [first]
        while pending:
            index = pending.pop()
            if index in cluster:
                continue
            cluster.add(index)
            pending.extend((neighbors[index] & small) - cluster)
        visited.update(cluster)
        clusters.append(tuple(sorted(cluster)))

    joins: list[_Join] = []
    for members in clusters:
        if selected_tips is not None and not selected_tips.intersection(members):
            continue
        parts = [polygons[i] for i in members]
        if any(p.interiors for p in parts):
            skipped["tip_has_holes"] += 1
            continue
        if _extent(union_all(parts)) > 2 * scale:
            skipped["cluster_too_large"] += 1
            continue
        parents = {j for i in members for j in neighbors[i] if j not in small}
        if len(parents) != 1:
            skipped["no_unique_parent"] += 1
            continue
        parent = parents.pop()
        if polygons[parent].area <= sum(p.area for p in parts):
            skipped["parent_not_larger"] += 1
            continue
        allowed = {parent, *members}
        contacts: set[Point2] = set()
        parent_contacts: set[Point2] = set()
        non_point = False
        for child in members:
            for neighbor in neighbors[child]:
                points = _points(polygons[child].intersection(polygons[neighbor]))
                if points is None:
                    non_point = True
                    break
                contacts.update(points)
                if neighbor == parent:
                    parent_contacts.update(points)
            if non_point:
                break
        if non_point:
            skipped["non_point_contact"] += 1
            continue
        if any(p.intersects(silhouette.boundary) or _hits(other, p) for p in parts):
            skipped["tip_touches_other_boundary"] += 1
            continue

        bridges = [
            Polygon(
                [
                    (x, y + half_size),
                    (x + half_size, y),
                    (x, y - half_size),
                    (x - half_size, y),
                ]
            )
            for x, y in sorted(contacts)
        ]
        if not all(
            _clear(bridge, allowed, tree, other, silhouette) for bridge in bridges
        ):
            skipped["bridge_touches_other_boundary"] += 1
            continue
        merged = union_all([polygons[parent], *parts, *bridges])
        if not isinstance(merged, Polygon) or not merged.is_valid:
            skipped["invalid_merged_topology"] += 1
            continue
        holes = _holes(polygons[parent], merged)
        if holes is None:
            skipped["parent_hole_changed"] += 1
            continue
        _, counters = holes
        if any(
            _extent(counter) > scale
            or not _clear(counter, allowed, tree, other, silhouette)
            for counter in counters
        ):
            skipped["contact_counter_blocked"] += 1
            continue
        point = min(parent_contacts)
        apex = max(
            ((float(x), float(y)) for p in parts for x, y in p.exterior.coords[:-1]),
            key=lambda p: ((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2, p),
        )
        joins.append(
            _Join(
                members[0],
                parent,
                point,
                apex,
                union_all([*bridges, *counters]),
                members,
                tuple(sorted(parent_contacts)),
                tuple(counters),
            )
        )

    # Independent bridges must not create a new relationship between parents.
    bridges = STRtree([join.bridge for join in joins])
    conflicting: set[int] = set()
    for index, join in enumerate(joins):
        for neighbor in _hits(bridges, join.bridge):
            if neighbor > index and joins[neighbor].parent != join.parent:
                conflicting.update((join.parent, joins[neighbor].parent))
    groups: dict[int, list[_Join]] = defaultdict(list)
    for join in joins:
        if join.parent in conflicting:
            skipped["bridge_conflict"] += 1
        else:
            groups[join.parent].append(join)

    replacements: dict[int, tuple[Polygon, tuple[int, ...]]] = {}
    accepted: list[_Join] = []
    consumed: set[int] = set()
    for parent, children in groups.items():
        indices = tuple(
            sorted([parent, *(i for join in children for i in join.members)])
        )
        pieces: list[BaseGeometry] = [
            *(polygons[i] for i in indices),
            *(j.bridge for j in children),
        ]
        ordered = sorted(pieces, key=lambda p: (p.bounds, p.wkb))
        merged = union_all(ordered)
        if (
            not isinstance(merged, Polygon)
            or not merged.is_valid
            or not silhouette.covers(merged)
            or (holes := _holes(polygons[parent], merged)) is None
            or holes[1]
        ):
            skipped["invalid_merged_topology"] += len(children)
            continue
        replacements[parent] = (orient(cast("Polygon", merged.normalize())), indices)
        consumed.update(i for join in children for i in join.members)
        accepted.extend(children)
    if not replacements:
        return polygons, tuple((i,) for i in range(len(polygons))), [], skipped
    selected = [
        replacements.get(i, (polygon, (i,)))
        for i, polygon in enumerate(polygons)
        if i not in consumed
    ]
    selected.sort(key=lambda item: (item[0].bounds, item[0].area, item[0].wkb))
    return (
        [polygon for polygon, _ in selected],
        tuple(indices for _, indices in selected),
        accepted,
        skipped,
    )


def before_graph(
    source: PanelContours,
    *,
    scale_mm: float,
    tolerance_mm: float,
    selected_tips: frozenset[tuple[Color, int]] | None = None,
) -> ContactInterpretation:
    """Join bounded point-connected small groups to one substantial parent.

    A diamond extending at most half a source pixel resolves a diagonal contact
    with finite area. A parent must exceed the tip-size class as well as have
    greater area; two small islands do not select a parent by rounding noise.
    Only new sub-printable contact-cycle counters may be filled. Every original
    parent hole must survive unambiguously; the exact silhouette and unrelated
    components remain. Neither this interpretation nor its diamonds authorize
    extra fitting error against the original source paths.
    The returned corner candidates guide fitting, rather than authorize a fit.
    Source groups let the caller enforce the original total error budget.
    Selecting any member resolves its whole group; all other parts still
    participate in neighborhood checks and remain separate source components.
    """
    scale = _number(scale_mm, "contact scale")
    tolerance = _number(tolerance_mm, "contact tolerance")
    pitch = _number(source.mm_per_pixel, "source pixel pitch")
    half_size = min(pitch / 2, tolerance)
    if half_size <= 0:
        raise ValueError("source pixel pitch must represent a positive half step")
    polygons = [source.silhouette, *source.orange, *source.white]
    if any(
        not p.is_valid or p.is_empty or p.has_z or not source.silhouette.covers(p)
        for p in polygons
    ):
        raise ValueError("contact interpretation requires valid contained polygons")
    colored = polygons[1:]
    tree = STRtree(colored)
    if any(
        polygon.intersection(colored[j]).area > 0
        for i, polygon in enumerate(colored)
        for j in _hits(tree, polygon)
        if j > i
    ):
        raise ValueError("source color regions must have disjoint interiors")
    orange, orange_sources, orange_joins, orange_skipped = _color(
        source.orange,
        source.white,
        source.silhouette,
        scale,
        half_size,
        None
        if selected_tips is None
        else frozenset(i for color, i in selected_tips if color == "orange"),
    )
    white, white_sources, white_joins, white_skipped = _color(
        source.white,
        source.orange,
        source.silhouette,
        scale,
        half_size,
        None
        if selected_tips is None
        else frozenset(i for color, i in selected_tips if color == "white"),
    )
    # Cross-color bridges can meet even though neither meets an original part.
    conflicts = any(
        a.bridge.intersects(b.bridge) for a in orange_joins for b in white_joins
    )
    if conflicts:
        # Such a saddle has no unambiguous foreground. Preserve the source.
        orange, white = source.orange, source.white
        orange_sources = tuple((i,) for i in range(len(orange)))
        white_sources = tuple((i,) for i in range(len(white)))
        orange_skipped["cross_color_bridge_conflict"] += len(orange_joins)
        white_skipped["cross_color_bridge_conflict"] += len(white_joins)
        orange_joins, white_joins = [], []
    joins = [*orange_joins, *white_joins]
    contours = source
    if joins:
        contours = PanelContours(
            source.silhouette,
            orange,
            white,
            source.mm_per_pixel,
            source.source_bounds_px,
            sum(_vertices(p) for p in [source.silhouette, *orange, *white]),
        )
    selections: tuple[tuple[Color, list[_Join]], ...] = (
        ("orange", orange_joins),
        ("white", white_joins),
    )
    diagnostics: dict[str, object] = {
        "contact_interpretation": {
            "method": "bounded same-color point-contact graph joins",
            "scale_mm": scale,
            "bridge_half_size_mm": half_size,
            "joined_tips": sum(len(join.members) for join in joins),
            "joined_groups": len(joins),
            "joins": [
                {
                    "color": color,
                    "source_tip": j.child,
                    "source_parent": j.parent,
                    "contact": j.point,
                    "tip_corner": j.apex,
                    "source_members": j.members,
                    "contact_points": j.points,
                    "filled_contact_counters": len(j.counters),
                    "filled_contact_counter_area_mm2": sum(p.area for p in j.counters),
                    "filled_contact_counter_bounds_mm": [p.bounds for p in j.counters],
                }
                for color, selected in selections
                for j in selected
            ],
            "skipped": {"orange": dict(orange_skipped), "white": dict(white_skipped)},
            "source_error_policy": "caller must certify fitted merged boundaries against their original component paths",
        }
    }
    contacts = tuple(
        PointContact(
            color,
            join.child,
            join.parent,
            join.point,
            join.apex,
            join.members if len(join.members) > 1 or len(join.points) > 1 else (),
            join.points if len(join.members) > 1 or len(join.points) > 1 else (),
        )
        for color, selected in selections
        for join in selected
    )
    return ContactInterpretation(
        contours=contours,
        orange_sources=orange_sources,
        white_sources=white_sources,
        tip_corners=tuple(sorted({join.apex for join in joins})),
        diagnostics=diagnostics,
        contacts=contacts,
    )
