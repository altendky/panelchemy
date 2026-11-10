"""Resolve bounded point-contact fragments without refitting other outlines."""

# Exact proof primitives are shared with the ordinary outline fitter.
# pyright: reportPrivateUsage=false

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import chain, pairwise
from math import hypot, isfinite, ulp
from time import monotonic
from typing import cast

from shapely import STRtree
from shapely.geometry import LineString, Polygon

from panelchemy import boundary_tapers, corner_refinement, curvature, geometric_distance
from panelchemy import contact_interpretation as ci
from panelchemy import curves as c
from panelchemy.contours import PanelContours

type _Key = tuple[bool, c.Controls]
type _Selected = frozenset[tuple[ci.Color, int]]
type _Path = tuple[c.Point2, ...]
type _DistanceKey = tuple[tuple[_Path, ...], tuple[_Key, ...], float]
_EMPTY_POINTS: frozenset[c.Point2] = frozenset()


@dataclass(frozen=True)
class _ReferenceTopology:
    """Lazy, unchanged topology expectations for one proposed reference."""

    reference: PanelContours

    @cached_property
    def polygons(self) -> dict[tuple[int, int], str]:
        r = self.reference
        return c._relationships([r.silhouette, *r.orange, *r.white])

    @cached_property
    def rings(self) -> dict[tuple[int, int], str]:
        r = self.reference
        return c._relationships(
            [
                LineString(ring.coords)
                for polygon in (r.silhouette, *r.orange, *r.white)
                for ring in (polygon.exterior, *polygon.interiors)
            ]
        )


def _key(span: c.BezierSpan) -> _Key:
    p = span.control_points
    if span.is_line:
        p = (p[0], p[0], p[-1], p[-1])
    return span.is_line, min(p, cast("c.Controls", tuple(reversed(p))))


def _bad(
    contours: c.CurveContours,
    runs: tuple[float, float, float],
    proofs: boundary_tapers.ProofCache | None = None,
) -> set[_Key]:
    try:
        boundary_tapers.require_curvature(contours, runs, proofs)
    except boundary_tapers.CurvatureLimitError as error:
        return {_key(span) for span, _ in error.failures.values()}
    return set()


def _groups(reference: ci.ContactInterpretation, color: ci.Color) -> ci.SourceGroups:
    return reference.orange_sources if color == "orange" else reference.white_sources


def _source(source: PanelContours, color: ci.Color) -> list[Polygon]:
    return source.orange if color == "orange" else source.white


def _paths(polygons: Sequence[Polygon]) -> list[list[c.Point2]]:
    return [
        [(float(x), float(y)) for x, y in ring.coords]
        for polygon in polygons
        for ring in (polygon.exterior, *polygon.interiors)
    ]


def _source_certificate(
    paths: Sequence[_Path],
    spans: Sequence[c.BezierSpan],
    tolerance: float,
    cache: dict[_DistanceKey, bool],
    forward: geometric_distance.ForwardProofCache | None = None,
) -> bool:
    if forward is not None and (
        forward.tolerance_mm != tolerance
        or tuple(paths[: len(forward.base_paths)]) != forward.base_paths
    ):
        raise ValueError(
            "forward proofs require their original source paths and tolerance"
        )
    # This cache lives for one refine call. Both original paths and selected
    # coefficients are part of the key, so an accepted edit cannot reuse an
    # older proof for the changed ring or its enlarged original source group.
    key = (
        tuple(paths),
        tuple((s.is_line, s.control_points) for s in spans),
        tolerance,
    )
    if key not in cache:
        cache[key] = (
            geometric_distance.within_paths(paths, spans, tolerance)
            if forward is None
            else forward.within_extra_paths(paths[len(forward.base_paths) :], spans)
        )
    return cache[key]


def _ordered_reference(
    source: PanelContours, reference: ci.ContactInterpretation
) -> PanelContours | None:
    """Keep source ring identities when a union normalizes its hole order."""
    colors: dict[ci.Color, list[Polygon]] = {}
    for color in ("orange", "white"):
        colors[color] = []
        originals = _source(source, color)
        for polygon, indices in zip(
            _source(reference.contours, color), _groups(reference, color), strict=True
        ):
            if len(indices) == 1 or not polygon.interiors:
                colors[color].append(polygon)
                continue
            parent = max(indices, key=lambda i: originals[i].area)
            old = [Polygon(ring) for ring in originals[parent].interiors]
            new = [Polygon(ring) for ring in polygon.interiors]
            if len(old) != len(new):
                return None
            tree = STRtree(new)
            # Match the surviving hole interiors one-to-one. This only orders
            # existing reference rings; geometry, source bounds and complete
            # topology still pass their ordinary independent proof gates.
            unused = set(range(len(new)))
            order: list[int] = []
            for ring in old:
                matches = [
                    i
                    for i in cast("list[int]", tree.query(ring).tolist())
                    if i in unused and ring.relate_pattern(new[i], "T********")
                ]
                if len(matches) != 1:
                    return None
                order.append(matches[0])
                unused.remove(matches[0])
            colors[color].append(
                Polygon(
                    polygon.exterior.coords,
                    [polygon.interiors[i].coords for i in order],
                )
            )
    return replace(reference.contours, orange=colors["orange"], white=colors["white"])


def _notches(
    source: Polygon, point: c.Point2, reach: float, ring_index: int = 0
) -> list[c.Point2]:
    ring = (source.exterior, *source.interiors)[ring_index]
    vertices = [(float(x), float(y)) for x, y in ring.coords[:-1]]
    # A hole has material outside its enclosed area. Classify its material
    # corners accordingly, independently of the supplied ring winding.
    sign = (1 if ring.is_ccw else -1) * (1 if ring_index == 0 else -1)
    result: list[c.Point2] = []
    for i, b in enumerate(vertices):
        a, d = vertices[i - 1], vertices[(i + 1) % len(vertices)]
        cross = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])
        if sign * cross < 0 and hypot(b[0] - point[0], b[1] - point[1]) <= reach:
            result.append(b)
    result.sort(key=lambda p: (hypot(p[0] - point[0], p[1] - point[1]), p))
    nearby = result[:8]
    return [
        *nearby,
        *[((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in pairwise(nearby)],
    ]


def _patch(
    parent: c.CurvePolygon, index: int, points: list[c.Point2], ring_index: int = 0
) -> c.CurvePolygon:
    spans = (parent.exterior, *parent.interiors)[ring_index].spans
    patch = tuple(c._line(a, b) for a, b in pairwise(points))
    ring = c.CurveRing((*spans[:index], *patch, *spans[index + 1 :]))
    if ring_index == 0:
        return replace(parent, exterior=ring)
    interiors = list(parent.interiors)
    interiors[ring_index - 1] = ring
    return replace(parent, interiors=tuple(interiors))


def _degenerate(points: Sequence[c.Point2]) -> bool:
    """Reject zero-length steps and roundoff-width out-and-back spikes."""
    if any(a == b for a, b in pairwise(points)):
        return True
    for a, b, d in zip(points, points[1:], points[2:], strict=False):
        u, v = (b[0] - a[0], b[1] - a[1]), (d[0] - b[0], d[1] - b[1])
        if u[0] * v[0] + u[1] * v[1] >= 0:
            continue
        # Cover coordinate/subtraction and determinant rounding, not a physical
        # detail allowance. Resolved sharp tips remain eligible candidates.
        roundoff = 8 * max(ulp(value) for point in (a, b, d) for value in point)
        if abs(u[0] * v[1] - u[1] * v[0]) <= roundoff * (hypot(*u) + hypot(*v)):
            return True
    return False


def _ring_candidates(
    contact: ci.PointContact,
    parent: c.CurvePolygon,
    source: Polygon,
    bad: set[_Key],
    scale: float,
    tolerance: float,
    ring_index: int,
) -> Iterator[c.CurvePolygon]:
    spans = (parent.exterior, *parent.interiors)[ring_index].spans
    source_ring = (source.exterior, *source.interiors)[ring_index]
    for i, span in enumerate(spans):
        a, b = span.control_points[0], span.control_points[-1]
        if _key(span) not in bad or contact.point not in (a, b):
            continue
        if len(contact.members) > 1:
            points = [a, contact.apex, b]
            if not _degenerate(points):
                yield _patch(parent, i, points, ring_index)
        reach = max(2 * scale, 2 * tolerance, hypot(b[0] - a[0], b[1] - a[1]))
        for notch in _notches(source, contact.point, reach, ring_index):
            points = (
                [contact.apex, notch] if contact.point == a else [notch, contact.apex]
            )
            selected = [a, *points, b]
            if _degenerate(selected):
                continue
            # The interpreted attachment permits explicit G0 corners at splice
            # anchors too; all outside polynomial curves remain unchanged.
            yield _patch(parent, i, selected, ring_index)
        before, after = spans[i - 1], spans[(i + 1) % len(spans)]
        u = tuple(
            a[j] - before.control_points[0 if before.is_line else 2][j] for j in (0, 1)
        )
        v = tuple(
            b[j] - after.control_points[-1 if after.is_line else 1][j] for j in (0, 1)
        )
        lengths = hypot(*u), hypot(*v)
        if min(lengths) == 0:
            continue
        rays = (
            (u[0] / lengths[0], u[1] / lengths[0]),
            (v[0] / lengths[1], v[1] / lengths[1]),
        )
        tip_index = 0 if a == contact.point else 1
        anchors = a, b
        tip_ray = rays[tip_index]
        tip_length = sum(
            (contact.apex[j] - contact.point[j]) * tip_ray[j] for j in (0, 1)
        )
        if tip_length <= 0:
            continue
        notch_index = 1 - tip_index
        anchor, ray = anchors[notch_index], rays[notch_index]
        projections = [
            sum((point[j] - anchor[j]) * ray[j] for j in (0, 1))
            for point in source_ring.coords
            if hypot(point[0] - contact.point[0], point[1] - contact.point[1]) <= reach
        ]
        notch_length = min(reach, max(projections, default=0.0))
        if notch_length <= 0:
            continue
        # A bounded search grid guides construction; it does not relax a proof.
        for tip_fraction in (1.0, 0.75, 1.25, 0.5):
            for notch_fraction in (1.0, 0.75, 1.25, 0.5):
                selected = [0.0, 0.0]
                selected[tip_index] = tip_length * tip_fraction
                selected[notch_index] = notch_length * notch_fraction
                points = [
                    (
                        anchors[k][0] + selected[k] * rays[k][0],
                        anchors[k][1] + selected[k] * rays[k][1],
                    )
                    for k in (0, 1)
                ]
                selected_points = [a, *points, b]
                if _degenerate(selected_points):
                    continue
                yield _patch(parent, i, selected_points, ring_index)


def _candidates(
    contact: ci.PointContact,
    parent: c.CurvePolygon,
    source: Polygon,
    bad: set[_Key],
    scale: float,
    tolerance: float,
) -> Iterator[c.CurvePolygon]:
    points = contact.points or (contact.point,)
    for ring_index in range(1 + len(parent.interiors)):
        yield from _short_arc_candidates(contact, parent, scale, ring_index)
        for point in points:
            yield from _ring_candidates(
                replace(contact, point=point),
                parent,
                source,
                bad,
                scale,
                tolerance,
                ring_index,
            )


def _short_arc_candidates(
    contact: ci.PointContact,
    parent: c.CurvePolygon,
    scale: float,
    ring_index: int,
) -> Iterator[c.CurvePolygon]:
    """Shortcut only bounded arcs between authorized contact junctions."""
    junctions = frozenset(contact.points or (contact.point,))
    if len(junctions) < 2:
        return
    spans = (parent.exterior, *parent.interiors)[ring_index].spans
    for start, span in enumerate(spans):
        a = span.control_points[0]
        if a not in junctions:
            continue
        rotated = (*spans[start:], *spans[:start])
        length = 0.0
        for count, current in enumerate(rotated[:-1], 1):
            length += sum(
                hypot(b[0] - a[0], b[1] - a[1])
                for a, b in pairwise(current.control_points)
            )
            # Control-polygon length bounds the whole cubic arc, not just its
            # endpoint chord. Wrapping does not authorize a longer edit.
            if length > 2 * scale:
                break
            b = current.control_points[-1]
            if b not in junctions or a == b:
                continue
            for points in ([a, b], [a, contact.apex, b]):
                if _degenerate(points):
                    continue
                patch = tuple(c._line(p, q) for p, q in pairwise(points))
                ring = c.CurveRing((*patch, *rotated[count:]))
                if ring_index == 0:
                    yield replace(parent, exterior=ring)
                else:
                    holes = list(parent.interiors)
                    holes[ring_index - 1] = ring
                    yield replace(parent, interiors=tuple(holes))


def _assemble(
    polygons: list[c.CurvePolygon],
    reference: PanelContours,
    tolerance: float,
    expected: _ReferenceTopology | None = None,
    proofs: c.ProofCache | None = None,
) -> tuple[c.CurveContours, PanelContours] | None:
    if expected is None or expected.reference is not reference:
        expected = _ReferenceTopology(reference)
    pieces: list[c._Piece] = []
    selected: dict[_Key, int] = {}
    uses: list[list[tuple[int, bool]]] = []
    ranges: list[range] = []
    for polygon in polygons:
        first = len(uses)
        for ring in (polygon.exterior, *polygon.interiors):
            if any(
                a.control_points[-1] != b.control_points[0]
                for a, b in pairwise((*ring.spans, ring.spans[0]))
            ):
                return None
            owners: list[tuple[int, bool]] = []
            for span in ring.spans:
                key = _key(span)
                index = selected.get(key)
                if index is None:
                    index = len(pieces)
                    selected[key] = index
                    pieces.append(c._Piece(index, 0, 1, span))
                owners.append(
                    (
                        index,
                        span.control_points[0] == pieces[index].span.control_points[0],
                    )
                )
            uses.append(owners)
        ranges.append(range(first, len(uses)))
    try:
        unsafe, cuts = c._unsafe_pair(pieces, proofs)
        if unsafe:
            return None
        curves, proxy, rings, proxy_unsafe = c._assemble(
            pieces, uses, ranges, reference, tolerance, cuts, proofs
        )
    except ValueError as error:
        if str(error) not in {
            "curve topology exceeds 1,000,000 candidate pairs",
            "curve proxy cannot meet its continuous error budget",
            "rounded curve proxy cannot preserve monotone embedding",
            "curve/proxy topology exceeds 1,000,000 candidate pairs",
        }:
            raise
        return None
    original_polygons = [reference.silhouette, *reference.orange, *reference.white]
    actual = [proxy.silhouette, *proxy.orange, *proxy.white]
    if (
        proxy_unsafe
        or any(
            not (
                p.is_valid and p.area > 0 and reference.silhouette.covers(p)
                if proofs is None
                else proofs.valid_polygon(p, reference.silhouette)
            )
            or len(p.interiors) != len(original.interiors)
            for p, original in zip(actual, original_polygons, strict=True)
        )
        or c._relationships(actual, proofs) != expected.polygons
        or c._relationships(rings, proofs) != expected.rings
    ):
        return None
    return curves, proxy


def _corner_rings(
    ring: c.CurveRing,
    index: int,
    source_paths: Sequence[_Path] = (),
    tolerance_mm: float = 0.0,
    *,
    protected_points: frozenset[c.Point2] = _EMPTY_POINTS,
) -> Iterator[c.CurveRing]:
    """Offer one bend, neighbors, then its corner-bounded smooth run."""
    spans, size = ring.spans, len(ring.spans)
    before, span, after = spans[index - 1], spans[index], spans[(index + 1) % size]
    start_free = not corner_refinement.smooth_join(before, span)
    end_free = not corner_refinement.smooth_join(span, after)
    windows = [(index, 1, start_free, end_free)]
    if size > 2:
        if not start_free and not corner_refinement.smooth_join(
            spans[(index - 2) % size], before
        ):
            windows.append(((index - 1) % size, 2, True, end_free))
        if not end_free and not corner_refinement.smooth_join(
            after, spans[(index + 2) % size]
        ):
            windows.append((index, 2, start_free, True))

    def surrounding_run() -> Iterator[tuple[int, int, bool, bool]]:
        first, count = index, 1
        while count < size and corner_refinement.smooth_join(
            spans[first - 1], spans[first]
        ):
            if spans[first].control_points[0] in protected_points:
                break
            first = (first - 1) % size
            count += 1
        while count < size:
            last = (first + count - 1) % size
            if (
                not corner_refinement.smooth_join(spans[last], spans[(last + 1) % size])
                or spans[last].control_points[-1] in protected_points
            ):
                break
            count += 1
        if 2 < count < size:
            last = (first + count - 1) % size
            yield (
                first,
                count,
                not corner_refinement.smooth_join(spans[first - 1], spans[first]),
                not corner_refinement.smooth_join(
                    spans[last], spans[(last + 1) % size]
                ),
            )

    for first, count, free_start, free_end in chain(windows, surrounding_run()):
        rotated = (*spans[first:], *spans[:first])
        if any(
            part.control_points[-1] in protected_points for part in rotated[: count - 1]
        ):
            continue
        left, right = rotated[0], rotated[count - 1]
        guide = (
            left
            if count == 1
            else c.BezierSpan(
                (
                    left.control_points[0],
                    left.control_points[-1 if left.is_line else 1],
                    right.control_points[0 if right.is_line else 2],
                    right.control_points[-1],
                ),
                False,
                max(part.certified_error_mm for part in rotated[:count]),
            )
        )
        for replacement in corner_refinement.corner_variants(
            guide,
            free_start=free_start,
            free_end=free_end,
            apices=(
                tuple(part.control_points[-1] for part in rotated[: count - 1])
                if count <= 2
                else (span.control_points[0], span.control_points[-1])
            ),
        ):
            yield c.CurveRing((*replacement, *rotated[count:]))
        if free_start and free_end and tolerance_mm > 0:
            for replacement in corner_refinement.source_corners(
                guide, source_paths, 0.9 * tolerance_mm
            ):
                yield c.CurveRing((*replacement, *rotated[count:]))


def _repair_corners(
    source: PanelContours,
    contours: c.CurveContours,
    proxy: PanelContours,
    reference: ci.ContactInterpretation,
    source_groups: dict[tuple[ci.Color, int], list[tuple[_Path, ...]]],
    *,
    tolerance_mm: float,
    material_runs: tuple[float, float, float],
    proxy_tolerance_mm: float,
    proofs: c.ProofCache,
    radius_proofs: boundary_tapers.ProofCache,
) -> tuple[c.CurveContours, PanelContours, dict[str, int]]:
    """Refine actual selected bends against their original source path groups."""
    by_color: dict[ci.Color, dict[int, c.CurvePolygon]] = {}
    colors: tuple[tuple[ci.Color, tuple[c.CurvePolygon, ...]], ...] = (
        ("orange", contours.orange),
        ("white", contours.white),
    )
    for color, regions in colors:
        originals = _source(source, color)
        by_color[color] = {
            max(indices, key=lambda i: originals[i].area): polygon
            for indices, polygon in zip(_groups(reference, color), regions, strict=True)
        }
    certificates: dict[_DistanceKey, bool] = {}
    # Contact repairs can leave short, genuinely over-curved bends away from
    # their splice. Offer one explicit corner per such bend, preserving outer
    # endpoint rays. Original source paths and the selected palette still own
    # acceptance; a failed construction never authorizes feature deletion.
    corner_limits: dict[_Key, boundary_tapers.BoundaryTaper] = {}
    try:
        boundary_tapers.require_curvature(contours, material_runs, radius_proofs)
    except boundary_tapers.CurvatureLimitError as error:
        corner_limits = {_key(span): limit for span, limit in error.failures.values()}
    remaining_bad = set(corner_limits)
    ownership: Counter[_Key] = (
        Counter(
            _key(span)
            for polygon in (contours.silhouette, *contours.orange, *contours.white)
            for ring in (polygon.exterior, *polygon.interiors)
            for span in ring.spans
        )
        if corner_limits
        else Counter()
    )
    point_owners = Counter(
        point
        for polygon in (contours.silhouette, *contours.orange, *contours.white)
        for ring in (polygon.exterior, *polygon.interiors)
        for point in {span.control_points[0] for span in ring.spans}
        if corner_limits
    )
    protected_points = frozenset(
        point for point, owners in point_owners.items() if owners > 1
    )
    corner_report: dict[str, int] = dict(
        attempted=0, repaired=0, source_rejected=0, topology_rejected=0
    )
    owners: list[tuple[ci.Color, int]] = [
        (color, max(indices, key=lambda i: _source(source, color)[i].area))
        for color in by_color
        for indices in _groups(reference, color)
        if corner_limits
    ]
    expected = _ReferenceTopology(reference.contours)
    for color, owner in owners:
        parent: c.CurvePolygon = by_color[color][owner]
        for ring_index, ring in enumerate((parent.exterior, *parent.interiors)):
            if not any(_key(span) in remaining_bad for span in ring.spans):
                continue
            source_key = color, owner
            if source_key not in source_groups:
                original = _source(source, color)[owner]
                source_groups[source_key] = [
                    (tuple((float(x), float(y)) for x, y in path.coords),)
                    for path in (original.exterior, *original.interiors)
                ]
            original_paths = source_groups[source_key][ring_index]
            forward = geometric_distance.ForwardProofCache(original_paths, tolerance_mm)
            # Address original failing geometry, not indices that a repair or
            # wrapped neighbor window can shift. Outside windows remain intact.
            for key in [_key(span) for span in ring.spans]:
                if key not in remaining_bad:
                    continue
                index = next(
                    (i for i, span in enumerate(ring.spans) if _key(span) == key), None
                )
                if index is None:
                    continue
                corner_report["attempted"] += 1
                span = ring.spans[index]
                limit = corner_limits[key]
                direction = (
                    limit.inward_sign
                    if span.control_points == key[1]
                    else -limit.inward_sign
                )
                for candidate_ring in _corner_rings(
                    ring,
                    index,
                    original_paths,
                    tolerance_mm,
                    protected_points=protected_points,
                ):
                    original_spans = set(ring.spans)
                    candidate_spans = set(candidate_ring.spans)
                    if any(
                        ownership[_key(part)] > 1
                        for part in original_spans - candidate_spans
                    ):
                        continue
                    changed_spans = [
                        part
                        for part in candidate_ring.spans
                        if part not in original_spans
                    ]
                    if not all(
                        corner_refinement.forward_progress(part)
                        and curvature.within_curvature(part, limit.offset_mm, direction)
                        for part in changed_spans
                    ):
                        continue
                    if not _source_certificate(
                        original_paths,
                        candidate_ring.spans,
                        tolerance_mm,
                        certificates,
                        forward,
                    ):
                        corner_report["source_rejected"] += 1
                        continue
                    candidate: c.CurvePolygon = (
                        replace(parent, exterior=candidate_ring)
                        if ring_index == 0
                        else replace(
                            parent,
                            interiors=(
                                *parent.interiors[: ring_index - 1],
                                candidate_ring,
                                *parent.interiors[ring_index:],
                            ),
                        )
                    )
                    polygons: list[c.CurvePolygon] = [
                        candidate
                        if (selected_color, selected_owner) == (color, owner)
                        else by_color[selected_color][selected_owner]
                        for selected_color, selected_owner in owners
                    ]
                    checked = _assemble(
                        [contours.silhouette, *polygons],
                        reference.contours,
                        proxy_tolerance_mm,
                        expected,
                        proofs,
                    )
                    failures = (
                        _bad(checked[0], material_runs, radius_proofs)
                        if checked is not None
                        else remaining_bad
                    )
                    if checked is None or not failures < remaining_bad:
                        corner_report["topology_rejected"] += 1
                        continue
                    contours, proxy = checked
                    parent = candidate
                    ring = candidate_ring
                    by_color[color][owner] = parent
                    corner_report["repaired"] += len(remaining_bad) - len(failures)
                    remaining_bad = failures
                    break
    return contours, proxy, corner_report


def refine(
    source: PanelContours,
    contours: c.CurveContours,
    proxy: PanelContours,
    *,
    scale_mm: float,
    tolerance_mm: float,
    material_runs: tuple[float, float, float],
    proxy_tolerance_mm: float,
    repair_corners: bool = False,
) -> tuple[c.CurveContours, PanelContours, dict[str, object]]:
    """Interpret failing tips and explicitly authorized complex contact groups.

    Point-contact merges are explicit topology changes. The reference helper
    authorizes just that change; original multi-ring source distance and the
    complete palette embedding independently certify each selected geometry.
    """
    if not isfinite(proxy_tolerance_mm) or proxy_tolerance_mm <= 0:
        raise ValueError("contact refinement requires positive proxy tolerance")
    if type(cast("object", repair_corners)) is not bool:
        raise ValueError("repair_corners must be a boolean")
    plan = ci.prepare(source, scale_mm=scale_mm, tolerance_mm=tolerance_mm)
    interpretation = plan.select()
    if len(contours.orange) != len(source.orange) or len(contours.white) != len(
        source.white
    ):
        raise ValueError("contact refinement requires original component indexing")
    radius_proofs = boundary_tapers.ProofCache()
    old_bad = _bad(contours, material_runs, radius_proofs)
    by_color: dict[ci.Color, dict[int, c.CurvePolygon]] = {
        "orange": dict(enumerate(contours.orange)),
        "white": dict(enumerate(contours.white)),
    }
    accepted: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    selected: _Selected = frozenset()
    reference = plan.select(selected)
    source_groups: dict[tuple[ci.Color, int], list[tuple[_Path, ...]]] = {}
    forward_proofs: dict[
        tuple[ci.Color, int], list[geometric_distance.ForwardProofCache]
    ] = {}
    certificates: dict[_DistanceKey, bool] = {}
    proofs = c.ProofCache()
    preparation_seconds = 0.0
    paired_checks = union_checks = 0
    for contact in interpretation.contacts:
        parent = by_color[contact.color][contact.parent]
        points = contact.points or (contact.point,)
        members = contact.members or (contact.tip,)
        complex_contact = len(members) > 1 or len(points) > 1
        incident = {
            _key(span)
            for ring in (parent.exterior, *parent.interiors)
            for span in ring.spans
            if any(
                point in (span.control_points[0], span.control_points[-1])
                for point in points
            )
        }
        if not complex_contact and not any(
            _key(span) in old_bad
            and contact.point in (span.control_points[0], span.control_points[-1])
            for ring in (parent.exterior, *parent.interiors)
            for span in ring.spans
        ):
            continue
        preparation_started = monotonic()
        candidates = _candidates(
            contact,
            parent,
            _source(source, contact.color)[contact.parent],
            old_bad | incident if complex_contact else old_bad,
            scale_mm,
            tolerance_mm,
        )
        attempted = 0
        choice: _Selected = selected | frozenset[tuple[ci.Color, int]](
            ((contact.color, contact.tip),)
        )
        proposed_reference = plan.select(choice)
        reference_tips: _Selected = frozenset(
            (v.color, v.tip) for v in proposed_reference.contacts
        )
        if reference_tips != choice:
            preparation_seconds += monotonic() - preparation_started
            continue
        ordered = _ordered_reference(source, proposed_reference)
        if ordered is None:
            preparation_seconds += monotonic() - preparation_started
            blocked.append(
                dict(
                    color=contact.color,
                    tip=contact.tip,
                    parent=contact.parent,
                    candidates=0,
                    reason="unmatched-reference-rings",
                )
            )
            continue
        proposed_reference = replace(proposed_reference, contours=ordered)
        expected = _ReferenceTopology(ordered)
        groups = _groups(proposed_reference, contact.color)
        group = next(indices for indices in groups if contact.parent in indices)
        originals = [_source(source, contact.color)[i] for i in group]
        parent_rings = (parent.exterior, *parent.interiors)
        source_key = contact.color, contact.parent
        if source_key not in source_groups:
            original = _source(source, contact.color)[contact.parent]
            source_groups[source_key] = [
                (tuple((float(x), float(y)) for x, y in ring.coords),)
                for ring in (original.exterior, *original.interiors)
            ]
            forward_proofs[source_key] = [
                geometric_distance.ForwardProofCache(group, tolerance_mm)
                for group in source_groups[source_key]
            ]
        paths = source_groups[source_key]
        forward = forward_proofs[source_key]
        baseline = [
            _source_certificate(group, ring.spans, tolerance_mm, certificates, stored)
            for group, ring, stored in zip(paths, parent_rings, forward, strict=True)
        ]
        tip_paths = tuple(
            tuple(path)
            for path in _paths([_source(source, contact.color)[i] for i in members])
        )
        # Baseline proofs are reusable preprocessing, accounted separately.
        # They must not consume the bounded candidate search's wall time.
        preparation_seconds += monotonic() - preparation_started
        deadline = monotonic() + 12.0
        deferred: list[c.CurvePolygon] = []
        # Pairing is a sufficient certificate, not a new rejection rule. Try
        # those cheap candidates first, then retain the original whole-union
        # predicate for every conservatively rejected/inconclusive pairing.
        searches = chain(
            ((candidate, True) for candidate in candidates),
            ((candidate, False) for candidate in deferred),
        )
        for candidate, paired in searches:
            if monotonic() >= deadline:
                break
            attempted += 1
            rings = (candidate.exterior, *candidate.interiors)
            changed = [
                i
                for i, (before, after) in enumerate(
                    zip(parent_rings, rings, strict=True)
                )
                if before != after
            ]
            if len(changed) != 1:
                raise ValueError("attachment candidate must change exactly one ring")
            ring_index = changed[0]
            original_paths = (*paths[ring_index], *tip_paths)
            if paired:
                paired_checks += 1
                if not all(
                    ok for i, ok in enumerate(baseline) if i != ring_index
                ) or not _source_certificate(
                    original_paths,
                    rings[ring_index].spans,
                    tolerance_mm,
                    certificates,
                    forward[ring_index],
                ):
                    deferred.append(candidate)
                    continue
            else:
                union_checks += 1
                if not _source_certificate(
                    tuple(tuple(path) for path in _paths(originals)),
                    [s for ring in rings for s in ring.spans],
                    tolerance_mm,
                    certificates,
                ):
                    continue
            polygons = [contours.silhouette]
            for color in by_color:
                for indices in _groups(proposed_reference, color):
                    parts = _source(source, color)
                    owner = max(indices, key=lambda i, parts=parts: parts[i].area)
                    polygons.append(
                        candidate
                        if color == contact.color and owner == contact.parent
                        else by_color[color][owner]
                    )
            checked = _assemble(
                polygons,
                proposed_reference.contours,
                proxy_tolerance_mm,
                expected,
                proofs,
            )
            if (
                checked is None
                or not _bad(checked[0], material_runs, radius_proofs) <= old_bad
            ):
                continue
            contours, proxy = checked
            by_color[contact.color][contact.parent] = candidate
            paths[ring_index] = original_paths
            selected = frozenset(choice)
            reference = proposed_reference
            accepted.append(
                dict(
                    color=contact.color,
                    tip=contact.tip,
                    parent=contact.parent,
                    point=contact.point,
                    points=points,
                    members=members,
                    source_group=group,
                    candidates=attempted,
                )
            )
            break
        if choice != selected:
            blocked.append(
                dict(
                    color=contact.color,
                    tip=contact.tip,
                    parent=contact.parent,
                    candidates=attempted,
                    reason="time-budget"
                    if monotonic() >= deadline
                    else "no-certified-candidate",
                )
            )
    corner_report: dict[str, int] = dict(
        attempted=0, repaired=0, source_rejected=0, topology_rejected=0
    )
    if repair_corners:
        contours, proxy, corner_report = _repair_corners(
            source,
            contours,
            proxy,
            reference,
            source_groups,
            tolerance_mm=tolerance_mm,
            material_runs=material_runs,
            proxy_tolerance_mm=proxy_tolerance_mm,
            proofs=proofs,
            radius_proofs=radius_proofs,
        )
    return (
        contours,
        proxy,
        {
            "changed": bool(accepted) or bool(corner_report["repaired"]),
            "corner_refinement": corner_report,
            "selected_unique_spans": len(
                {
                    _key(span)
                    for polygon in (
                        contours.silhouette,
                        *contours.orange,
                        *contours.white,
                    )
                    for ring in (polygon.exterior, *polygon.interiors)
                    for span in ring.spans
                }
            ),
            "contact_interpretation": {
                "method": "local pointed attachment repair",
                "joined_tips": len(accepted),
                "joins": accepted,
                "blocked": blocked,
                "orange_sources": reference.orange_sources,
                "white_sources": reference.white_sources,
                "authorized_topology": reference.diagnostics.get(
                    "contact_interpretation", {}
                ),
                "raw_source_distance_certified": bool(accepted),
                "raw_source_tolerance_mm": tolerance_mm,
                "certificate": "original multi-ring source distance, exact curve separation, complete palette and ring topology; no new radius failures",
                "splice_policy": "outside curves unchanged; float-generated patch rays are not claimed exact G1",
                "source_preparation_seconds": preparation_seconds,
                "paired_source_checks": paired_checks,
                "whole_union_source_checks": union_checks,
            },
        },
    )
