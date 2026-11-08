"""Resolve bounded point-contact fragments without refitting other outlines."""

# Exact proof primitives are shared with the ordinary outline fitter.
# pyright: reportPrivateUsage=false

from collections.abc import Iterator, Sequence
from dataclasses import replace
from itertools import chain, pairwise
from math import hypot, isfinite, ulp
from time import monotonic
from typing import cast

from shapely.geometry import LineString, Polygon

from panelchemy import boundary_tapers, geometric_distance
from panelchemy import contact_interpretation as ci
from panelchemy import curves as c
from panelchemy.contours import PanelContours

type _Key = tuple[bool, c.Controls]
type _Selected = frozenset[tuple[ci.Color, int]]
type _Path = tuple[c.Point2, ...]
type _DistanceKey = tuple[tuple[_Path, ...], tuple[_Key, ...], float]


def _key(span: c.BezierSpan) -> _Key:
    p = span.control_points
    if span.is_line:
        p = (p[0], p[0], p[-1], p[-1])
    return span.is_line, min(p, cast("c.Controls", tuple(reversed(p))))


def _bad(contours: c.CurveContours, runs: tuple[float, float, float]) -> set[_Key]:
    try:
        boundary_tapers.require_curvature(contours, runs)
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


def _distance_key(
    paths: Sequence[_Path],
    spans: Sequence[c.BezierSpan],
    tolerance: float,
) -> _DistanceKey:
    # This cache lives for one refine call. Both original paths and selected
    # coefficients are part of the key, so an accepted edit cannot reuse an
    # older proof for the changed ring or its enlarged original source group.
    return (
        tuple(paths),
        tuple((s.is_line, s.control_points) for s in spans),
        tolerance,
    )


def _source_certificate(
    paths: Sequence[_Path],
    spans: Sequence[c.BezierSpan],
    tolerance: float,
    cache: dict[_DistanceKey, bool],
) -> bool:
    key = _distance_key(paths, spans, tolerance)
    if key not in cache:
        cache[key] = geometric_distance.within_paths(paths, spans, tolerance)
    return cache[key]


def _source_certificates(
    jobs: Sequence[geometric_distance.DistanceJob], cache: dict[_DistanceKey, bool]
) -> list[bool]:
    keys = [
        _distance_key(tuple(tuple(path) for path in paths), spans, tolerance)
        for paths, spans, tolerance in jobs
    ]
    missing: dict[_DistanceKey, geometric_distance.DistanceJob] = {}
    for key, job in zip(keys, jobs, strict=True):
        if key not in cache:
            _ = missing.setdefault(key, job)
    if missing:
        results = geometric_distance.within_paths_many(list(missing.values()))
        cache.update(zip(missing, results, strict=True))
    return [cache[key] for key in keys]


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
            # Match the surviving hole interiors one-to-one. This only orders
            # existing reference rings; geometry, source bounds and complete
            # topology still pass their ordinary independent proof gates.
            unused = set(range(len(new)))
            order: list[int] = []
            for ring in old:
                matches = [
                    i for i in unused if ring.relate_pattern(new[i], "T********")
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
    polygons: list[c.CurvePolygon], reference: PanelContours, tolerance: float
) -> tuple[c.CurveContours, PanelContours] | None:
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
        unsafe, cuts = c._unsafe_pair(pieces)
        if unsafe:
            return None
        curves, proxy, rings, proxy_unsafe = c._assemble(
            pieces, uses, ranges, reference, tolerance, cuts
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
    expected = [reference.silhouette, *reference.orange, *reference.white]
    actual = [proxy.silhouette, *proxy.orange, *proxy.white]
    if (
        proxy_unsafe
        or any(
            not p.is_valid
            or p.area <= 0
            or len(p.interiors) != len(original.interiors)
            or not reference.silhouette.covers(p)
            for p, original in zip(actual, expected, strict=True)
        )
        or c._relationships(actual) != c._relationships(expected)
        or c._relationships(rings)
        != c._relationships(
            [
                LineString(r.coords)
                for polygon in expected
                for r in (polygon.exterior, *polygon.interiors)
            ]
        )
    ):
        return None
    return curves, proxy


def refine(
    source: PanelContours,
    contours: c.CurveContours,
    proxy: PanelContours,
    *,
    scale_mm: float,
    tolerance_mm: float,
    material_runs: tuple[float, float, float],
    proxy_tolerance_mm: float,
) -> tuple[c.CurveContours, PanelContours, dict[str, object]]:
    """Interpret failing tips and explicitly authorized complex contact groups.

    Point-contact merges are explicit topology changes. The reference helper
    authorizes just that change; original multi-ring source distance and the
    complete palette embedding independently certify each selected geometry.
    """
    if not isfinite(proxy_tolerance_mm) or proxy_tolerance_mm <= 0:
        raise ValueError("contact refinement requires positive proxy tolerance")
    interpretation = ci.before_graph(
        source, scale_mm=scale_mm, tolerance_mm=tolerance_mm
    )
    if len(contours.orange) != len(source.orange) or len(contours.white) != len(
        source.white
    ):
        raise ValueError("contact refinement requires original component indexing")
    old_bad = _bad(contours, material_runs)
    by_color: dict[ci.Color, dict[int, c.CurvePolygon]] = {
        "orange": dict(enumerate(contours.orange)),
        "white": dict(enumerate(contours.white)),
    }
    accepted: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    selected: _Selected = frozenset()
    reference = ci.before_graph(
        source, scale_mm=scale_mm, tolerance_mm=tolerance_mm, selected_tips=selected
    )
    source_groups: dict[tuple[ci.Color, int], list[tuple[_Path, ...]]] = {}
    certificates: dict[_DistanceKey, bool] = {}
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
        proposed_reference = ci.before_graph(
            source,
            scale_mm=scale_mm,
            tolerance_mm=tolerance_mm,
            selected_tips=choice,
        )
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
        paths = source_groups[source_key]
        baseline = _source_certificates(
            [
                (group, ring.spans, tolerance_mm)
                for group, ring in zip(paths, parent_rings, strict=True)
            ],
            certificates,
        )
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
                    original_paths, rings[ring_index].spans, tolerance_mm, certificates
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
                polygons, proposed_reference.contours, proxy_tolerance_mm
            )
            if checked is None or not _bad(checked[0], material_runs) <= old_bad:
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
    return (
        contours,
        proxy,
        {
            "changed": bool(accepted),
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
