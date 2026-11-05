"""Shared cubic boundaries with continuous error bounds and local refinement."""

import itertools
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import fsum, hypot, inf, isfinite, nextafter
from typing import Literal, cast

from shapely import STRtree, get_parts, union_all
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from panelchemy.contours import PanelContours
from panelchemy.separation import Palette, SeparationConfig

type Point2 = tuple[float, float]
type Controls = tuple[Point2, Point2, Point2, Point2]
type RationalPoint = tuple[Fraction, Fraction]
type RationalControls = tuple[
    RationalPoint, RationalPoint, RationalPoint, RationalPoint
]
type Edge = tuple[Point2, Point2]
_HALF = Fraction(1, 2)
_UNIT_INTERVAL = (Fraction(0), Fraction(1))


@dataclass(frozen=True)
class BezierSpan:
    control_points: Controls
    is_line: bool
    certified_error_mm: float = 0.0


@dataclass(frozen=True)
class CurveRing:
    spans: tuple[BezierSpan, ...]


@dataclass(frozen=True)
class CurvePolygon:
    exterior: CurveRing
    interiors: tuple[CurveRing, ...] = ()


@dataclass(frozen=True)
class CurveContours:
    silhouette: CurvePolygon
    orange: tuple[CurvePolygon, ...]
    white: tuple[CurvePolygon, ...]
    mm_per_pixel: float
    source_bounds_px: tuple[float, float, float, float]

    @property
    def edge_count(self) -> int:
        return sum(
            len(ring.spans)
            for polygon in (self.silhouette, *self.orange, *self.white)
            for ring in (polygon.exterior, *polygon.interiors)
        )


@dataclass(frozen=True)
class CurveFitResult:
    contours: CurveContours
    proxy: PanelContours
    diagnostics: dict[str, object]
    comparison_svg: str
    proxy_deviation_mm: float


@dataclass(frozen=True)
class _Chain:
    points: tuple[Point2, ...]
    fixed: bool
    free_loop: bool = False


@dataclass(frozen=True)
class _Piece:
    chain: int
    start: int
    stop: int
    span: BezierSpan


def _point(coordinate: Sequence[float]) -> Point2:
    return float(coordinate[0]), float(coordinate[1])


def _edge(a: Point2, b: Point2) -> Edge:
    return (a, b) if a < b else (b, a)


def _line(a: Point2, b: Point2, error: float = 0.0) -> BezierSpan:
    return BezierSpan(
        (
            a,
            ((2 * a[0] + b[0]) / 3, (2 * a[1] + b[1]) / 3),
            ((a[0] + 2 * b[0]) / 3, (a[1] + 2 * b[1]) / 3),
            b,
        ),
        True,
        error,
    )


def polygon_lines(polygon: Polygon) -> CurvePolygon:
    """Represent unchanged straight edges; native adapters use line endpoints."""

    def ring(coordinates: Sequence[Point2]) -> CurveRing:
        return CurveRing(tuple(_line(a, b) for a, b in itertools.pairwise(coordinates)))

    return CurvePolygon(
        ring([_point(p) for p in polygon.exterior.coords]),
        tuple(
            ring([_point(p) for p in interior.coords]) for interior in polygon.interiors
        ),
    )


@lru_cache(maxsize=131_072)
def _rational(point: Point2) -> RationalPoint:
    return Fraction(point[0]), Fraction(point[1])


@lru_cache(maxsize=65_536)
def _controls(span: BezierSpan) -> RationalControls:
    a, b, c, d = tuple(_rational(point) for point in span.control_points)
    if span.is_line:
        b = ((2 * a[0] + d[0]) / 3, (2 * a[1] + d[1]) / 3)
        c = ((a[0] + 2 * d[0]) / 3, (a[1] + 2 * d[1]) / 3)
    return a, b, c, d


def _mix(a: RationalPoint, b: RationalPoint, t: Fraction) -> RationalPoint:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _split(
    controls: RationalControls, t: Fraction = _HALF
) -> tuple[RationalControls, RationalControls]:
    a, b, c, d = controls
    ab, bc, cd = _mix(a, b, t), _mix(b, c, t), _mix(c, d, t)
    abc, bcd = _mix(ab, bc, t), _mix(bc, cd, t)
    middle = _mix(abc, bcd, t)
    return (a, ab, abc, middle), (middle, bcd, cd, d)


def _restrict(
    controls: RationalControls, start: Fraction, stop: Fraction
) -> RationalControls:
    left, _ = _split(controls, stop)
    if start == 0:
        return left
    _, selected = _split(left, start / stop)
    return selected


def _squared(point: RationalPoint) -> Fraction:
    return point[0] ** 2 + point[1] ** 2


def _bounded(
    controls: RationalControls, budget_squared: Fraction, depth: int = 0
) -> bool:
    """A Bernstein convex-hull bound on every parameter, with exact arithmetic."""
    if all(_squared(point) <= budget_squared for point in controls):
        return True
    if (
        _squared(controls[0]) > budget_squared
        or _squared(controls[-1]) > budget_squared
        or depth == 8
    ):
        return False
    left, right = _split(controls)
    return _bounded(left, budget_squared, depth + 1) and _bounded(
        right, budget_squared, depth + 1
    )


def _parameters(points: Sequence[Point2]) -> tuple[float, ...]:
    lengths = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points)]
    total = fsum(lengths)
    if not isfinite(total) or total <= 0:
        raise ValueError("curve source paths must have finite positive length")
    accumulated = 0.0
    values = [0.0]
    for length in lengths[:-1]:
        accumulated += length
        values.append(accumulated / total)
    values.append(1.0)
    if any(a >= b for a, b in itertools.pairwise(values)):
        raise ValueError("curve source parameters cannot be represented distinctly")
    return tuple(values)


def span_within_tolerance(
    span: BezierSpan, source: Sequence[Point2], tolerance_mm: float
) -> bool:
    """Certify both directed distances by a common continuous parameterization.

    Each source line is degree-elevated to a cubic. Difference-control hulls
    bound the entire matching Bézier interval, not merely sampled vertices.
    """
    if not isfinite(tolerance_mm) or tolerance_mm < 0 or len(source) < 2:
        raise ValueError("a finite nonnegative tolerance and source path are required")
    if span.control_points[0] != source[0] or span.control_points[-1] != source[-1]:
        return False
    parameters = _parameters(source)
    controls = _controls(span)
    budget = Fraction(tolerance_mm) ** 2
    if span.is_line:
        # On each source interval the matched difference is affine, so its
        # squared norm is convex and attains its maximum at an endpoint.
        a, d = controls[0], controls[-1]
        return all(
            _squared((point[0] - selected[0], point[1] - selected[1])) <= budget
            for point, selected in (
                (_rational(point), _mix(a, d, Fraction(t)))
                for point, t in zip(source, parameters, strict=True)
            )
        )
    for index, (a, b) in enumerate(itertools.pairwise(source)):
        selected = _restrict(
            controls, Fraction(parameters[index]), Fraction(parameters[index + 1])
        )
        line = _controls(_line(a, b))
        difference = cast(
            "RationalControls",
            tuple(
                (p[0] - q[0], p[1] - q[1]) for p, q in zip(selected, line, strict=True)
            ),
        )
        if not _bounded(difference, budget):
            return False
    return True


def _unit(a: Point2, b: Point2) -> Point2:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = hypot(dx, dy)
    return dx / length, dy / length


def _proposal(points: Sequence[Point2], tolerance: float) -> BezierSpan:
    a, d = points[0], points[-1]
    window = min(3, max(1, (len(points) - 1) // 3))
    left = _unit(a, points[window])
    right = _unit(d, points[-window - 1])
    c00 = c01 = c11 = x0 = x1 = 0.0
    for point, t in zip(points, _parameters(points), strict=True):
        u = 1 - t
        b0, b1, b2, b3 = u**3, 3 * t * u**2, 3 * t**2 * u, t**3
        base = (
            a[0] * (b0 + b1) + d[0] * (b2 + b3),
            a[1] * (b0 + b1) + d[1] * (b2 + b3),
        )
        v0, v1 = (left[0] * b1, left[1] * b1), (right[0] * b2, right[1] * b2)
        delta = (point[0] - base[0], point[1] - base[1])
        c00 += v0[0] ** 2 + v0[1] ** 2
        c01 += v0[0] * v1[0] + v0[1] * v1[1]
        c11 += v1[0] ** 2 + v1[1] ** 2
        x0 += delta[0] * v0[0] + delta[1] * v0[1]
        x1 += delta[0] * v1[0] + delta[1] * v1[1]
    determinant = c00 * c11 - c01**2
    chord = hypot(d[0] - a[0], d[1] - a[1])
    alpha = (x0 * c11 - x1 * c01) / determinant if determinant > 0 else chord / 3
    beta = (c00 * x1 - c01 * x0) / determinant if determinant > 0 else chord / 3
    if alpha <= 0 or not isfinite(alpha):
        alpha = chord / 3
    if beta <= 0 or not isfinite(beta):
        beta = chord / 3
    return BezierSpan(
        (
            a,
            (a[0] + alpha * left[0], a[1] + alpha * left[1]),
            (d[0] + beta * right[0], d[1] + beta * right[1]),
            d,
        ),
        False,
        tolerance,
    )


def _monotone(span: BezierSpan) -> bool:
    controls = _controls(span)
    direction = (controls[-1][0] - controls[0][0], controls[-1][1] - controls[0][1])
    return any(direction) and all(
        (b[0] - a[0]) * direction[0] + (b[1] - a[1]) * direction[1] >= 0
        for a, b in itertools.pairwise(controls)
    )


def _fit_chain(
    chain_id: int,
    chain: _Chain,
    tolerance: float,
    start: int = 0,
    stop: int | None = None,
    *,
    allow_curves: bool = True,
) -> list[_Piece]:
    final = len(chain.points) - 1 if stop is None else stop
    pending = [(start, final)]
    output: list[_Piece] = []
    while pending:
        first, last = pending.pop()
        points = chain.points[first : last + 1]
        line = _line(points[0], points[-1], tolerance if len(points) > 2 else 0.0)
        if len(points) == 2 or (
            not chain.fixed
            and tolerance > 0
            and span_within_tolerance(line, points, tolerance)
        ):
            selected = line
        elif allow_curves and not chain.fixed and tolerance > 0:
            candidate = _proposal(points, tolerance)
            selected = (
                candidate
                if _monotone(candidate)
                and span_within_tolerance(candidate, points, tolerance)
                else None
            )
        else:
            selected = None
        if selected is not None:
            output.append(_Piece(chain_id, first, last, selected))
            continue
        middle = (first + last) // 2
        pending.extend([(middle, last), (first, middle)])
    return output


def _graph(
    rings: Sequence[LineString],
    *,
    split_loops: bool = True,
) -> tuple[list[_Chain], list[list[tuple[int, bool]]]]:
    segments: list[LineString] = []
    owners: list[int] = []
    for owner, ring in enumerate(rings):
        coordinates = [_point(point) for point in ring.coords]
        for a, b in itertools.pairwise(coordinates):
            segments.append(LineString([a, b]))
            owners.append(owner)
    source_tree = STRtree(segments)
    parts = cast("list[BaseGeometry]", get_parts(union_all(segments)).tolist())
    edge_owners: dict[Edge, frozenset[int]] = {}
    for part in parts:
        if not isinstance(part, LineString):
            raise ValueError("source boundaries did not form a line network")
        coordinates = [_point(point) for point in part.coords]
        for a, b in itertools.pairwise(coordinates):
            edge = _edge(a, b)
            candidates = cast(
                "list[int]",
                source_tree.query(LineString(edge), predicate="covered_by").tolist(),
            )
            if not candidates:
                raise ValueError("noding changed a source boundary")
            edge_owners[edge] = frozenset(owners[index] for index in candidates)
    adjacency: dict[Point2, list[Edge]] = defaultdict(list)
    for edge in edge_owners:
        for point in edge:
            adjacency[point].append(edge)

    def terminal(point: Point2) -> bool:
        edges = adjacency[point]
        return len(edges) != 2 or edge_owners[edges[0]] != edge_owners[edges[1]]

    visited: set[Edge] = set()
    paths: list[tuple[list[Point2], bool, bool]] = []
    for initial in sorted(
        edge_owners, key=lambda edge: (not any(terminal(p) for p in edge), edge)
    ):
        if initial in visited:
            continue
        first = next((p for p in initial if terminal(p)), initial[0])
        path = [first]
        current = initial
        while current not in visited:
            visited.add(current)
            following = current[1] if path[-1] == current[0] else current[0]
            path.append(following)
            if terminal(following) or following == first:
                break
            current = next(edge for edge in adjacency[following] if edge != current)
        fixed = 0 in edge_owners[initial]
        if path[0] == path[-1] and split_loops:
            anchor = max(
                range(1, len(path) - 1),
                key=lambda i: (
                    hypot(path[i][0] - path[0][0], path[i][1] - path[0][1]),
                    path[i],
                ),
            )
            paths.extend(
                [(path[: anchor + 1], fixed, False), (path[anchor:], fixed, False)]
            )
        else:
            paths.append((path, fixed, path[0] == path[-1] and not terminal(first)))
    chains = [
        _Chain(tuple(path if path[0] < path[-1] else reversed(path)), fixed, free_loop)
        for path, fixed, free_loop in paths
    ]
    lookup: dict[Edge, int] = {}
    ring_edges: list[list[Edge]] = [[] for _ in rings]
    for edge, edge_owners_set in edge_owners.items():
        for owner in edge_owners_set:
            ring_edges[owner].append(edge)
    for index, chain in enumerate(chains):
        for a, b in zip(chain.points[:-1], chain.points[1:], strict=True):
            lookup[_edge(a, b)] = index
    uses: list[list[tuple[int, bool]]] = []
    for ring, owned in zip(rings, ring_edges, strict=True):
        ring_adjacency: dict[Point2, list[Point2]] = defaultdict(list)
        for a, b in owned:
            ring_adjacency[a].append(b)
            ring_adjacency[b].append(a)
        if any(len(neighbors) != 2 for neighbors in ring_adjacency.values()):
            raise ValueError("a source ring does not have a simple boundary")
        source_points = [_point(point) for point in ring.coords]
        first = source_points[0]
        first_segment = LineString(source_points[:2])
        following = next(
            point
            for point in ring_adjacency[first]
            if first_segment.covers(LineString([first, point]))
        )
        ordered = [first]
        previous = first
        while following != first:
            ordered.append(following)
            if len(ordered) > len(owned):
                raise ValueError("source ring boundary traversal did not close")
            previous, following = (
                following,
                next(point for point in ring_adjacency[following] if point != previous),
            )
        start = next(
            index
            for index, point in enumerate(ordered)
            if point
            in (
                chains[
                    lookup[_edge(point, ordered[(index + 1) % len(ordered)])]
                ].points[0],
                chains[
                    lookup[_edge(point, ordered[(index + 1) % len(ordered)])]
                ].points[-1],
            )
        )
        ordered = ordered[start:] + ordered[:start]
        if not LineString([*ordered, ordered[0]]).equals(ring):
            raise ValueError("shared-chain extraction changed a source ring")
        selected: list[tuple[int, bool]] = []
        offset = 0
        while offset < len(ordered):
            chain_id = lookup[
                _edge(ordered[offset], ordered[(offset + 1) % len(ordered)])
            ]
            chain = chains[chain_id]
            forward = ordered[offset] == chain.points[0]
            if chain.points[0] == chain.points[-1]:
                forward = ordered[(offset + 1) % len(ordered)] == chain.points[1]
            expected = chain.points if forward else tuple(reversed(chain.points))
            if any(
                ordered[(offset + index) % len(ordered)] != point
                for index, point in enumerate(expected)
            ):
                raise ValueError(
                    "source ring does not follow one shared boundary chain"
                )
            selected.append((chain_id, forward))
            offset += len(chain.points) - 1
        uses.append(selected)
    return chains, uses


def _separated(
    a: Sequence[RationalPoint], b: Sequence[RationalPoint], allowed: set[RationalPoint]
) -> bool:
    axes: set[RationalPoint] = set()
    for points in (a, b):
        for i, p in enumerate(points):
            for q in points[i + 1 :]:
                dx, dy = q[0] - p[0], q[1] - p[1]
                if dx or dy:
                    axes.update(((dx, dy), (-dy, dx)))
    for x, y in axes:
        av = [p[0] * x + p[1] * y for p in a]
        bv = [p[0] * x + p[1] * y for p in b]
        for first, fv, second, sv in ((a, av, b, bv), (b, bv, a, av)):
            maximum, minimum = max(fv), min(sv)
            if maximum < minimum:
                return True
            if maximum == minimum:
                boundary_first = {
                    p for p, value in zip(first, fv, strict=True) if value == maximum
                }
                boundary_second = {
                    p for p, value in zip(second, sv, strict=True) if value == minimum
                }
                if (len(boundary_first) == 1 and boundary_first <= allowed) or (
                    len(boundary_second) == 1 and boundary_second <= allowed
                ):
                    return True
    return False


def _disjoint(
    a: RationalControls,
    b: RationalControls,
    allowed: set[RationalPoint],
    depth: int = 0,
    cuts_a: set[Fraction] | None = None,
    cuts_b: set[Fraction] | None = None,
    range_a: tuple[Fraction, Fraction] = _UNIT_INTERVAL,
    range_b: tuple[Fraction, Fraction] = _UNIT_INTERVAL,
) -> bool:
    if _separated(a, b, allowed):
        return True
    if depth == 12:
        return False

    def extent(points: RationalControls) -> Fraction:
        return (
            max(p[0] for p in points)
            - min(p[0] for p in points)
            + max(p[1] for p in points)
            - min(p[1] for p in points)
        )

    if extent(a) >= extent(b):
        left, right = _split(a)
        middle = (range_a[0] + range_a[1]) / 2
        if cuts_a is not None:
            cuts_a.add(middle)
        return _disjoint(
            left, b, allowed, depth + 1, cuts_a, cuts_b, (range_a[0], middle), range_b
        ) and _disjoint(
            right, b, allowed, depth + 1, cuts_a, cuts_b, (middle, range_a[1]), range_b
        )
    left, right = _split(b)
    middle = (range_b[0] + range_b[1]) / 2
    if cuts_b is not None:
        cuts_b.add(middle)
    return _disjoint(
        a, left, allowed, depth + 1, cuts_a, cuts_b, range_a, (range_b[0], middle)
    ) and _disjoint(
        a, right, allowed, depth + 1, cuts_a, cuts_b, range_a, (middle, range_b[1])
    )


def _unsafe_pair(
    pieces: Sequence[_Piece],
) -> tuple[list[tuple[int, int]], list[set[Fraction]]]:
    boxes = [
        box(
            min(p[0] for p in piece.span.control_points),
            min(p[1] for p in piece.span.control_points),
            max(p[0] for p in piece.span.control_points),
            max(p[1] for p in piece.span.control_points),
        )
        for piece in pieces
    ]
    tree = STRtree(boxes)
    controls = [_controls(piece.span) for piece in pieces]
    cuts: list[set[Fraction]] = [set() for _ in pieces]
    unsafe: list[tuple[int, int]] = []
    checked = 0
    for index, bounds in enumerate(boxes):
        for neighbor in cast("list[int]", tree.query(bounds).tolist()):
            if neighbor <= index:
                continue
            checked += 1
            if checked > 1_000_000:
                raise ValueError("curve topology exceeds 1,000,000 candidate pairs")
            a, b = controls[index], controls[neighbor]
            allowed = {a[0], a[-1]} & {b[0], b[-1]}
            if not _disjoint(a, b, allowed, cuts_a=cuts[index], cuts_b=cuts[neighbor]):
                unsafe.append((index, neighbor))
    return unsafe, cuts


def _flatten(
    span: BezierSpan, tolerance: float, cuts: set[Fraction]
) -> tuple[tuple[Point2, ...], list[tuple[RationalPoint, ...]]]:
    if span.is_line:
        endpoints = (span.control_points[0], span.control_points[-1])
        return endpoints, [tuple(_rational(point) for point in endpoints)]
    result = [span.control_points[0]]
    hulls: list[tuple[RationalPoint, ...]] = []
    controls = _controls(span)
    parameters = [Fraction(0), *sorted(cuts), Fraction(1)]
    pending = [
        (_restrict(controls, a, b), 0)
        for a, b in reversed(list(itertools.pairwise(parameters)))
    ]
    while pending:
        controls, depth = pending.pop()
        a, b = (
            tuple(float(v) for v in controls[0]),
            tuple(float(v) for v in controls[-1]),
        )
        chord = _controls(_line(cast("Point2", a), cast("Point2", b)))
        difference = cast(
            "RationalControls",
            tuple(
                (p[0] - q[0], p[1] - q[1]) for p, q in zip(controls, chord, strict=True)
            ),
        )
        if _bounded(difference, Fraction(tolerance) ** 2):
            result.append(cast("Point2", b))
            hulls.append(
                (*controls, _rational(cast("Point2", a)), _rational(cast("Point2", b)))
            )
        elif depth == 24:
            raise ValueError("curve proxy cannot meet its continuous error budget")
        else:
            left, right = _split(controls)
            pending.extend([(right, depth + 1), (left, depth + 1)])
    original = _controls(span)
    direction = (original[-1][0] - original[0][0], original[-1][1] - original[0][1])
    if any(
        (b[0] - a[0]) * direction[0] + (b[1] - a[1]) * direction[1] < 0
        for a, b in itertools.pairwise([_rational(point) for point in result])
    ):
        raise ValueError("rounded curve proxy cannot preserve monotone embedding")
    return tuple(result), hulls


def _proxy_unsafe(
    pieces: Sequence[_Piece], hulls: Sequence[Sequence[tuple[RationalPoint, ...]]]
) -> list[tuple[int, int]]:
    """Certify the complete native-to-proxy homotopy, including float rounding."""
    records = [(index, hull) for index, part in enumerate(hulls) for hull in part]
    boxes = [
        box(
            nextafter(float(min(p[0] for p in hull)), -inf),
            nextafter(float(min(p[1] for p in hull)), -inf),
            nextafter(float(max(p[0] for p in hull)), inf),
            nextafter(float(max(p[1] for p in hull)), inf),
        )
        for _, hull in records
    ]
    tree = STRtree(boxes)
    checked = 0
    unsafe: set[tuple[int, int]] = set()
    for i, bounds in enumerate(boxes):
        left, a = records[i]
        for j in cast("list[int]", tree.query(bounds).tolist()):
            right, b = records[j]
            if j <= i or left == right:
                continue
            checked += 1
            if checked > 1_000_000:
                raise ValueError(
                    "curve/proxy topology exceeds 1,000,000 candidate pairs"
                )
            sa, sb = pieces[left].span, pieces[right].span
            allowed = {
                _rational(sa.control_points[0]),
                _rational(sa.control_points[-1]),
            } & {_rational(sb.control_points[0]), _rational(sb.control_points[-1])}
            if not _separated(a, b, allowed):
                unsafe.add((min(left, right), max(left, right)))
    return sorted(unsafe)


def _relationships(geometries: Sequence[BaseGeometry]) -> dict[tuple[int, int], str]:
    tree = STRtree(geometries)
    return {
        (i, j): geometry.relate(geometries[j])
        for i, geometry in enumerate(geometries)
        for j in cast(
            "list[int]", tree.query(geometry, predicate="intersects").tolist()
        )
        if i < j
    }


def _assemble(
    pieces: Sequence[_Piece],
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    source: PanelContours,
    proxy_tolerance: float,
    topology_cuts: Sequence[set[Fraction]],
) -> tuple[CurveContours, PanelContours, list[LineString], list[tuple[int, int]]]:
    by_chain: dict[int, list[_Piece]] = defaultdict(list)
    flattened: dict[_Piece, tuple[Point2, ...]] = {}
    hulls: list[list[tuple[RationalPoint, ...]]] = []
    for piece, cuts in zip(pieces, topology_cuts, strict=True):
        by_chain[piece.chain].append(piece)
        flattened[piece], leaf_hulls = _flatten(piece.span, proxy_tolerance, cuts)
        hulls.append(leaf_hulls)
    proxy_unsafe = _proxy_unsafe(pieces, hulls)
    for chain_pieces in by_chain.values():
        chain_pieces.sort(key=lambda piece: piece.start)
    rings: list[CurveRing] = []
    proxy_rings: list[LineString] = []
    for ring_uses in uses:
        spans: list[BezierSpan] = []
        points: list[Point2] = []
        for chain_id, forward in ring_uses:
            chain_pieces = by_chain[chain_id]
            for piece in chain_pieces if forward else reversed(chain_pieces):
                span = piece.span
                vertices = flattened[piece]
                if not forward:
                    span = BezierSpan(
                        cast("Controls", tuple(reversed(span.control_points))),
                        span.is_line,
                        span.certified_error_mm,
                    )
                    vertices = tuple(reversed(vertices))
                spans.append(span)
                points.extend(vertices[:-1])
        points.append(points[0])
        rings.append(CurveRing(tuple(spans)))
        proxy_rings.append(LineString(points))
    polygons = [
        CurvePolygon(
            rings[indices.start], tuple(rings[index] for index in list(indices)[1:])
        )
        for indices in ranges
    ]
    proxy_polygons = [
        Polygon(
            proxy_rings[indices.start].coords,
            [proxy_rings[index].coords for index in list(indices)[1:]],
        )
        for indices in ranges
    ]
    split = 1 + len(source.orange)
    curves = CurveContours(
        polygons[0],
        tuple(polygons[1:split]),
        tuple(polygons[split:]),
        source.mm_per_pixel,
        source.source_bounds_px,
    )
    # Preserve the original silhouette object as well as its exact geometry.
    proxy = PanelContours(
        source.silhouette,
        proxy_polygons[1:split],
        proxy_polygons[split:],
        source.mm_per_pixel,
        source.source_bounds_px,
        sum(
            len(p.exterior.coords) - 1 + sum(len(r.coords) - 1 for r in p.interiors)
            for p in [source.silhouette, *proxy_polygons[1:]]
        ),
    )
    return curves, proxy, proxy_rings, proxy_unsafe


def _comparison(source: PanelContours, curves: CurveContours, tolerance: float) -> str:
    _, _, width, height = source.silhouette.bounds
    margin = max(width, height) / 20
    stroke = max(width, height) / 500
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-margin} {-margin} {width + 2 * margin} {height + 2 * margin}">',
        f"<desc>Original blue; selected cubic curves orange; tolerance {tolerance:g} mm</desc>",
        f'<g transform="translate(0,{height}) scale(1,-1)" fill="none" stroke-width="{stroke:.12g}">',
    ]
    for polygon in (*source.orange, *source.white):
        for ring in (polygon.exterior, *polygon.interiors):
            content.append(
                '<path stroke="#1166cc" d="M '
                + " L ".join(f"{x:.12g},{y:.12g}" for x, y in ring.coords)
                + ' Z"/>'
            )
    for polygon in (*curves.orange, *curves.white):
        for ring in (polygon.exterior, *polygon.interiors):
            a = ring.spans[0].control_points[0]
            commands = [f"M {a[0]:.12g},{a[1]:.12g}"]
            for span in ring.spans:
                _, b, c, d = span.control_points
                commands.append(
                    f"L {d[0]:.12g},{d[1]:.12g}"
                    if span.is_line
                    else f"C {b[0]:.12g},{b[1]:.12g} {c[0]:.12g},{c[1]:.12g} {d[0]:.12g},{d[1]:.12g}"
                )
            content.append('<path stroke="#e07700" d="' + " ".join(commands) + ' Z"/>')
    return "".join([*content, "</g></svg>"])


def _validated_mode(value: object) -> Literal["curves", "lines"]:
    if value == "curves":
        return "curves"
    if value == "lines":
        return "lines"
    raise ValueError("contour mode must be 'curves' or 'lines'")


def fit_contours(
    source: PanelContours,
    tolerance_mm: float,
    *,
    palette: Palette | None = None,
    mode: Literal["curves", "lines"] = "curves",
) -> CurveFitResult:
    """Fit each shared boundary once; refine only spans that cannot be certified.

    Rings keep their component/counter identities and endpoint contacts. The
    silhouette is exact. A polygonal proxy is separately certified for display;
    native construction must consume the returned cubic controls instead.
    Line mode uses the same certification and local splitting, without cubic
    proposals. Curve mode remains the default.
    """
    mode = _validated_mode(mode)
    allow_curves = mode == "curves"
    raw = cast("object", tolerance_mm)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("curve tolerance must be finite and nonnegative")
    try:
        tolerance = float(raw)
    except OverflowError as error:
        raise ValueError("curve tolerance must be finite and nonnegative") from error
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("curve tolerance must be finite and nonnegative")
    if palette is not None:
        _ = SeparationConfig(palette=palette)
    polygons = [source.silhouette, *source.orange, *source.white]
    if any(
        not p.is_valid or p.is_empty or p.has_z or not source.silhouette.covers(p)
        for p in polygons
    ):
        raise ValueError("curve fitting requires valid contained planar polygons")
    relationships = _relationships(polygons)
    if any(relation[0] != "F" for (a, _), relation in relationships.items() if a != 0):
        raise ValueError("source color regions must have disjoint interiors")
    rings: list[LineString] = []
    ranges: list[range] = []
    for polygon in polygons:
        start = len(rings)
        rings.extend(
            LineString(ring.coords) for ring in (polygon.exterior, *polygon.interiors)
        )
        ranges.append(range(start, len(rings)))
    source_ring_relationships = _relationships(rings)
    chains, uses = _graph(rings)
    pieces = [
        piece
        for index, chain in enumerate(chains)
        for piece in _fit_chain(index, chain, tolerance, allow_curves=allow_curves)
    ]
    local_refinements = 0
    maximum_pieces = sum(len(chain.points) - 1 for chain in chains)

    def choose(pair: Sequence[int]) -> int:
        candidates = [
            index
            for index in pair
            if pieces[index].stop - pieces[index].start > 1
            and not chains[pieces[index].chain].fixed
        ]
        if not candidates:
            raise ValueError(
                "curve topology cannot be certified without changing source features"
            )
        return max(candidates, key=lambda i: (pieces[i].stop - pieces[i].start, -i))

    while True:
        unsafe, topology_cuts = _unsafe_pair(pieces)
        proxy_tolerance = (
            tolerance / 64 if any(not p.span.is_line for p in pieces) else 0.0
        )
        refinements: set[int] = set()
        if unsafe:
            refinements.update(choose(pair) for pair in unsafe)
        else:
            curves, proxy, proxy_rings, proxy_unsafe = _assemble(
                pieces, uses, ranges, source, proxy_tolerance, topology_cuts
            )
            selected_polygons = [proxy.silhouette, *proxy.orange, *proxy.white]
            invalid = [
                i
                for i, p in enumerate(selected_polygons)
                if not p.is_valid
                or p.area <= 0
                or len(p.interiors) != len(polygons[i].interiors)
                or not source.silhouette.covers(p)
            ]
            selected_ring_relationships = _relationships(proxy_rings)
            selected_polygon_relationships = _relationships(selected_polygons)
            changed_rings = {
                index
                for pair in source_ring_relationships.keys()
                | selected_ring_relationships.keys()
                if source_ring_relationships.get(pair)
                != selected_ring_relationships.get(pair)
                for index in pair
            }
            changed_polygons = {
                index
                for pair in relationships.keys() | selected_polygon_relationships.keys()
                if relationships.get(pair) != selected_polygon_relationships.get(pair)
                for index in pair
            }
            affected = {chain for index in changed_rings for chain, _ in uses[index]}
            affected.update(
                chain
                for index in set(invalid) | changed_polygons
                for ring_index in ranges[index]
                for chain, _ in uses[ring_index]
            )
            refinements.update(choose(pair) for pair in proxy_unsafe)
            if not affected and not refinements:
                break
            for chain in affected:
                refinements.add(
                    choose(
                        [
                            index
                            for index, piece in enumerate(pieces)
                            if piece.chain == chain
                        ]
                    )
                )
        for index in sorted(refinements, reverse=True):
            piece = pieces[index]
            middle = (piece.start + piece.stop) // 2
            replacements = [
                *_fit_chain(
                    piece.chain,
                    chains[piece.chain],
                    tolerance,
                    piece.start,
                    middle,
                    allow_curves=allow_curves,
                ),
                *_fit_chain(
                    piece.chain,
                    chains[piece.chain],
                    tolerance,
                    middle,
                    piece.stop,
                    allow_curves=allow_curves,
                ),
            ]
            pieces[index : index + 1] = replacements
        local_refinements += len(refinements)
        if len(pieces) > maximum_pieces:
            raise ValueError("local curve refinement exceeded the source edge budget")
    curved_count = sum(not piece.span.is_line for piece in pieces)
    diagnostics: dict[str, object] = {
        "method": "shared-chain cubic Bezier fitting"
        if allow_curves
        else "shared-chain line fitting",
        "mode": mode,
        "tolerance_mm": tolerance,
        "continuous_boundary_error_upper_bound_mm": tolerance if tolerance > 0 else 0.0,
        "error_certificate": "exact rational Bernstein bounds under a common continuous parameterization",
        "topology_certificate": "exact rational control-hull separation certifies the native-to-proxy homotopy and pinned source graph contacts",
        "shared_chains": len(chains),
        "source_unique_edges": maximum_pieces,
        "selected_unique_spans": len(pieces),
        "selected_unique_curved_spans": curved_count,
        "native_edges": curves.edge_count,
        "local_topology_refinements": local_refinements,
        "whole_ring_rollbacks": 0,
        "proxy_deviation_mm": proxy_tolerance if curved_count else 0.0,
        "silhouette": "exact source lines",
    }
    return CurveFitResult(
        curves,
        proxy,
        diagnostics,
        _comparison(source, curves, tolerance),
        proxy_tolerance if curved_count else 0.0,
    )
