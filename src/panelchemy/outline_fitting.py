"""Physical-scale corners and jointly fitted smooth 2D outlines.

This API produces reviewable outlines before native CAD construction. Refining a
fit adds smooth spline knots; it never restores individual raster edges.
"""

# These two fitting policies share the core's exact internal certification.
# pyright: reportPrivateUsage=false

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import hypot, isfinite, pi
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly
from scipy.sparse import csr_array
from shapely.geometry import LineString

from panelchemy import curves as c
from panelchemy.contours import PanelContours
from panelchemy.separation import Palette, SeparationConfig
from panelchemy.spline_solver import solve_coefficients

type FloatArray = NDArray[np.float64]
type FloatSparse = csr_array[np.float64, tuple[int, int]]


def _scalar(values: FloatArray, index: int | tuple[int, int]) -> np.float64:
    return cast("np.float64", values[index])


def _row(values: FloatArray, index: int) -> FloatArray:
    return cast("FloatArray", values[index])


@dataclass
class _Fit:
    chain: c._Chain
    parameters: FloatArray
    corners: list[float]
    knots: list[float]
    pieces: list[c._Piece]
    intervals: list[tuple[float, float]]
    fairing_mm: float = 0.25


class OutlineFitError(ValueError):
    """An unaccepted 2D candidate, retained explicitly for visual diagnosis."""

    def __init__(
        self,
        message: str,
        fit: _Fit,
        piece: int,
        fits: Sequence[_Fit],
        uses: Sequence[Sequence[tuple[int, bool]]],
    ) -> None:
        super().__init__(message)
        self.source_points: tuple[c.Point2, ...] = fit.chain.points
        self.interval: tuple[float, float] = fit.intervals[piece]
        self.failed_span: c.BezierSpan = fit.pieces[piece].span
        self.candidate_rings: list[c.CurveRing] = []
        for ring_uses in uses:
            spans: list[c.BezierSpan] = []
            for chain, forward in ring_uses:
                pieces = fits[chain].pieces
                for p in pieces if forward else reversed(pieces):
                    span = p.span
                    if not forward:
                        span = c.BezierSpan(
                            cast("c.Controls", tuple(reversed(span.control_points))),
                            span.is_line,
                            span.certified_error_mm,
                        )
                    spans.append(span)
            self.candidate_rings.append(c.CurveRing(tuple(spans)))


def _sample(points: FloatArray, distances: FloatArray, at: FloatArray) -> FloatArray:
    return np.column_stack(
        [np.interp(at, distances, points[:, axis]) for axis in (0, 1)]
    )


def _corners(points: FloatArray, distances: FloatArray, scale: float) -> list[int]:
    """Keep concentrated turns that persist at two physical neighborhood sizes."""
    closed = bool(np.array_equal(_row(points, 0), _row(points, -1)))
    length = float(_scalar(distances, -1))
    window = min(scale, length / 4)
    positions = distances[:-1] if closed else distances

    def at(values: FloatArray) -> FloatArray:
        selected = values % length if closed else np.clip(values, 0, length)
        return _sample(points, distances, selected)

    def turns(radius: float) -> FloatArray:
        middle = at(positions)
        left, right = middle - at(positions - radius), at(positions + radius) - middle
        return np.abs(
            np.arctan2(
                left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
                np.sum(left * right, axis=1),
            )
        )

    broad, narrow = turns(window), turns(window / 2)
    candidates = [
        i
        for i, angle in enumerate(cast("Iterable[np.float64]", broad))
        if angle >= pi / 5
        and _scalar(narrow, i) >= angle * 0.76
        and (closed or window <= _scalar(positions, i) <= length - window)
    ]
    selected: list[int] = []
    for index in sorted(
        candidates, key=lambda i: (-_scalar(narrow, i), -_scalar(broad, i), i)
    ):
        if all(
            min(
                abs(_scalar(positions, index) - _scalar(positions, other)),
                length - abs(_scalar(positions, index) - _scalar(positions, other))
                if closed
                else length,
            )
            >= window / 2
            for other in selected
        ):
            selected.append(index)
    return sorted(selected)


def _spline(fit: _Fit) -> BSpline[np.float64]:
    """Solve all control points together, with actual graph junctions pinned."""
    points = np.asarray(fit.chain.points)
    origin = np.mean(points, axis=0)
    length_mm = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    points = points - origin
    parameters = fit.parameters
    knots = np.asarray(
        [0.0] * 4
        + sorted(
            value
            for value in fit.knots
            for _ in range(3 if value in fit.corners else 1)
        )
        + [1.0] * 4
    )
    count = len(knots) - 4
    closed = bool(np.array_equal(_row(points, 0), _row(points, -1)))
    smooth_seam = closed and fit.chain.free_loop and 0.0 not in fit.corners
    fixed: dict[int, FloatArray] = {}
    if not fit.chain.free_loop:
        fixed[0], fixed[count - 1] = _row(points, 0), _row(points, -1)
    dependent: set[int] = {count - 1} if closed else set()
    if smooth_seam:
        dependent.add(1)
    free = [i for i in range(count) if i not in fixed and i not in dependent]
    offsets = np.zeros((count, 2))
    rows: list[dict[int, float]] = [{} for _ in range(count)]
    for i, value in fixed.items():
        offsets[i] = value
    for j, i in enumerate(free):
        rows[i][j] = 1
    if closed:
        rows[-1], offsets[-1] = rows[0].copy(), offsets[0]
    if smooth_seam:
        ratio = (_scalar(knots, 4) - _scalar(knots, 3)) / (
            _scalar(knots, count) - _scalar(knots, count - 1)
        )
        rows[1] = {
            column: (1 + ratio) * rows[0].get(column, 0)
            - ratio * rows[-2].get(column, 0)
            for column in rows[0].keys() | rows[-2].keys()
        }
        offsets[1] = (1 + ratio) * offsets[0] - ratio * offsets[-2]
    entries = [
        (i, column, value)
        for i, row in enumerate(rows)
        for column, value in row.items()
    ]
    transform: FloatSparse = csr_array(
        (
            [v for _, _, v in entries],
            ([i for i, _, _ in entries], [j for _, j, _ in entries]),
        ),
        shape=(count, len(free)),
    )
    # Include line interiors, not just raster vertices: a long straight source
    # edge must constrain the entire stretch even when it has only two vertices.
    sample_count = min(20_000, max(32, 2 * len(points)))
    interval_samples = [np.linspace(a, b, 6) for a, b in pairwise(knots) if b > a]
    sample_times = np.unique(
        np.concatenate([parameters, np.linspace(0, 1, sample_count), *interval_samples])
    )
    points = _sample(points, parameters, sample_times)
    parameters = sample_times
    lengths = np.diff(parameters)
    weights = np.sqrt(
        np.concatenate([lengths[:1], lengths[:-1] + lengths[1:], lengths[-1:]]) / 2
    )
    basis = BSpline.design_matrix(parameters, knots, 3)
    weighted = (basis @ transform).multiply(weights[:, None])
    matrix = weighted.tocsr()
    target = (points - basis @ offsets) * weights[:, None]
    coefficients = solve_coefficients(
        matrix,
        target,
        transform,
        offsets,
        knots,
        parameters,
        points,
        length_mm,
        fit.fairing_mm,
    )
    return BSpline(knots, origin + offsets + transform @ coefficients, 3)


def _pieces(
    fit: _Fit, index: int, tolerance: float
) -> tuple[list[c._Piece], list[tuple[float, float]]]:
    spline = _spline(fit)
    polynomials = [
        PPoly.from_spline((spline.t, spline.c[:, axis], 3)) for axis in (0, 1)
    ]
    spans: list[c.BezierSpan] = []
    intervals: list[tuple[float, float]] = []
    grid = 2.0**-30

    def snap(point: Sequence[float] | FloatArray) -> c.Point2:
        return (
            round(float(cast("float | np.float64", point[0])) / grid) * grid,
            round(float(cast("float | np.float64", point[1])) / grid) * grid,
        )

    knot_values = spline.t.tolist()
    # Each knot denotes one shared point, including the two parameter values
    # at a periodic seam. Canonicalize before distance/topology certification.
    junctions = {value: snap(spline(value)) for value in knot_values}
    closed = fit.chain.points[0] == fit.chain.points[-1]
    if closed:
        junctions[1.0] = junctions[0.0]
    for i, (start, stop) in enumerate(pairwise(knot_values)):
        if stop <= start:
            continue
        h = stop - start
        a, d = junctions[start], junctions[stop]
        left = tuple(float(_scalar(p.c, (2, i))) for p in polynomials)
        right = tuple(
            float(
                3 * _scalar(p.c, (0, i)) * h**2
                + 2 * _scalar(p.c, (1, i)) * h
                + _scalar(p.c, (2, i))
            )
            for p in polynomials
        )
        controls = (
            a,
            snap([a[j] + left[j] * h / 3 for j in (0, 1)]),
            snap([d[j] - right[j] * h / 3 for j in (0, 1)]),
            d,
        )
        spans.append(c.BezierSpan(controls, False, tolerance))
        intervals.append((float(start), float(stop)))
    # Represent shared tangents on a binary lattice. Both handles are exact
    # integer multiples of one direction, avoiding microscopic opposing joins
    # introduced solely by independent floating-point rounding.
    pairs = [(i - 1, i) for i in range(1, len(spans))]
    if closed and fit.chain.free_loop:
        pairs.append((len(spans) - 1, 0))
    for previous, following in pairs:
        join_parameter = intervals[following][0]
        if join_parameter in fit.corners:
            continue
        before, after = spans[previous], spans[following]
        point = before.control_points[-1]
        u = np.asarray(point) - before.control_points[2]
        v = np.asarray(after.control_points[1]) - point
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if min(lu, lv) < grid:
            raise ValueError(
                f"smooth outline chain {index} has a stationary join at {join_parameter:.12g} (handle lengths {lu:.6g}, {lv:.6g})"
            )
        direction = u / lu + v / lv
        direction /= np.linalg.norm(direction)
        lattice = np.rint(direction * 65536).astype(np.int64)
        unit_length = float(np.linalg.norm(lattice)) * grid
        legs = [max(1, round(length / unit_length)) for length in (lu, lv)]
        b = before.control_points
        a = after.control_points
        spans[previous] = c.BezierSpan(
            (
                b[0],
                b[1],
                snap(
                    np.asarray(
                        np.asarray(point) - lattice * legs[0] * grid, dtype=np.float64
                    )
                ),
                point,
            ),
            False,
            tolerance,
        )
        spans[following] = c.BezierSpan(
            (
                point,
                snap(
                    np.asarray(
                        np.asarray(point) + lattice * legs[1] * grid, dtype=np.float64
                    )
                ),
                a[2],
                a[3],
            ),
            False,
            tolerance,
        )
    if not fit.chain.free_loop:
        first = spans[0]
        spans[0] = c.BezierSpan(
            (fit.chain.points[0], *first.control_points[1:]), False, tolerance
        )
        # A one-span chain is both first and last. Read it after pinning its
        # start, otherwise an earlier snapshot restores the rounded endpoint.
        last = spans[-1]
        spans[-1] = c.BezierSpan(
            (*last.control_points[:-1], fit.chain.points[-1]), False, tolerance
        )
    # Straight stretches bounded by intentional corners remain true CAD lines.
    for i, span in enumerate(spans):
        start, stop = intervals[i]
        left_corner = start in fit.corners or (start == 0 and not closed)
        right_corner = stop in fit.corners or (stop == 1 and not closed)
        if closed and stop == 1:
            right_corner = 0.0 in fit.corners
        if left_corner and right_corner:
            line = c._line(span.control_points[0], span.control_points[-1], tolerance)
            a, b = (
                np.asarray(line.control_points[0]),
                np.asarray(line.control_points[-1]),
            )
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length and all(
                abs(
                    float(
                        _scalar(direction, 0) * (p[1] - _scalar(a, 1))
                        - _scalar(direction, 1) * (p[0] - _scalar(a, 0))
                    )
                )
                / length
                < 1e-6
                for p in span.control_points[1:3]
            ):
                spans[i] = line
    return [c._Piece(index, i, i + 1, span) for i, span in enumerate(spans)], intervals


def _certified(fit: _Fit, piece: c._Piece, tolerance: float) -> bool:
    """Continuous source-to-curve matching, including moved knot endpoints."""
    from fractions import Fraction

    start_value, stop_value = fit.intervals[piece.start]
    source_parameters = cast("list[float]", fit.parameters.tolist())
    first = bisect_right(source_parameters, start_value) - 1
    last = bisect_left(source_parameters, stop_value)
    start, stop = Fraction(start_value), Fraction(stop_value)
    times = [
        start,
        *(Fraction(value) for value in source_parameters[first + 1 : last]),
        stop,
    ]
    controls = c._controls(piece.span)
    budget = Fraction(tolerance) ** 2
    for index, (a, b) in enumerate(pairwise(times), first):
        selected = c._restrict(
            controls,
            (a - start) / (stop - start),
            (b - start) / (stop - start),
        )
        source_start = Fraction(source_parameters[index])
        source_stop = Fraction(source_parameters[index + 1])
        left = c._rational(fit.chain.points[index])
        right = c._rational(fit.chain.points[index + 1])
        # Restrict the original float-defined source segment exactly. Rounding
        # a newly inserted source point before certification would instead
        # certify a slightly different polyline, particularly at tiny budgets.
        p = c._mix(left, right, (a - source_start) / (source_stop - source_start))
        q = c._mix(left, right, (b - source_start) / (source_stop - source_start))
        source: c.RationalControls = (
            p,
            c._mix(p, q, Fraction(1, 3)),
            c._mix(p, q, Fraction(2, 3)),
            q,
        )
        difference = cast(
            "c.RationalControls",
            tuple(
                (p[0] - q[0], p[1] - q[1])
                for p, q in zip(selected, source, strict=True)
            ),
        )
        if not c._bounded(difference, budget):
            return False
    return True


def fit_outlines(
    source: PanelContours,
    tolerance_mm: float,
    *,
    corner_scale_mm: float = 1.0,
    fairing_mm: float = 0.25,
    palette: Palette | None = None,
) -> c.CurveFitResult:
    """Fit shared outlines with movable smooth joins and physical-scale corners.

    No source islands/counters are removed. If a smooth fit cannot satisfy the
    requested distance and topology bounds, report it instead of restoring a
    pixel staircase. Zero tolerance requests the existing exact representation.
    """
    for value, name, positive in (
        (cast("object", tolerance_mm), "tolerance", False),
        (cast("object", corner_scale_mm), "corner scale", True),
        (cast("object", fairing_mm), "fairing scale", False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            )
    if tolerance_mm == 0:
        return c.fit_contours(source, 0, palette=palette)
    if palette is not None:
        _ = SeparationConfig(palette=palette)
    polygons = [source.silhouette, *source.orange, *source.white]
    if any(
        not p.is_valid or p.is_empty or p.has_z or not source.silhouette.covers(p)
        for p in polygons
    ):
        raise ValueError("outline fitting requires valid contained planar polygons")
    relationships = c._relationships(polygons)
    if any(relation[0] != "F" for (a, _), relation in relationships.items() if a != 0):
        raise ValueError("source color regions must have disjoint interiors")
    rings: list[LineString] = []
    ranges: list[range] = []
    for polygon in polygons:
        start = len(rings)
        rings.extend(
            LineString(r.coords) for r in (polygon.exterior, *polygon.interiors)
        )
        ranges.append(range(start, len(rings)))
    ring_relationships = c._relationships(rings)
    chains, uses = c._graph(rings, split_loops=False)
    fits: list[_Fit] = []
    for index, chain in enumerate(chains):
        points = np.asarray(chain.points)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distances = np.concatenate([np.array([0.0]), np.cumsum(lengths)])
        parameters = distances / _scalar(distances, -1)
        corner_indices = (
            _corners(points, distances, corner_scale_mm) if not chain.fixed else []
        )
        corners = [float(_scalar(parameters, i)) for i in corner_indices]
        knots = {value for value in corners if 0 < value < 1}
        if chain.points[0] == chain.points[-1]:
            for knot in (0.25, 0.5, 0.75):
                if not any(abs(value - knot) < 1e-10 for value in knots):
                    knots.add(knot)
        fit = _Fit(chain, parameters, corners, sorted(knots), [], [], fairing_mm)
        if chain.fixed:
            fit.pieces = [
                c._Piece(index, i, i + 1, c._line(a, b))
                for i, (a, b) in enumerate(
                    zip(chain.points[:-1], chain.points[1:], strict=True)
                )
            ]
        fits.append(fit)
    refinement_count = 0
    for iteration in range(14):
        unresolved: set[tuple[int, int]] = set()
        for index, fit in enumerate(fits):
            if fit.chain.fixed:
                continue
            fit.pieces, fit.intervals = _pieces(fit, index, tolerance_mm)
            for piece in fit.pieces:
                if not c._monotone(piece.span) or not _certified(
                    fit, piece, tolerance_mm
                ):
                    unresolved.add((index, piece.start))
        pieces = [p for fit in fits for p in fit.pieces]
        unsafe, cuts = c._unsafe_pair(pieces)
        for pair in unsafe:
            for i in pair:
                if not fits[pieces[i].chain].chain.fixed:
                    unresolved.add((pieces[i].chain, pieces[i].start))
        if not unresolved:
            proxy_tolerance = tolerance_mm / 64
            curves, proxy, proxy_rings, proxy_unsafe = c._assemble(
                pieces, uses, ranges, source, proxy_tolerance, cuts
            )
            if any(
                before.control_points[-1] != after.control_points[0]
                for polygon in (curves.silhouette, *curves.orange, *curves.white)
                for ring in (polygon.exterior, *polygon.interiors)
                for before, after in pairwise((*ring.spans, ring.spans[0]))
            ):
                raise ValueError("fitted outline spans must join exactly and close")
            selected = [proxy.silhouette, *proxy.orange, *proxy.white]
            changed_rings = {
                i
                for pair in ring_relationships.keys()
                | c._relationships(proxy_rings).keys()
                if ring_relationships.get(pair)
                != c._relationships(proxy_rings).get(pair)
                for i in pair
            }
            changed_polygons = {
                i
                for pair in relationships.keys() | c._relationships(selected).keys()
                if relationships.get(pair) != c._relationships(selected).get(pair)
                for i in pair
            }
            invalid = {
                i
                for i, p in enumerate(selected)
                if not p.is_valid
                or p.area <= 0
                or len(p.interiors) != len(polygons[i].interiors)
                or not source.silhouette.covers(p)
            }
            affected = {chain for i in changed_rings for chain, _ in uses[i]}
            affected.update(
                chain
                for i in changed_polygons | invalid
                for r in ranges[i]
                for chain, _ in uses[r]
            )
            for pair in proxy_unsafe:
                affected.update(pieces[i].chain for i in pair)
            if not affected:
                diagnostics: dict[str, object] = dict(
                    method="joint cubic B-spline outline fitting",
                    mode="curves",
                    corner_scale_mm=corner_scale_mm,
                    fairing_mm=fairing_mm,
                    forward_derivative_constraints=True,
                    tolerance_mm=tolerance_mm,
                    continuous_boundary_error_upper_bound_mm=tolerance_mm,
                    proxy_deviation_mm=proxy_tolerance,
                    source_unique_edges=sum(len(f.chain.points) - 1 for f in fits),
                    selected_unique_spans=len(pieces),
                    native_edges=curves.edge_count,
                    physical_corners=sum(len(f.corners) for f in fits),
                    smooth_refinements=refinement_count,
                    pixel_fallbacks=0,
                    corner_points=[
                        cast(
                            "list[float]",
                            _row(
                                _sample(
                                    np.asarray(f.chain.points),
                                    f.parameters,
                                    np.array([t]),
                                ),
                                0,
                            ).tolist(),
                        )
                        for f in fits
                        for t in f.corners
                    ],
                    error_certificate="exact rational Bernstein bounds including moved spline endpoints",
                    topology_certificate="native-to-proxy control-hull homotopy and original palette relationships",
                )
                return c.CurveFitResult(
                    curves,
                    proxy,
                    diagnostics,
                    c._comparison(source, curves, tolerance_mm),
                    proxy_tolerance,
                )
            unresolved.update(
                (i, p.start)
                for i in affected
                if not fits[i].chain.fixed
                for p in fits[i].pieces
            )
        for index, interval in sorted(unresolved):
            fit = fits[index]
            a, b = fit.intervals[interval]
            length = float(
                sum(
                    hypot(q[0] - p[0], q[1] - p[1])
                    for p, q in zip(
                        fit.chain.points[:-1], fit.chain.points[1:], strict=True
                    )
                )
            )
            if (b - a) * length < min(
                corner_scale_mm / 32, length / 128
            ) or iteration == 13:
                raise OutlineFitError(
                    f"cannot certify smooth outline chain {index}, interval {a:.6g}..{b:.6g}; no pixel-edge fallback",
                    fit,
                    interval,
                    fits,
                    uses,
                )
            fit.knots.append((a + b) / 2)
            fit.knots.sort()
            refinement_count += 1
    raise ValueError("smooth outline fitting exceeded its refinement budget")
