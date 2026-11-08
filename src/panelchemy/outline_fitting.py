"""Physical-scale corners and jointly fitted smooth 2D outlines.

This API produces reviewable outlines before native CAD construction. Refining a
fit adds smooth spline knots; it never restores individual raster edges.
"""

# These two fitting policies share the core's exact internal certification.
# pyright: reportPrivateUsage=false

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from time import monotonic
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly
from scipy.sparse import csr_array
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient

from panelchemy import (
    boundary_tapers,
    contact_refinement,
    curvature,
    curvature_fitting,
    edge_detail,
    fitting_feedback,
    geometric_distance,
    local_windows,
    outline_simplification,
    outline_softening,
)
from panelchemy import curves as c
from panelchemy.contours import PanelContours
from panelchemy.separation import Palette, SeparationConfig
from panelchemy.spline_solver import SplineRefinementNeeded, solve_coefficients

type FloatArray = NDArray[np.float64]
type FloatSparse = csr_array[np.float64, tuple[int, int]]
_FIT_TARGET_FRACTION = 0.9


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
    source_corners: tuple[float, ...] = ()
    texture_intervals: tuple[tuple[float, float], ...] = ()
    texture_barriers: tuple[float, ...] = ()


def _detail_guidance(
    fits: Sequence[_Fit], source: PanelContours, line_width: float
) -> dict[str, object]:
    """Classify source geometry; labels choose fits, never enlarge error bounds."""
    counts: Counter[str] = Counter()
    regions: list[dict[str, object]] = []
    for index, fit in enumerate(fits):
        if fit.chain.fixed:
            continue
        points = fit.chain.points
        samples = edge_detail.classify_chain(
            points,
            pixel_pitch_mm=source.mm_per_pixel,
            line_width_mm=line_width,
            protected_indices=(
                frozenset({0, len(points) - 1})
                if not fit.chain.free_loop
                else frozenset()
            ),
        )
        counts.update(sample.kind for sample in samples)
        fine = {sample.index for sample in samples if sample.kind == "fine_texture"}
        structural = {sample.index for sample in samples if sample.kind == "structural"}
        fit.texture_barriers = tuple(
            float(_scalar(fit.parameters, i)) for i in sorted(structural)
        )
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i in range(len(points)):
            selected = (
                i + 1 < len(points)
                and (i in fine or i + 1 in fine)
                and i not in structural
                and i + 1 not in structural
            )
            if selected and start is None:
                start = i
            elif not selected and start is not None:
                runs.append((start, i))
                start = None
        fit.texture_intervals = tuple(
            (float(_scalar(fit.parameters, a)), float(_scalar(fit.parameters, b)))
            for a, b in runs
        )
        for (a, b), interval in zip(runs, fit.texture_intervals, strict=True):
            selected_samples = [sample for sample in samples if a <= sample.index <= b]
            xs, ys = zip(*points[a : b + 1], strict=True)
            regions.append(
                dict(
                    chain=index,
                    source_interval=interval,
                    first_source_vertex=a,
                    last_source_vertex=b,
                    bounds_mm=(min(xs), min(ys), max(xs), max(ys)),
                    backtracks=max(sample.backtracks for sample in selected_samples),
                    normal_reversals=max(
                        sample.normal_reversals for sample in selected_samples
                    ),
                    path_excess_mm=max(
                        sample.path_excess_mm for sample in selected_samples
                    ),
                    max_event_mm=max(
                        sample.max_event_mm for sample in selected_samples
                    ),
                )
            )
    return {
        "edge_detail": dict(
            line_width_mm=line_width,
            pixel_pitch_mm=source.mm_per_pixel,
            source_vertex_classes=dict(counts),
            fine_texture_regions=regions,
            policy="fine texture prefers a compact smooth fit; source and taper limits remain unchanged",
            grants_error_allowance=False,
        )
    }


@dataclass
class _RejectedFit:
    spline: BSpline
    fit: _Fit
    source_failure: bool = False


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


def _corners(
    points: FloatArray, distances: FloatArray, scale: float, prominence_mm: float
) -> list[int]:
    """Keep persistent turns with a physically meaningful excursion."""
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
    # Measure the turn against a physical neighborhood, rather than its two
    # immediate pixel edges. Segment distance also retains a substantial V tip
    # whose projection lies beyond the neighborhood's chord endpoints.
    middle = at(positions)
    before, after = at(positions - window), at(positions + window)
    chord = after - before
    squared = np.sum(chord * chord, axis=1)
    fraction = np.divide(
        np.sum((middle - before) * chord, axis=1),
        squared,
        out=np.zeros_like(squared),
        where=squared > 0,
    )
    closest = before + np.clip(fraction, 0, 1)[:, None] * chord
    prominence = np.linalg.norm(middle - closest, axis=1)
    # A small closed island/counter is retained by this fitting policy. Keep
    # enough structure to fit it before attempting bounded local softening;
    # a manufacturing scale larger than the whole loop must not erase its
    # corner constraints and make its otherwise valid fit impossible.
    threshold = min(prominence_mm, window / 4) if closed else prominence_mm
    candidates = [
        i
        for i, angle in enumerate(cast("Iterable[np.float64]", broad))
        if angle >= pi / 5
        and _scalar(narrow, i) >= angle * 0.76
        and _scalar(prominence, i) >= threshold
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


def _spline(fit: _Fit, fit_target_mm: float) -> BSpline[np.float64]:
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
        fit_target_mm=fit_target_mm,
    )
    return BSpline(knots, origin + offsets + transform @ coefficients, 3)


def _pieces(
    fit: _Fit, index: int, tolerance: float
) -> tuple[list[c._Piece], list[tuple[float, float]]]:
    spline = _spline(fit, _FIT_TARGET_FRACTION * tolerance)
    return _spline_pieces(fit, index, tolerance, spline)


def _spline_pieces(
    fit: _Fit, index: int, tolerance: float, spline: BSpline
) -> tuple[list[c._Piece], list[tuple[float, float]]]:
    """Canonicalize both ordinary and curvature-constrained spline proposals."""
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
        if stop <= start or start < 0 or stop > 1:
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
    """Sufficient continuous source matching at corresponding parameters."""
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
        # The intervals are consecutive: retain the exact unconsumed suffix
        # instead of restricting the original cubic twice for every segment.
        if b == stop:
            selected = controls
        else:
            selected, controls = c._split(controls, (b - a) / (stop - a))
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


def _orientation_preserved(fit: _Fit) -> bool:
    """Keep closed-chain material sides using exact signed boundary integrals."""
    from fractions import Fraction

    if fit.chain.points[0] != fit.chain.points[-1]:
        return True
    source = [c._rational(point) for point in fit.chain.points]
    source_area = sum(a[0] * b[1] - a[1] * b[0] for a, b in pairwise(source))
    curve_area = Fraction(0)
    for piece in fit.pieces:
        controls = c._controls(piece.span)
        velocity = [
            (3 * (b[0] - a[0]), 3 * (b[1] - a[1])) for a, b in pairwise(controls)
        ]
        xy = curvature._product([p[0] for p in controls], [v[1] for v in velocity])
        yx = curvature._product([p[1] for p in controls], [v[0] for v in velocity])
        # All degree-five Bernstein basis functions integrate to one sixth.
        curve_area += sum(a - b for a, b in zip(xy, yx, strict=True)) / 6
    return source_area * curve_area > 0


def _source_failures(fit: _Fit, tolerance: float) -> set[int]:
    """Certify geometric distance; old parameter failures only guide refinement."""
    if (
        not fit.pieces
        or len(fit.pieces) != len(fit.intervals)
        or fit.intervals[0][0] != 0
        or fit.intervals[-1][1] != 1
        or any(a >= b for a, b in fit.intervals)
        or any(a[1] != b[0] for a, b in pairwise(fit.intervals))
        or [p.start for p in fit.pieces] != list(range(len(fit.pieces)))
    ):
        raise ValueError("source certification requires a complete chain partition")
    if not _orientation_preserved(fit):
        return {piece.start for piece in fit.pieces}
    failed = {
        piece.start for piece in fit.pieces if not _certified(fit, piece, tolerance)
    }
    if failed and geometric_distance.within_distance(
        fit.chain.points, [piece.span for piece in fit.pieces], tolerance
    ):
        return set()
    return failed


def _soften_verified(
    fits: Sequence[_Fit],
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    source: PanelContours,
    tolerance: float,
    radius: float,
    original_curves: c.CurveContours,
    original_proxy: PanelContours,
    *,
    material_taper_runs_mm: tuple[float, float, float] | None = None,
) -> tuple[c.CurveContours, PanelContours, dict[str, object], list[_Fit]]:
    """Try local bridges; retain sharp geometry where any acceptance gate fails."""
    blocked: set[tuple[int, float]] = set()
    attempted: set[tuple[int, float]] = set()
    rejection: str | None = None
    polygons = [source.silhouette, *source.orange, *source.white]
    source_rings = [
        LineString(ring.coords)
        for polygon in polygons
        for ring in (polygon.exterior, *polygon.interiors)
    ]
    relationships, ring_relationships = (
        c._relationships(polygons),
        c._relationships(source_rings),
    )
    limits = (
        boundary_tapers.boundaries(
            uses, ranges, len(source.orange), material_taper_runs_mm
        )
        if material_taper_runs_mm is not None
        else {}
    )
    curvature_blocked: set[tuple[int, float]] = set()
    for _ in range(8):
        selected_fits: list[_Fit] = []
        pieces: list[c._Piece] = []
        dependencies: list[set[tuple[int, float]]] = []
        applied: set[tuple[int, float]] = set()
        failed: set[tuple[int, float]] = set()
        for index, fit in enumerate(fits):
            if fit.chain.fixed:
                selected_fits.append(fit)
                pieces.extend(fit.pieces)
                dependencies.extend(set() for _ in fit.pieces)
                continue
            softened = outline_softening.soften(
                [piece.span for piece in fit.pieces],
                fit.intervals,
                fit.corners,
                radius,
                excluded=frozenset(t for chain, t in blocked if chain == index),
                periodic=fit.chain.free_loop,
            )
            attempted.update((index, t) for t in softened.candidates)
            applied.update((index, t) for t in softened.applied)
            selected = [
                c._Piece(index, i, i + 1, span) for i, span in enumerate(softened.spans)
            ]
            neighbors = [(i - 1, i) for i in range(1, len(selected))]
            if fit.chain.points[0] == fit.chain.points[-1]:
                neighbors.append((len(selected) - 1, 0))
            for a, b in neighbors:
                if (
                    selected[a].span.control_points[-1]
                    != selected[b].span.control_points[0]
                ):
                    owners = softened.dependencies[a] | softened.dependencies[b]
                    if not owners:
                        raise ValueError("unmodified outlines lost an exact join")
                    failed.update((index, t) for t in owners)
            checked = replace(
                fit,
                pieces=selected,
                intervals=list(softened.intervals),
                corners=[t for t in fit.corners if t not in softened.applied],
            )
            selected_fits.append(checked)
            if _source_failures(checked, tolerance):
                if not softened.applied:
                    raise ValueError(
                        "unmodified outline failed its existing certificate"
                    )
                failed.update((index, t) for t in softened.applied)
            for piece, owners in zip(selected, softened.dependencies, strict=True):
                tags = {(index, t) for t in owners}
                pieces.append(piece)
                dependencies.append(tags)
                if not c._monotone(piece.span):
                    if not tags:
                        raise ValueError(
                            "unmodified outline failed its existing certificate"
                        )
                    failed.update(tags)
                limit = limits.get(index)
                if (
                    tags
                    and limit is not None
                    and limit.offset_mm > 0
                    and not curvature.within_curvature(
                        piece.span, limit.offset_mm, limit.inward_sign
                    )
                ):
                    # Restore the detected corner and its original incident
                    # pieces, rather than passing a tiny rounded connector on
                    # to the smooth curvature optimizer. Dependencies include
                    # both corners when their trims affect the same piece.
                    failed.update(tags)
                    curvature_blocked.update(tags)
        if not applied:
            break
        if failed:
            blocked.update(failed)
            continue
        try:
            unsafe, cuts = c._unsafe_pair(pieces)
        except ValueError as error:
            if str(error) != "curve topology exceeds 1,000,000 candidate pairs":
                raise
            rejection = str(error)
            break
        for a, b in unsafe:
            failed.update(dependencies[a] | dependencies[b])
        if unsafe and not failed:
            raise ValueError(
                "unmodified outlines failed their existing separation proof"
            )
        if failed:
            blocked.update(failed)
            continue
        try:
            curves, proxy, rings, proxy_unsafe = c._assemble(
                pieces, uses, ranges, source, tolerance / 64, cuts
            )
        except ValueError as error:
            if str(error) not in {
                "curve proxy cannot meet its continuous error budget",
                "rounded curve proxy cannot preserve monotone embedding",
                "curve/proxy topology exceeds 1,000,000 candidate pairs",
            }:
                raise
            rejection = str(error)
            break
        for a, b in proxy_unsafe:
            failed.update(dependencies[a] | dependencies[b])
        selected_polygons = [proxy.silhouette, *proxy.orange, *proxy.white]
        current_relationships = c._relationships(selected_polygons)
        current_ring_relationships = c._relationships(rings)
        changed_rings = {
            i
            for pair in ring_relationships.keys() | current_ring_relationships.keys()
            if ring_relationships.get(pair) != current_ring_relationships.get(pair)
            for i in pair
        }
        changed_polygons = {
            i
            for pair in relationships.keys() | current_relationships.keys()
            if relationships.get(pair) != current_relationships.get(pair)
            for i in pair
        }
        invalid = {
            i
            for i, polygon in enumerate(selected_polygons)
            if not polygon.is_valid
            or polygon.area <= 0
            or len(polygon.interiors) != len(polygons[i].interiors)
            or not source.silhouette.covers(polygon)
        }
        affected = {chain for ring in changed_rings for chain, _ in uses[ring]}
        affected.update(
            chain
            for i in changed_polygons | invalid
            for ring in ranges[i]
            for chain, _ in uses[ring]
        )
        failed.update(tag for tag in applied if tag[0] in affected)
        if failed:
            blocked.update(failed)
            continue
        if proxy_unsafe or affected:
            raise ValueError("unmodified outlines failed their existing topology proof")
        return (
            curves,
            proxy,
            dict(
                corner_softening_mm=radius,
                softened_corners=len(applied),
                softening_candidates=len(attempted),
                softening_blocked=len(attempted - applied),
                curvature_blocked_corners=len(curvature_blocked),
                selected_unique_spans=len(pieces),
                softening_support_certificate="removed tails and cubic bridge control hulls contained in corner neighborhood",
            ),
            selected_fits,
        )
    return (
        original_curves,
        original_proxy,
        dict(
            corner_softening_mm=radius,
            softened_corners=0,
            softening_candidates=len(attempted),
            softening_blocked=len(attempted),
            curvature_blocked_corners=len(curvature_blocked),
            softening_rejection=rejection
            or "local edits could not pass the safety gates",
        ),
        list(fits),
    )


@dataclass
class _VerifiedRefits:
    curves: c.CurveContours
    proxy: PanelContours
    fits: list[_Fit]
    accepted: set[int]
    blocked: set[int]
    rejection: str | None


def _verify_refits(
    fits: Sequence[_Fit],
    candidates: Sequence[_Fit],
    changed: set[int],
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    source: PanelContours,
    tolerance: float,
    original_curves: c.CurveContours,
    original_proxy: PanelContours,
) -> _VerifiedRefits:
    """Recertify edited chains together; selectively restore unsafe proposals."""
    candidates = list(candidates)
    changed = changed.copy()
    blocked: set[int] = set()
    polygons = [source.silhouette, *source.orange, *source.white]
    relationships = c._relationships(polygons)
    ring_relationships = c._relationships(
        [
            LineString(ring.coords)
            for polygon in polygons
            for ring in (polygon.exterior, *polygon.interiors)
        ]
    )
    curves, proxy = original_curves, original_proxy
    rejection: str | None = None
    for _ in range(8):
        if not changed:
            break
        pieces = [piece for fit in candidates for piece in fit.pieces]
        try:
            unsafe, cuts = c._unsafe_pair(pieces)
        except ValueError as error:
            if str(error) != "curve topology exceeds 1,000,000 candidate pairs":
                raise
            rejection = str(error)
            blocked.update(changed)
            changed.clear()
            break
        affected = {pieces[i].chain for pair in unsafe for i in pair}
        if not affected:
            try:
                selected_curves, selected_proxy, rings, proxy_unsafe = c._assemble(
                    pieces, uses, ranges, source, tolerance / 64, cuts
                )
            except ValueError as error:
                if str(error) not in {
                    "curve proxy cannot meet its continuous error budget",
                    "rounded curve proxy cannot preserve monotone embedding",
                    "curve/proxy topology exceeds 1,000,000 candidate pairs",
                }:
                    raise
                rejection = str(error)
                affected = set(changed)
            else:
                selected = [
                    selected_proxy.silhouette,
                    *selected_proxy.orange,
                    *selected_proxy.white,
                ]
                current = c._relationships(selected)
                current_rings = c._relationships(rings)
                changed_rings = {
                    i
                    for pair in ring_relationships.keys() | current_rings.keys()
                    if ring_relationships.get(pair) != current_rings.get(pair)
                    for i in pair
                }
                changed_polygons = {
                    i
                    for pair in relationships.keys() | current.keys()
                    if relationships.get(pair) != current.get(pair)
                    for i in pair
                }
                invalid = {
                    i
                    for i, polygon in enumerate(selected)
                    if not polygon.is_valid
                    or polygon.area <= 0
                    or len(polygon.interiors) != len(polygons[i].interiors)
                    or not source.silhouette.covers(polygon)
                }
                affected.update(
                    chain for ring in changed_rings for chain, _ in uses[ring]
                )
                affected.update(
                    chain
                    for i in changed_polygons | invalid
                    for ring in ranges[i]
                    for chain, _ in uses[ring]
                )
                affected.update(pieces[i].chain for pair in proxy_unsafe for i in pair)
                if not affected:
                    curves, proxy = selected_curves, selected_proxy
                    break
        rollback = affected & changed
        if not rollback:
            raise ValueError("unmodified outlines failed their existing topology proof")
        for index in rollback:
            candidates[index] = fits[index]
        changed.difference_update(rollback)
        blocked.update(rollback)
    else:
        rejection = "refinement exceeded its topology rollback budget"
        blocked.update(changed)
        changed.clear()
    if curves is original_curves:
        changed.clear()
    return _VerifiedRefits(
        curves,
        proxy,
        [candidates[i] if i in changed else fit for i, fit in enumerate(fits)],
        changed,
        blocked,
        rejection,
    )


def _simplify_verified(
    fits: Sequence[_Fit],
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    source: PanelContours,
    tolerance: float,
    edit_budget: float,
    original_curves: c.CurveContours,
    original_proxy: PanelContours,
    *,
    material_taper_runs_mm: tuple[float, float, float] | None = None,
) -> tuple[c.CurveContours, PanelContours, dict[str, object], list[_Fit]]:
    """Reduce actual polynomials, certifying original artwork and palette topology."""
    candidates = list(fits)
    changed: set[int] = set()
    limits = (
        boundary_tapers.boundaries(
            uses, ranges, len(source.orange), material_taper_runs_mm
        )
        if material_taper_runs_mm is not None
        else {}
    )
    for index, fit in enumerate(fits):
        if fit.chain.fixed or len(fit.pieces) < 2:
            continue

        def accept(
            span: c.BezierSpan,
            start: float,
            stop: float,
            fit: _Fit = fit,
            index: int = index,
        ) -> bool:
            if any(start < corner < stop for corner in fit.corners):
                return False
            limit = limits.get(index)
            if (
                limit is not None
                and limit.offset_mm > 0
                and not curvature.within_curvature(
                    span, limit.offset_mm, limit.inward_sign
                )
            ):
                return False
            checked = replace(fit, intervals=[(start, stop)])
            if _certified(checked, c._Piece(index, 0, 1, span), tolerance):
                return True
            # A merge may slide correspondence across old knot positions.
            # Include unchanged spans when proving reverse source coverage.
            selected = [
                piece.span
                for piece, (a, b) in zip(fit.pieces, fit.intervals, strict=True)
                if b <= start or a >= stop
            ]
            return geometric_distance.within_distance(
                fit.chain.points, [*selected, span], tolerance
            )

        reduced = outline_simplification.simplify(
            [piece.span for piece in fit.pieces],
            fit.intervals,
            edit_budget,
            accept=accept,
        )
        if len(reduced.spans) < len(fit.pieces):
            candidate = replace(
                fit,
                pieces=[
                    c._Piece(index, i, i + 1, span)
                    for i, span in enumerate(reduced.spans)
                ],
                intervals=list(reduced.intervals),
            )
            # Independently approved merges can have overlapping coverage;
            # recertify their combined replacement against the complete source.
            if not _source_failures(candidate, tolerance):
                candidates[index] = candidate
                changed.add(index)
    before = sum(len(fit.pieces) for fit in fits)
    verified = _verify_refits(
        fits,
        candidates,
        changed,
        uses,
        ranges,
        source,
        tolerance,
        original_curves,
        original_proxy,
    )
    after = before - sum(
        len(fits[i].pieces) - len(candidates[i].pieces) for i in verified.accepted
    )
    return (
        verified.curves,
        verified.proxy,
        dict(
            span_simplification_mm=edit_budget,
            spans_before_simplification=before,
            selected_unique_spans=after,
            removed_polynomial_spans=before - after,
            simplified_chains=len(verified.accepted),
            simplification_blocked_chains=len(verified.blocked),
            simplification_rejection=verified.rejection,
            simplification_certificate="exact continuous displacement from accepted curves and original raster; complete palette topology recertified",
        ),
        verified.fits,
    )


def _curvature_feedback(
    failed: _RejectedFit,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
) -> fitting_feedback.Refinement:
    checked = failed.fit
    if failed.source_failure:
        return fitting_feedback.distance_refinement(
            checked.chain.points,
            checked.parameters,
            [p.span for p in checked.pieces],
            checked.intervals,
            tolerance,
            fraction=curvature_fitting._SOURCE_FRACTION,
        )
    locations: set[float] = set()
    # Include peaks near the search target, so a correction cannot merely
    # move the violation onto an adjoining piece.
    for piece, (a, b) in zip(checked.pieces, checked.intervals, strict=True):
        locations.update(
            a + t * (b - a)
            for t in fitting_feedback.curvature_locations(
                piece.span,
                limit.offset_mm,
                limit.inward_sign,
                fraction=curvature_fitting._SAMPLED_FRACTION,
            )
        )
    return fitting_feedback.Refinement(tuple(sorted(locations)))


def _forward_progress(span: c.BezierSpan) -> bool:
    """Certify positive interior progress, not just a forward control polygon."""
    controls = c._controls(span)
    direction = (controls[-1][0] - controls[0][0], controls[-1][1] - controls[0][1])
    a, b, d = (
        (b[0] - a[0]) * direction[0] + (b[1] - a[1]) * direction[1]
        for a, b in pairwise(controls)
    )
    # These are quadratic Bernstein coefficients of projected velocity. A
    # negative middle coefficient still has positive minimum iff a*d > b*b.
    # Strictness excludes a newly admitted interior stationary point.
    return any(direction) and a >= 0 and d >= 0 and (b >= 0 or a * d > b * b)


def _run_corners(fit: _Fit) -> set[float]:
    # Unsplit graph chains end at branch/ownership terminals. These positions
    # stay pinned, but need not retain a smooth inside ray across that junction.
    return set(fit.corners) | (
        {0.0, 1.0} if not fit.chain.free_loop and not fit.chain.fixed else set()
    )


def _local_window_refit(
    fit: _Fit,
    index: int,
    guide: local_windows.Guide,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    seconds: float,
    *,
    compact: bool = False,
    apex: float | None = None,
    free_corner_rays: bool = False,
) -> tuple[_Fit | None, dict[str, object]]:
    if apex is not None and (compact or not isfinite(apex) or not 0 < apex < 1):
        raise ValueError(
            "a local apex must be interior and separate from compact fitting"
        )
    corners = _run_corners(fit)
    if free_corner_rays and not (
        guide.start_t in corners
        and (
            guide.end_t_original in corners
            or (
                guide.end_t_original == 1
                and fit.chain.points[0] == fit.chain.points[-1]
                and 0 in corners
            )
        )
    ):
        return None, dict(status="endpoint-not-corner")
    start = (
        float(_scalar(guide.endpoint_positions, (0, 0))),
        float(_scalar(guide.endpoint_positions, (0, 1))),
    )
    end = (
        float(_scalar(guide.endpoint_positions, (1, 0))),
        float(_scalar(guide.endpoint_positions, (1, 1))),
    )
    conversion = _Fit(
        c._Chain((start, end), fixed=False),
        np.array([0.0, 1.0]),
        [],
        [],
        [],
        [],
    )
    accepted: list[_Fit] = []
    rejected: list[_RejectedFit] = []
    corner_origins: dict[float, float] = {}

    def accept(spline: BSpline) -> bool:
        rejected.clear()
        selected = spline
        if guide.seam is not None and guide.seam not in spline.t:
            selected = spline.insert_knot(guide.seam)
        try:
            pieces, intervals = _spline_pieces(conversion, index, tolerance, selected)
        except ValueError as error:
            if not str(error).startswith("smooth outline chain"):
                raise
            return False
        # Preserve smooth splice rays. Established corner endpoints may change
        # their inside rays while their exact positions remain pinned.
        for position, anchor, inherited, pole in (
            (0, start, guide.inherited_handles[0], 1),
            (-1, end, guide.inherited_handles[1], 2),
        ):
            if free_corner_rays:
                continue
            controls = pieces[position].span.control_points
            length = hypot(controls[pole][0] - anchor[0], controls[pole][1] - anchor[1])
            handle = outline_simplification._handle(anchor, inherited, length)
            if handle is None:
                return False
            controls = (
                (anchor, handle, controls[2], controls[3])
                if position == 0
                else (controls[0], controls[1], handle, anchor)
            )
            pieces[position] = replace(
                pieces[position], span=c.BezierSpan(controls, False, tolerance)
            )
        local = replace(conversion, pieces=pieces, intervals=intervals)
        if not all(
            _forward_progress(p.span) if free_corner_rays else c._monotone(p.span)
            for p in pieces
        ):
            return False
        if not all(
            curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
            for p in pieces
        ):
            rejected.append(_RejectedFit(spline, local))
            return False
        spans, mapped = guide.splice(
            [p.span for p in fit.pieces],
            fit.intervals,
            [p.span for p in pieces],
            intervals,
        )
        references: dict[float, float] = {}
        for q, original in corner_origins.items():
            value = guide.start_t + q * (guide.end_t - guide.start_t)
            if q == guide.seam:
                value = 0.0
            elif guide.seam is not None and value > 1:
                value -= 1
            references[value] = original
        for i, (a, b) in enumerate(mapped):
            replacement = references.get(a, a), references.get(b, b)
            if replacement != mapped[i]:
                mapped[i] = replacement
        if any(a >= b for a, b in mapped):
            return False
        checked = replace(
            fit,
            pieces=[c._Piece(index, i, i + 1, s) for i, s in enumerate(spans)],
            intervals=mapped,
            corners=sorted(
                {
                    t
                    for t in fit.corners
                    if (q := guide.to_local(t)) is None or not 0 < q < 1
                }
                | {corner_origins[q] for q in conversion.corners}
            ),
        )
        if _source_failures(checked, tolerance):
            rejected.append(_RejectedFit(spline, checked, source_failure=True))
            return False
        try:
            unsafe, _ = c._unsafe_pair(checked.pieces)
        except ValueError as error:
            if str(error) != "curve topology exceeds 1,000,000 candidate pairs":
                raise
            return False
        if unsafe:
            return False
        accepted.append(checked)
        return True

    def refine(spline: BSpline) -> fitting_feedback.Refinement:
        if not rejected or rejected[-1].spline is not spline:
            return fitting_feedback.Refinement()
        failed = rejected[-1]
        hints = _curvature_feedback(failed, limit, tolerance)
        if not failed.source_failure:
            return hints  # Converted window intervals already use local0..1.
        return fitting_feedback.Refinement(
            tuple(
                t
                for p in hints.curve_parameters
                if (t := guide.to_local(p)) is not None
            ),
            tuple(
                t
                for p in hints.source_parameters
                if (t := guide.to_local(p)) is not None
            ),
        )

    indices = guide.window.indices(len(fit.pieces))
    breaks = np.asarray(
        [
            0.0,
            *[guide.to_local(fit.intervals[i][1]) for i in indices[:-1]],
            1.0,
        ],
        dtype=np.float64,
    )
    if 2 * len(indices) <= curvature_fitting._MAX_SPANS:
        breaks = np.unique(np.concatenate((breaks, (breaks[:-1] + breaks[1:]) / 2)))
    # Existing intervals locate the small bends even beside long stretches.
    # Reserve half the window's budget for the original uniform search.
    deadline = monotonic() + max(0.0, seconds)
    searches: list[dict[str, object]] = []
    bad = [
        i
        for i in indices
        if not curvature.within_curvature(
            fit.pieces[i].span, limit.offset_mm, limit.inward_sign
        )
    ]
    choices = _curvature_corner_choices(fit, limit, bad)
    plans: list[tuple[FloatArray | None, dict[float, float]]] = []
    for choice in choices:
        origins: dict[float, float] = {}
        for t in [*fit.corners, *choice]:
            if (q := guide.to_local(t)) is not None and 0 < q < 1:
                _ = origins.setdefault(q, t)
        corners = sorted(origins)
        selected_breaks = np.unique(np.concatenate((breaks, corners)))
        if len(selected_breaks) > curvature_fitting._MAX_SPANS + 1:
            if not corners:
                continue
            selected_breaks = np.asarray([0.0, *corners, 1.0])
        if len(selected_breaks) <= curvature_fitting._MAX_SPANS + 1 and (
            not plans or plans[-1][1] != origins
        ):
            plans.append((selected_breaks, origins))
    # Offer the existing smooth family before soft-budget corner proofs can
    # consume its reserved half. All acceptance gates remain unchanged.
    plans.insert(0, (None, {}))
    if compact:
        if any((q := guide.to_local(t)) is not None and 0 < q < 1 for t in fit.corners):
            return None, dict(status="protected-corner")
        plans = [(np.array([0.0, 0.5, 1.0]), {})]
    elif apex is not None:
        if any((q := guide.to_local(t)) is not None and 0 < q < 1 for t in fit.corners):
            return None, dict(status="protected-corner")
        reference = guide.start_t + apex * (guide.end_t - guide.start_t)
        if apex == guide.seam:
            reference = 0.0
        elif reference > 1:
            reference -= 1
        plans = [(np.array([0.0, apex, 1.0]), {apex: reference})]
    for position, (selected_breaks, corner_origins) in enumerate(plans):
        remaining = max(0.0, deadline - monotonic())
        if remaining <= 0:
            break
        accepted.clear()
        rejected.clear()
        corners = sorted(corner_origins)
        conversion.corners = corners
        proposal, diagnostics = curvature_fitting.refit(
            guide.source_points,
            guide.parameters,
            limit.offset_mm,
            limit.inward_sign,
            tolerance,
            periodic=False,
            accept=accept,
            refine=refine,
            endpoint_positions=guide.endpoint_positions,
            tangent_directions=None if free_corner_rays else guide.tangent_directions,
            knot_breaks=selected_breaks,
            corner_parameters=np.asarray(corners) if corners else None,
            c1_parameters=np.array([0.5]) if compact else None,
            source_search_fraction=1.0 if compact else None,
            time_budget_seconds=(
                remaining / 2
                if position == 0 and selected_breaks is None and len(plans) > 1
                else remaining / (len(plans) - position)
            ),
        )
        basis = (
            "compact-c1"
            if compact
            else "movable-apex"
            if apex is not None
            else "corner-intervals"
            if corners
            else "approved-intervals"
            if selected_breaks is not None
            else "uniform"
        )
        searches.append(dict(basis=basis, search=diagnostics))
        if proposal is not None and accepted:
            return accepted[-1], dict(diagnostics, basis=basis, searches=searches)
    return None, dict(
        status="time-budget" if monotonic() >= deadline else "unrefined",
        searches=searches,
    )


def _apex_window_refit(
    fit: _Fit,
    index: int,
    guide: local_windows.Guide,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    seconds: float,
    bad: Sequence[int],
) -> tuple[_Fit | None, dict[str, object]]:
    """Try the failing group's center, then its existing curvature witnesses."""
    deadline = monotonic() + max(0.0, seconds)
    selected = set(guide.window.indices(len(fit.pieces)))
    failing = [i for i in bad if i in selected]

    def lift(t: float) -> float:
        return t + int(guide.seam is not None and t < guide.start_t)

    endpoints = [lift(t) for i in failing for t in fit.intervals[i]]
    middle = (min(endpoints) + max(endpoints)) / 2
    witnesses = {
        lift(a + t * (b - a))
        for i in failing
        for a, b in [fit.intervals[i]]
        for t in fitting_feedback.curvature_locations(
            fit.pieces[i].span, limit.offset_mm, limit.inward_sign
        )
    }
    references = [middle, *sorted(witnesses - {middle})]
    trials: list[dict[str, object]] = []
    for position, reference in enumerate(references):
        apex = guide.to_local(reference - 1 if reference >= 1 else reference)
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        if apex is None or not 0 < apex < 1:
            continue
        proposal, search = _local_window_refit(
            fit,
            index,
            guide,
            limit,
            tolerance,
            remaining / (len(references) - position),
            apex=apex,
        )
        trials.append(dict(source_parameter=reference % 1, search=search))
        if proposal is not None or search.get("status") == "protected-corner":
            return proposal, dict(search, apex_trials=trials)
    return None, dict(
        status="time-budget" if monotonic() >= deadline else "unrefined",
        apex_trials=trials,
    )


def _local_curvature_refit(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    deadline: float,
    *,
    compact: bool = False,
    texture_only: bool = False,
    inferred_corner: bool = False,
) -> tuple[_Fit | None, dict[str, object]]:
    started = monotonic()
    working = fit
    attempts: list[dict[str, object]] = []
    repaired = 0
    paddings = (1, 2, 4) if compact else (0, 1, 2, 4)
    for level, padding in enumerate(paddings):
        # Outside intervals survive splicing exactly. Remember failed regions
        # by their source references, rather than indices that a repair shifts.
        deferred: set[tuple[float, float]] = set()
        while True:
            bad = [
                i
                for i, piece in enumerate(working.pieces)
                if not curvature.within_curvature(
                    piece.span, limit.offset_mm, limit.inward_sign
                )
            ]
            if not bad:
                return working, dict(
                    status="accepted",
                    windows_repaired=repaired,
                    attempts=attempts,
                    elapsed_seconds=monotonic() - started,
                )
            if texture_only and any(
                not any(
                    max(a, working.intervals[i][0]) < min(b, working.intervals[i][1])
                    for a, b in working.texture_intervals
                )
                for i in bad
            ):
                break  # Other failure neighborhoods retain ordinary repair.
            if monotonic() >= deadline:
                break
            pending = [i for i in bad if working.intervals[i] not in deferred]
            if not pending:
                break
            plans = local_windows.windows(
                len(working.pieces),
                pending,
                periodic=working.chain.free_loop,
                neighbors=padding,
            )
            improved = False
            for position, window in enumerate(plans):
                indices = set(window.indices(len(working.pieces)))
                deferred.update(working.intervals[i] for i in bad if i in indices)
                if (
                    2 * window.count >= len(working.pieces) and not inferred_corner
                ) or (window.count == len(working.pieces) and working.chain.free_loop):
                    # Broad ordinary groups belong to whole-chain search. The
                    # fixed two-cubic family stays bounded even on short chains.
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    guide = local_windows.prepare(
                        window,
                        working.chain.points,
                        working.parameters,
                        [p.span for p in working.pieces],
                        working.intervals,
                        periodic=working.chain.free_loop,
                    )
                except ValueError as error:
                    if str(error) != "local repair has an unavailable endpoint tangent":
                        raise
                    continue
                if texture_only and any(
                    guide.start_t < barrier + shift < guide.end_t
                    for barrier in working.texture_barriers
                    for shift in (0, 1)
                ):
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                # Give remaining windows equal shares and leave a share for
                # each later growth level, so a small failed fit cannot starve
                # the larger support that may make it feasible.
                shares = len(plans) - position + len(paddings) - level - 1
                tick = monotonic()
                if inferred_corner:
                    proposal, search = _apex_window_refit(
                        working,
                        index,
                        guide,
                        limit,
                        tolerance,
                        remaining / shares,
                        bad,
                    )
                elif compact:
                    proposal, search = _local_window_refit(
                        working,
                        index,
                        guide,
                        limit,
                        tolerance,
                        remaining / shares,
                        compact=True,
                    )
                else:
                    proposal, search = _local_window_refit(
                        working, index, guide, limit, tolerance, remaining / shares
                    )
                attempts.append(
                    dict(
                        first=window.first,
                        count=window.count,
                        padding=padding,
                        source_points=len(guide.source_points),
                        search=search,
                        wall_seconds=monotonic() - tick,
                    )
                )
                if proposal is not None:
                    working = proposal
                    repaired += 1
                    improved = True
                    break  # Recompute windows: splicing changed their indices.
            if not improved:
                break
        if monotonic() >= deadline:
            break
    return None, dict(
        status="time-budget" if monotonic() >= deadline else "unrefined",
        windows_repaired=repaired,
        attempts=attempts,
        elapsed_seconds=monotonic() - started,
    )


def _compact_curvature_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    *,
    texture_only: bool = False,
) -> tuple[_Fit | None, dict[str, object]]:
    """Try a small smooth family, optionally guided by local texture labels.

    Two C1 cubics let curvature change across a shared tangent without adding a
    sharp corner. Search can approach the full source limit; the unchanged exact
    source, radius and palette proofs remain the acceptance authority.
    """
    return _local_curvature_refit(
        fit,
        index,
        limit,
        tolerance,
        monotonic() + curvature_fitting._SECONDS / 3,
        compact=True,
        texture_only=texture_only,
    )


def _apex_curvature_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
) -> tuple[_Fit | None, dict[str, object]]:
    """Offer one inferred corner only after the ordinary smooth families fail."""
    return _local_curvature_refit(
        fit,
        index,
        limit,
        tolerance,
        monotonic() + curvature_fitting._SECONDS / 3,
        inferred_corner=True,
    )


def _corner_run_curvature_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
) -> tuple[_Fit | None, dict[str, object]]:
    """Refit smooth runs bounded by established corners or graph terminals."""
    deadline = monotonic() + curvature_fitting._SECONDS / 3
    working = fit
    attempts: list[dict[str, object]] = []
    repaired = 0
    while True:
        bad = {
            i
            for i, piece in enumerate(working.pieces)
            if not curvature.within_curvature(
                piece.span, limit.offset_mm, limit.inward_sign
            )
        }
        if not bad:
            return working, dict(
                status="accepted", windows_repaired=repaired, attempts=attempts
            )
        if monotonic() >= deadline:
            break
        corners = _run_corners(working)
        stops = [i for i, (a, _) in enumerate(working.intervals) if a in corners]
        size = len(working.pieces)
        if working.chain.free_loop:
            if len(stops) < 2:
                break
            stops.append(stops[0] + size)
        elif 1 in corners or (
            working.chain.points[0] == working.chain.points[-1] and 0 in corners
        ):
            stops.append(size)
        windows = [
            local_windows.Window(a, b - a)
            for a, b in pairwise(stops)
            if bad.intersection(range(a, b))
            or (b > size and bad.intersection(range(b - size)))
        ]
        improved = False
        for position, window in enumerate(windows):
            guide = local_windows.prepare(
                window,
                working.chain.points,
                working.parameters,
                [p.span for p in working.pieces],
                working.intervals,
                periodic=working.chain.free_loop,
            )
            searches: list[dict[str, object]] = []
            for compact in (True, False):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                shares = (len(windows) - position) * (2 if compact else 1)
                proposal, search = _local_window_refit(
                    working,
                    index,
                    guide,
                    limit,
                    tolerance,
                    remaining / shares,
                    compact=compact,
                    free_corner_rays=True,
                )
                searches.append(dict(compact=compact, search=search))
                if proposal is not None:
                    working = proposal
                    repaired += 1
                    improved = True
                    break
            attempts.append(
                dict(first=window.first, count=window.count, searches=searches)
            )
            if improved:
                break  # Recompute run indices after splicing.
        if not improved:
            break
    return working if repaired else None, dict(
        status="partial" if repaired else "unrefined",
        stop_reason="time-budget" if monotonic() >= deadline else "unrefined",
        windows_repaired=repaired,
        attempts=attempts,
    )


def _polygon_curvature_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
) -> tuple[_Fit | None, dict[str, object]]:
    """Retain exact straight sides of a simple closed source primitive."""
    points = fit.chain.points
    unsupported: dict[str, object] = dict(status="unsupported-source")
    if fit.chain.fixed or len(points) < 4 or points[0] != points[-1]:
        return None, unsupported
    vertices = [c._rational(point) for point in points[:-1]]
    corners: list[int] = []
    direction = 0
    for i, b in enumerate(vertices):
        a, d = vertices[i - 1], vertices[(i + 1) % len(vertices)]
        u, v = (b[0] - a[0], b[1] - a[1]), (d[0] - b[0], d[1] - b[1])
        turn = u[0] * v[1] - u[1] * v[0]
        if not turn:
            if u[0] * v[0] + u[1] * v[1] <= 0:
                return None, unsupported
            continue
        sign = 1 if turn > 0 else -1
        if direction and direction != sign:
            return None, unsupported
        direction = sign
        corners.append(i)
        if len(corners) > 4:
            return None, unsupported
    if len(corners) < 3:
        return None, unsupported
    # Keep the graph pin even when it is midway along a straight source edge.
    indices = sorted({0, *corners, len(points) - 1})
    parameters = [float(_scalar(fit.parameters, i)) for i in indices]
    proposal = replace(
        fit,
        corners=[float(_scalar(fit.parameters, i)) for i in corners],
        knots=parameters[1:-1],
        pieces=[
            c._Piece(index, i, i + 1, c._line(points[a], points[b]))
            for i, (a, b) in enumerate(pairwise(indices))
        ],
        intervals=list(pairwise(parameters)),
    )
    if (
        _source_failures(proposal, tolerance)
        or not all(
            c._monotone(p.span)
            and curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
            for p in proposal.pieces
        )
        or c._unsafe_pair(proposal.pieces)[0]
    ):
        return None, dict(status="uncertified")
    return proposal, dict(
        status="accepted",
        corners=len(corners),
        source_edges=len(proposal.pieces),
        source_displacement_mm=0.0,
    )


def _whole_curvature_refit(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    seconds: float,
) -> tuple[_Fit | None, dict[str, object]]:
    deadline = monotonic() + max(0.0, seconds)
    if seconds <= 0:
        return None, dict(status="time-budget")
    if not fit.chain.free_loop and fit.chain.points[0] == fit.chain.points[-1]:
        # A closed chain pinned at a graph junction must keep its approved
        # branch directions. Position alone permits a tiny self-crossing there.
        try:
            guide = local_windows.prepare(
                local_windows.Window(0, len(fit.pieces)),
                fit.chain.points,
                fit.parameters,
                [p.span for p in fit.pieces],
                fit.intervals,
            )
        except ValueError as error:
            if str(error) != "local repair has an unavailable endpoint tangent":
                raise
            return None, dict(status="unrefined", reason=str(error))
        return _local_window_refit(
            fit, index, guide, limit, tolerance, max(0.0, deadline - monotonic())
        )
    accepted: list[_Fit] = []
    rejected: list[_RejectedFit] = []
    selected_corners: list[float] = []

    def accept(spline: BSpline) -> bool:
        rejected.clear()
        checked = replace(fit, corners=selected_corners.copy())
        try:
            checked.pieces, checked.intervals = _spline_pieces(
                checked, index, tolerance, spline
            )
        except ValueError as error:
            if not str(error).startswith("smooth outline chain"):
                raise
            return False
        if not all(c._monotone(p.span) for p in checked.pieces):
            return False
        if not all(
            curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
            for p in checked.pieces
        ):
            rejected.append(_RejectedFit(spline, checked))
            return False
        if _source_failures(checked, tolerance):
            rejected.append(_RejectedFit(spline, checked, source_failure=True))
            return False
        accepted.append(checked)
        return True

    def refine(spline: BSpline) -> fitting_feedback.Refinement:
        if not rejected or rejected[-1].spline is not spline:
            return fitting_feedback.Refinement()
        return _curvature_feedback(rejected[-1], limit, tolerance)

    bad = [
        i
        for i, p in enumerate(fit.pieces)
        if not curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
    ]
    corners = _curvature_corner_choices(fit, limit, bad)[-1]
    interior_corners = [t for t in corners if 0 < t < 1]
    plans: list[tuple[FloatArray | None, list[float]]] = []
    if corners:
        breaks = np.unique(np.asarray([0.0, *fit.knots, *interior_corners, 1.0]))
        if len(breaks) > curvature_fitting._MAX_SPANS + 1:
            breaks = np.asarray([0.0, *interior_corners, 1.0])
        if len(breaks) <= curvature_fitting._MAX_SPANS + 1:
            plans.append((breaks, corners))
    plans.append((None, []))
    searches: list[dict[str, object]] = []
    diagnostics: dict[str, object] = dict(status="unrefined")
    for position, (breaks, selected_corners) in enumerate(plans):
        accepted.clear()
        rejected.clear()
        proposal, diagnostics = curvature_fitting.refit(
            np.asarray(fit.chain.points),
            fit.parameters,
            limit.offset_mm,
            limit.inward_sign,
            tolerance,
            periodic=fit.chain.free_loop and breaks is None,
            accept=accept,
            refine=refine,
            knot_breaks=breaks,
            corner_parameters=np.asarray(interior_corners)
            if selected_corners and interior_corners
            else None,
            time_budget_seconds=max(0.0, deadline - monotonic())
            / (len(plans) - position),
        )
        searches.append(
            dict(
                diagnostics, basis="corner-intervals" if selected_corners else "smooth"
            )
        )
        if proposal is not None and accepted:
            return accepted[-1], dict(diagnostics, searches=searches)
    return None, dict(diagnostics, searches=searches)


def _pinned_circle_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
) -> _Fit | None:
    """Try a smooth small loop without moving its shared graph junction."""
    from fractions import Fraction

    if (
        fit.chain.free_loop
        or len(fit.chain.points) < 4
        or fit.chain.points[0] != fit.chain.points[-1]
    ):
        return None
    polygon = Polygon(fit.chain.points)
    radius = limit.offset_mm / curvature_fitting._SAMPLED_FRACTION * 1.01
    if not polygon.is_valid or polygon.is_empty or not isfinite(radius) or radius <= 0:
        return None
    xmin, ymin, xmax, ymax = polygon.bounds
    if max(xmax - xmin, ymax - ymin) > 2 * (radius + tolerance):
        return None
    pin = fit.chain.points[0]
    dx, dy = polygon.centroid.x - pin[0], polygon.centroid.y - pin[1]
    length = hypot(dx, dy)
    if not isfinite(length) or length == 0:
        return None
    center = pin[0] + radius * dx / length, pin[1] + radius * dy / length
    theta = atan2(pin[1] - center[1], pin[0] - center[0])
    orientation = 1 if polygon.exterior.is_ccw else -1
    angles = [theta + orientation * i * pi / 2 for i in range(5)]
    anchors = [
        (center[0] + radius * cos(a), center[1] + radius * sin(a)) for a in angles
    ]
    anchors[0] = anchors[-1] = pin
    poles = [pin]
    handle = orientation * radius * 4 * (sqrt(2) - 1) / 3
    for i, (a, b) in enumerate(pairwise(angles)):
        poles.extend(
            [
                (anchors[i][0] - handle * sin(a), anchors[i][1] + handle * cos(a)),
                (
                    anchors[i + 1][0] + handle * sin(b),
                    anchors[i + 1][1] - handle * cos(b),
                ),
                anchors[i + 1],
            ]
        )
    spline = BSpline(
        [0.0] * 4 + [0.25] * 3 + [0.5] * 3 + [0.75] * 3 + [1.0] * 4, poles, 3
    )
    # Canonicalize a smooth seam, then restore the source junction exactly.
    temporary = replace(fit, chain=replace(fit.chain, free_loop=True), corners=[])
    try:
        pieces, intervals = _spline_pieces(temporary, index, tolerance, spline)
    except ValueError as error:
        if not str(error).startswith("smooth outline chain"):
            raise
        return None
    seam = pieces[0].span.control_points[0]
    for position, pole in ((0, 1), (-1, 2)):
        controls = list(pieces[position].span.control_points)
        exact = tuple(
            Fraction(pin[j]) + Fraction(controls[pole][j]) - Fraction(seam[j])
            for j in (0, 1)
        )
        point = float(exact[0]), float(exact[1])
        if c._rational(point) != exact:
            return None
        controls[pole], controls[0 if position == 0 else 3] = point, pin
        pieces[position] = replace(
            pieces[position],
            span=c.BezierSpan(cast("c.Controls", tuple(controls)), False, tolerance),
        )
    proposal = replace(
        fit, pieces=pieces, intervals=intervals, corners=[], knots=[0.25, 0.5, 0.75]
    )
    if (
        not all(c._monotone(p.span) for p in pieces)
        or not all(
            curvature.within_curvature(p.span, limit.offset_mm, limit.inward_sign)
            for p in pieces
        )
        or _source_failures(proposal, tolerance)
        or c._unsafe_pair(pieces)[0]
    ):
        return None
    return proposal


def _curvature_candidate(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    *,
    local: bool = True,
    time_budget_seconds: float | None = None,
) -> tuple[_Fit | None, dict[str, object]]:
    started = monotonic()
    seconds = (
        curvature_fitting._SECONDS
        if time_budget_seconds is None
        else max(0.0, min(curvature_fitting._SECONDS, time_budget_seconds))
    )
    deadline = started + seconds
    if seconds == 0:
        return None, dict(status="time-budget", total_elapsed_seconds=0.0)
    # Certify a fallback within the same budget before the optimizer can spend
    # it. Preferred searches retain inherited endpoint rays; this alternative
    # may rotate them, while preserving the pin and all later topology gates.
    circle = _pinned_circle_candidate(fit, index, limit, tolerance)
    local_report: dict[str, object] | None = None
    if local:
        candidate, local_report = _local_curvature_refit(
            fit, index, limit, tolerance, deadline
        )
        if candidate is not None:
            return candidate, dict(
                status="accepted",
                strategy="local",
                elapsed_seconds=monotonic() - started,
                local_search=local_report,
            )
    candidate, diagnostics = _whole_curvature_refit(
        fit, index, limit, tolerance, max(0.0, deadline - monotonic())
    )
    if candidate is None and circle is not None:
        return circle, dict(
            status="accepted",
            strategy="pinned-circle",
            local_search=local_report,
            whole_search=diagnostics,
            total_elapsed_seconds=monotonic() - started,
        )
    return candidate, dict(
        diagnostics,
        strategy="whole-chain",
        local_search=local_report,
        total_elapsed_seconds=monotonic() - started,
    )


def _curvature_corner_choices(
    fit: _Fit,
    limit: boundary_tapers.BoundaryTaper,
    bad: Sequence[int],
) -> list[list[float]]:
    """Offer existing corners, then additional source turns near bad spans."""
    points = np.asarray(fit.chain.points)
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    intervals = [fit.intervals[i] for i in bad]
    detected = _corners(points, distances, limit.offset_mm / 0.9, 0.0)
    possible = {
        *fit.source_corners,
        *(float(_scalar(fit.parameters, i)) for i in detected),
    }
    added = {t for t in possible if any(a <= t <= b for a, b in intervals)} - set(
        fit.corners
    )
    variants = [sorted(fit.corners)]
    if added:
        variants.append(sorted(set(fit.corners) | added))
    return variants


def _corner_curvature_candidates(
    fit: _Fit,
    index: int,
    limit: boundary_tapers.BoundaryTaper,
    tolerance: float,
    bad: Sequence[int],
) -> list[tuple[_Fit, str]]:
    """Certify a bounded corner family, awaiting full shared-palette checks.

    Whole-chain alternatives may rotate graph-pin rays, like the existing
    circle fallback. Ordinary local repairs retain their inherited rays.
    """
    points = np.asarray(fit.chain.points)
    variants = _curvature_corner_choices(fit, limit, bad)
    deadline = monotonic() + curvature_fitting._SECONDS
    result: list[tuple[_Fit, str]] = []
    seen: set[tuple[c.BezierSpan, ...]] = set()

    def retain(proposal: _Fit, method: str) -> None:
        spans = tuple(p.span for p in proposal.pieces)
        if (
            monotonic() >= deadline
            or spans in seen
            or not all(c._monotone(span) for span in spans)
            or not all(
                curvature.within_curvature(span, limit.offset_mm, limit.inward_sign)
                for span in spans
            )
            or _source_failures(proposal, tolerance)
        ):
            return
        if fit.chain.free_loop and 0.0 not in proposal.corners:
            before, after = c._controls(spans[-1]), c._controls(spans[0])
            u = (before[3][0] - before[2][0], before[3][1] - before[2][1])
            v = (after[1][0] - after[0][0], after[1][1] - after[0][1])
            if u[0] * v[1] != u[1] * v[0] or u[0] * v[0] + u[1] * v[1] <= 0:
                return
        try:
            unsafe, _ = c._unsafe_pair(proposal.pieces)
        except ValueError as error:
            if str(error) != "curve topology exceeds 1,000,000 candidate pairs":
                raise
            return
        if not unsafe:
            seen.add(spans)
            result.append((proposal, method))

    # These are resource bounds on a fallback, not geometric acceptance limits.
    maximum = 2 * curvature_fitting._MAX_SPANS
    for corners in variants:
        if monotonic() >= deadline:
            break
        nodes = sorted({0.0, 1.0, *corners})
        knots = sorted({*fit.knots, *corners})
        if len(nodes) - 1 > maximum:
            continue
        base = replace(fit, corners=corners, knots=[t for t in knots if 0 < t < 1])
        sampled = _sample(points, fit.parameters, np.asarray(nodes))
        anchors = [
            (float(_scalar(sampled, (i, 0))), float(_scalar(sampled, (i, 1))))
            for i in range(len(nodes))
        ]
        chords = replace(
            base,
            pieces=[
                c._Piece(index, i, i + 1, c._line(a, b, tolerance))
                for i, (a, b) in enumerate(pairwise(anchors))
            ],
            intervals=list(pairwise(nodes)),
        )
        retain(chords, "source-corner-chords")
        if (
            len(base.knots) + 1 > maximum
            or len(points) > curvature_fitting._MAX_POINTS
            or monotonic() >= deadline
        ):
            continue
        try:
            pieces, times = _pieces(base, index, tolerance)
        except SplineRefinementNeeded:
            continue
        except ValueError as error:
            if (
                not str(error).startswith("smooth outline chain")
                and str(error) != "spline interval has no source forward direction"
            ):
                raise
            continue
        curved = replace(base, pieces=pieces, intervals=times)
        retain(curved, "source-corner-curves")
        stops = set(nodes)
        mixed = replace(
            curved,
            pieces=[
                replace(
                    p,
                    span=c._line(
                        p.span.control_points[0], p.span.control_points[-1], tolerance
                    ),
                )
                if a in stops and b in stops
                else p
                for p, (a, b) in zip(pieces, times, strict=True)
            ],
        )
        retain(mixed, "source-corner-mixed")
    return result


def _curvature_verified(
    fits: Sequence[_Fit],
    uses: Sequence[Sequence[tuple[int, bool]]],
    ranges: Sequence[range],
    source: PanelContours,
    tolerance: float,
    material_runs: tuple[float, float, float],
    original_curves: c.CurveContours,
    original_proxy: PanelContours,
) -> tuple[c.CurveContours, PanelContours, dict[str, object]]:
    """Try signed curvature refits without dropping features or weakening proofs."""
    limits = boundary_tapers.boundaries(uses, ranges, len(source.orange), material_runs)
    candidates = list(fits)
    changed: set[int] = set()
    attempted: set[int] = set()
    records: list[dict[str, object]] = []
    pools: dict[int, list[tuple[_Fit, str]]] = {}
    budgets: dict[int, float] = {}
    smooth_attempted: set[int] = set()
    preferred_candidates: dict[int, _Fit] = {}
    for index, limit in limits.items():
        fit = fits[index]
        if limit.offset_mm == 0:
            continue
        bad = [
            i
            for i, piece in enumerate(fit.pieces)
            if not curvature.within_curvature(
                piece.span, limit.offset_mm, limit.inward_sign
            )
        ]
        if not bad:
            continue
        attempted.add(index)
        tick = monotonic()
        pool = _corner_curvature_candidates(fit, index, limit, tolerance, bad)
        budgets[index] = max(0.0, curvature_fitting._SECONDS - (monotonic() - tick))
        detail_search: dict[str, object] | None = None
        if any(
            max(a, fit.intervals[i][0]) < min(b, fit.intervals[i][1])
            for a, b in fit.texture_intervals
            for i in bad
        ):
            preferred, detail_search = _compact_curvature_candidate(
                fit, index, limit, tolerance, texture_only=True
            )
            if preferred is not None:
                preferred_candidates[index] = preferred
                pool.insert(0, (preferred, "fine-texture-smooth"))
        if pool:
            proposal, method = pool.pop(0)
            pools[index] = pool
            diagnostics: dict[str, object] = dict(status="accepted", strategy=method)
        else:
            smooth_attempted.add(index)
            proposal, diagnostics = _curvature_candidate(
                fit, index, limit, tolerance, time_budget_seconds=budgets[index]
            )
        if proposal is not None:
            candidates[index] = proposal
            changed.add(index)
        record: dict[str, object] = dict(
            chain=index,
            left_material=limit.left,
            right_material=limit.right,
            offset_mm=limit.offset_mm,
            inward_sign=limit.inward_sign,
            search=diagnostics,
        )
        if detail_search is not None:
            record["detail_search"] = detail_search
        records.append(record)
    verified = _verify_refits(
        fits,
        candidates,
        changed,
        uses,
        ranges,
        source,
        tolerance,
        original_curves,
        original_proxy,
    )
    accepted = verified.accepted.copy()
    blocked = verified.blocked.copy()
    # A simpler individual fit may close a gap against a different chain.
    # Retry only the rolled-back chains with their next certified alternative;
    # retain unrelated accepted repairs. Exhausted corner pools still get one
    # ordinary repair attempt using the boundary's remaining search budget.
    for _ in range(max((len(pool) for pool in pools.values()), default=0) + 1):
        retry = {
            i for i in verified.blocked if pools.get(i) or i not in smooth_attempted
        }
        if not retry:
            break
        next_fits = list(verified.fits)
        proposed: set[int] = set()
        for i in retry:
            if pools.get(i):
                next_fits[i], method = pools[i].pop(0)
            else:
                smooth_attempted.add(i)
                proposal, search = _curvature_candidate(
                    fits[i], i, limits[i], tolerance, time_budget_seconds=budgets[i]
                )
                next(r for r in records if r["chain"] == i)["fallback_search"] = search
                if proposal is None:
                    continue
                next_fits[i] = proposal
                method = "curvature-search"
            proposed.add(i)
            next(r for r in records if r["chain"] == i)["palette_retry_strategy"] = (
                method
            )
        if not proposed:
            break
        verified = _verify_refits(
            verified.fits,
            next_fits,
            proposed,
            uses,
            ranges,
            source,
            tolerance,
            verified.curves,
            verified.proxy,
        )
        accepted.update(verified.accepted)
        blocked.difference_update(verified.accepted)
        blocked.update(verified.blocked)
    for record in records:
        if record["chain"] in accepted:
            record["selected_strategy"] = record.get(
                "palette_retry_strategy",
                cast("dict[str, object]", record["search"]).get("strategy"),
            )
    # Later families only touch unresolved chains. Recheck cached preferred
    # fits against stabilized neighbors before offering one inferred corner.
    for family, method, refit in (
        ("compact", "compact-local-c1", _compact_curvature_candidate),
        ("apex", "inferred-local-corner", _apex_curvature_candidate),
        ("polygon", "source-convex-polygon", _polygon_curvature_candidate),
        ("corner_run", "corner-delimited-run", _corner_run_curvature_candidate),
    ):
        next_fits = list(verified.fits)
        next_changed: set[int] = set()
        for index in sorted(attempted - accepted):
            proposal = preferred_candidates.get(index) if family == "compact" else None
            if proposal is None:
                proposal, search = refit(
                    verified.fits[index], index, limits[index], tolerance
                )
                next(r for r in records if r["chain"] == index)[family + "_search"] = (
                    search
                )
            if proposal is not None:
                next_fits[index] = proposal
                next_changed.add(index)
        if not next_changed:
            continue
        verified = _verify_refits(
            verified.fits,
            next_fits,
            next_changed,
            uses,
            ranges,
            source,
            tolerance,
            verified.curves,
            verified.proxy,
        )
        accepted.update(verified.accepted)
        blocked.difference_update(verified.accepted)
        blocked.update(verified.blocked)
        for index in verified.accepted:
            next(r for r in records if r["chain"] == index)["selected_strategy"] = (
                "fine-texture-smooth"
                if family == "compact" and index in preferred_candidates
                else method
            )
    # Palette acceptance can retain successful local runs while other spans
    # still need contact cleanup. Report actual selected controls and intervals;
    # neither a selected partial repair nor a rollback certifies the chain.
    uncertified = {
        index: bad
        for index in sorted(attempted)
        if (
            bad := [
                i
                for i, piece in enumerate(verified.fits[index].pieces)
                if not curvature.within_curvature(
                    piece.span, limits[index].offset_mm, limits[index].inward_sign
                )
            ]
        )
    }
    remaining = [
        dict(
            chain=index,
            offset_mm=limits[index].offset_mm,
            inward_sign=limits[index].inward_sign,
            spans=[
                dict(piece=i, interval=verified.fits[index].intervals[i]) for i in bad
            ],
        )
        for index, bad in sorted(uncertified.items())
    ]
    return (
        verified.curves,
        verified.proxy,
        {
            "boundary_curvature": {
                "maximum_curvature_times_offset": 0.9,
                "margin_status": "unresolved" if remaining else "certified",
                "unresolved_boundaries": remaining,
                "material_taper_runs_mm": material_runs,
                "attempted_chains": len(attempted),
                "refined_chains": len(attempted - uncertified.keys()),
                "partially_refined_chains": sorted(accepted & uncertified.keys()),
                "unrefined_chains": sorted(uncertified),
                "topology_rejected_chains": sorted(blocked),
                "topology_rejection": verified.rejection,
                "boundaries": records,
                "policy": "retain source features when a bounded refit is unavailable; early counter closure remains permitted",
                "certificate": "continuous curvature certified for repaired local runs and fully refined chains; remaining selected spans reported as unresolved; original source error and complete palette topology certified for all selected outlines",
            },
            "selected_unique_spans": sum(len(f.pieces) for f in verified.fits),
        },
    )


def fit_outlines(
    source: PanelContours,
    tolerance_mm: float,
    *,
    corner_scale_mm: float = 1.0,
    corner_prominence_mm: float = 0.25,
    corner_softening_mm: float = 0.0,
    fairing_mm: float = 0.25,
    span_simplification_mm: float = 0.0,
    point_contact_scale_mm: float = 0.0,
    edge_detail_scale_mm: float = 0.0,
    material_taper_runs_mm: tuple[float, float, float] | None = None,
    palette: Palette | None = None,
) -> c.CurveFitResult:
    """Fit shared outlines with movable smooth joins and physical-scale corners.

    Optional point-contact repair attaches small raster fragments after tracing,
    retaining the original source-distance limit and other fitted curves.
    Otherwise source islands/counters retain their original topology. Zero
    tolerance requests the existing exact representation.
    """
    for value, name, positive in (
        (cast("object", tolerance_mm), "tolerance", False),
        (cast("object", corner_scale_mm), "corner scale", True),
        (cast("object", corner_prominence_mm), "corner prominence", False),
        (cast("object", corner_softening_mm), "corner softening", False),
        (cast("object", fairing_mm), "fairing scale", False),
        (cast("object", span_simplification_mm), "span simplification", False),
        (cast("object", point_contact_scale_mm), "point-contact scale", False),
        (cast("object", edge_detail_scale_mm), "edge-detail scale", False),
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
    if material_taper_runs_mm is not None:
        _ = boundary_tapers.boundaries([[]], [range(1)], 0, material_taper_runs_mm)
    if tolerance_mm == 0:
        return c.fit_contours(source, 0, palette=palette)
    if palette is not None:
        _ = SeparationConfig(palette=palette)
    if material_taper_runs_mm is not None:
        # Public callers need not supply vectorize's normalized winding. Side
        # ownership requires each colored ring to keep its material on the left.
        source = replace(
            source,
            silhouette=orient(source.silhouette),
            orange=[orient(polygon) for polygon in source.orange],
            white=[orient(polygon) for polygon in source.white],
        )
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
            _corners(points, distances, corner_scale_mm, corner_prominence_mm)
            if not chain.fixed
            else []
        )
        corners = [float(_scalar(parameters, i)) for i in corner_indices]
        knots = {value for value in corners if 0 < value < 1}
        if chain.points[0] == chain.points[-1]:
            for knot in (0.25, 0.5, 0.75):
                if not any(abs(value - knot) < 1e-10 for value in knots):
                    knots.add(knot)
        fit = _Fit(
            chain,
            parameters,
            corners,
            sorted(knots),
            [],
            [],
            fairing_mm,
            source_corners=tuple(corners),
        )
        if chain.fixed:
            fit.pieces = [
                c._Piece(index, i, i + 1, c._line(a, b))
                for i, (a, b) in enumerate(
                    zip(chain.points[:-1], chain.points[1:], strict=True)
                )
            ]
        fits.append(fit)
    detail_diagnostics = (
        _detail_guidance(fits, source, edge_detail_scale_mm)
        if edge_detail_scale_mm > 0
        else {}
    )
    refinement_count = 0
    constraint_refinements = 0
    certificates: dict[
        int, tuple[list[c._Piece], list[tuple[float, float]], set[int]]
    ] = {}
    solve_attempts: Counter[int] = Counter()

    def refine(index: int, interval: int, exhausted: bool) -> None:
        nonlocal refinement_count
        fit = fits[index]
        a, b = fit.intervals[interval]
        length = sum(
            hypot(q[0] - p[0], q[1] - p[1]) for p, q in pairwise(fit.chain.points)
        )
        if (b - a) * length < min(corner_scale_mm / 32, length / 128) or exhausted:
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

    for iteration in range(14):
        unresolved: set[tuple[int, int]] = set()
        for index, fit in enumerate(fits):
            if fit.chain.fixed:
                continue
            # Source/direction proofs belong to one chain. Another chain's
            # constraint retries must not consume its refinement opportunities.
            # Count actual solves across topology rounds, keeping the same cap.
            while True:
                try:
                    if fit.intervals != list(pairwise([0.0, *fit.knots, 1.0])):
                        solve_attempts[index] += 1
                        fit.pieces, fit.intervals = _pieces(fit, index, tolerance_mm)
                except SplineRefinementNeeded as error:
                    if solve_attempts[index] == 14:
                        raise
                    a, b = error.interval or max(
                        pairwise([0.0, *fit.knots, 1.0]), key=lambda p: p[1] - p[0]
                    )
                    middle = (a + b) / 2
                    if not a < middle < b:
                        raise
                    fit.knots.append(middle)
                    fit.knots.sort()
                    refinement_count += 1
                    constraint_refinements += 1
                    continue
                cached = certificates.get(index)
                if (
                    cached is None
                    or cached[0] is not fit.pieces
                    or cached[1] is not fit.intervals
                ):
                    failed = _source_failures(fit, tolerance_mm)
                    failed.update(
                        p.start for p in fit.pieces if not c._monotone(p.span)
                    )
                    certificates[index] = fit.pieces, fit.intervals, failed
                else:
                    failed = cached[2]
                if not failed:
                    break
                for interval in sorted(failed):
                    refine(index, interval, solve_attempts[index] == 14)
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
                softening_diagnostics: dict[str, object] = {}
                selected_fits = fits
                if corner_softening_mm > 0:
                    curves, proxy, softening_diagnostics, selected_fits = (
                        _soften_verified(
                            fits,
                            uses,
                            ranges,
                            source,
                            tolerance_mm,
                            corner_softening_mm,
                            curves,
                            proxy,
                            material_taper_runs_mm=material_taper_runs_mm,
                        )
                    )
                simplification_diagnostics: dict[str, object] = {}
                if span_simplification_mm > 0:
                    curves, proxy, simplification_diagnostics, selected_fits = (
                        _simplify_verified(
                            selected_fits,
                            uses,
                            ranges,
                            source,
                            tolerance_mm,
                            span_simplification_mm,
                            curves,
                            proxy,
                            material_taper_runs_mm=material_taper_runs_mm,
                        )
                    )
                curvature_diagnostics: dict[str, object] = {}
                if material_taper_runs_mm is not None and any(material_taper_runs_mm):
                    curves, proxy, curvature_diagnostics = _curvature_verified(
                        selected_fits,
                        uses,
                        ranges,
                        source,
                        tolerance_mm,
                        material_taper_runs_mm,
                        curves,
                        proxy,
                    )
                contact_diagnostics: dict[str, object] = {}
                if point_contact_scale_mm > 0 and material_taper_runs_mm is not None:
                    curves, proxy, contact_diagnostics = contact_refinement.refine(
                        source,
                        curves,
                        proxy,
                        scale_mm=point_contact_scale_mm,
                        tolerance_mm=tolerance_mm,
                        material_runs=material_taper_runs_mm,
                        proxy_tolerance_mm=proxy_tolerance,
                    )
                    if contact_diagnostics.get("changed"):
                        curvature_diagnostics["pre_contact_boundary_curvature"] = (
                            curvature_diagnostics.pop("boundary_curvature", {})
                        )
                        try:
                            boundary_tapers.require_curvature(
                                curves, material_taper_runs_mm
                            )
                        except boundary_tapers.CurvatureLimitError as error:
                            failures = sorted(error.failures)
                        else:
                            failures = []
                        curvature_diagnostics["boundary_curvature"] = {
                            "maximum_curvature_times_offset": 0.9,
                            "margin_status": "unresolved" if failures else "certified",
                            "unresolved_selected_spans": failures,
                            "material_taper_runs_mm": material_taper_runs_mm,
                            "scope": "after local point-contact interpretation",
                        }
                diagnostics: dict[str, object] = dict(
                    method="joint cubic B-spline outline fitting",
                    mode="curves",
                    corner_scale_mm=corner_scale_mm,
                    corner_prominence_mm=corner_prominence_mm,
                    fairing_mm=fairing_mm,
                    forward_derivative_constraints=True,
                    tolerance_mm=tolerance_mm,
                    fit_target_mm=_FIT_TARGET_FRACTION * tolerance_mm,
                    continuous_boundary_error_upper_bound_mm=tolerance_mm,
                    proxy_deviation_mm=proxy_tolerance,
                    source_unique_edges=sum(len(f.chain.points) - 1 for f in fits),
                    selected_unique_spans=len(pieces),
                    native_edges=curves.edge_count,
                    physical_corners=sum(len(f.corners) for f in fits),
                    smooth_refinements=refinement_count,
                    constraint_refinements=constraint_refinements,
                    solver_calls=sum(solve_attempts.values()),
                    maximum_chain_solver_calls=max(solve_attempts.values(), default=0),
                    topology_checks=iteration + 1,
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
                    error_certificate="continuous bidirectional source distance with exact rational bounds",
                    topology_certificate="native-to-proxy control-hull homotopy and original palette relationships",
                )
                diagnostics.update(softening_diagnostics)
                diagnostics.update(simplification_diagnostics)
                diagnostics.update(curvature_diagnostics)
                diagnostics.update(contact_diagnostics)
                diagnostics.update(detail_diagnostics)
                if contact_diagnostics.get("changed"):
                    diagnostics["topology_certificate"] = (
                        "native-to-proxy control-hull homotopy and interpreted palette relationships"
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
            refine(index, interval, iteration == 13 or solve_attempts[index] == 14)
    raise ValueError("smooth outline fitting exceeded its refinement budget")
