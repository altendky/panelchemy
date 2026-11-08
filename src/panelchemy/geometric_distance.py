"""Conservative continuous distance in both directions, without parameter matching."""

# This certificate shares the conversion core's exact rational primitives.
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import pairwise
from math import inf, isfinite, nextafter
from typing import cast, final

from shapely import STRtree
from shapely.geometry import box

from panelchemy import curves as c
from panelchemy._native import within_paths as _native_within_paths

type _Point = c.RationalPoint
type _Segment = tuple[_Point, _Point]
_MAX_DEPTH = 24
_MAX_NODES = 100_000
_INDEX_MIN_SEGMENTS = 32
_PROXY_REFINEMENTS = 3
type DistanceJob = tuple[Sequence[Sequence[c.Point2]], Sequence[c.BezierSpan], float]


@final
class _Budget:
    def __init__(self) -> None:
        self.remaining = _MAX_NODES

    def spend(self) -> bool:
        self.remaining -= 1
        return self.remaining >= 0


def _inside(point: _Point, segment: _Segment, radius_squared: Fraction) -> bool:
    a, b = segment
    dx, dy = b[0] - a[0], b[1] - a[1]
    px, py = point[0] - a[0], point[1] - a[1]
    length = dx * dx + dy * dy
    projection = px * dx + py * dy
    if projection <= 0:
        return px * px + py * py <= radius_squared
    if projection >= length:
        return c._squared((point[0] - b[0], point[1] - b[1])) <= radius_squared
    return (px * dy - py * dx) ** 2 <= radius_squared * length


def _straight(controls: c.RationalControls) -> bool:
    a, d = controls[0], controls[-1]
    direction = d[0] - a[0], d[1] - a[1]
    length = c._squared(direction)
    if length == 0:
        return all(point == a for point in controls)
    return all(
        (point[0] - a[0]) * direction[1] == (point[1] - a[1]) * direction[0]
        and 0
        <= (point[0] - a[0]) * direction[0] + (point[1] - a[1]) * direction[1]
        <= length
        for point in controls[1:3]
    )


@final
class _Capsules:
    def __init__(self, segments: Sequence[_Segment], radii: Sequence[Fraction]) -> None:
        self.segments = segments
        self.squared = [radius**2 for radius in radii]
        self.radius = max(radii)
        self.bounds = [
            (
                nextafter(float(min(a[0], b[0])), -inf),
                nextafter(float(min(a[1], b[1])), -inf),
                nextafter(float(max(a[0], b[0])), inf),
                nextafter(float(max(a[1], b[1])), inf),
            )
            for a, b in segments
        ]
        self.index = (
            STRtree([box(*bounds) for bounds in self.bounds])
            if len(segments) >= _INDEX_MIN_SEGMENTS
            and all(isfinite(value) for bounds in self.bounds for value in bounds)
            else None
        )

    def candidates(self, points: Sequence[_Point]) -> list[int]:
        # Round both conversions and arithmetic outward. These boxes only
        # select possible witnesses; acceptance always uses exact arithmetic.
        radius = nextafter(float(self.radius), inf)
        lower = [
            nextafter(nextafter(float(min(p[j] for p in points)), -inf) - radius, -inf)
            for j in (0, 1)
        ]
        upper = [
            nextafter(nextafter(float(max(p[j] for p in points)), inf) + radius, inf)
            for j in (0, 1)
        ]
        indices = (
            sorted(
                cast(
                    "list[int]",
                    self.index.query(
                        box(lower[0], lower[1], upper[0], upper[1])
                    ).tolist(),
                )
            )
            if self.index is not None
            and all(isfinite(value) for value in (*lower, *upper))
            else range(len(self.bounds))
        )
        return [
            i
            for i in indices
            for left, bottom, right, top in (self.bounds[i],)
            if right >= lower[0]
            and left <= upper[0]
            and top >= lower[1]
            and bottom <= upper[1]
        ]

    def covered(self, points: Sequence[_Point], candidates: Sequence[int]) -> bool:
        # A segment capsule is convex, so it contains every convex combination
        # of these controls/endpoints when it contains the controls themselves.
        return any(all(self.inside(point, i) for point in points) for i in candidates)

    def inside(self, point: _Point, index: int) -> bool:
        return _inside(point, self.segments[index], self.squared[index])

    def parallel_cover(self, a: _Point, b: _Point, candidates: Sequence[int]) -> bool:
        """Cover a line using exact rectangular strips of parallel capsules.

        This also proves equality at zero radius when several target segments
        partition the same line at nondyadic locations.
        """
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        if length == 0:
            return self.covered((a,), candidates)
        intervals: list[tuple[Fraction, Fraction]] = []
        for index in candidates:
            p, q = self.segments[index]
            if dx * (q[1] - p[1]) != dy * (q[0] - p[0]):
                continue
            if ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) ** 2 > (
                self.squared[index] * length
            ):
                continue
            u, v = (
                ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length
                for point in (p, q)
            )
            intervals.append((max(Fraction(0), min(u, v)), min(Fraction(1), max(u, v))))
        reached = Fraction(0)
        for start, stop in sorted(intervals):
            if start > reached:
                return False
            reached = max(reached, stop)
            if reached >= 1:
                return True
        return False


def _curve_to_source(
    controls: c.RationalControls, target: _Capsules, budget: _Budget
) -> bool:
    pending = [(controls, 0)]
    while pending:
        controls, depth = pending.pop()
        if not budget.spend():
            return False
        candidates = target.candidates(controls)
        if target.covered(controls, candidates) or (
            _straight(controls)
            and target.parallel_cover(controls[0], controls[-1], candidates)
        ):
            continue
        if depth == _MAX_DEPTH or any(
            not any(target.inside(point, i) for i in candidates)
            for point in (controls[0], controls[-1])
        ):
            return False
        left, right = c._split(controls)
        pending.extend(((right, depth + 1), (left, depth + 1)))
    return True


def _proxy(
    controls: c.RationalControls, epsilon: Fraction, budget: _Budget
) -> list[_Segment] | None:
    result: list[_Segment] = []
    epsilon_squared = epsilon**2
    pending = [(controls, 0)]
    while pending:
        controls, depth = pending.pop()
        if not budget.spend():
            return None
        a, d = controls[0], controls[-1]
        chord = (a, c._mix(a, d, Fraction(1, 3)), c._mix(a, d, Fraction(2, 3)), d)
        if all(
            c._squared((p[0] - q[0], p[1] - q[1])) <= epsilon_squared
            for p, q in zip(controls, chord, strict=True)
        ):
            result.append((a, d))
        elif depth == _MAX_DEPTH:
            return None
        else:
            left, right = c._split(controls)
            pending.extend(((right, depth + 1), (left, depth + 1)))
    return result


def _source_to_proxy(segment: _Segment, target: _Capsules, budget: _Budget) -> bool:
    pending = [(segment, 0)]
    while pending:
        (a, b), depth = pending.pop()
        if not budget.spend():
            return False
        candidates = target.candidates((a, b))
        if target.covered((a, b), candidates) or target.parallel_cover(
            a, b, candidates
        ):
            continue
        if depth == _MAX_DEPTH or any(
            not any(target.inside(point, i) for i in candidates) for point in (a, b)
        ):
            return False
        middle = c._mix(a, b, Fraction(1, 2))
        pending.extend((((middle, b), depth + 1), ((a, middle), depth + 1)))
    return True


def within_distance(
    source_points: Sequence[c.Point2],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Certify continuous bidirectional distance to one source path."""
    return within_paths((source_points,), spans, tolerance_mm)


def _validate_paths(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> None:
    if (
        isinstance(tolerance_mm, bool)
        or not isfinite(tolerance_mm)
        or tolerance_mm < 0
        or not source_paths
        or any(len(path) < 2 for path in source_paths)
        or not spans
        or any(len(span.control_points) != 4 for span in spans)
        or any(
            len(point) != 2 or not all(isfinite(v) for v in point)
            for path in source_paths
            for point in path
        )
        or any(
            len(point) != 2 or not all(isfinite(v) for v in point)
            for span in spans
            for point in span.control_points
        )
    ):
        raise ValueError(
            "distance certification requires finite planar paths and nonnegative tolerance"
        )


def within_paths(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Certify one job with the faster serial Python rational implementation."""
    return _within_paths_python(source_paths, spans, tolerance_mm)


def within_paths_many(jobs: Sequence[DistanceJob], *, workers: int = 4) -> list[bool]:
    """Certify independent jobs in stable order, each with its own proof budget.

    Native computations release the interpreter lock. Isolated checks retain
    the faster serial reference; a batch amortizes executor overhead. No shared
    geometry is edited here, and every result needs the caller's topology gate.
    """
    count = cast("object", workers)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("certificate workers must be a positive integer")
    if len(jobs) < 2 or count == 1:
        return [within_paths(*job) for job in jobs]

    def certify(job: DistanceJob) -> bool:
        return _within_paths_native(*job)

    with ThreadPoolExecutor(max_workers=min(count, len(jobs))) as executor:
        return list(executor.map(certify, jobs))


def _within_paths_native(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Certify continuous bidirectional distance in the Rust computation core.

    Disconnected paths, exact line handles, proxy headroom, and shared resource
    limits retain the reference algorithm's semantics. Non-binary64 values use
    the Python rational implementation to preserve their exact numeric value.
    False includes inconclusive proofs; topology remains a separate check.
    """
    _validate_paths(source_paths, spans, tolerance_mm)
    values = (
        tolerance_mm,
        *(v for path in source_paths for point in path for v in point),
        *(v for span in spans for point in span.control_points for v in point),
    )
    if (
        any(float(value) != value for value in values)
        or _MAX_DEPTH < 0
        or _PROXY_REFINEMENTS < 0
    ):
        return _within_paths_python(source_paths, spans, tolerance_mm)
    return _native_within_paths(
        [[(float(x), float(y)) for x, y in path] for path in source_paths],
        [
            (
                cast(
                    "c.Controls",
                    tuple((float(x), float(y)) for x, y in span.control_points),
                ),
                bool(span.is_line),
            )
            for span in spans
        ],
        float(tolerance_mm),
        max(0, _MAX_NODES),
        _MAX_DEPTH,
        _PROXY_REFINEMENTS,
        max(0, _INDEX_MIN_SEGMENTS),
    )


def _within_paths_python(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Exact Python reference, also preserving non-binary64 input values."""
    _validate_paths(source_paths, spans, tolerance_mm)
    segments = tuple(
        segment
        for path in source_paths
        for segment in pairwise(tuple(c._rational(point) for point in path))
    )
    tolerance = Fraction(tolerance_mm)
    target = _Capsules(segments, [tolerance] * len(segments))
    budget = _Budget()
    controls = [c._controls(span) for span in spans]
    if not all(_curve_to_source(span, target, budget) for span in controls):
        return False
    first_unproved = 0
    nonlinear = any(not _straight(span) for span in controls)
    for refinement in range(_PROXY_REFINEMENTS + 1):
        proxy: list[_Segment] = []
        radii: list[Fraction] = []
        epsilon = tolerance / (16 * cast("int", 4**refinement))
        for span in controls:
            if _straight(span):
                proxy.append((span[0], span[-1]))
                radii.append(tolerance)
            else:
                if tolerance == 0:
                    return False
                selected = _proxy(span, epsilon, budget)
                if selected is None:
                    return False
                proxy.extend(selected)
                radii.extend([tolerance - epsilon] * len(selected))
        target = _Capsules(proxy, radii)
        while first_unproved < len(segments) and _source_to_proxy(
            segments[first_unproved], target, budget
        ):
            # A completed certificate remains valid for the same true curves
            # even when later intervals require a more accurate proxy.
            first_unproved += 1
        if first_unproved == len(segments):
            return True
        if budget.remaining < 0 or not nonlinear:
            return False
        # A point outside the enlarged proxy tube is provably farther from
        # the true curves than the requested tolerance. Refinement cannot help.
        outer = _Capsules(proxy, [2 * tolerance - radius for radius in radii])
        if any(
            not outer.covered((point,), outer.candidates((point,)))
            for point in segments[first_unproved]
        ):
            return False
    return False
