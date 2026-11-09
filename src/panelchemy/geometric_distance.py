"""Conservative continuous distance in both directions, without parameter matching."""

# This certificate shares the conversion core's exact rational primitives.
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from math import inf, isfinite, nextafter
from typing import cast, final

from shapely import STRtree
from shapely.geometry import box

from panelchemy import curves as c

type _Point = c.RationalPoint
type _Segment = tuple[_Point, _Point]
_MAX_DEPTH = 24
_MAX_NODES = 100_000
_INDEX_MIN_SEGMENTS = 32
_PROXY_REFINEMENTS = 3
_MAX_CACHE_UNITS = 100_000


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
    def __init__(
        self,
        segments: Sequence[_Segment],
        radii: Sequence[Fraction],
        owners: Sequence[c.RationalControls] | None = None,
    ) -> None:
        self.segments = segments
        self.radii = radii
        self.owners = owners
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
        return self.covering_index(points, candidates) is not None

    def covering_index(
        self, points: Sequence[_Point], candidates: Sequence[int]
    ) -> int | None:
        return next(
            (i for i in candidates if all(self.inside(point, i) for point in points)),
            None,
        )

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


def _source_to_proxy(
    segment: _Segment,
    target: _Capsules,
    budget: _Budget,
    receipt: set[c.RationalControls] | None = None,
) -> bool:
    owners = target.owners
    if receipt is not None and owners is None:
        raise ValueError("reverse receipts require actual curve owners")
    pending = [(segment, 0)]
    while pending:
        (a, b), depth = pending.pop()
        if not budget.spend():
            return False
        candidates = target.candidates((a, b))
        witness = target.covering_index((a, b), candidates)
        if witness is not None:
            if receipt is not None and owners is not None:
                receipt.add(owners[witness])
            continue
        if target.parallel_cover(a, b, candidates):
            if receipt is not None and owners is not None:
                # All candidates is a conservative superset of the capsules
                # whose parallel strips actually cover this subdivision leaf.
                receipt.update(owners[i] for i in candidates)
            continue
        if depth == _MAX_DEPTH or any(
            not any(target.inside(point, i) for i in candidates) for point in (a, b)
        ):
            return False
        middle = c._mix(a, b, Fraction(1, 2))
        pending.extend((((middle, b), depth + 1), ((a, middle), depth + 1)))
    return True


def _local_proxy(
    controls: Sequence[c.RationalControls],
    owners: Sequence[int],
    tolerance: Fraction,
    refinement: int,
    budget: _Budget,
    cache: dict[tuple[int, int], tuple[_Segment, ...]],
    proof_cache: "ForwardProofCache | None" = None,
) -> _Capsules | None:
    proxy: list[_Segment] = []
    radii: list[Fraction] = []
    curve_owners: list[c.RationalControls] = []
    epsilon = tolerance / (16 * cast("int", 4**refinement))
    for owner in owners:
        span = controls[owner]
        if _straight(span):
            proxy.append((span[0], span[-1]))
            radii.append(tolerance)
            curve_owners.append(span)
            continue
        if tolerance == 0:
            return None
        key = owner, refinement
        if key not in cache:
            selected = (
                _proxy(span, epsilon, budget)
                if proof_cache is None
                else proof_cache._proxy(span, epsilon, budget)
            )
            if selected is None:
                return None
            cache[key] = tuple(selected)
        # Cached subdivisions describe the same immutable span and epsilon.
        # The node budget bounds computation, so reuse spends no extra nodes.
        selected = cache[key]
        proxy.extend(selected)
        radii.extend([tolerance - epsilon] * len(selected))
        curve_owners.extend([span] * len(selected))
    return _Capsules(proxy, radii, curve_owners)


def within_distance(
    source_points: Sequence[c.Point2],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Certify continuous bidirectional distance to one source path."""
    return within_paths((source_points,), spans, tolerance_mm)


def _valid_source(paths: Sequence[Sequence[c.Point2]], tolerance_mm: float) -> bool:
    return (
        not isinstance(tolerance_mm, bool)
        and isfinite(tolerance_mm)
        and tolerance_mm >= 0
        and bool(paths)
        and all(len(path) >= 2 for path in paths)
        and all(
            len(point) == 2 and all(isfinite(v) for v in point)
            for path in paths
            for point in path
        )
    )


@final
@dataclass(frozen=True, init=False)
class ForwardProofCache:
    """Reuse positive distance facts for one immutable original source.

    Only base-only calls teach forward proofs. Reverse receipts cover exact
    source primitives using actual curve owners that must all remain present.
    Scope this bounded context to one refinement operation, not a global cache.
    """

    _base_paths: tuple[tuple[c.Point2, ...], ...] = field(repr=False)
    _tolerance_mm: float
    _positive: set[c.RationalControls] = field(repr=False, compare=False)
    _proxies: dict[tuple[c.RationalControls, Fraction], tuple[_Segment, ...]] = field(
        repr=False, compare=False
    )
    _reverse: dict[_Segment, frozenset[c.RationalControls]] = field(
        repr=False, compare=False
    )
    _cache_size: int = field(repr=False, compare=False)
    _policy: tuple[int, ...] = field(repr=False, compare=False)

    def __init__(
        self, base_paths: Sequence[Sequence[c.Point2]], tolerance_mm: float
    ) -> None:
        if not _valid_source(base_paths, tolerance_mm):
            raise ValueError(
                "distance certification requires finite planar paths and nonnegative tolerance"
            )
        frozen = tuple(
            tuple((point[0], point[1]) for point in path) for path in base_paths
        )
        object.__setattr__(self, "_base_paths", frozen)
        object.__setattr__(self, "_tolerance_mm", tolerance_mm)
        object.__setattr__(self, "_positive", set())
        object.__setattr__(self, "_proxies", {})
        object.__setattr__(self, "_reverse", {})
        object.__setattr__(self, "_cache_size", 0)
        object.__setattr__(self, "_policy", self._current_policy())

    @staticmethod
    def _current_policy() -> tuple[int, ...]:
        return (
            _MAX_NODES,
            _MAX_DEPTH,
            _PROXY_REFINEMENTS,
            _INDEX_MIN_SEGMENTS,
            _MAX_CACHE_UNITS,
        )

    def _prepare(
        self, controls: set[c.RationalControls], segments: set[_Segment]
    ) -> None:
        policy = self._current_policy()
        if policy != self._policy:
            self._positive.clear()
            self._proxies.clear()
            self._reverse.clear()
            object.__setattr__(self, "_policy", policy)
        # Forward facts depend only on the immutable base and tolerance, so a
        # rejected candidate must not erase proofs for curves restored later.
        for key in list(self._proxies):
            if key[0] not in controls:
                del self._proxies[key]
        for segment, owners in list(self._reverse.items()):
            if segment not in segments or not owners <= controls:
                del self._reverse[segment]
        # Bound retained entries, proxy segments and witness owner references.
        # Proxy/reverse facts require surviving owners; forward facts remain
        # bounded by the same capacity until policy reset or context release.
        object.__setattr__(
            self,
            "_cache_size",
            len(self._positive)
            + sum(1 + len(proxy) for proxy in self._proxies.values())
            + sum(1 + len(owners) for owners in self._reverse.values()),
        )

    def _reserve(self, units: int) -> bool:
        if self._cache_size + units > _MAX_CACHE_UNITS:
            return False
        object.__setattr__(self, "_cache_size", self._cache_size + units)
        return True

    def _remember_forward(self, controls: c.RationalControls) -> None:
        if controls not in self._positive and self._reserve(1):
            self._positive.add(controls)

    def _proxy(
        self, controls: c.RationalControls, epsilon: Fraction, budget: _Budget
    ) -> tuple[_Segment, ...] | None:
        key = controls, epsilon
        if key in self._proxies:
            return self._proxies[key]
        selected = _proxy(controls, epsilon, budget)
        if selected is None:
            return None
        frozen = tuple(selected)
        if self._reserve(1 + len(frozen)):
            self._proxies[key] = frozen
        return frozen

    def _remember_reverse(
        self, segment: _Segment, owners: set[c.RationalControls]
    ) -> None:
        if owners and segment not in self._reverse and self._reserve(1 + len(owners)):
            self._reverse[segment] = frozenset(owners)

    @property
    def base_paths(self) -> tuple[tuple[c.Point2, ...], ...]:
        return self._base_paths

    @property
    def tolerance_mm(self) -> float:
        return self._tolerance_mm

    def within_extra_paths(
        self,
        extra_paths: Sequence[Sequence[c.Point2]],
        spans: Sequence[c.BezierSpan],
    ) -> bool:
        """Certify the complete base-plus-extra source against current curves."""
        return _within_paths(
            (*self._base_paths, *extra_paths),
            spans,
            self._tolerance_mm,
            self,
        )


def within_paths(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
) -> bool:
    """Certify bidirectional Hausdorff distance without matching parameters.

    Paths remain disconnected; no artificial segment joins their endpoints.
    False includes inconclusive bounds or exhausted resources. Topology and
    exact joins are checked by the caller. No sampled estimate authorizes a fit.
    """
    return _within_paths(source_paths, spans, tolerance_mm)


def _reverse_certificate(
    segment: _Segment,
    target: _Capsules,
    budget: _Budget,
    proof_cache: ForwardProofCache | None,
) -> bool:
    if proof_cache is None:
        return _source_to_proxy(segment, target, budget)
    receipt: set[c.RationalControls] = set()
    if not _source_to_proxy(segment, target, budget, receipt):
        return False
    # Partial subdivision success from a False result never becomes a receipt.
    proof_cache._remember_reverse(segment, receipt)
    return True


def _within_paths(
    source_paths: Sequence[Sequence[c.Point2]],
    spans: Sequence[c.BezierSpan],
    tolerance_mm: float,
    forward_cache: ForwardProofCache | None = None,
) -> bool:
    if (
        not _valid_source(source_paths, tolerance_mm)
        or not spans
        or any(len(span.control_points) != 4 for span in spans)
        or any(
            len(point) != 2 or not all(isfinite(v) for v in point)
            for span in spans
            for point in span.control_points
        )
    ):
        raise ValueError(
            "distance certification requires finite planar paths and nonnegative tolerance"
        )
    segments = tuple(
        segment
        for path in source_paths
        for segment in pairwise(tuple(c._rational(point) for point in path))
    )
    tolerance = Fraction(tolerance_mm)
    target = _Capsules(segments, [tolerance] * len(segments))
    budget = _Budget()
    controls = [c._controls(span) for span in spans]
    if forward_cache is not None:
        forward_cache._prepare(set(controls), set(segments))
    for span in controls:
        if forward_cache is not None and span in forward_cache._positive:
            continue
        if not _curve_to_source(span, target, budget):
            return False
        if forward_cache is not None and len(source_paths) == len(
            forward_cache._base_paths
        ):
            # Only the source-bound method supplies this context. Without
            # extras, this proof is against exactly its immutable base paths.
            # Cached work is not recomputed or charged to the node budget.
            forward_cache._remember_forward(span)
    coarse: _Capsules | None = None
    outer: _Capsules | None = None
    cache: dict[tuple[int, int], tuple[_Segment, ...]] = {}
    nonlinear = any(not _straight(span) for span in controls)
    owner_indices = {span: i for i, span in enumerate(controls)}
    if forward_cache is not None:
        base_count = sum(len(path) - 1 for path in forward_cache.base_paths)
        reverse_segments = (*segments[base_count:], *segments[:base_count])
    else:
        reverse_segments = segments
    for segment in reverse_segments:
        # _prepare invalidated receipts with any missing/changed actual owner.
        # Every current source primitive is still visited, including extras.
        if forward_cache is not None and segment in forward_cache._reverse:
            continue
        if coarse is None:
            coarse = _local_proxy(
                controls,
                list(range(len(controls))),
                tolerance,
                0,
                budget,
                cache,
                forward_cache,
            )
            if coarse is None:
                return False
        if _reverse_certificate(segment, coarse, budget, forward_cache):
            continue
        if budget.remaining < 0 or not nonlinear:
            return False
        if outer is None:
            outer = _Capsules(
                coarse.segments,
                [2 * tolerance - radius for radius in coarse.radii],
                coarse.owners,
            )
        if any(
            not outer.covered((point,), outer.candidates((point,))) for point in segment
        ):
            return False
        assert coarse.owners is not None
        nearby = sorted(
            {owner_indices[coarse.owners[i]] for i in outer.candidates(segment)}
        )
        if not nearby:
            return False
        for refinement in range(1, _PROXY_REFINEMENTS + 1):
            local = _local_proxy(
                controls, nearby, tolerance, refinement, budget, cache, forward_cache
            )
            if local is None:
                return False
            # A positive proof with a subset of the actual curves suffices.
            # Resume the coarse target for the next source primitive; a local
            # near-limit endpoint must not refine every distant curve.
            if _reverse_certificate(segment, local, budget, forward_cache):
                break
            if budget.remaining < 0:
                return False
        else:
            return False
    return True
