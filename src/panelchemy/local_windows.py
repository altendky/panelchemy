"""Index windows and source guidance for local repairs of approved curves."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np
from numpy.typing import NDArray

from panelchemy.curves import BezierSpan, Point2

type FloatArray = NDArray[np.float64]
type Interval = tuple[float, float]


@dataclass(frozen=True)
class Window:
    first: int
    count: int

    def indices(self, size: int) -> tuple[int, ...]:
        if (
            type(size) is not int
            or size < 1
            or type(self.first) is not int
            or type(self.count) is not int
            or not 0 <= self.first < size
            or not 1 <= self.count <= size
        ):
            raise ValueError("local window indices must fit the chain")
        return tuple((self.first + i) % size for i in range(self.count))


def windows(
    size: int,
    bad_indices: Iterable[int],
    *,
    periodic: bool,
    neighbors: int,
) -> tuple[Window, ...]:
    """Merge connected padded groups, including the freely movable seam."""
    bad = set(bad_indices)
    if (
        type(size) is not int
        or size < 1
        or type(periodic) is not bool
        or type(neighbors) is not int
        or neighbors < 0
        or any(type(i) is not int or not 0 <= i < size for i in bad)
    ):
        raise ValueError("window planning requires valid chain indices and padding")
    if not bad:
        return ()
    padding = min(neighbors, size)
    selected = {
        j % size if periodic else j
        for i in bad
        for j in range(i - padding, i + padding + 1)
        if periodic or 0 <= j < size
    }
    if len(selected) == size:
        return (Window(0, size),)
    cut = next(i for i in range(size) if i not in selected) if periodic else -1
    order = tuple((cut + 1 + i) % size for i in range(size))
    result: list[Window] = []
    run: list[int] = []
    for index in (*order, -1):
        if index in selected:
            run.append(index)
        elif run:
            result.append(Window(run[0], len(run)))
            run = []
    first_bad = min(bad)
    return tuple(
        sorted(result, key=lambda w: (first_bad not in w.indices(size), w.first))
    )


def _partition(intervals: Sequence[Interval], size: int) -> None:
    if (
        len(intervals) != size
        or not intervals
        or intervals[0][0] != 0
        or intervals[-1][1] != 1
        or any(not np.isfinite((a, b)).all() or a >= b for a, b in intervals)
        or any(a[1] != b[0] for a, b in pairwise(intervals))
    ):
        raise ValueError("local repair requires a complete parameter partition")


@dataclass(frozen=True)
class Guide:
    """Interpolated samples guide search; the unchanged full source proves it."""

    window: Window
    total_spans: int
    source_points: FloatArray
    parameters: FloatArray
    endpoint_positions: FloatArray
    tangent_directions: FloatArray
    inherited_handles: tuple[Point2, Point2]
    start_t: float
    end_t: float
    end_t_original: float
    seam: float | None

    def to_local(self, parameter: float) -> float | None:
        if not np.isfinite(parameter) or not 0 <= parameter <= 1:
            return None
        if self.seam is not None:
            if parameter < self.start_t and parameter > self.end_t_original:
                return None
        elif parameter < self.start_t or parameter > self.end_t_original:
            return None
        lifted = parameter + int(self.seam is not None and parameter < self.start_t)
        if lifted < self.start_t or lifted > self.end_t:
            return None
        return float((lifted - self.start_t) / (self.end_t - self.start_t))

    def splice(
        self,
        original_spans: Sequence[BezierSpan],
        original_intervals: Sequence[Interval],
        replacement_spans: Sequence[BezierSpan],
        local_intervals: Sequence[Interval],
    ) -> tuple[list[BezierSpan], list[Interval]]:
        """Restore original parameter references and leave outside spans intact."""
        if len(original_spans) != self.total_spans or not replacement_spans:
            raise ValueError("local replacement must match its original chain")
        _partition(original_intervals, self.total_spans)
        _partition(local_intervals, len(replacement_spans))
        first, count = self.window.first, self.window.count
        stop = (first + count) % self.total_spans
        pins = cast("list[list[float]]", self.endpoint_positions.tolist())
        if (
            original_intervals[first][0] != self.start_t
            or original_intervals[(first + count - 1) % self.total_spans][1]
            != self.end_t_original
            or replacement_spans[0].control_points[0] != tuple(pins[0])
            or replacement_spans[-1].control_points[-1] != tuple(pins[1])
            or any(
                a.control_points[-1] != b.control_points[0]
                for a, b in pairwise(replacement_spans)
            )
        ):
            raise ValueError("local replacement changed a pinned endpoint or join")
        mapped = {
            t: self.start_t + t * (self.end_t - self.start_t)
            for interval in local_intervals
            for t in interval
        }
        mapped[0.0], mapped[1.0] = self.start_t, self.end_t
        if self.seam is None:
            end = first + count
            mapped[1.0] = self.end_t_original
            spans = [*original_spans[:first], *replacement_spans, *original_spans[end:]]
            intervals = [
                *original_intervals[:first],
                *((mapped[a], mapped[b]) for a, b in local_intervals),
                *original_intervals[end:],
            ]
        else:
            if self.seam not in {b for _, b in local_intervals[:-1]}:
                raise ValueError("wrapped replacement must split at the original seam")
            seam = next(i for i, (_, b) in enumerate(local_intervals) if b == self.seam)
            mapped[self.seam] = 1.0
            before = [(mapped[a], mapped[b]) for a, b in local_intervals[: seam + 1]]
            after = [
                (mapped[a] - 1, mapped[b] - 1) for a, b in local_intervals[seam + 1 :]
            ]
            after[0] = (0.0, after[0][1])
            after[-1] = (after[-1][0], self.end_t_original)
            spans = [
                *replacement_spans[seam + 1 :],
                *original_spans[stop:first],
                *replacement_spans[: seam + 1],
            ]
            intervals = [*after, *original_intervals[stop:first], *before]
        _partition(intervals, len(spans))
        if any(a.control_points[-1] != b.control_points[0] for a, b in pairwise(spans)):
            raise ValueError("spliced repair lost an exact curve join")
        return spans, intervals


def prepare(
    window: Window,
    source_points: Sequence[Point2],
    parameters: FloatArray,
    spans: Sequence[BezierSpan],
    intervals: Sequence[Interval],
    *,
    periodic: bool = False,
) -> Guide:
    """Prepare a subchain or entire pinned chain with inherited endpoint rays."""
    size = len(spans)
    indices = window.indices(size)
    _partition(intervals, size)
    points: FloatArray = np.asarray(source_points, dtype=np.float64)
    if (
        (window.count == size and (window.first != 0 or periodic))
        or type(periodic) is not bool
        or points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or parameters.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(parameters).all()
        or parameters[0] != 0
        or parameters[-1] != 1
        or not np.all(np.diff(parameters) > 0)
        or any(
            len(span.control_points) != 4 or not np.isfinite(span.control_points).all()
            for span in spans
        )
    ):
        raise ValueError("local guide requires finite source data and a proper window")
    wrapped = window.first + window.count > size
    if wrapped and (
        not periodic
        or not np.array_equal(
            cast("FloatArray", points[0]), cast("FloatArray", points[-1])
        )
    ):
        raise ValueError("only a freely periodic source can cross its seam")
    first, last = spans[indices[0]], spans[indices[-1]]
    start, end_original = intervals[indices[0]][0], intervals[indices[-1]][1]
    end = end_original + int(wrapped)
    extended = (
        np.concatenate((parameters, parameters[1:] + 1)) if wrapped else parameters
    )
    references = np.concatenate((parameters, parameters[1:])) if wrapped else parameters
    interior = (extended > start) & (extended < end)
    times = np.concatenate(([start], extended[interior], [end]))
    # Preserve original source parameter references; adding/subtracting one
    # need not round-trip a nonbinary seam-adjacent value.
    original_times = np.concatenate(([start], references[interior], [end_original]))
    local_source = np.column_stack(
        [np.interp(original_times, parameters, points[:, axis]) for axis in (0, 1)]
    )
    local_parameters = (times - start) / (end - start)
    local_parameters[0], local_parameters[-1] = 0.0, 1.0
    pins = np.asarray(
        (first.control_points[0], last.control_points[-1]), dtype=np.float64
    )
    inherited = (
        first.control_points[-1] if first.is_line else first.control_points[1],
        last.control_points[0] if last.is_line else last.control_points[2],
    )
    directions = np.asarray(
        (np.asarray(inherited[0]) - pins[0], pins[1] - inherited[1]), dtype=np.float64
    )
    lengths = cast("FloatArray", np.hypot(directions[:, 0], directions[:, 1]))
    if not np.isfinite(pins).all() or not np.isfinite(lengths).all() or 0.0 in lengths:
        raise ValueError("local repair has an unavailable endpoint tangent")
    directions /= lengths[:, None]
    return Guide(
        window,
        size,
        local_source,
        local_parameters,
        pins,
        directions,
        inherited,
        start,
        end,
        end_original,
        (1 - start) / (end - start) if wrapped else None,
    )
