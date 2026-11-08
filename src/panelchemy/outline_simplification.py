"""Bounded reduction of approved cubic pieces, awaiting source/topology checks.

Every proposal is compared to all ORIGINAL approved pieces in its group. The
edit bound cannot accumulate through repeated merges. The caller still owns
certification against the original raster and global palette topology.
"""

# Reuse the fitting core's exact Bernstein and monotonicity predicates.
# pyright: reportPrivateUsage=false

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import atan2, gcd, hypot, isfinite
from sys import float_info
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from panelchemy import curves as c

type Interval = tuple[float, float]
type Acceptance = Callable[[c.BezierSpan, float, float], bool]
type FloatArray = NDArray[np.float64]
_MAX_GROUP_SPANS = 32


@dataclass(frozen=True)
class SimplifiedChain:
    spans: tuple[c.BezierSpan, ...]
    intervals: tuple[Interval, ...]
    input_spans: int
    output_spans: int
    attempted_merges: int
    accepted_merges: int


def _direction(span: c.BezierSpan, *, end: bool) -> c.Point2:
    p = span.control_points
    a, b = (p[2], p[3]) if end else (p[0], p[1])
    return b[0] - a[0], b[1] - a[1]


def _smooth(a: c.BezierSpan, b: c.BezierSpan) -> bool:
    if a.is_line or b.is_line:
        return False
    u, v = _direction(a, end=True), _direction(b, end=False)
    lu, lv = hypot(*u), hypot(*v)
    if not isfinite(lu) or not isfinite(lv) or min(lu, lv) == 0:
        return False
    x, y, a0, b0 = u[0] / lu, u[1] / lu, v[0] / lv, v[1] / lv
    return abs(atan2(x * b0 - y * a0, x * a0 + y * b0)) <= 1e-8


def _handle(anchor: c.Point2, inherited: c.Point2, length: float) -> c.Point2 | None:
    """Choose an exactly representable positive multiple of inherited direction."""
    if (
        not isfinite(length)
        or length <= 0
        or not all(isfinite(v) for v in (*anchor, *inherited))
    ):
        return None
    a = c._rational(anchor)
    p = c._rational(inherited)
    dx, dy = p[0] - a[0], p[1] - a[1]
    denominator = max(dx.denominator, dy.denominator)
    ix, iy = int(dx * denominator), int(dy * denominator)
    common = gcd(abs(ix), abs(iy))
    if not common:
        return None
    ix, iy = ix // common, iy // common
    # The inherited coordinates may use a much coarser dyadic grid than IEEE
    # points near the anchor. Refine only the scalar, keeping the exact ray.
    for _ in range(float_info.mant_dig):
        refined = denominator * 2
        exact = (
            a[0] + Fraction(ix, refined),
            a[1] + Fraction(iy, refined),
        )
        point = float(exact[0]), float(exact[1])
        if (
            point == anchor
            or not all(isfinite(v) for v in point)
            or c._rational(point) != exact
        ):
            break
        denominator = refined
    unit = hypot(ix, iy) / denominator
    if not isfinite(unit) or unit <= 0 or not isfinite(length / unit):
        return None
    desired = max(1, round(length / unit))
    # Coarsen only the scalar lattice until the point is exactly representable.
    # This retains the inherited tangent direction without float-rounding kinks.
    for power in range(min(64, desired.bit_length() + 1)):
        stride = 1 << power
        multiple = max(stride, ((desired + stride // 2) // stride) * stride)
        exact = (
            a[0] + Fraction(multiple * ix, denominator),
            a[1] + Fraction(multiple * iy, denominator),
        )
        point = float(exact[0]), float(exact[1])
        if all(isfinite(v) for v in point) and c._rational(point) == exact:
            return point
    return None


def _fit(
    spans: Sequence[c.BezierSpan], intervals: Sequence[Interval]
) -> c.BezierSpan | None:
    first, last = spans[0], spans[-1]
    a, d = np.asarray(first.control_points[0]), np.asarray(last.control_points[-1])
    u, v = (
        np.asarray(_direction(first, end=False)),
        np.asarray(_direction(last, end=True)),
    )
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if min(lu, lv) == 0 or not isfinite(lu) or not isfinite(lv):
        return None
    u, v = u / lu, v / lv
    start, stop = intervals[0][0], intervals[-1][1]
    duration = stop - start
    matrices: list[FloatArray] = []
    targets: list[FloatArray] = []
    local = np.linspace(0, 1, 7)
    quadrature = np.ones(7)
    quadrature[[0, -1]] = 0.5
    for span, (left, right) in zip(spans, intervals, strict=True):
        t = np.clip((left + (right - left) * local - start) / duration, 0, 1)
        b0, b1, b2, b3 = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t**3
        p = np.asarray(span.control_points)
        q = local[:, None]
        values = cast(
            "FloatArray",
            (1 - q) ** 3 * p[0]
            + 3 * (1 - q) ** 2 * q * p[1]
            + 3 * (1 - q) * q * q * p[2]
            + q**3 * p[3],
        )
        weight = np.sqrt(quadrature * ((right - left) / duration))
        matrix = np.stack((b1[:, None] * u, -b2[:, None] * v), axis=2)
        matrices.append((matrix * weight[:, None, None]).reshape(-1, 2))
        base = (b0 + b1)[:, None] * a + (b2 + b3)[:, None] * d
        targets.append(((values - base) * weight[:, None]).ravel())
    try:
        handles, _ = nnls(np.vstack(matrices), np.concatenate(targets), maxiter=30)
    except RuntimeError:
        return None
    b = _handle(
        first.control_points[0],
        first.control_points[1],
        float(cast("np.float64", handles[0])),
    )
    e = _handle(
        last.control_points[-1],
        last.control_points[2],
        float(cast("np.float64", handles[1])),
    )
    if b is None or e is None:
        return None
    return c.BezierSpan(
        (first.control_points[0], b, e, last.control_points[-1]),
        False,
        max(s.certified_error_mm for s in spans),
    )


def _within(
    candidate: c.BezierSpan,
    originals: Sequence[c.BezierSpan],
    intervals: Sequence[Interval],
    budget: float,
) -> bool:
    start, stop = Fraction(intervals[0][0]), Fraction(intervals[-1][1])
    controls = c._controls(candidate)
    for span, (left, right) in zip(originals, intervals, strict=True):
        restricted = c._restrict(
            controls,
            (Fraction(left) - start) / (stop - start),
            (Fraction(right) - start) / (stop - start),
        )
        difference = cast(
            "c.RationalControls",
            tuple(
                (p[0] - q[0], p[1] - q[1])
                for p, q in zip(restricted, c._controls(span), strict=True)
            ),
        )
        if not c._bounded(difference, Fraction(budget) ** 2):
            return False
    return True


def simplify(
    spans: Sequence[c.BezierSpan],
    intervals: Sequence[Interval],
    edit_budget_mm: float,
    *,
    accept: Acceptance | None = None,
) -> SimplifiedChain:
    """Replace bounded smooth groups with fewer cubics; preserve original joins."""
    if (
        isinstance(edit_budget_mm, bool)
        or not isfinite(edit_budget_mm)
        or edit_budget_mm < 0
    ):
        raise ValueError("simplification budget must be finite and nonnegative")
    if not spans or len(spans) != len(intervals):
        raise ValueError(
            "simplification needs a nonempty chain with matching intervals"
        )
    if any(not isfinite(v) for pair in intervals for v in pair) or any(
        a >= b for a, b in intervals
    ):
        raise ValueError("source intervals must be finite and increasing")
    if any(not isfinite(v) for s in spans for p in s.control_points for v in p):
        raise ValueError("source cubic controls must be finite")
    if any(
        intervals[i - 1][1] != intervals[i][0]
        or spans[i - 1].control_points[-1] != spans[i].control_points[0]
        for i in range(1, len(spans))
    ):
        raise ValueError("source intervals and cubic spans must join exactly")
    if edit_budget_mm == 0:
        return SimplifiedChain(
            tuple(spans), tuple(intervals), len(spans), len(spans), 0, 0
        )
    output: list[c.BezierSpan] = []
    times: list[Interval] = []
    attempted = accepted = 0
    start = 0
    while start < len(spans):
        best = spans[start]
        stop = start + 1
        while (
            stop < len(spans)
            and stop - start < _MAX_GROUP_SPANS
            and _smooth(spans[stop - 1], spans[stop])
        ):
            attempted += 1
            originals, parameters = spans[start : stop + 1], intervals[start : stop + 1]
            proposal = _fit(originals, parameters)
            if (
                proposal is None
                or proposal.control_points[0] == proposal.control_points[1]
                or proposal.control_points[2] == proposal.control_points[3]
                or not c._monotone(proposal)
                or not _within(proposal, originals, parameters, edit_budget_mm)
                or (
                    accept is not None
                    and not accept(proposal, parameters[0][0], parameters[-1][1])
                )
            ):
                break
            best = proposal
            stop += 1
            accepted += 1
        output.append(best)
        times.append((intervals[start][0], intervals[stop - 1][1]))
        start = stop
    return SimplifiedChain(
        tuple(output), tuple(times), len(spans), len(output), attempted, accepted
    )
