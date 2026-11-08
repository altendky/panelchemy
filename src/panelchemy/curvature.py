"""Continuous, exact bounds on the signed curvature of individual spans.

This certifies local offset regularity. Callers separately check joins, global
topology, and the exposed height on each material boundary.
"""

from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from math import comb, isfinite

from panelchemy.curves import BezierSpan

type _Polynomial = list[Fraction]
type _Number = float | Fraction

_MAX_DEPTH = 24
_MAX_NODES = 100_000
_DEFAULT_FRACTION = Fraction(9, 10)


def _rational(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"{name} must be a finite real number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return Fraction(value)


def _signed_direction(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1):
        raise ValueError("curvature direction must be -1, 0, or 1")
    return value


def _product(a: Sequence[Fraction], b: Sequence[Fraction]) -> _Polynomial:
    """Multiply Bernstein polynomials without converting to power coefficients."""
    m, n = len(a) - 1, len(b) - 1
    return [
        sum(
            (
                a[i] * b[k - i] * Fraction(comb(m, i) * comb(n, k - i), comb(m + n, k))
                for i in range(max(0, k - n), min(m, k) + 1)
            ),
            Fraction(0),
        )
        for k in range(m + n + 1)
    ]


def _elevate(a: Sequence[Fraction], degree: int) -> _Polynomial:
    """Represent the same polynomial in a higher-degree Bernstein basis."""
    n = len(a) - 1
    return [
        sum(
            (
                a[i] * Fraction(comb(n, i) * comb(degree - n, k - i), comb(degree, k))
                for i in range(max(0, k - degree + n), min(n, k) + 1)
            ),
            Fraction(0),
        )
        for k in range(degree + 1)
    ]


def _split(a: Sequence[Fraction]) -> tuple[_Polynomial, _Polynomial]:
    """Exact midpoint de Casteljau subdivision of a scalar polynomial."""
    current: _Polynomial = list(a)
    left, right = [current[0]], [current[-1]]
    while len(current) > 1:
        current = [(x + y) / 2 for x, y in pairwise(current)]
        left.append(current[0])
        right.append(current[-1])
    return left, right[::-1]


def within_curvature(
    span: BezierSpan,
    offset_mm: _Number,
    inward_sign: int,
    fraction: _Number = _DEFAULT_FRACTION,
) -> bool:
    """Prove regular speed and ``offset * signed curvature <= fraction``.

    Sign +1 checks positive curvature, -1 negative curvature, and 0 both.
    The default leaves ten percent headroom before local offset collapse.
    Bounds apply continuously, including endpoints; inconclusive bounds or
    exhausted subdivision resources return False. No sampled acceptance is used.
    """
    offset = _rational(offset_mm, "curvature offset")
    margin = _rational(fraction, "curvature fraction")
    if offset < 0 or not 0 < margin <= 1:
        raise ValueError("curvature needs a nonnegative offset and fraction in (0, 1]")
    direction = _signed_direction(inward_sign)
    if len(span.control_points) != 4 or any(len(p) != 2 for p in span.control_points):
        raise ValueError("curvature needs four planar cubic controls")
    points = [
        tuple(_rational(v, "curvature coordinate") for v in p)
        for p in span.control_points
    ]
    if span.is_line:
        # Line metadata uses only its endpoints, matching native construction.
        return points[0] != points[-1]
    if _MAX_NODES < 1:
        return False
    velocity = [[3 * (b[j] - a[j]) for a, b in pairwise(points)] for j in (0, 1)]
    acceleration = [[2 * (b - a) for a, b in pairwise(axis)] for axis in velocity]
    margin_squared, offset_squared = margin**2, offset**2
    # Endpoints are necessary conditions, so an exact violation can reject the
    # span before constructing any Bernstein products. Zero speed is forbidden
    # even when that curvature sign or a zero offset would otherwise be safe.
    for i in (0, -1):
        vx, vy = velocity[0][i], velocity[1][i]
        ax, ay = acceleration[0][i], acceleration[1][i]
        speed_squared = vx**2 + vy**2
        signed_cross = vx * ay - vy * ax
        if speed_squared == 0 or (
            (direction == 0 or direction * signed_cross > 0)
            and margin_squared * speed_squared**3 < offset_squared * signed_cross**2
        ):
            return False
    speed = [
        x + y
        for x, y in zip(
            _product(velocity[0], velocity[0]),
            _product(velocity[1], velocity[1]),
            strict=True,
        )
    ]
    cross = [
        x - y
        for x, y in zip(
            _product(velocity[0], acceleration[1]),
            _product(velocity[1], acceleration[0]),
            strict=True,
        )
    ]
    # Bernstein coefficient extrema enclose the entire scalar polynomial.
    # These coarse certificates imply acceptance at the original proof's root;
    # undecided cases retain the unchanged degree-twelve subdivision below.
    minimum_speed = min(speed)
    if minimum_speed > 0:
        if (direction == 1 and max(cross) <= 0) or (
            direction == -1 and min(cross) >= 0
        ):
            return True
        maximum_cross = max(abs(value) for value in cross)
        if margin_squared * minimum_speed**3 >= offset_squared * maximum_cross**2:
            return True
    speed_cubed = _product(_product(speed, speed), speed)
    cross_squared = _elevate(_product(cross, cross), len(speed_cubed) - 1)
    # kappa = cross(r', r'') / |r'|^3. With positive speed the
    # squared inequality is equivalent to this degree-twelve polynomial bound.
    bound = [
        margin_squared * s - offset_squared * n
        for s, n in zip(speed_cubed, cross_squared, strict=True)
    ]
    pending = [(speed, cross, bound, 0)]
    visited = 0
    while pending:
        s, n, b, depth = pending.pop()
        visited += 1
        if visited > _MAX_NODES:
            return False
        opposite = (direction == 1 and max(n) <= 0) or (direction == -1 and min(n) >= 0)
        positive_speed = min(s) > 0
        if positive_speed and (opposite or min(b) >= 0):
            continue
        if max(s) <= 0 or (positive_speed and max(b) < 0 and not opposite):
            return False
        if depth >= _MAX_DEPTH:
            return False
        sl, sr = _split(s)
        nl, nr = _split(n)
        bl, br = _split(b)
        pending.extend(((sl, nl, bl, depth + 1), (sr, nr, br, depth + 1)))
    return True
