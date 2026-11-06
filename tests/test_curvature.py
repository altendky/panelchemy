"""Curvature acceptance requires continuous exact bounds, never samples."""

from fractions import Fraction
from typing import cast

import pytest

from panelchemy import curvature
from panelchemy.curves import BezierSpan, Controls

F = Fraction


def _span(points: tuple[tuple[float | Fraction, float | Fraction], ...]) -> BezierSpan:
    return BezierSpan(cast("Controls", points), False)


def _parabola() -> BezierSpan:
    # Exact cubic representation of x=t, y=t²; maximum curvature is2 at t=0.
    return _span(((F(0), F(0)), (F(1, 3), F(0)), (F(2, 3), F(1, 3)), (F(1), F(1))))


def test_analytic_parabola_and_exact_ninety_percent_threshold() -> None:
    span = _parabola()
    assert curvature.within_curvature(span, F(2, 5), 0)
    assert curvature.within_curvature(span, F(9, 20), 0)
    assert not curvature.within_curvature(span, F(1, 2), 0)
    assert not curvature.within_curvature(span, F(9, 20) + F(1, 10**18), 0)


def test_signed_curvature_and_reversing_parameter_direction() -> None:
    span = _parabola()
    reversed_span = BezierSpan(
        cast("Controls", tuple(reversed(span.control_points))), False
    )
    assert not curvature.within_curvature(span, 1, 1)
    assert curvature.within_curvature(span, 1, -1)
    assert curvature.within_curvature(reversed_span, 1, 1)
    assert not curvature.within_curvature(reversed_span, 1, -1)


def test_translation_and_scale_preserve_the_curvature_offset_product() -> None:
    span = _parabola()
    translated = _span(
        tuple((F(x) + 10000, F(y) - 20000) for x, y in span.control_points)
    )
    scaled = _span(tuple((F(x) / 1000, F(y) / 1000) for x, y in span.control_points))
    assert curvature.within_curvature(translated, F(2, 5), 0)
    assert curvature.within_curvature(scaled, F(2, 5000), 0)
    assert not curvature.within_curvature(scaled, F(1, 2000), 0)


def test_lines_zero_offset_and_stationary_curves() -> None:
    straight = _span(((0, 0), (1, 0), (2, 0), (3, 0)))
    assert curvature.within_curvature(straight, 1000, 0)
    assert curvature.within_curvature(
        BezierSpan(straight.control_points, True), 1000, 0
    )
    stationary = _span(((0, 0),) * 4)
    assert not curvature.within_curvature(stationary, 0, 0)
    assert not curvature.within_curvature(
        BezierSpan(stationary.control_points, True), 0, 0
    )
    # Zero speed at an interior point also fails even with no offset.
    assert not curvature.within_curvature(
        _span(((0, 0), (1, 0), (-1, 0), (0, 0))), 0, 0
    )
    assert curvature.within_curvature(_parabola(), 0, 0)


def test_float_fraction_cannot_make_an_exact_violation_pass() -> None:
    threshold = 0.9
    offset = F(threshold) / 2 + F(1, 10**18)
    assert 2 * offset > F(threshold)
    assert not curvature.within_curvature(_parabola(), offset, 0, threshold)
    assert not curvature.within_curvature(_parabola(), offset, 0, F(threshold))


@pytest.mark.parametrize("limit", ["_MAX_DEPTH", "_MAX_NODES"])
def test_resource_limits_reject_inconclusive_bounds(
    monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    # This regular S curve has negative initial Bernstein speed coefficients;
    # subdivision is needed to prove its strictly positive speed.
    span = _span(((0, 0), (0, 1), (1, 0), (1, 1)))
    assert curvature.within_curvature(span, 0, 0)
    monkeypatch.setattr(curvature, limit, 0 if limit == "_MAX_DEPTH" else 1)
    assert not curvature.within_curvature(span, 0, 0)


@pytest.mark.parametrize("offset", [-1, True, float("nan"), float("inf")])
def test_invalid_offsets_raise(offset: float) -> None:
    with pytest.raises(ValueError, match="curvature"):
        _ = curvature.within_curvature(_parabola(), offset, 0)


@pytest.mark.parametrize("fraction", [0, -1, 1.01, True, float("nan"), float("inf")])
def test_invalid_fraction_raises(fraction: float) -> None:
    with pytest.raises(ValueError, match="curvature"):
        _ = curvature.within_curvature(_parabola(), 1, 0, fraction)


@pytest.mark.parametrize("sign", [-2, 2, True, 1.0])
def test_invalid_curvature_sign_raises(sign: int) -> None:
    with pytest.raises(ValueError, match="direction"):
        _ = curvature.within_curvature(_parabola(), 1, sign)


def test_nonfinite_controls_are_rejected_even_for_line_metadata() -> None:
    span = _span(((0, 0), (float("inf"), 0), (2, 0), (3, 0)))
    for is_line in (False, True):
        with pytest.raises(ValueError, match="coordinate"):
            _ = curvature.within_curvature(
                BezierSpan(span.control_points, is_line), 1, 0
            )
