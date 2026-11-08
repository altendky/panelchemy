"""Continuous geometric distance allows sliding, while retaining both directions."""

# Synthetic fixtures use the conversion core's line constructor.
# pyright: reportPrivateUsage=false

from fractions import Fraction
from math import cos, hypot, inf, nextafter, pi, sin
from random import Random
from typing import cast

import pytest

from panelchemy import geometric_distance
from panelchemy.curves import (
    BezierSpan,
    Controls,
    _line,
)
from panelchemy.geometric_distance import within_distance, within_paths


def test_disconnected_paths_do_not_gain_a_fictitious_connecting_segment() -> None:
    paths = (((0.0, 0.0), (1.0, 0.0)), ((4.0, 0.0), (5.0, 0.0)))
    matching = (_line((0, 0), (1, 0)), _line((4, 0), (5, 0)))
    assert within_paths(paths, matching, 0)
    assert not within_paths(paths, (*matching, _line((1, 0), (4, 0))), 0.2)
    assert not within_paths(paths, matching[:1], 0.2)


def test_paths_share_the_original_total_distance_budget() -> None:
    paths = (((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)))
    shifted = (_line((0, 0.25), (1, 0.25)), _line((1, 0.25), (1, 1.25)))
    assert within_paths(paths, shifted, 0.25)
    assert not within_paths(paths, shifted, nextafter(0.25, 0))


@pytest.mark.parametrize("paths", [(), ((),), (((0.0, 0.0),),)])
def test_invalid_disconnected_paths_are_rejected(
    paths: tuple[tuple[tuple[float, float], ...], ...],
) -> None:
    with pytest.raises(ValueError, match="finite planar paths"):
        _ = within_paths(paths, (_line((0, 0), (1, 0)),), 0.2)


@pytest.mark.parametrize("radius", [0.0, 0.2])
def test_spatial_index_selects_exactly_the_old_outward_boxes(
    monkeypatch: pytest.MonkeyPatch, radius: float
) -> None:
    random = Random(83)
    segments = [
        (
            (Fraction(i, 7), Fraction(random.randrange(-10, 11), 3)),
            (Fraction(i, 7), Fraction(random.randrange(-10, 11), 3)),
        )
        for i in range(80)
    ]
    monkeypatch.setattr(geometric_distance, "_INDEX_MIN_SEGMENTS", 10_000)
    linear = geometric_distance._Capsules(segments, [Fraction(radius)] * len(segments))
    monkeypatch.setattr(geometric_distance, "_INDEX_MIN_SEGMENTS", 0)
    indexed = geometric_distance._Capsules(segments, [Fraction(radius)] * len(segments))
    assert indexed.index is not None
    for segment in segments:
        queries = [segment, (segment[0],), (segment[1],)]
        queries.extend(
            ((point[0] + dx, point[1] + dy),)
            for point in segment
            for dx, dy in (
                (Fraction(radius), Fraction(0)),
                (Fraction(0), Fraction(radius)),
            )
        )
        for query in queries:
            assert indexed.candidates(query) == linear.candidates(query)


def test_spatial_index_retains_continuous_certificate_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tuple((cos(i * pi / 400), sin(i * pi / 400)) for i in range(201))
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    results: list[tuple[bool, ...]] = []
    for threshold in (10_000, 0):
        monkeypatch.setattr(geometric_distance, "_INDEX_MIN_SEGMENTS", threshold)
        results.append(
            tuple(within_distance(source, (curve,), e) for e in (0.001, 0.0001))
        )
    assert results == [(True, False), (True, False)]


def test_nonfinite_outward_box_uses_the_original_linear_fallback() -> None:
    value = Fraction(float.fromhex("0x1.fffffffffffffp+1023"))
    target = geometric_distance._Capsules(
        [((value, Fraction(0)), (value, Fraction(1)))] * 40, [Fraction(0)] * 40
    )
    assert target.index is None
    assert target.candidates(((value, Fraction(0)),)) == list(range(40))


def test_near_limit_curved_source_gets_a_more_accurate_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tuple(
        (1.195 * cos(i * pi / 400), 1.195 * sin(i * pi / 400)) for i in range(201)
    )
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    monkeypatch.setattr(geometric_distance, "_PROXY_REFINEMENTS", 0)
    assert not within_distance(source, (curve,), 0.2)
    monkeypatch.setattr(geometric_distance, "_PROXY_REFINEMENTS", 3)
    assert within_distance(source, (curve,), 0.2)
    assert not within_distance(source, (curve,), 0.194)


def test_proxy_refinement_does_not_accept_actual_out_of_tolerance_source() -> None:
    source = tuple(
        (1.205 * cos(i * pi / 400), 1.205 * sin(i * pi / 400)) for i in range(201)
    )
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    assert not within_distance(source, (curve,), 0.2)


def test_tangential_reparameterization_accepts_identical_line() -> None:
    curve = BezierSpan(((0, 0), (0.001, 0), (0.002, 0), (10, 0)), False)
    assert within_distance(((0, 0), (0.2, 0), (4.7, 0), (10, 0)), (curve,), 0)


@pytest.mark.parametrize("tolerance", [0.0, 1e-20, 0.25])
def test_source_line_can_match_multiple_curve_pieces(tolerance: float) -> None:
    source = ((0.0, 0.0), (1.0, 0.0))
    curves = (
        _line((0, tolerance), (0.27, tolerance)),
        _line((0.27, tolerance), (1, tolerance)),
    )
    assert within_distance(source, curves, tolerance)
    if tolerance:
        assert not within_distance(source, curves, nextafter(tolerance, 0))


def test_extra_fitted_excursion_is_rejected() -> None:
    curve = BezierSpan(((0, 0), (1, 2), (3, 2), (4, 0)), False)
    assert not within_distance(((0, 0), (4, 0)), (curve,), 0.2)


def test_reverse_direction_rejects_a_source_feature_left_behind() -> None:
    source = ((0, 0), (2, 0), (2, 2), (2, 0), (4, 0))
    assert not within_distance(source, (_line((0, 0), (4, 0)),), 0.2)


def test_source_interval_can_be_covered_by_a_neighboring_piece() -> None:
    source = ((0, 0), (1, 0), (2, 0), (2, 1))
    curves = (_line((0, 0), (1.7, 0)), _line((1.7, 0), (2, 0)), _line((2, 0), (2, 1)))
    assert within_distance(source, curves, 1e-20)


def test_smooth_quarter_circle_approximates_original_polyline() -> None:
    source = tuple((cos(i * pi / 400), sin(i * pi / 400)) for i in range(201))
    source = ((1.0, 0.0), *source[1:-1], (0.0, 1.0))
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    assert within_distance(source, (curve,), 0.001)
    assert not within_distance(source, (curve,), 0.0001)


def test_node_budget_exhaustion_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geometric_distance, "_MAX_NODES", 0)
    assert not within_distance(((0, 0), (1, 0)), (_line((0, 0), (1, 0)),), 0.2)


def test_depth_budget_exhaustion_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    source = tuple((cos(i * pi / 80), sin(i * pi / 80)) for i in range(41))
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    assert within_distance(source, (curve,), 0.002)
    monkeypatch.setattr(geometric_distance, "_MAX_DEPTH", 0)
    assert not within_distance(source, (curve,), 0.002)


def test_curved_reverse_coverage_reserves_proxy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    source: list[tuple[float, float]] = []
    for i in range(257):
        t = i / 256
        a = 1 - t
        point = (
            a**3 + 3 * a * a * t + 3 * a * t * t * k,
            3 * a * a * t * k + 3 * a * t * t + t**3,
        )
        if i == 8:
            # This inward source tip lies more than 0.05 from the curve, but
            # less than 0.05 from its inward chord proxy. Spending the full
            # source budget on that proxy would incorrectly accept the tip.
            radius = hypot(*point)
            point = point[0] * 0.9495 / radius, point[1] * 0.9495 / radius
        source.append(point)
    reverse_reached = False
    original = geometric_distance._source_to_proxy

    def reverse(
        segment: geometric_distance._Segment,
        target: geometric_distance._Capsules,
        budget: geometric_distance._Budget,
    ) -> bool:
        nonlocal reverse_reached
        reverse_reached = True
        return original(segment, target, budget)

    monkeypatch.setattr(geometric_distance, "_source_to_proxy", reverse)
    assert not within_distance(source, (curve,), 0.05)
    assert reverse_reached  # Forward coverage and nonlinear proxy construction passed.


def test_curved_distance_survives_translation_and_scaling() -> None:
    scale = 7.5
    dx, dy = 17.25, -31.5
    source = tuple(
        (dx + scale * cos(i * pi / 200), dy + scale * sin(i * pi / 200))
        for i in range(101)
    )
    k = 0.5522847498307936
    controls: Controls = tuple(  # pyright: ignore[reportAssignmentType]
        (dx + scale * x, dy + scale * y) for x, y in ((1, 0), (1, k), (k, 1), (0, 1))
    )
    curve = BezierSpan(controls, False)
    assert within_distance(source, (curve,), scale * 0.001)
    assert not within_distance(source, (curve,), scale * 0.0001)


@pytest.mark.parametrize("tolerance", [-1, inf, float("nan"), True])
def test_invalid_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite planar"):
        _ = within_distance(((0, 0), (1, 0)), (_line((0, 0), (1, 0)),), tolerance)


def test_malformed_cubic_is_rejected() -> None:
    malformed = BezierSpan(cast("Controls", ((0, 0), (1, 0), (2, 0))), False)
    with pytest.raises(ValueError, match="finite planar"):
        _ = within_distance(((0, 0), (2, 0)), (malformed,), 0.2)
