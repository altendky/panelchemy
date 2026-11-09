"""Radius reuse retains exact inputs, strict validation, and fresh ownership."""

# pyright: reportPrivateUsage=false

import gc
import weakref
from dataclasses import replace
from fractions import Fraction
from math import sqrt
from typing import Never, cast

import pytest
from shapely.geometry import box

from panelchemy import boundary_tapers as bt
from panelchemy import curvature as cu
from panelchemy.curves import (
    BezierSpan,
    Controls,
    CurveContours,
    CurvePolygon,
    CurveRing,
    polygon_lines,
)


def arch() -> BezierSpan:
    return BezierSpan(((0, 0), (0, 1), (1, 1), (1, 0)), False)


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("an identical proof should have been reused")


def reverse(ring: CurveRing) -> CurveRing:
    return CurveRing(
        tuple(
            replace(
                s, control_points=cast("Controls", tuple(reversed(s.control_points)))
            )
            for s in reversed(ring.spans)
        )
    )


def selected() -> CurveContours:
    # Four cubic quadrants around a shared orange island / white hole.
    k = 4 * (sqrt(2) - 1) / 3
    quadrant = ((1.0, 0.0), (1.0, k), (k, 1.0), (0.0, 1.0))
    spans: list[BezierSpan] = []
    for _ in range(4):
        spans.append(
            BezierSpan(
                cast(
                    "Controls", tuple((5 + 0.3 * x, 5 + 0.3 * y) for x, y in quadrant)
                ),
                False,
            )
        )
        quadrant = tuple((-y, x) for x, y in quadrant)
    ring = CurveRing(tuple(spans))
    return CurveContours(
        polygon_lines(box(0, 0, 10, 10)),
        (CurvePolygon(ring),),
        (replace(polygon_lines(box(1, 1, 9, 9)), interiors=(reverse(ring),)),),
        0.1,
        (0, 0, 100, 100),
    )


def failures(
    curves: CurveContours,
    runs: tuple[float, float, float],
    cache: bt.ProofCache | None = None,
) -> dict[int, tuple[BezierSpan, bt.BoundaryTaper]]:
    try:
        bt.require_curvature(curves, runs, cache)
    except bt.CurvatureLimitError as error:
        assert error.contours is curves
        return error.failures
    return {}


def test_reuses_both_proved_and_unproved_scalar_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, span = bt.ProofCache(), arch()
    assert cache.within_curvature(span, 0.1, -1)
    assert not cache.within_curvature(span, 1, -1)
    monkeypatch.setattr(cu, "within_curvature", unexpected)
    assert cache.within_curvature(span, 0.1, -1)
    assert not cache.within_curvature(span, 1, -1)


def test_scalar_key_covers_geometry_line_role_offset_sign_and_exact_margin() -> None:
    cache, span = bt.ProofCache(), arch()
    assert not cache.within_curvature(span, 1, -1)
    assert cache.within_curvature(span, 0.1, -1)
    assert cache.within_curvature(span, 1, 1)
    assert cache.within_curvature(replace(span, is_line=True), 1, -1)
    larger = replace(span, control_points=((0, 0), (0, 10), (10, 10), (10, 0)))
    assert cache.within_curvature(larger, 1, -1)
    assert cache.within_curvature(span, Fraction(1, 10), -1, Fraction(9, 10))
    assert not cache.within_curvature(span, Fraction(1, 10), -1, Fraction(1, 10))
    flipped = replace(
        span, control_points=cast("Controls", tuple(reversed(span.control_points)))
    )
    assert cache.within_curvature(flipped, 1, -1)
    assert not cache.within_curvature(flipped, 1, 1)


@pytest.mark.parametrize("policy", ["_MAX_NODES", "_MAX_DEPTH"])
def test_resource_policy_invalidates_true_and_false(
    monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    span = BezierSpan(((0, 0), (0, 1), (1, 0), (1, 1)), False)
    cache = bt.ProofCache()
    assert cache.within_curvature(span, 0, 0)
    previous = cu._MAX_NODES if policy == "_MAX_NODES" else cu._MAX_DEPTH
    monkeypatch.setattr(cu, policy, 0)
    assert not cache.within_curvature(span, 0, 0)
    assert not cache.within_curvature(span, 0, 0)
    monkeypatch.setattr(cu, policy, previous)
    assert cache.within_curvature(span, 0, 0)


@pytest.mark.parametrize("line", [False, True])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_invalid_controls_are_validated_before_a_warm_lookup(
    line: bool, value: object
) -> None:
    cache = bt.ProofCache()
    span = replace(arch(), is_line=line)
    assert cache.within_curvature(span, 0.1, 1)
    # A bool at this coordinate compares equal to the original integer 1.
    invalid = replace(
        span, control_points=cast("Controls", ((0, 0), (0, value), (1, 1), (1, 0)))
    )
    with pytest.raises(ValueError, match="coordinate"):
        _ = cache.within_curvature(invalid, 0.1, 1)


def test_invalid_equal_value_offset_sign_and_margin_do_not_alias_valid_inputs() -> None:
    cache, span = bt.ProofCache(), arch()
    assert cache.within_curvature(span, 1, 1, 1)
    for offset, sign, margin in ((True, 1, 1), (1, True, 1), (1, 1, True)):
        with pytest.raises(ValueError):
            _ = cache.within_curvature(span, offset, sign, margin)
    invalid_shape = replace(span, control_points=cast("Controls", ((0, 0),) * 3))
    with pytest.raises(ValueError, match="four planar"):
        _ = cache.within_curvature(invalid_shape, 1, 1)


def test_orientation_uses_exact_full_ring_and_retains_unused_line_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring, cache = polygon_lines(box(0, 0, 2, 2)).exterior, bt.ProofCache()
    expected = bt._positive_area(ring)
    assert cache.positive_area(ring) == expected
    assert cache.positive_area(reverse(ring)) != expected
    span = ring.spans[0]
    invalid = replace(
        span,
        control_points=(
            span.control_points[0],
            (float("nan"), 0),
            span.control_points[2],
            span.control_points[3],
        ),
    )
    with pytest.raises(ValueError):
        _ = cache.positive_area(CurveRing((invalid, *ring.spans[1:])))
    monkeypatch.setattr(bt, "_positive_area", unexpected)
    assert cache.positive_area(ring) == expected
    assert cache.positive_area(reverse(ring)) != expected


def test_cached_gate_reconstructs_roles_runs_and_full_failure_details() -> None:
    curves, cache = selected(), bt.ProofCache()
    black = replace(curves, orange=())
    swapped = replace(curves, orange=curves.white, white=curves.orange)
    backwards = replace(
        black,
        white=tuple(
            CurvePolygon(reverse(p.exterior), tuple(map(reverse, p.interiors)))
            for p in black.white
        ),
    )
    for value, runs in (
        (curves, (0, 0.2, 0.4)),
        (black, (0, 0.2, 0.4)),
        (swapped, (0, 0.2, 0.4)),
        (backwards, (0, 0.2, 0.4)),
        (curves, (0, 0.4, 0.8)),
        (black, (0, 0, 0)),
    ):
        expected = failures(value, runs)
        assert failures(value, runs, cache) == expected
        assert failures(value, runs, cache) == expected
    assert failures(black, (0, 0.2, 0.4), cache)
    for runs in ((False, 0.2, 0.4), (0, 0.4, 0.2)):
        with pytest.raises(ValueError, match="material taper runs"):
            bt.require_curvature(curves, runs, cache)


def test_ordinary_gate_stays_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    curves = selected()
    bt.require_curvature(curves, (0, 0.2, 0.4))
    monkeypatch.setattr(cu, "within_curvature", unexpected)
    with pytest.raises(AssertionError, match="reused"):
        bt.require_curvature(curves, (0, 0.2, 0.4))


def test_cache_is_bounded_and_does_not_retain_context_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bt, "_PROOF_CACHE_LIMIT", 2)
    cache, span = bt.ProofCache(), arch()
    for offset in (0.1, 0.2, 0.3):
        _ = cache.within_curvature(span, offset, -1)
    for size in (1, 2, 3):
        _ = cache.positive_area(polygon_lines(box(0, 0, size, size)).exterior)
    assert len(cache._bounds) == len(cache._orientations) == 2
    pointer = weakref.ref(cache)
    del cache
    _ = gc.collect()
    assert pointer() is None
