"""Pointed alternatives preserve inherited rays without authorizing a fit."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from fractions import Fraction
from math import inf
from typing import cast

import pytest

from panelchemy import curvature
from panelchemy import curves as c
from panelchemy.corner_refinement import (
    corner_variants,
    forward_progress,
    smooth_join,
    source_corners,
    tangent_corner,
)


def cap() -> c.BezierSpan:
    return c.BezierSpan(((0, 0), (1, 1), (1, 1), (2, 0)), False, 0.125)


def assert_rays(
    original: c.BezierSpan, selected: tuple[c.BezierSpan, c.BezierSpan]
) -> None:
    before = c._controls(original)
    left, right = map(c._controls, selected)
    assert left[0] == before[0] and right[-1] == before[-1]
    assert left[-1] == right[0]
    for anchor, inherited, handle in (
        (before[0], before[1], left[1]),
        (before[-1], before[-2], right[-2]),
    ):
        u = inherited[0] - anchor[0], inherited[1] - anchor[1]
        v = handle[0] - anchor[0], handle[1] - anchor[1]
        assert u[0] * v[1] == u[1] * v[0]
        assert u[0] * v[0] + u[1] * v[1] > 0


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("shift", [(0, 0), (16, -8)])
def test_rotated_translated_corner_preserves_exact_endpoints_rays_and_metadata(
    turns: int, shift: c.Point2
) -> None:
    def transform(point: c.Point2) -> c.Point2:
        x, y = point
        for _ in range(turns):
            x, y = -y, x
        return x + shift[0], y + shift[1]

    original = replace(
        cap(),
        control_points=cast("c.Controls", tuple(map(transform, cap().control_points))),
    )
    selected = tangent_corner(original)
    assert selected is not None
    assert_rays(original, selected)
    assert selected[0].control_points[-1] == transform((1, 1))
    assert selected[0].control_points[-1] is selected[1].control_points[0]
    assert all(s.certified_error_mm == original.certified_error_mm for s in selected)
    assert all(not s.is_line for s in selected)
    backwards = replace(
        original,
        control_points=cast("c.Controls", tuple(reversed(original.control_points))),
    )
    reverse = tangent_corner(backwards)
    assert reverse is not None
    assert reverse == tuple(
        replace(s, control_points=cast("c.Controls", tuple(reversed(s.control_points))))
        for s in reversed(selected)
    )


@pytest.mark.parametrize(
    "controls",
    [
        ((0, 0), (0, 0), (1, 1), (2, 0)),  # zero start ray
        ((0, 0), (1, 1), (2, 0), (2, 0)),  # zero end ray
        ((0, 0), (1, 0), (1, 0), (2, 0)),  # parallel rays
        ((0, 0), (-1, 1), (3, 1), (2, 0)),  # rays meet behind the endpoints
        ((0, 0), (1, 1), (1, 0), (2, 0)),  # apex is an endpoint
        ((0, 0), (1, 1), (-1, 1), (0, 0)),  # closed degenerate span
        ((64, 0), (65, 0), (65, -1e308), (64, 1)),  # rounded apex collapses
        (
            (0, -1),
            (1, -1),
            (2, -float.fromhex("0x0.0000000000001p-1022")),
            (1, 0),
        ),  # apex overflows
    ],
)
def test_unavailable_positive_finite_apex_returns_none(controls: c.Controls) -> None:
    assert tangent_corner(c.BezierSpan(controls, False)) is None


@pytest.mark.parametrize("value", [True, inf, -inf, float("nan"), "1", 1j])
def test_invalid_coordinates_do_not_escape_construction(value: object) -> None:
    original = cap()
    controls = (original.control_points[0], (value, 1), *original.control_points[2:])
    assert (
        tangent_corner(replace(original, control_points=cast("c.Controls", controls)))
        is None
    )


def test_lines_and_malformed_controls_are_not_corner_candidates() -> None:
    assert tangent_corner(replace(cap(), is_line=True)) is None
    assert (
        tangent_corner(replace(cap(), control_points=cast("c.Controls", ((0, 0),) * 3)))
        is None
    )
    assert (
        tangent_corner(
            replace(cap(), control_points=cast("c.Controls", ((0, 0, 0),) * 4))
        )
        is None
    )


def test_coarse_dyadic_ray_preserves_real_progress_despite_negative_control_step() -> (
    None
):
    # Decimal inputs at a translated origin can make exact inherited direction
    # multiples coarse: the nearest representable handle exceeds side/3.
    original = c.BezierSpan(
        ((64, 128), (64.3, 128.1), (64.1, 128.7), (64, 129)), False, 0.2
    )
    selected = tangent_corner(original)
    assert selected is not None
    assert_rays(original, selected)
    assert not c._monotone(selected[0])
    assert all(forward_progress(s) for s in selected)
    assert all(curvature.within_curvature(s, Fraction(2, 5), 0) for s in selected)


@pytest.mark.parametrize(
    "controls,expected",
    [
        (((0, 0), (0, 0), (0, 0), (0, 0)), False),
        (((0, 0), (1, 1), (1, -1), (0, 0)), False),
        (((0, 0), (1, 0), (0, 0), (1, 0)), False),  # interior stop at t=1/2
        (((0, 0), (2, 0), (-1, 0), (1, 0)), False),  # true backtracking
        (((0, 0), (-1, 0), (0.5, 0), (1, 0)), False),
        (((0, 0), (0.5, 0), (2, 0), (1, 0)), False),
        (((0, 0), (0.75, 0), (0.5, 0), (1, 0)), True),
        (((0, 0), (1, 0), (2**-60, 0), (1, 0)), True),
    ],
)
def test_exact_forward_progress_handles_stops_backtracking_and_reversal(
    controls: c.Controls, expected: bool
) -> None:
    span = c.BezierSpan(controls, False)
    assert forward_progress(span) == expected
    reverse = replace(
        span, control_points=cast("c.Controls", tuple(reversed(controls)))
    )
    assert forward_progress(reverse) == expected


def test_forward_progress_is_not_an_endpoint_speed_or_radius_certificate() -> None:
    span = c.BezierSpan(((0, 0), (0, 0), (1, 0), (2, 0)), False)
    assert forward_progress(span)
    assert not curvature.within_curvature(span, 0, 0)
    # Straight metadata uses exact canonical line controls, not its handles.
    line = c.BezierSpan(((0, 0), (-10, 8), (20, -8), (1, 0)), True)
    assert forward_progress(line)


def parallel_offset() -> c.BezierSpan:
    # Exact binary-coordinate regression: the endpoint rays are parallel even
    # though their common direction differs from the endpoint chord.
    return c.BezierSpan(
        (
            (61.244853935204446, 104.44422921072692),
            (61.27622961997986, 104.45599311124533),
            (61.314749067649245, 104.44699095096439),
            (61.34618190303445, 104.4587762793526),
        ),
        False,
        0.2,
    )


def assert_locked_rays(
    original: c.BezierSpan,
    candidate: tuple[c.BezierSpan, ...],
    free_start: bool,
    free_end: bool,
) -> None:
    p, a, b = (
        c._controls(original),
        c._controls(candidate[0]),
        c._controls(candidate[-1]),
    )
    assert a[0] == p[0] and b[-1] == p[-1]
    for free, anchor, inherited, selected in (
        (free_start, p[0], p[1], a[1]),
        (free_end, p[-1], p[-2], b[-2]),
    ):
        if free:
            continue
        u, v = (
            (inherited[0] - anchor[0], inherited[1] - anchor[1]),
            (selected[0] - anchor[0], selected[1] - anchor[1]),
        )
        assert u[0] * v[1] == u[1] * v[0]
        assert u[0] * v[0] + u[1] * v[1] > 0


@pytest.mark.parametrize(
    "free_start,free_end", [(True, True), (True, False), (False, True)]
)
def test_parallel_tangent_case_gets_only_authorized_corner_variants(
    free_start: bool, free_end: bool
) -> None:
    original = parallel_offset()
    assert tangent_corner(original) is None
    assert list(corner_variants(original)) == []
    variants = list(corner_variants(original, free_start=free_start, free_end=free_end))
    assert variants
    if free_start and free_end:
        assert len(variants[0]) == 1  # direct chord precedes pointed alternatives
    else:
        assert all(len(candidate) == 2 for candidate in variants)
    for candidate in variants:
        assert_locked_rays(original, candidate, free_start, free_end)
        assert all(s.is_line and s.certified_error_mm == 0.2 for s in candidate)


def test_variant_order_deduplication_and_degenerate_apices() -> None:
    original = cap()
    variants = list(
        corner_variants(
            original,
            free_start=True,
            free_end=True,
            apices=((0, 0), (2, 0), (-1, 0), (3, 0), (1, 2), (1, 2)),
        )
    )
    assert variants[0] == tangent_corner(original)
    assert len(variants[1]) == 1 and variants[1][0].is_line
    apices = [candidate[0].control_points[-1] for candidate in variants[2:]]
    assert apices == [(1, 2), (1, 0.75), (1, 1)]
    assert len(variants) == len(set(variants))
    assert all(
        s.certified_error_mm == original.certified_error_mm
        for candidate in variants
        for s in candidate
    )
    assert list(corner_variants(original, apices=((100, 100),))) == [
        tangent_corner(original)
    ]


@pytest.mark.parametrize(
    "free_start,free_end", [(True, False), (False, True), (True, True)]
)
def test_variant_reversal_swaps_endpoint_freedom(
    free_start: bool, free_end: bool
) -> None:
    original = parallel_offset()
    backwards = replace(
        original,
        control_points=cast("c.Controls", tuple(reversed(original.control_points))),
    )
    selected = list(corner_variants(original, free_start=free_start, free_end=free_end))
    reversed_variants = set(
        corner_variants(backwards, free_start=free_end, free_end=free_start)
    )
    assert reversed_variants == {
        tuple(
            replace(
                s, control_points=cast("c.Controls", tuple(reversed(s.control_points)))
            )
            for s in reversed(candidate)
        )
        for candidate in selected
    }


@pytest.mark.parametrize("free_start,free_end", [(True, False), (False, True)])
def test_one_free_endpoint_ignores_apices_that_would_release_the_other_ray(
    free_start: bool, free_end: bool
) -> None:
    original = parallel_offset()
    assert list(
        corner_variants(
            original, free_start=free_start, free_end=free_end, apices=((100, 100),)
        )
    ) == list(corner_variants(original, free_start=free_start, free_end=free_end))
    line = c.BezierSpan(((0, 0), (88, 99), (-88, -99), (1, 0)), True)
    for candidate in corner_variants(line, free_start=free_start, free_end=free_end):
        assert_locked_rays(line, candidate, free_start, free_end)


@pytest.mark.parametrize(
    "before_line,after_line",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_smooth_join_recognizes_lines_and_curves(
    before_line: bool, after_line: bool
) -> None:
    before = c.BezierSpan(
        (
            (0, 0),
            (88, 99) if before_line else (0.25, 0),
            (-88, -99) if before_line else (0.75, 0),
            (1, 0),
        ),
        before_line,
    )
    after = c.BezierSpan(
        (
            (1, 0),
            (-88, 99) if after_line else (1.25, 0),
            (88, -99) if after_line else (1.75, 0),
            (2, 0),
        ),
        after_line,
    )
    assert smooth_join(before, after)


@pytest.mark.parametrize(
    "height,expected", [(1e-12, True), (-1e-12, True), (1e-6, False)]
)
def test_near_smooth_roundoff_does_not_authorize_ray_release(
    height: float, expected: bool
) -> None:
    before = c._line((0, 0), (1, 0))
    after = c.BezierSpan(((1, 0), (1.25, height), (1.75, 0), (2, 0)), False)
    assert smooth_join(before, after) == expected


def test_smooth_join_rejects_missing_reversed_and_zero_tangents() -> None:
    before = c._line((0, 0), (1, 0))
    assert not smooth_join(before, c._line((1, 0), (0, 0)))
    assert not smooth_join(before, c._line((1 + 2**-52, 0), (2, 0)))
    assert not smooth_join(
        before, c.BezierSpan(((1, 0), (1, 0), (1.75, 0), (2, 0)), False)
    )
    assert not smooth_join(
        c.BezierSpan(((0, 0), (0.25, 0), (1, 0), (1, 0)), False),
        c._line((1, 0), (2, 0)),
    )


def test_corner_ray_freedom_requires_explicit_booleans() -> None:
    with pytest.raises(ValueError, match="flags must be booleans"):
        _ = list(corner_variants(cap(), free_start=cast("bool", cast("object", 1))))


def source_arc() -> tuple[c.Point2, ...]:
    return (
        (0, 0),
        (0, 0.1),
        (0, 0.2),
        (0, 0.3),
        (0.3, 0.3),
        (0.6, 0.3),
        (0.9, 0.3),
        (1, 0.3),
        (1, 0.5),
        (1, 0.75),
        (1, 1),
        (2, 1),
        (2, -1),
        (0, -1),
        (0, 0),
    )


def source_span() -> c.BezierSpan:
    return c.BezierSpan(((0.02, 0.01), (0, 1), (1, 0), (0.99, 0.98)), False, 0.2)


def test_source_corners_simplifies_raw_steps_with_exact_fixed_ends_and_metadata() -> (
    None
):
    original, path = source_span(), source_arc()
    selected = list(source_corners(original, [path], 0.05))
    assert len(selected) == 1 and len(selected[0]) == 3
    assert [s.control_points[0] for s in selected[0]] == [
        original.control_points[0],
        (0, 0.3),
        (1, 0.3),
    ]
    assert selected[0][-1].control_points[-1] == original.control_points[-1]
    assert all(s.is_line and s.certified_error_mm == 0.2 for s in selected[0])
    duplicate_steps = tuple(p for point in path for p in (point, point))
    assert list(source_corners(original, [duplicate_steps, path], 0.05)) == selected


def test_source_corners_handles_seam_wrapping_and_reversed_winding() -> None:
    original, path = source_span(), source_arc()
    expected = list(source_corners(original, [path], 0.05))
    vertices = path[:-1]
    wrapped = (*vertices[5:], *vertices[:5], vertices[5])
    assert list(source_corners(original, [wrapped], 0.05)) == expected
    reverse = replace(
        original,
        control_points=cast("c.Controls", tuple(reversed(original.control_points))),
    )
    reversed_candidates = list(source_corners(reverse, [tuple(reversed(path))], 0.05))
    assert reversed_candidates == [
        tuple(
            replace(
                s, control_points=cast("c.Controls", tuple(reversed(s.control_points)))
            )
            for s in reversed(candidate)
        )
        for candidate in expected
    ]


def test_source_corners_rejects_wrong_or_far_arcs_and_open_paths() -> None:
    original, path = source_span(), source_arc()
    assert list(source_corners(original, [tuple(reversed(path))], 0.05)) == []
    assert (
        list(source_corners(original, [tuple((x + 10, y + 10) for x, y in path)], 0.05))
        == []
    )
    assert list(source_corners(original, [path[:-1]], 0.05)) == []


def test_source_corners_caps_complexity_and_does_not_repeat_the_chord_family() -> None:
    original = c.BezierSpan(((0, 0), (0, 1.2), (1.2, 0), (1.2, 1.2)), False)
    path = (
        (0, 0),
        (0, 0.4),
        (0.4, 0.4),
        (0.4, 0.8),
        (0.8, 0.8),
        (0.8, 1.2),
        (1.2, 1.2),
        (3, 1.2),
        (3, -1),
        (0, -1),
        (0, 0),
    )
    assert list(source_corners(original, [path], 0.01)) == []
    straight = ((0, 0), (0.4, 0.4), (1.2, 1.2), (3, 1.2), (3, -1), (0, -1), (0, 0))
    assert list(source_corners(original, [straight], 0.01)) == []


def test_source_corners_offers_one_corner_and_skips_collapsed_raw_paths() -> None:
    original = c.BezierSpan(((0, 0), (0, 1), (1, 0), (1, 1)), False, 0.125)
    path = (
        (0, 0),
        (0, 0.25),
        (0, 0.75),
        (0, 1),
        (0.5, 1),
        (1, 1),
        (2, 1),
        (2, -1),
        (0, -1),
        (0, 0),
    )
    selected = list(source_corners(original, [((0, 0),) * 4, path], 0.05))
    assert len(selected) == 1 and len(selected[0]) == 2
    assert selected[0][0].control_points[-1] == (0, 1)
    assert all(s.is_line and s.certified_error_mm == 0.125 for s in selected[0])


@pytest.mark.parametrize("target", [0, -1, True, inf, float("nan")])
def test_source_corner_target_must_be_positive_and_finite(target: float) -> None:
    with pytest.raises(ValueError, match="fit target must be finite and positive"):
        _ = list(source_corners(source_span(), [source_arc()], target))
