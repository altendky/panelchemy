"""Local windows preserve approved surroundings and exact source references."""

# Synthetic fixtures use the core's line representation.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from itertools import pairwise
from math import inf, nextafter
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from panelchemy.curves import BezierSpan, _line
from panelchemy.local_windows import Window, prepare, windows

type FloatArray = NDArray[np.float64]


def fixture() -> tuple[
    tuple[tuple[float, float], ...],
    FloatArray,
    list[BezierSpan],
    list[tuple[float, float]],
]:
    source = (
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (-1.0, 1.0),
        (0.0, 0.0),
    )
    parameters = np.array([0.0, 0.1, 0.3, 0.55, 0.7, 0.85, 1.0])
    spans = [_line(a, b) for a, b in pairwise(source)]
    values = cast("list[float]", parameters.tolist())
    intervals = list(pairwise(values))
    return source, parameters, spans, intervals


def test_padding_merges_connected_groups_and_orders_lowest_failure_first() -> None:
    assert windows(10, {2, 4, 8}, periodic=False, neighbors=1) == (
        Window(1, 5),
        Window(7, 3),
    )
    assert windows(10, {0, 9, 5}, periodic=True, neighbors=1) == (
        Window(8, 4),
        Window(4, 3),
    )


def test_pinned_closed_chain_does_not_wrap() -> None:
    assert windows(8, {0, 7}, periodic=False, neighbors=1) == (
        Window(0, 2),
        Window(6, 2),
    )


def test_full_coverage_is_explicit_and_left_to_global_search() -> None:
    assert windows(5, {0}, periodic=True, neighbors=1000) == (Window(0, 5),)
    source, parameters, spans, intervals = fixture()
    with pytest.raises(ValueError, match="proper window"):
        _ = prepare(Window(0, 6), source, parameters, spans, intervals, periodic=True)


def test_entire_pinned_chain_keeps_its_seam_and_source_references() -> None:
    source, parameters, spans, intervals = fixture()
    guide = prepare(Window(0, len(spans)), source, parameters, spans, intervals)
    assert guide.seam is None
    assert np.array_equal(guide.source_points, source)
    assert np.array_equal(guide.parameters, parameters)
    assert guide.inherited_handles == (
        spans[0].control_points[-1],
        spans[-1].control_points[0],
    )
    assert guide.splice(spans, intervals, spans, intervals) == (spans, intervals)
    with pytest.raises(ValueError, match="proper window"):
        _ = prepare(Window(1, len(spans)), source, parameters, spans, intervals)


def test_empty_failures_need_no_windows() -> None:
    assert windows(5, (), periodic=False, neighbors=2) == ()


@pytest.mark.parametrize("bad", [{-1}, {5}, {True}])
def test_invalid_bad_indices_are_rejected(bad: set[int]) -> None:
    with pytest.raises(ValueError, match="valid chain"):
        _ = windows(5, bad, periodic=True, neighbors=1)


def test_wrapped_guide_preserves_source_vertices_and_original_inputs() -> None:
    source, parameters, spans, intervals = fixture()
    saved = parameters.copy()
    guide = prepare(Window(5, 2), source, parameters, spans, intervals, periodic=True)
    assert guide.seam is not None
    assert guide.start_t == 0.85 and guide.end_t_original == 0.1
    assert guide.end_t - 1 != guide.end_t_original  # A real round-trip hazard.
    assert np.array_equal(
        guide.source_points, np.array([source[5], source[6], source[1]])
    )
    assert np.array_equal(parameters, saved)
    assert guide.parameters[0] == 0 and guide.parameters[-1] == 1
    assert guide.to_local(0) == guide.to_local(1) == guide.seam
    assert guide.to_local(0.5) is None
    assert guide.to_local(nextafter(guide.end_t_original, inf)) is None
    assert guide.to_local(float("nan")) is None


def test_wrap_requires_freely_periodic_source_not_merely_closed_geometry() -> None:
    source, parameters, spans, intervals = fixture()
    with pytest.raises(ValueError, match="freely periodic"):
        _ = prepare(Window(5, 2), source, parameters, spans, intervals)


def test_guide_inherits_true_line_chords_not_dummy_handles() -> None:
    source, parameters, spans, intervals = fixture()
    altered = replace(
        spans[1],
        control_points=(
            spans[1].control_points[0],
            (9, 2),
            (9, 3),
            spans[1].control_points[-1],
        ),
    )
    selected = [spans[0], altered, *spans[2:]]
    guide = prepare(Window(1, 2), source, parameters, selected, intervals)
    assert guide.inherited_handles[0] == altered.control_points[-1]
    direction = np.array(altered.control_points[-1]) - altered.control_points[0]
    assert np.allclose(
        cast("FloatArray", guide.tangent_directions[0]),
        direction / np.linalg.norm(direction),
    )
    assert np.array_equal(
        cast("FloatArray", guide.endpoint_positions[0]), altered.control_points[0]
    )


def test_nonwrapped_splice_keeps_outside_span_and_interval_objects() -> None:
    source, parameters, spans, intervals = fixture()
    guide = prepare(Window(1, 2), source, parameters, spans, intervals)
    middle = (intervals[1][1] - guide.start_t) / (guide.end_t - guide.start_t)
    joined, times = guide.splice(
        spans, intervals, spans[1:3], [(0, middle), (middle, 1)]
    )
    assert joined == spans and times == intervals
    for index in (0, 3, 4, 5):
        assert joined[index] is spans[index]
        assert times[index] is intervals[index]


def test_wrapped_splice_restores_exact_zero_one_and_nondyadic_end_reference() -> None:
    source, parameters, spans, intervals = fixture()
    guide = prepare(Window(5, 2), source, parameters, spans, intervals, periodic=True)
    assert guide.seam is not None
    joined, times = guide.splice(
        spans, intervals, [spans[5], spans[0]], [(0, guide.seam), (guide.seam, 1)]
    )
    assert joined == spans and times == intervals
    for index in (1, 2, 3, 4):
        assert joined[index] is spans[index]
        assert times[index] is intervals[index]


def test_wrapped_replacement_must_contain_original_seam_knot() -> None:
    source, parameters, spans, intervals = fixture()
    guide = prepare(Window(5, 2), source, parameters, spans, intervals, periodic=True)
    with pytest.raises(ValueError, match="split at the original seam"):
        _ = guide.splice(spans, intervals, [spans[5], spans[0]], [(0, 0.5), (0.5, 1)])


def test_disconnected_replacement_or_stationary_endpoint_is_rejected() -> None:
    source, parameters, spans, intervals = fixture()
    guide = prepare(Window(1, 2), source, parameters, spans, intervals)
    with pytest.raises(ValueError, match="pinned endpoint"):
        _ = guide.splice(spans, intervals, [_line((99, 99), (99, 100))], [(0, 1)])
    stationary = replace(
        spans[1],
        is_line=False,
        control_points=(
            spans[1].control_points[0],
            spans[1].control_points[0],
            spans[1].control_points[2],
            spans[1].control_points[-1],
        ),
    )
    with pytest.raises(ValueError, match="unavailable endpoint tangent"):
        _ = prepare(
            Window(1, 2),
            source,
            parameters,
            [spans[0], stationary, *spans[2:]],
            intervals,
        )


@pytest.mark.parametrize(
    "bad_parameters",
    [np.array([0.0, 1.0]), np.array([0.0, 0.5, 0.5, 0.7, 0.8, 0.9, 1.0])],
)
def test_invalid_source_parameter_shapes_and_order_are_rejected(
    bad_parameters: FloatArray,
) -> None:
    source, _parameters, spans, intervals = fixture()
    with pytest.raises(ValueError, match="finite source"):
        _ = prepare(Window(1, 2), source, bad_parameters, spans, intervals)


def test_open_source_cannot_be_wrapped_even_with_periodic_request() -> None:
    source, parameters, spans, intervals = fixture()
    source = (*source[:-1], (9.0, 9.0))
    with pytest.raises(ValueError, match="freely periodic"):
        _ = prepare(Window(5, 2), source, parameters, spans, intervals, periodic=True)
