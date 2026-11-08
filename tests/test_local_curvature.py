"""Local search orchestration must finish repairs within one chain budget."""

# These regressions exercise internal orchestration with deterministic searches.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from itertools import pairwise
from math import nextafter

import numpy as np
import pytest

from panelchemy import (
    boundary_tapers,
    curvature,
    curvature_fitting,
    fitting_feedback,
    local_windows,
)
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting

_LIMIT = boundary_tapers.BoundaryTaper(1, 2, 0.2, 1)


def _fit(size: int, bad: set[int], *, closed: bool = False) -> fitting._Fit:
    points = tuple((float(i), 0.0) for i in range(size))
    points = (*points, points[0] if closed else (float(size), 0.0))
    pieces = [
        c._Piece(
            0, i, i + 1, replace(c._line(a, b), certified_error_mm=float(i in bad))
        )
        for i, (a, b) in enumerate(pairwise(points))
    ]
    return fitting._Fit(
        c._Chain(points, fixed=False),
        np.linspace(0, 1, len(points)),
        [],
        [],
        pieces,
        [(i / size, (i + 1) / size) for i in range(size)],
    )


@pytest.fixture(autouse=True)
def controlled_curvature(monkeypatch: pytest.MonkeyPatch) -> None:
    def approved(span: c.BezierSpan, _offset: float, _sign: int) -> bool:
        # The certificate field tags synthetic failures; geometry proof is
        # covered separately from these scheduling and fallback regressions.
        return span.certified_error_mm == 0

    monkeypatch.setattr(curvature, "within_curvature", approved)
    monkeypatch.setattr(fitting, "monotonic", lambda: 0.0)


def _repair(fit: fitting._Fit, guide: local_windows.Guide) -> fitting._Fit:
    selected = guide.window.indices(len(fit.pieces))
    span = c._line(
        fit.pieces[selected[0]].span.control_points[0],
        fit.pieces[selected[-1]].span.control_points[-1],
    )
    spans, intervals = guide.splice(
        [p.span for p in fit.pieces], fit.intervals, [span], [(0.0, 1.0)]
    )
    return replace(
        fit,
        pieces=[c._Piece(0, i, i + 1, span) for i, span in enumerate(spans)],
        intervals=intervals,
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_corner_run_stops_at_active_corners_and_replans_after_splicing(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    fit = _fit(8, {0, 7} if wrapped else {3, 4}, closed=wrapped)
    fit.chain = replace(fit.chain, free_loop=wrapped)
    fit.corners = [0.25, 0.75]
    calls: list[local_windows.Window] = []

    def search(
        current: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
        *,
        compact: bool,
        free_corner_rays: bool,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        assert compact and free_corner_rays
        calls.append(guide.window)
        if wrapped:
            selected = set(guide.window.indices(len(current.pieces)))
            return replace(
                current,
                pieces=[
                    replace(p, span=replace(p.span, certified_error_mm=0))
                    if i in selected
                    else p
                    for i, p in enumerate(current.pieces)
                ],
            ), {}
        return _repair(current, guide), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._corner_run_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is not None and report["windows_repaired"] == 1
    assert calls == [local_windows.Window(6 if wrapped else 2, 4)]
    selected = set(calls[0].indices(len(fit.pieces)))
    assert all(
        any(p.span is q.span for q in result.pieces)
        for i, p in enumerate(fit.pieces)
        if i not in selected
    )


def test_graph_terminal_runs_keep_pins_and_internal_outside_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(8, {0, 7})
    fit.corners = [0.25, 0.75]
    calls: list[local_windows.Window] = []

    def search(
        current: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        *_args: object,
        **kwargs: object,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        assert kwargs["free_corner_rays"] is True
        calls.append(guide.window)
        return _repair(current, guide), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._corner_run_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is not None and report["windows_repaired"] == 2
    assert calls == [local_windows.Window(0, 2), local_windows.Window(5, 2)]
    assert result.pieces[0].span.control_points[0] == fit.chain.points[0]
    assert result.pieces[-1].span.control_points[-1] == fit.chain.points[-1]
    assert all(any(p.span is q.span for q in result.pieces) for p in fit.pieces[2:6])


@pytest.mark.parametrize("free,fixed", [(True, False), (False, True)])
def test_graph_terminal_rule_does_not_free_periodic_seams_or_fixed_boundaries(
    free: bool, fixed: bool
) -> None:
    fit = _fit(8, {0}, closed=free)
    fit.chain = replace(fit.chain, free_loop=free, fixed=fixed)
    fit.corners = [0.25, 0.75]
    assert fitting._run_corners(fit) == {0.25, 0.75}


def test_corner_run_retains_success_without_certifying_the_remaining_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(8, {0, 3})
    fit.corners = [0.25, 0.75]

    def search(
        current: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        if guide.start_t == 0:
            return None, {}
        return _repair(current, guide), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._corner_run_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is not None and report["windows_repaired"] == 1
    assert report["status"] == "partial"
    assert result.pieces[0].span is fit.pieces[0].span
    assert result.pieces[0].span.certified_error_mm == 1
    assert all(p.span.certified_error_mm == 0 for p in result.pieces[1:])


def test_corner_run_shares_budget_between_compact_and_ordinary_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(8, {3})
    fit.corners = [0.25, 0.75]
    clock = [0.0]
    calls: list[tuple[bool, float]] = []

    def search(
        _fit: fitting._Fit,
        _index: int,
        _guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
        *,
        compact: bool,
        free_corner_rays: bool,
    ) -> tuple[None, dict[str, object]]:
        assert free_corner_rays
        calls.append((compact, seconds))
        clock[0] += seconds
        return None, {}

    monkeypatch.setattr(fitting, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._corner_run_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is None and report["stop_reason"] == "time-budget"
    assert calls == [(True, 2.0), (False, 2.0)]


def test_window_covering_half_the_chain_defers_to_whole_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a broad local window should defer to whole-chain fitting")

    monkeypatch.setattr(fitting, "_local_window_refit", unexpected)
    result, report = fitting._local_curvature_refit(
        _fit(6, {2, 3, 4}), 0, _LIMIT, 0.2, 12
    )
    assert result is None
    assert report["attempts"] == []


def test_inferred_corner_can_cover_most_of_a_short_pinned_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(5, {0, 1, 2})
    calls: list[local_windows.Window] = []

    def search(
        current: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
        *,
        apex: float,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        assert apex == 0.5
        calls.append(guide.window)
        return _repair(current, guide), {"status": "accepted"}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._apex_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is not None and calls == [local_windows.Window(0, 3)]
    assert result.pieces[-1].span is fit.pieces[-1].span
    assert result.pieces[-2].span is fit.pieces[-2].span


def test_inferred_corner_still_defers_a_whole_freely_periodic_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(5, {0, 1, 2, 3, 4}, closed=True)
    fit.chain = replace(fit.chain, free_loop=True)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("whole freely periodic windows require their own solver")

    monkeypatch.setattr(fitting, "_local_window_refit", unexpected)
    result, report = fitting._apex_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is None and report["attempts"] == []


def test_inferred_corner_tries_curvature_witness_after_midpoint_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(8, {2})
    guide = local_windows.prepare(
        local_windows.Window(2, 1),
        fit.chain.points,
        fit.parameters,
        [p.span for p in fit.pieces],
        fit.intervals,
    )
    clock = [0.0]
    calls: list[tuple[float, float]] = []

    def locations(*_args: object) -> tuple[float, ...]:
        return (0.75,)

    def search(
        current: fitting._Fit,
        _index: int,
        selected: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
        *,
        apex: float,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append((apex, seconds))
        clock[0] += seconds
        return (_repair(current, selected) if apex == 0.75 else None), {
            "status": "unrefined" if apex == 0.5 else "accepted"
        }

    monkeypatch.setattr(fitting, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fitting_feedback, "curvature_locations", locations)
    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._apex_window_refit(fit, 0, guide, _LIMIT, 0.2, 2, [2])
    assert result is not None and calls == [(0.5, 1), (0.75, 1)]
    assert clock[0] == 2


def test_compact_retry_starts_with_support_and_keeps_outside_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit(12, {5})
    calls: list[tuple[local_windows.Window, float, bool]] = []

    def search(
        fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
        *,
        compact: bool = False,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        calls.append((guide.window, seconds, compact))
        return _repair(fit, guide), {"status": "accepted"}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._compact_curvature_candidate(original, 0, _LIMIT, 0.2)
    assert result is not None
    assert calls == [(local_windows.Window(4, 3), 4 / 3, True)]
    assert result.pieces[0].span is original.pieces[0].span
    assert result.pieces[-1].span is original.pieces[-1].span
    assert result.intervals[0] is original.intervals[0]
    assert all(span.span.certified_error_mm == 0 for span in result.pieces)


def test_inferred_apex_stays_on_the_failure_when_padding_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(12, {0})
    calls: list[tuple[int, float, float]] = []

    def search(
        current: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
        *,
        apex: float,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append((guide.window.count, apex, seconds))
        reference = guide.start_t + apex * (guide.end_t - guide.start_t)
        assert reference == pytest.approx(1 / 24)
        return (_repair(current, guide) if guide.window.count > 1 else None), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._apex_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is not None
    assert [(count, apex) for count, apex, _ in calls] == [(1, 0.5), (2, 0.25)]
    assert all(0 < seconds <= 4 for _, _, seconds in calls)
    assert result.pieces[-1].span is fit.pieces[-1].span


def test_inferred_apex_uses_the_exact_wrapped_seam_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(12, {0, 11}, closed=True)
    fit.chain = replace(fit.chain, free_loop=True)
    seen: list[float] = []

    def search(
        _fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
        *,
        apex: float,
    ) -> tuple[None, dict[str, object]]:
        assert guide.seam is not None
        assert apex == guide.seam
        seen.append(apex)
        return None, {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._apex_curvature_candidate(fit, 0, _LIMIT, 0.2)
    assert result is None and seen


def test_compact_retry_cannot_smooth_across_a_retained_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(8, {3})
    fit.corners = [0.5]
    guide = local_windows.prepare(
        local_windows.Window(2, 3),
        fit.chain.points,
        fit.parameters,
        [p.span for p in fit.pieces],
        fit.intervals,
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("compact smooth repair must retain an active interior corner")

    monkeypatch.setattr(curvature_fitting, "refit", unexpected)
    result, report = fitting._local_window_refit(
        fit, 0, guide, _LIMIT, 0.2, 1.0, compact=True
    )
    assert result is None and report["status"] == "protected-corner"


@pytest.mark.parametrize("periodic", [False, True])
def test_texture_preference_cannot_cross_an_unpinned_structural_sample(
    monkeypatch: pytest.MonkeyPatch, periodic: bool
) -> None:
    fit = _fit(12, {0, 11} if periodic else {5}, closed=periodic)
    fit.chain = replace(fit.chain, free_loop=periodic)
    fit.texture_intervals = ((0, 1),)
    fit.texture_barriers = (0.0 if periodic else 5.5 / 12,)
    assert fit.corners == []

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("texture-guided smoothing must stop at structural source samples")

    monkeypatch.setattr(fitting, "_local_window_refit", unexpected)
    result, report = fitting._compact_curvature_candidate(
        fit, 0, _LIMIT, 0.2, texture_only=True
    )
    assert result is None and report["attempts"] == []


def test_texture_preference_defers_a_chain_with_other_failure_neighborhoods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = _fit(12, {2, 9})
    fit.texture_intervals = ((0, 0.4),)

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "a local texture label must not select smoothing of unrelated failures"
        )

    monkeypatch.setattr(fitting, "_local_window_refit", unexpected)
    result, report = fitting._compact_curvature_candidate(
        fit, 0, _LIMIT, 0.2, texture_only=True
    )
    assert result is None and report["attempts"] == []


def test_partial_repair_cannot_escape_with_other_bad_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit(12, {2, 9})
    calls: list[fitting._Fit] = []

    def search(
        fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append(fit)
        return (_repair(fit, guide) if len(calls) == 1 else None), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._local_curvature_refit(original, 0, _LIMIT, 0.2, 12.0)
    assert result is None
    assert report["windows_repaired"] == 1
    assert len(calls) > 1 and calls[0] is original
    assert all(fit is calls[1] for fit in calls[1:])
    assert any(p.span.certified_error_mm for p in calls[1].pieces)
    assert len(original.pieces) == 12  # Failed local work leaves the input intact.


def test_window_growth_recomputes_shifted_failures_after_each_splice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit(12, {2, 9})
    calls: list[tuple[int, local_windows.Window]] = []

    def search(
        fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append((len(fit.pieces), guide.window))
        return (_repair(fit, guide) if guide.window.count > 1 else None), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._local_curvature_refit(original, 0, _LIMIT, 0.2, 12.0)
    assert calls == [
        (12, local_windows.Window(2, 1)),
        (12, local_windows.Window(9, 1)),
        (12, local_windows.Window(1, 3)),
        (10, local_windows.Window(6, 3)),
    ]
    assert result is not None
    assert result.chain is original.chain
    assert all(p.span.certified_error_mm == 0 for p in result.pieces)
    assert report["windows_repaired"] == 2


def test_closed_pinned_chain_never_uses_a_periodic_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit(12, {0, 11}, closed=True)
    calls: list[local_windows.Window] = []

    def search(
        _fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
    ) -> tuple[None, dict[str, object]]:
        assert guide.seam is None
        calls.append(guide.window)
        return None, {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._local_curvature_refit(original, 0, _LIMIT, 0.2, 12.0)
    assert result is None
    assert calls == [
        local_windows.Window(first, size)
        for size in (1, 2, 3, 5)
        for first in (0, 12 - size)
    ]


def test_failed_first_group_does_not_starve_or_replay_after_other_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fit(18, {2, 8, 14})
    calls: list[local_windows.Window] = []

    def search(
        fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
    ) -> tuple[fitting._Fit | None, dict[str, object]]:
        calls.append(guide.window)
        blocked = original.intervals[2]
        contains_blocked = any(
            fit.intervals[i] == blocked for i in guide.window.indices(len(fit.pieces))
        )
        return (None if contains_blocked else _repair(fit, guide)), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._local_curvature_refit(original, 0, _LIMIT, 0.2, 12)
    assert result is None  # Two successful groups cannot hide the blocked one.
    assert report["windows_repaired"] == 2
    assert calls[:3] == [local_windows.Window(i, 1) for i in (2, 8, 14)]
    assert calls.count(local_windows.Window(2, 1)) == 1


def test_broad_first_group_does_not_block_an_independent_small_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[local_windows.Window] = []

    def search(
        fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        calls.append(guide.window)
        return _repair(fit, guide), {}

    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, report = fitting._local_curvature_refit(
        _fit(20, {*range(10), 15}), 0, _LIMIT, 0.2, 12
    )
    assert result is None
    assert calls == [local_windows.Window(15, 1)]
    assert report["windows_repaired"] == 1


def test_failed_small_windows_leave_time_for_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    calls: list[tuple[local_windows.Window, float]] = []

    def search(
        _fit: fitting._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
    ) -> tuple[None, dict[str, object]]:
        nonlocal now
        calls.append((guide.window, seconds))
        now += seconds
        return None, {}

    monkeypatch.setattr(fitting, "monotonic", lambda: now)
    monkeypatch.setattr(fitting, "_local_window_refit", search)
    result, _ = fitting._local_curvature_refit(_fit(12, {2, 9}), 0, _LIMIT, 0.2, 12)
    assert result is None
    assert [seconds for _, seconds in calls[:2]] == pytest.approx([2.4, 2.4])
    assert any(window.count > 1 for window, _ in calls)
    assert now < 12  # The broad last growth level also leaves fallback time.


@pytest.mark.parametrize("local_seconds,remaining", [(5.0, 7.0), (15.0, 0.0)])
def test_local_and_whole_fallback_share_one_deadline(
    monkeypatch: pytest.MonkeyPatch, local_seconds: float, remaining: float
) -> None:
    original, completed = _fit(8, {2}), _fit(8, set())
    now = 100.0
    deadlines: list[float] = []
    whole_budgets: list[float] = []

    def local(
        fit: fitting._Fit,
        _index: int,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        deadline: float,
    ) -> tuple[None, dict[str, object]]:
        nonlocal now
        assert fit is original
        deadlines.append(deadline)
        now += local_seconds
        return None, {"status": "unrefined"}

    def whole(
        fit: fitting._Fit,
        _index: int,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        seconds: float,
    ) -> tuple[fitting._Fit, dict[str, object]]:
        assert fit is original
        whole_budgets.append(seconds)
        return completed, {"status": "accepted"}

    monkeypatch.setattr(fitting, "monotonic", lambda: now)
    monkeypatch.setattr(curvature_fitting, "_SECONDS", 12.0)
    monkeypatch.setattr(fitting, "_local_curvature_refit", local)
    monkeypatch.setattr(fitting, "_whole_curvature_refit", whole)
    result, report = fitting._curvature_candidate(original, 0, _LIMIT, 0.2)
    assert result is completed
    assert deadlines == [112.0]
    assert whole_budgets == [remaining]
    assert report["strategy"] == "whole-chain"
    assert report["total_elapsed_seconds"] == local_seconds


def test_adjacent_float_interval_midpoints_keep_guided_and_uniform_searches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0))
    boundaries = [0.0, 0.5, nextafter(0.5, 1.0), 1.0]
    original = fitting._Fit(
        c._Chain(points, fixed=False),
        np.asarray(boundaries),
        [],
        [],
        [
            c._Piece(0, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        list(pairwise(boundaries)),
    )
    guide = local_windows.prepare(
        local_windows.Window(0, len(original.pieces)),
        original.chain.points,
        original.parameters,
        [p.span for p in original.pieces],
        original.intervals,
    )
    now = 0.0
    calls: list[tuple[str, float]] = []

    def search(
        _points: curvature_fitting.FloatArray,
        _parameters: curvature_fitting.FloatArray,
        _offset: float,
        _sign: int,
        _tolerance: float,
        *,
        knot_breaks: curvature_fitting.FloatArray | None,
        time_budget_seconds: float,
        **_kwargs: object,
    ) -> tuple[None, dict[str, object]]:
        nonlocal now
        if knot_breaks is not None:
            assert knot_breaks[0] == 0 and knot_breaks[-1] == 1
            assert np.all(knot_breaks[1:] > knot_breaks[:-1])
            assert all(value in knot_breaks for value in boundaries)
            calls.append(("guided", time_budget_seconds))
            now += 1.0
        else:
            calls.append(("uniform", time_budget_seconds))
        return None, {"status": "unrefined"}

    monkeypatch.setattr(fitting, "monotonic", lambda: now)
    monkeypatch.setattr(curvature_fitting, "refit", search)
    candidate, report = fitting._local_window_refit(
        original, 0, guide, _LIMIT, 0.2, 4.0
    )
    assert candidate is None
    assert calls == [("uniform", 2.0), ("guided", 4.0)]
    assert report["status"] == "unrefined"


@pytest.mark.parametrize("expected_rejection", [True, False])
def test_optional_circle_rejects_only_expected_conversion_failures(
    monkeypatch: pytest.MonkeyPatch, expected_rejection: bool
) -> None:
    points = ((0.0, 0.0), (0.18, 0.0), (0.18, 0.18), (0.0, 0.18), (0.0, 0.0))
    original = fitting._Fit(
        c._Chain(points, fixed=False),
        np.linspace(0, 1, len(points)),
        [],
        [],
        [
            c._Piece(0, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        [(i / 4, (i + 1) / 4) for i in range(4)],
    )
    message = (
        "smooth outline chain 0 has a stationary join at 0.25"
        if expected_rejection
        else "unexpected conversion invariant"
    )
    calls = 0

    def conversion(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise ValueError(message)

    monkeypatch.setattr(fitting, "_spline_pieces", conversion)
    if expected_rejection:
        assert fitting._pinned_circle_candidate(original, 0, _LIMIT, 0.2) is None
    else:
        with pytest.raises(ValueError, match=message):
            _ = fitting._pinned_circle_candidate(original, 0, _LIMIT, 0.2)
    assert calls == 1
