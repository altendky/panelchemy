"""Independent initial searches preserve ordering and central acceptance gates."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from itertools import pairwise
from multiprocessing.context import BaseContext
from typing import ClassVar, Never, Self, cast, override

import numpy as np
import pytest
from shapely.geometry import LineString, box

from panelchemy import boundary_tapers as bt
from panelchemy import curvature, curvature_fitting
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours


def job(index: int) -> fitting._CurvatureJob:
    points = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0))
    parameters = [0.0, 0.25, 0.5, 0.75, 1.0]
    fit = fitting._Fit(
        c._Chain(points, fixed=False, free_loop=True),
        np.asarray(parameters),
        parameters[:-1],
        parameters[1:-1],
        [
            c._Piece(index, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        list(pairwise(parameters)),
        fairing_mm=0,
    )
    return fit, index, bt.BoundaryTaper(0, 2, 0.2, 1), 0.2, [0]


def fake_trial(value: fitting._CurvatureJob) -> fitting._CurvatureTrial:
    return fitting._CurvatureTrial(value[0], [], 1, False, None, {"chain": value[1]})


def unexpected(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("this path must not start an executor")


class InlinePool:
    configurations: ClassVar[list[tuple[int, str]]] = []

    def __init__(self, *, max_workers: int, mp_context: BaseContext) -> None:
        self.configurations.append((max_workers, mp_context.get_start_method()))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def map(
        self,
        function: Callable[[fitting._CurvatureJob], fitting._CurvatureTrial],
        jobs: Sequence[fitting._CurvatureJob],
    ) -> Iterator[fitting._CurvatureTrial]:
        return map(function, jobs)


@pytest.mark.parametrize("workers,indices", [(1, [5, 2]), (8, [5]), (8, [])])
def test_serial_or_small_batches_do_not_start_a_pool(
    monkeypatch: pytest.MonkeyPatch, workers: int, indices: list[int]
) -> None:
    monkeypatch.setattr(fitting, "ProcessPoolExecutor", unexpected)
    monkeypatch.setattr(fitting, "_initial_curvature_trial", fake_trial)
    trials = fitting._curvature_trials([job(i) for i in indices], workers)
    assert [trial.record["chain"] for trial in trials] == indices


@pytest.mark.parametrize("workers,indices,cap", [(8, [5, 2], 2), (2, [7, 1, 4], 2)])
def test_pool_uses_spawn_caps_workers_and_preserves_job_order(
    monkeypatch: pytest.MonkeyPatch, workers: int, indices: list[int], cap: int
) -> None:
    InlinePool.configurations.clear()
    monkeypatch.setattr(fitting, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(fitting, "_initial_curvature_trial", fake_trial)
    jobs = [job(i) for i in indices]
    trials = fitting._curvature_trials(jobs, workers)
    assert InlinePool.configurations == [(cap, "spawn")]
    assert [trial.record["chain"] for trial in trials] == indices
    assert all(
        trial.proposal is value[0] for trial, value in zip(trials, jobs, strict=True)
    )


def fit_signature(fit: fitting._Fit | None) -> object:
    if fit is None:
        return None
    return (
        fit.chain,
        tuple(cast("list[float]", fit.parameters.tolist())),
        fit.corners,
        fit.knots,
        fit.pieces,
        fit.intervals,
        fit.fairing_mm,
        fit.source_corners,
        fit.texture_intervals,
        fit.texture_barriers,
    )


def test_real_spawned_trials_match_serial_geometry_and_result_addresses() -> None:
    jobs = [job(7), job(2)]
    serial = fitting._curvature_trials(jobs, 1)
    parallel = fitting._curvature_trials(jobs, 2)
    for value, direct, spawned in zip(jobs, serial, parallel, strict=True):
        assert fit_signature(spawned.proposal) == fit_signature(direct.proposal)
        assert [(fit_signature(fit), method) for fit, method in spawned.pool] == [
            (fit_signature(fit), method) for fit, method in direct.pool
        ]
        assert fit_signature(spawned.preferred) == fit_signature(direct.preferred)
        assert spawned.smooth_attempted == direct.smooth_attempted
        for key in (
            "chain",
            "left_material",
            "right_material",
            "offset_mm",
            "inward_sign",
        ):
            assert spawned.record[key] == direct.record[key]
        spawned_search = cast("dict[str, object]", spawned.record["search"])
        direct_search = cast("dict[str, object]", direct.record["search"])
        for key in ("status", "strategy"):
            assert spawned_search.get(key) == direct_search.get(key)
        assert spawned.record["chain"] == value[1]
        assert 0 < spawned.budget <= curvature_fitting._SECONDS


def test_spawned_child_failure_propagates_to_the_caller() -> None:
    broken = (*job(1)[:-1], [999])
    with pytest.raises(IndexError):
        _ = fitting._curvature_trials([broken, job(2)], 2)


def test_queue_wait_does_not_consume_a_childs_search_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    budgets: list[float | None] = []

    class DelayedPool(InlinePool):
        @override
        def map(
            self,
            function: Callable[[fitting._CurvatureJob], fitting._CurvatureTrial],
            jobs: Sequence[fitting._CurvatureJob],
        ) -> Iterator[fitting._CurvatureTrial]:
            clock[0] += 1000  # simulate queue wait before the child begins
            return super().map(function, jobs)

    def corners(*_args: object) -> list[tuple[fitting._Fit, str]]:
        clock[0] += 2
        return []

    def smooth(
        _fit: fitting._Fit,
        _index: int,
        _limit: bt.BoundaryTaper,
        _tolerance: float,
        *,
        time_budget_seconds: float | None,
    ) -> tuple[None, dict[str, object]]:
        budgets.append(time_budget_seconds)
        return None, {"status": "unrefined"}

    monkeypatch.setattr(fitting, "ProcessPoolExecutor", DelayedPool)
    monkeypatch.setattr(fitting, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fitting, "_corner_curvature_candidates", corners)
    monkeypatch.setattr(fitting, "_curvature_candidate", smooth)
    trials = fitting._curvature_trials([job(5), job(2)], 2)
    expected = curvature_fitting._SECONDS - 2
    assert budgets == [expected, expected]
    assert [trial.budget for trial in trials] == [expected, expected]
    assert all(trial.smooth_attempted for trial in trials)


def test_worker_proposal_is_applied_to_its_chain_then_rejected_by_palette_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PanelContours(
        box(0, 0, 10, 10), [], [box(1, 1, 3, 3), box(5, 1, 7, 3)], 1, (0, 0, 10, 10), 12
    )
    rings = [LineString(p.exterior.coords) for p in (source.silhouette, *source.white)]
    chains, uses = c._graph(rings, split_loops=False)
    fits: list[fitting._Fit] = []
    for index, chain in enumerate(chains):
        parameters = c._parameters(chain.points)
        fits.append(
            fitting._Fit(
                chain,
                np.asarray(parameters),
                [],
                [],
                [
                    c._Piece(index, i, i + 1, c._line(a, b))
                    for i, (a, b) in enumerate(pairwise(chain.points))
                ],
                list(pairwise(parameters)),
            )
        )
    pieces = [piece for fit in fits for piece in fit.pieces]
    _, cuts = c._unsafe_pair(pieces)
    ranges = [range(1), range(1, 2), range(2, 3)]
    original, proxy, _, _ = c._assemble(pieces, uses, ranges, source, 0.01, cuts)
    target = uses[2][0][0]
    bad_span = fits[target].pieces[0].span
    candidate = replace(
        fits[target],
        pieces=[
            replace(
                piece,
                span=replace(
                    piece.span,
                    control_points=cast(
                        "c.Controls",
                        tuple((x - 4, y) for x, y in piece.span.control_points),
                    ),
                ),
            )
            for piece in fits[target].pieces
        ],
    )
    seen: list[int] = []

    def trials(
        jobs: Sequence[fitting._CurvatureJob], workers: int
    ) -> list[fitting._CurvatureTrial]:
        assert workers == 2
        seen.extend(value[1] for value in jobs)
        assert seen == [target]
        return [
            fitting._CurvatureTrial(
                candidate,
                [],
                0,
                True,
                None,
                {"chain": target, "search": {"status": "accepted"}},
            )
        ]

    def no_repair(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
        return None, {"status": "unrefined"}

    monkeypatch.setattr(fitting, "_curvature_trials", trials)

    def radius(span: c.BezierSpan, *_args: object) -> bool:
        return span is not bad_span

    monkeypatch.setattr(curvature, "within_curvature", radius)
    for name in (
        "_compact_curvature_candidate",
        "_apex_curvature_candidate",
        "_polygon_curvature_candidate",
        "_corner_run_curvature_candidate",
    ):
        monkeypatch.setattr(fitting, name, no_repair)
    selected, actual, report = fitting._curvature_verified(
        fits, uses, ranges, source, 0.2, (0, 0.2, 0.4), original, proxy, workers=2
    )
    assert selected is original and actual is proxy
    summary = cast("dict[str, object]", report["boundary_curvature"])
    assert summary["topology_rejected_chains"] == [target]
    assert summary["unrefined_chains"] == [target]


@pytest.mark.parametrize("workers", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_worker_counts_are_rejected_before_any_fitting(workers: object) -> None:
    source = PanelContours(box(0, 0, 2, 2), [], [], 1, (0, 0, 2, 2), 4)
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = fitting.fit_outlines(source, 0, workers=cast("int", workers))
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        _ = fitting._curvature_trials([], cast("int", workers))
