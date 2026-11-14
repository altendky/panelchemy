"""Objective guidance retains the original constraints and all hard gates."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from scipy.interpolate import BSpline

from panelchemy import (
    boundary_tapers,
    curvature_fitting,
    curves,
    local_windows,
    spline_solver,
)
from panelchemy import outline_fitting as of
from panelchemy.shade_guides import GuideSamples


def fit() -> of._Fit:
    return of._Fit(
        curves._Chain(((0.0, 0.0), (1.0, 0.1), (2.0, 0.0)), False),
        np.array([0.0, 0.5, 1.0]),
        [],
        [0.5],
        [],
        [],
    )


def test_spline_guide_changes_only_rhs_not_source_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fit()
    calls: list[tuple[of.FloatArray | float, ...]] = []
    real = spline_solver.solve_coefficients

    def solve(
        matrix: of.FloatSparse,
        target: of.FloatArray,
        transform: of.FloatSparse,
        offsets: of.FloatArray,
        knots: of.FloatArray,
        source_parameters: of.FloatArray,
        source_points: of.FloatArray,
        length_mm: float,
        fairing_mm: float,
        *,
        fit_target_mm: float = 0,
    ) -> of.FloatArray:
        calls.append(
            (
                matrix.toarray(),
                target.copy(),
                transform.toarray(),
                offsets.copy(),
                knots.copy(),
                source_parameters.copy(),
                source_points.copy(),
                length_mm,
                fairing_mm,
                fit_target_mm,
            )
        )
        return real(
            matrix,
            target,
            transform,
            offsets,
            knots,
            source_parameters,
            source_points,
            length_mm,
            fairing_mm,
            fit_target_mm=fit_target_mm,
        )

    monkeypatch.setattr(of, "solve_coefficients", solve)
    hard = of._spline(original, 0.18)
    guided = replace(
        original,
        guide=GuideSamples(((0.0, 0.0), (1.0, 0.15), (2.0, 0.0)), (0.0, 0.5, 1.0)),
    )
    shaded = of._spline(guided, 0.18)
    assert not np.array_equal(calls[0][1], calls[1][1])
    for index in (0, 2, 3, 4, 5, 6, 7, 8, 9):
        assert np.array_equal(calls[0][index], calls[1][index])
    assert original.chain == guided.chain and original.parameters is guided.parameters
    assert original.corners is guided.corners and original.knots is guided.knots
    assert hard(0).tolist() == shaded(0).tolist() == [0, 0]
    assert hard(1).tolist() == shaded(1).tolist() == [2, 0]


def test_identical_nonbinary_guide_centers_before_sampling_bit_exactly() -> None:
    points = ((1000.1, 2000.2), (1001.4, 2000.27), (1003.7, 2000.13))
    original = of._Fit(
        curves._Chain(points, False),
        np.array([0.0, 0.37, 1.0]),
        [],
        [0.23, 0.71],
        [],
        [],
    )
    hard = of._spline(original, 0.18)
    guided = of._spline(
        replace(original, guide=GuideSamples(points, (0.0, 0.37, 1.0))), 0.18
    )
    assert np.array_equal(hard.t, guided.t)
    assert np.array_equal(hard.c, guided.c)


def test_source_certificate_rejects_guide_that_strays_from_original():
    original = fit()
    shifted = curves.BezierSpan(
        ((0.0, 1.0), (0.5, 1.0), (1.5, 1.0), (2.0, 1.0)), False, 0.2
    )
    guided = replace(
        original,
        guide=GuideSamples(((0.0, 1.0), (2.0, 1.0)), (0.0, 1.0)),
        pieces=[curves._Piece(0, 0, 1, shifted)],
        intervals=[(0, 1)],
    )
    assert of._source_failures(guided, 0.2)


def test_local_search_uses_guide_but_source_and_radius_still_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fit()
    original.pieces, original.intervals = of._pieces(original, 0, 0.2)
    guided = replace(
        original,
        guide=GuideSamples(((0.0, 0.0), (1.0, 0.15), (2.0, 0.0)), (0.0, 0.5, 1.0)),
    )
    points, parameters = of._search_samples(guided)
    assert np.array_equal(points[1], (1, 0.15))
    assert np.array_equal(parameters, original.parameters)
    guide = local_windows.prepare(
        local_windows.Window(0, len(guided.pieces)),
        points,
        parameters,
        [p.span for p in guided.pieces],
        guided.intervals,
    )
    real_refit = curvature_fitting.refit

    def inspect_search(
        points: of.FloatArray,
        _parameters: of.FloatArray,
        *_args: object,
        **kwargs: object,
    ):
        assert np.array_equal(points, guide.source_points)
        accept = kwargs["accept"]
        assert callable(accept)
        bad = BSpline(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([[0.0, 0.0], [0.0, 10.0], [2.0, 10.0], [2.0, 0.0]]),
            3,
        )
        assert not accept(bad)
        return None, {"status": "test-rejected"}

    monkeypatch.setattr(curvature_fitting, "refit", inspect_search)
    candidate, _ = of._local_window_refit(
        guided,
        0,
        guide,
        boundary_tapers.BoundaryTaper(0, 2, 0.35, 1),
        0.2,
        0.05,
        compact=True,
    )
    assert candidate is None
    assert original.chain == guided.chain
    monkeypatch.setattr(curvature_fitting, "refit", real_refit)


def test_periodic_guide_preserves_free_basis_and_original_seam_constraints() -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    original = of._Fit(
        curves._Chain(points, False, True),
        np.linspace(0, 1, 5),
        [],
        [0.25, 0.5, 0.75],
        [],
        [],
    )
    shaded = replace(
        original,
        guide=GuideSamples(
            ((0.125, 0.125), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.125, 0.125)),
            (0.0, 0.25, 0.5, 0.75, 1.0),
        ),
    )
    hard = of._spline(original, 0.2)
    gray = of._spline(shaded, 0.2)
    assert np.array_equal(hard(0), hard(1))
    assert np.array_equal(gray(0), gray(1))
    assert not np.array_equal(gray(0), points[0])
    assert shaded.chain.points == points


def test_source_window_fallback_keeps_source_corners_and_hard_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fit()
    original.pieces, original.intervals = of._pieces(original, 0, 0.2)
    original.guide = GuideSamples(
        ((0.0, 0.0), (1.0, 0.15), (2.0, 0.0)), (0.0, 0.5, 1.0)
    )
    real_failures = of._source_failures
    checks = 0

    def failures(candidate: of._Fit, tolerance: float) -> set[int]:
        nonlocal checks
        checks += 1
        return {0} if checks == 1 else real_failures(candidate, tolerance)

    def local(
        candidate: of._Fit,
        _index: int,
        guide: local_windows.Guide,
        _limit: boundary_tapers.BoundaryTaper,
        _tolerance: float,
        _seconds: float,
        *,
        compact: bool = False,
        apex: float | None = None,
    ):
        assert candidate.chain is original.chain
        assert compact and apex is None
        # Original midpoint objective (y=.1), rather than RGB objective (y=.15).
        assert np.max(guide.source_points[:, 1]) <= 0.1
        return candidate, {"status": "test-candidate"}

    monkeypatch.setattr(of, "_source_failures", failures)
    monkeypatch.setattr(of, "_local_window_refit", local)
    result = of._source_window_refit(
        original, 0, boundary_tapers.BoundaryTaper(0, 1, 0.0, 1), 0.2
    )
    assert result is not None and result.chain is original.chain
    assert result.corners == original.corners


def test_source_window_fallback_rejects_radius_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = fit()
    original.pieces, original.intervals = of._pieces(original, 0, 0.2)
    real_failures = of._source_failures
    checks = 0

    def failures(candidate: of._Fit, tolerance: float) -> set[int]:
        nonlocal checks
        checks += 1
        return {0} if checks == 1 else real_failures(candidate, tolerance)

    def local(candidate: of._Fit, *_args: object, **_kwargs: object):
        return candidate, {"status": "test-source-only"}

    monkeypatch.setattr(of, "_source_failures", failures)
    monkeypatch.setattr(of, "_local_window_refit", local)
    # This shallow source arch cannot satisfy such a large inward taper run.
    assert (
        of._source_window_refit(
            original, 0, boundary_tapers.BoundaryTaper(0, 2, 100.0, 1), 0.2
        )
        is None
    )


def test_source_window_corner_at_wrapped_seam_keeps_complete_parameter_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0))
    intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    original = of._Fit(
        curves._Chain(points, False, True),
        np.linspace(0, 1, 5),
        [0.0],
        [0.25, 0.5, 0.75],
        [
            curves._Piece(0, i, i + 1, curves._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        intervals,
        source_corners=(0.0,),
    )
    real_failures = of._source_failures
    checks = 0

    def failures(candidate: of._Fit, tolerance: float) -> set[int]:
        nonlocal checks
        checks += 1
        return {0, 3} if checks == 1 else real_failures(candidate, tolerance)

    def local(
        candidate: of._Fit,
        index: int,
        guide: local_windows.Guide,
        limit: boundary_tapers.BoundaryTaper,
        tolerance: float,
        seconds: float,
        *,
        compact: bool = False,
        apex: float | None = None,
    ):
        assert candidate.chain is original.chain and not candidate.corners
        assert index == 0 and limit.offset_mm == 0 and tolerance == 0.2 and seconds > 0
        assert guide.seam is not None and apex == guide.seam and not compact
        return original, {"status": "test-wrapped-corner"}

    monkeypatch.setattr(of, "_source_failures", failures)
    monkeypatch.setattr(of, "_local_window_refit", local)
    result = of._source_window_refit(
        original, 0, boundary_tapers.BoundaryTaper(0, 1, 0.0, 1), 0.2
    )
    assert (
        result is not None and result.intervals == intervals and result.corners == [0.0]
    )
