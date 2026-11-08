"""Each chain receives its bounded refinement attempts before topology checks."""

# These tests inject independent local failures into the actual fitting driver.
# pyright: reportPrivateUsage=false

from collections import Counter
from itertools import pairwise
from typing import cast

import pytest
from shapely.geometry import box
from shapely.geometry.polygon import orient

from panelchemy import geometric_distance
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours
from panelchemy.curves import Point2
from panelchemy.spline_solver import SplineRefinementNeeded

_TOLERANCE = 0.2


def source() -> PanelContours:
    return PanelContours(
        box(0, 0, 16, 16),
        [],
        [orient(box(2, 2, 6, 6)), orient(box(9, 9, 13, 13))],
        0.1,
        (0, 0, 160, 160),
        12,
    )


def snapshot(contours: PanelContours) -> tuple[bytes, ...]:
    return tuple(
        polygon.wkb
        for polygon in (contours.silhouette, *contours.orange, *contours.white)
    )


def longest_interval(fit: fitting._Fit) -> tuple[float, float]:
    return max(pairwise([0.0, *fit.knots, 1.0]), key=lambda p: p[1] - p[0])


def test_later_constraint_retries_do_not_starve_earlier_source_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = source()
    before = snapshot(original)
    pieces = fitting._pieces
    source_failures = fitting._source_failures
    roles: dict[tuple[Point2, ...], str] = {}
    indices: dict[str, int] = {}
    attempts: Counter[str] = Counter()
    forced_source_failures = 0

    def controlled_pieces(fit: fitting._Fit, index: int, tolerance: float):
        if fit.chain.points not in roles:
            role = "early" if not roles else "late"
            roles[fit.chain.points] = role
            indices[role] = index
        role = roles[fit.chain.points]
        attempts[role] += 1
        if role == "late" and attempts[role] <= 12:
            raise SplineRefinementNeeded(
                "later chain needs more basis freedom", longest_interval(fit)
            )
        return pieces(fit, index, tolerance)

    def controlled_source_failures(fit: fitting._Fit, tolerance: float) -> set[int]:
        nonlocal forced_source_failures
        failed = source_failures(fit, tolerance)
        if roles[fit.chain.points] == "early" and forced_source_failures < 2:
            forced_source_failures += 1
            interval = max(
                range(len(fit.intervals)),
                key=lambda i: fit.intervals[i][1] - fit.intervals[i][0],
            )
            return failed | {interval}
        return failed

    monkeypatch.setattr(fitting, "_pieces", controlled_pieces)
    monkeypatch.setattr(fitting, "_source_failures", controlled_source_failures)
    result = fitting.fit_outlines(original, _TOLERANCE, fairing_mm=0)

    assert len(roles) == 2
    assert indices["early"] < indices["late"]
    assert forced_source_failures == 2
    assert attempts == {"early": 3, "late": 13}
    assert result.diagnostics["constraint_refinements"] == 12
    assert snapshot(original) == before
    assert result.proxy.silhouette.equals(original.silhouette)
    assert not result.contours.orange
    assert len(result.contours.white) == len(result.proxy.white) == 2
    for polygon, curve, proxy in zip(
        original.white, result.contours.white, result.proxy.white, strict=True
    ):
        assert proxy.is_valid and original.silhouette.covers(proxy)
        assert not curve.interiors and not proxy.interiors
        path = tuple((float(x), float(y)) for x, y in polygon.exterior.coords)
        assert geometric_distance.within_paths(
            (path,), curve.exterior.spans, _TOLERANCE
        )
        assert all(
            a.control_points[-1] == b.control_points[0]
            for a, b in pairwise((*curve.exterior.spans, curve.exterior.spans[0]))
        )


def test_constraint_and_source_failures_share_one_fourteen_attempt_chain_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = source()
    before = snapshot(original)
    pieces = fitting._pieces
    source_failures = fitting._source_failures
    target: tuple[Point2, ...] | None = None
    attempts = constraint_failures = source_checks = 0
    last_fit: fitting._Fit | None = None

    def controlled_pieces(fit: fitting._Fit, index: int, tolerance: float):
        nonlocal target, attempts, constraint_failures, last_fit
        if target is None:
            target = fit.chain.points
        if fit.chain.points == target:
            attempts += 1
            last_fit = fit
            if constraint_failures < 7:
                constraint_failures += 1
                raise SplineRefinementNeeded(
                    "same chain needs more basis freedom", longest_interval(fit)
                )
        return pieces(fit, index, tolerance)

    def controlled_source_failures(fit: fitting._Fit, tolerance: float) -> set[int]:
        nonlocal source_checks
        failed = source_failures(fit, tolerance)
        if fit.chain.points == target:
            source_checks += 1
            interval = max(
                range(len(fit.intervals)),
                key=lambda i: fit.intervals[i][1] - fit.intervals[i][0],
            )
            return failed | {interval}
        return failed

    monkeypatch.setattr(fitting, "_pieces", controlled_pieces)
    monkeypatch.setattr(fitting, "_source_failures", controlled_source_failures)
    with pytest.raises(fitting.OutlineFitError) as stopped:
        _ = fitting.fit_outlines(original, _TOLERANCE, fairing_mm=0)

    assert attempts == 14
    assert constraint_failures == source_checks == 7
    assert snapshot(original) == before
    checked = cast("fitting._Fit | None", last_fit)
    assert checked is not None
    error = stopped.value
    assert error.source_points == target
    assert error.interval in checked.intervals
    failed_index = checked.intervals.index(error.interval)
    assert error.failed_span == checked.pieces[failed_index].span
    assert error.candidate_rings  # Unaccepted geometry remains available for review.
