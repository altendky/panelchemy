"""A soft-budget corner proof cannot starve the ordinary smooth repair family."""

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from itertools import pairwise
from typing import cast

import numpy as np
import pytest
from scipy.interpolate import BSpline

from panelchemy import boundary_tapers, curvature, curvature_fitting, local_windows
from panelchemy import curves as c
from panelchemy import outline_fitting as fitting

_LIMIT = boundary_tapers.BoundaryTaper(0, 2, 0.1, 1)


def corner_window() -> tuple[fitting._Fit, local_windows.Guide]:
    points: tuple[c.Point2, ...] = ((-1, 0), (0, 0), (1, 0), (1, 1), (1, 2))
    times = np.linspace(0, 1, len(points))
    fit = fitting._Fit(
        c._Chain(points, False),
        times,
        [0.5],
        [0.5],
        [
            c._Piece(0, i, i + 1, c._line(a, b))
            for i, (a, b) in enumerate(pairwise(points))
        ],
        list(pairwise(times.tolist())),
        source_corners=(0.5,),
    )
    guide = local_windows.prepare(
        local_windows.Window(1, 2),
        points,
        times,
        [piece.span for piece in fit.pieces],
        fit.intervals,
    )
    return fit, guide


def test_smooth_search_precedes_a_corner_proof_that_overruns_its_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, guide = corner_window()
    now = 0.0
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(fitting, "monotonic", lambda: now)

    def choices(*_args: object) -> list[list[float]]:
        return [[], [0.4]]

    monkeypatch.setattr(fitting, "_curvature_corner_choices", choices)

    def search(
        *_args: object,
        knot_breaks: curvature_fitting.FloatArray | None,
        time_budget_seconds: float,
        **_kwargs: object,
    ) -> tuple[None, dict[str, object]]:
        nonlocal now
        assert 0 < time_budget_seconds <= 4 - now
        if knot_breaks is None:
            calls.append(("uniform", time_budget_seconds))
            now += 0.5
        else:
            calls.append(("corner", time_budget_seconds))
            # Exact acceptance proofs can finish after a soft solver deadline.
            now += 10
        return None, {"status": "time-budget"}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    candidate, report = fitting._local_window_refit(fit, 0, guide, _LIMIT, 0.2, 4)
    assert candidate is None
    assert calls == [("uniform", 2.0), ("corner", 1.75)]
    assert report["status"] == "time-budget"


def test_uniform_only_plan_gets_the_whole_available_window_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, guide = corner_window()
    monkeypatch.setattr(fitting, "monotonic", lambda: 0.0)

    def no_choices(*_args: object) -> list[list[float]]:
        return []

    monkeypatch.setattr(fitting, "_curvature_corner_choices", no_choices)
    budgets: list[float] = []

    def search(
        *_args: object, time_budget_seconds: float, **kwargs: object
    ) -> tuple[None, dict[str, object]]:
        assert kwargs["knot_breaks"] is None
        budgets.append(time_budget_seconds)
        return None, {"status": "unrefined"}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    candidate, _ = fitting._local_window_refit(fit, 0, guide, _LIMIT, 0.2, 4)
    assert candidate is None and budgets == [4.0]


@pytest.mark.parametrize("family", ["compact", "apex"])
@pytest.mark.parametrize("protected", [False, True])
def test_single_family_overrides_keep_their_budget_and_corner_protection(
    monkeypatch: pytest.MonkeyPatch, family: str, protected: bool
) -> None:
    fit, guide = corner_window()
    if not protected:
        fit.corners = []
    monkeypatch.setattr(fitting, "monotonic", lambda: 0.0)
    calls: list[float] = []

    def search(
        *_args: object, time_budget_seconds: float, **kwargs: object
    ) -> tuple[None, dict[str, object]]:
        assert not protected
        calls.append(time_budget_seconds)
        assert np.array_equal(
            cast("curvature_fitting.FloatArray", kwargs["knot_breaks"]), [0, 0.5, 1]
        )
        if family == "compact":
            assert np.array_equal(
                cast("curvature_fitting.FloatArray", kwargs["c1_parameters"]), [0.5]
            )
            assert kwargs["corner_parameters"] is None
            assert kwargs["source_search_fraction"] == 1.0
        else:
            assert kwargs["c1_parameters"] is None
            assert np.array_equal(
                cast("curvature_fitting.FloatArray", kwargs["corner_parameters"]), [0.5]
            )
        return None, {"status": "unrefined"}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    candidate, report = fitting._local_window_refit(
        fit,
        0,
        guide,
        _LIMIT,
        0.2,
        4,
        compact=family == "compact",
        apex=0.5 if family == "apex" else None,
    )
    assert candidate is None
    assert calls == ([] if protected else [4.0])
    assert report["status"] == ("protected-corner" if protected else "unrefined")


def test_corner_family_remains_available_after_smooth_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit, guide = corner_window()
    original_pieces = tuple(fit.pieces)
    source_poles = [
        (0, 0),
        (1 / 3, 0),
        (2 / 3, 0),
        (1, 0),
        (1, 1 / 3),
        (1, 2 / 3),
        (1, 1),
    ]
    proposal: BSpline[np.float64] = BSpline(
        [0] * 4 + [0.5] * 3 + [1] * 4, np.asarray(source_poles), 3
    )
    calls: list[str] = []

    def search(
        *_args: object, **kwargs: object
    ) -> tuple[BSpline[np.float64] | None, dict[str, object]]:
        if kwargs["knot_breaks"] is None:
            calls.append("uniform")
            return None, {"status": "unrefined"}
        calls.append("corner")
        assert np.array_equal(
            cast("curvature_fitting.FloatArray", kwargs["corner_parameters"]), [0.5]
        )
        accept = cast("Callable[[BSpline[np.float64]], bool]", kwargs["accept"])
        assert accept(proposal)  # All original source/radius/separation checks run.
        return proposal, {"status": "accepted"}

    monkeypatch.setattr(curvature_fitting, "refit", search)
    candidate, _ = fitting._local_window_refit(fit, 0, guide, _LIMIT, 0.2, 2)
    assert candidate is not None and calls == ["uniform", "corner"]
    assert 0.5 in candidate.corners
    assert candidate.pieces[0].span is original_pieces[0].span
    assert candidate.pieces[-1].span is original_pieces[-1].span
    assert not fitting._source_failures(candidate, 0.2)
    assert all(
        curvature.within_curvature(p.span, _LIMIT.offset_mm, _LIMIT.inward_sign)
        for p in candidate.pieces
    )
