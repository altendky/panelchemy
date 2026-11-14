"""Oversized dense fits split only at already-declared source corners."""

# pyright: reportPrivateUsage=false

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from panelchemy import curves as c
from panelchemy import outline_fitting as of
from panelchemy.shade_guides import GuideSamples
from panelchemy.spline_solver import SplineRefinementNeeded


def oversized_fit(count: int = 227) -> of._Fit:
    points = tuple((float(i), float(i % 2)) for i in range(count))
    parameters = np.linspace(0, 1, count)
    corners = list(parameters[1:-1])
    return of._Fit(
        c._Chain(points, False),
        parameters,
        corners,
        corners.copy(),
        [],
        [],
        fairing_mm=0.0,
        source_corners=tuple(corners),
    )


def controls(fit: of._Fit) -> int:
    return 4 + sum(3 if value in fit.corners else 1 for value in fit.knots)


def test_oversized_initial_open_fit_retains_exact_source_corners_and_global_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = oversized_fit()
    assert controls(fit) == 679
    with pytest.raises(ValueError, match="exceeds 512 control points"):
        _ = of._spline(fit, 0.18)
    original = of._spline
    solves: list[of._Fit] = []

    def solve(selected: of._Fit, target: float):
        assert controls(selected) <= 512
        solves.append(selected)
        return original(selected, target)

    monkeypatch.setattr(of, "_spline", solve)
    pieces, intervals = of._pieces(fit, 37, 0.2)
    assert len(solves) == 2
    assert not fit.pieces and not fit.intervals
    assert intervals == list(pairwise([0.0, *fit.knots, 1.0]))
    assert pieces[0].span.control_points[0] == fit.chain.points[0]
    assert pieces[-1].span.control_points[-1] == fit.chain.points[-1]
    assert solves[0].chain.points[-1] == solves[1].chain.points[0]
    assert solves[0].chain.points[-1] in fit.chain.points[1:-1]
    for index, piece in enumerate(pieces):
        assert (piece.chain, piece.start, piece.stop) == (37, index, index + 1)
    assert all(
        a.span.control_points[-1] == b.span.control_points[0]
        for a, b in pairwise(pieces)
    )
    checked = replace(fit, pieces=pieces, intervals=intervals)
    assert not of._source_failures(checked, 0.2)
    assert all(c._monotone(piece.span) for piece in pieces)


def test_recursive_blocks_preserve_guide_registration_and_physical_fairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = oversized_fit(720)
    fit.fairing_mm = 0.25
    fit.guide = GuideSamples(
        ((0.0, 0.0), (90.0, 4.0), (480.0, -3.0), (719.0, 1.0)),
        (0.0, 0.13, 0.67, 1.0),
    )
    selected: list[of._Fit] = []

    def straight(block: of._Fit, index: int, tolerance: float):
        assert controls(block) <= 512
        selected.append(block)
        times = [0.0, *block.knots, 1.0]
        return (
            [
                c._Piece(index, i, i + 1, c._line(a, b, tolerance))
                for i, (a, b) in enumerate(pairwise(block.chain.points))
            ],
            list(pairwise(times)),
        )

    # Inspect leaf construction without spending time on eight dense solves.
    original_pieces = of._pieces

    def dispatch(block: of._Fit, index: int, tolerance: float):
        if controls(block) <= 512:
            return straight(block, index, tolerance)
        return original_pieces(block, index, tolerance)

    monkeypatch.setattr(of, "_pieces", dispatch)
    pieces, intervals = dispatch(fit, 0, 0.2)
    assert len(selected) == 8
    assert intervals == list(pairwise([0.0, *fit.knots, 1.0]))
    assert len(pieces) == len(intervals)
    for block in selected:
        assert block.fairing_mm == fit.fairing_mm
        assert block.guide is not None
        first = fit.chain.points.index(block.chain.points[0])
        last = fit.chain.points.index(block.chain.points[-1])
        a, b = of._scalar(fit.parameters, first), of._scalar(fit.parameters, last)
        local = np.array(block.guide.parameters)
        assert np.allclose(
            np.asarray(block.guide.points), fit.guide.sample(a + (b - a) * local)
        )
        assert all(t in block.corners for t in block.source_corners)


def test_child_refinement_failure_is_rebound_to_exact_original_knots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = oversized_fit()
    expected: tuple[float, float] | None = None

    def fail(block: of._Fit, _target: float):
        nonlocal expected
        first = fit.chain.points.index(block.chain.points[0])
        expected = fit.knots[first], fit.knots[first + 1]
        raise SplineRefinementNeeded(
            "child needs refinement", (block.knots[0], block.knots[1])
        )

    monkeypatch.setattr(of, "_spline", fail)
    with pytest.raises(SplineRefinementNeeded) as caught:
        _ = of._pieces(fit, 0, 0.2)
    assert caught.value.interval == expected


@pytest.mark.parametrize("unsupported", ["closed", "free-loop", "no-source-corners"])
def test_unsupported_fit_keeps_existing_solver_guard(unsupported: str) -> None:
    fit = oversized_fit()
    if unsupported == "closed":
        fit.chain = replace(
            fit.chain, points=(*fit.chain.points[:-1], fit.chain.points[0])
        )
    elif unsupported == "free-loop":
        fit.chain = replace(fit.chain, free_loop=True)
    else:
        fit.source_corners = ()
    with pytest.raises(ValueError, match="exceeds 512 control points"):
        _ = of._pieces(fit, 0, 0.2)


def test_small_fit_retains_the_original_single_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = oversized_fit(8)
    original = of._spline
    seen: list[of._Fit] = []

    def solve(selected: of._Fit, target: float):
        seen.append(selected)
        return original(selected, target)

    monkeypatch.setattr(of, "_spline", solve)
    _ = of._pieces(fit, 0, 0.2)
    assert seen == [fit]
