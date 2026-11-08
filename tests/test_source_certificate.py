"""Sequential subdivision preserves the original exact source certificate."""

# These differential tests exercise the exact internal certificate directly.
# pyright: reportPrivateUsage=false

from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import pairwise
from math import nextafter
from random import Random
from typing import cast

import numpy as np
import pytest

from panelchemy import curves as c
from panelchemy.outline_fitting import _certified, _Fit


def _independent_certificate(fit: _Fit, piece: c._Piece, tolerance: float) -> bool:
    """Previous algorithm: restrict every interval from the original cubic."""
    start_value, stop_value = fit.intervals[piece.start]
    parameters = cast("list[float]", fit.parameters.tolist())
    first = bisect_right(parameters, start_value) - 1
    last = bisect_left(parameters, stop_value)
    start, stop = Fraction(start_value), Fraction(stop_value)
    times = [start, *(Fraction(v) for v in parameters[first + 1 : last]), stop]
    controls = c._controls(piece.span)
    budget = Fraction(tolerance) ** 2
    for index, (a, b) in enumerate(pairwise(times), first):
        selected = c._restrict(
            controls, (a - start) / (stop - start), (b - start) / (stop - start)
        )
        source_start, source_stop = map(Fraction, parameters[index : index + 2])
        left, right = map(c._rational, fit.chain.points[index : index + 2])
        p = c._mix(left, right, (a - source_start) / (source_stop - source_start))
        q = c._mix(left, right, (b - source_start) / (source_stop - source_start))
        source = (p, c._mix(p, q, Fraction(1, 3)), c._mix(p, q, Fraction(2, 3)), q)
        difference = cast(
            "c.RationalControls",
            tuple(
                (p[0] - q[0], p[1] - q[1])
                for p, q in zip(selected, source, strict=True)
            ),
        )
        if not c._bounded(difference, budget):
            return False
    return True


def _case(random: Random, index: int) -> tuple[_Fit, c._Piece]:
    parameters = sorted([0.0, *(random.random() for _ in range(12)), 1.0])
    start = 0.0 if index % 3 == 0 else random.uniform(0.01, 0.3)
    stop = 1.0 if index % 4 == 0 else random.uniform(0.7, 0.99)
    points = tuple((t, random.uniform(-0.05, 0.05)) for t in parameters)
    controls: c.Controls = (
        (start, random.uniform(-0.2, 0.2)),
        (start + (stop - start) / 3, random.uniform(-0.2, 0.2)),
        (start + 2 * (stop - start) / 3, random.uniform(-0.2, 0.2)),
        (stop, random.uniform(-0.2, 0.2)),
    )
    piece = c._Piece(0, 0, 1, c.BezierSpan(controls, index % 2 == 0))
    fit = _Fit(
        c._Chain(points, fixed=False),
        np.asarray(parameters),
        [],
        [],
        [piece],
        [(start, stop)],
    )
    return fit, piece


def test_every_exact_interval_matches_independent_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random = Random(7295)
    calls: list[tuple[c.RationalControls, Fraction]] = []

    def record(controls: c.RationalControls, budget: Fraction) -> bool:
        calls.append((controls, budget))
        return True

    monkeypatch.setattr(c, "_bounded", record)
    for index in range(80):
        fit, piece = _case(random, index)
        assert _independent_certificate(fit, piece, 0.2)
        expected = calls.copy()
        calls.clear()
        assert _certified(fit, piece, 0.2)
        assert calls == expected
        calls.clear()


def test_acceptance_matches_independent_certificate() -> None:
    random = Random(1037)
    outcomes: set[bool] = set()
    for index in range(60):
        fit, piece = _case(random, index)
        for tolerance in (0.01, 0.1, 0.3):
            expected = _independent_certificate(fit, piece, tolerance)
            outcomes.add(expected)
            assert _certified(fit, piece, tolerance) == expected
    assert outcomes == {False, True}


@pytest.mark.parametrize("interval", [(0.0, 1.0), (0.03125, 0.9375), (0.25, 0.375)])
def test_exact_budget_equality_and_neighboring_limits(
    interval: tuple[float, float],
) -> None:
    parameters = [0.0, 0.001, 0.137, 0.5, 0.999, 1.0]
    start, stop = interval
    piece = c._Piece(0, 0, 1, c._line((start, 0.25), (stop, 0.25)))
    fit = _Fit(
        c._Chain(tuple((t, 0.0) for t in parameters), fixed=False),
        np.asarray(parameters),
        [],
        [],
        [piece],
        [interval],
    )
    for tolerance, expected in (
        (nextafter(0.25, 0.0), False),
        (0.25, True),
        (nextafter(0.25, 1.0), True),
    ):
        assert _independent_certificate(fit, piece, tolerance) == expected
        assert _certified(fit, piece, tolerance) == expected
