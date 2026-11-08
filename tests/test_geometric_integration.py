"""Geometric source checks permit sliding while preserving complete coverage."""

# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
from shapely.geometry import LinearRing, Polygon, box

from panelchemy import curves as c
from panelchemy import geometric_distance, outline_simplification
from panelchemy import outline_fitting as fitting
from panelchemy.contours import PanelContours
from panelchemy.outline_simplification import SimplifiedChain


def sliding_fit() -> fitting._Fit:
    return fitting._Fit(
        c._Chain(((0, 0), (1, 0), (2, 0)), fixed=False),
        np.array([0.0, 0.5, 1.0]),
        [],
        [0.5],
        [
            c._Piece(0, 0, 1, c._line((0, 0), (1.5, 0))),
            c._Piece(0, 1, 2, c._line((1.5, 0), (2, 0))),
        ],
        [(0.0, 0.5), (0.5, 1.0)],
    )


def test_same_outline_can_slide_across_source_parameter_boundary() -> None:
    fit = sliding_fit()
    assert not all(fitting._certified(fit, p, 0.01) for p in fit.pieces)
    assert not fitting._source_failures(fit, 0.01)


def test_source_feature_cannot_be_lost_behind_forward_distance() -> None:
    fit = sliding_fit()
    source = ((0, 0), (1, 0), (1, 1), (1, 0), (2, 0))
    fit = replace(
        fit,
        chain=c._Chain(source, fixed=False),
        parameters=np.linspace(0, 1, len(source)),
    )
    # Every point of the fitted line belongs to the source. Reverse coverage
    # must still reject omission of the one-millimeter source excursion.
    assert fitting._source_failures(fit, 0.2)


def test_existing_continuous_proof_is_a_sufficient_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = sliding_fit()
    fit = replace(
        fit,
        pieces=[
            c._Piece(0, 0, 1, c._line((0, 0), (1, 0))),
            c._Piece(0, 1, 2, c._line((1, 0), (2, 0))),
        ],
    )

    def unexpected(*_args: object) -> bool:
        raise AssertionError("existing exact proof already covers both directions")

    monkeypatch.setattr(geometric_distance, "within_distance", unexpected)
    assert not fitting._source_failures(fit, 0.01)


def test_partial_parameter_proof_cannot_stand_in_for_full_source_coverage() -> None:
    fit = sliding_fit()
    fit = replace(fit, pieces=fit.pieces[:1], intervals=fit.intervals[:1])
    with pytest.raises(ValueError, match="complete chain partition"):
        _ = fitting._source_failures(fit, 0.01)


def test_zero_geometric_distance_does_not_allow_reversed_material_sides() -> None:
    source = ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1), (0.0, 0.0))
    reversed_points = tuple(reversed(source))
    pieces = [
        c._Piece(0, i, i + 1, c._line(a, b))
        for i, (a, b) in enumerate(pairwise(reversed_points))
    ]
    fit = fitting._Fit(
        c._Chain(source, fixed=False, free_loop=True),
        np.linspace(0, 1, len(source)),
        [],
        [0.25, 0.5, 0.75],
        pieces,
        [(i / 4, (i + 1) / 4) for i in range(4)],
    )
    assert geometric_distance.within_distance(source, [p.span for p in pieces], 0)
    # Even the previous matched-parameter proof can pass on a tiny loop.
    assert all(fitting._certified(fit, p, 0.2) for p in pieces)
    assert fitting._source_failures(fit, 0.2) == set(range(4))
    assert fitting._orientation_preserved(
        replace(fit, chain=c._Chain(reversed_points, fixed=False, free_loop=True))
    )


def test_softening_rolls_back_when_whole_chain_exceeds_source_distance() -> None:
    source = PanelContours(
        box(0, 0, 10, 10), [box(2, 2, 8, 8)], [], 0.1, (0, 0, 100, 100), 8
    )
    original = fitting.fit_outlines(source, 0.01)
    softened = fitting.fit_outlines(source, 0.01, corner_softening_mm=0.25)
    assert softened.contours == original.contours
    assert softened.diagnostics["softened_corners"] == 0
    candidates = softened.diagnostics["softening_candidates"]
    assert isinstance(candidates, int) and candidates > 0


def test_combined_simplification_must_recertify_complete_source_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [Polygon([(1, 1), (3, 1), (3, 2), (2, 3), (1, 2)])],
        [],
        0.1,
        (0, 0, 100, 100),
        9,
    )
    original = fitting.fit_outlines(source, 0.05)
    calls = 0

    def incomplete_merges(
        spans: list[c.BezierSpan],
        _intervals: list[tuple[float, float]],
        _budget: float,
        **_kwargs: object,
    ) -> SimplifiedChain:
        nonlocal calls
        calls += 1
        # Simulate individually approved edits whose combination has lost the
        # roof tip. Island count and containment alone would allow this shape.
        points = [(1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
        if not LinearRing([s.control_points[0] for s in spans]).is_ccw:
            points.reverse()
        merged = tuple(c._line(a, b) for a, b in pairwise(points))
        intervals = tuple((i / 4, (i + 1) / 4) for i in range(4))
        return SimplifiedChain(merged, intervals, len(spans), 4, 1, 1)

    monkeypatch.setattr(outline_simplification, "simplify", incomplete_merges)
    simplified = fitting.fit_outlines(source, 0.05, span_simplification_mm=0.01)
    assert calls > 0
    assert simplified.contours == original.contours


def test_initial_proofs_reuse_only_unchanged_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PanelContours(
        box(0, 0, 10, 10),
        [box(1, 1, 3, 3), box(6, 6, 8, 8)],
        [],
        0.1,
        (0, 0, 100, 100),
        12,
    )
    original = fitting._source_failures
    unsafe = c._unsafe_pair
    checked: dict[tuple[c.Point2, ...], list[list[c._Piece]]] = {}
    forced = False
    global_checks = 0

    def source_check(fit: fitting._Fit, tolerance: float) -> set[int]:
        nonlocal forced
        checked.setdefault(fit.chain.points, []).append(fit.pieces)
        failed = original(fit, tolerance)
        if not forced:
            forced = True
            return failed | {0}  # Force one replacement; leave its neighbor alone.
        return failed

    def global_check(
        pieces: Sequence[c._Piece],
    ) -> tuple[list[tuple[int, int]], list[set[Fraction]]]:
        nonlocal global_checks
        global_checks += 1
        return unsafe(pieces)

    monkeypatch.setattr(fitting, "_source_failures", source_check)
    monkeypatch.setattr(c, "_unsafe_pair", global_check)
    fitted = fitting.fit_outlines(source, 0.2)
    assert len(fitted.contours.orange) == 2
    assert sorted(len(calls) for calls in checked.values()) == [1, 2]
    assert all(a is not b for calls in checked.values() for a, b in pairwise(calls))
    assert global_checks >= 2
