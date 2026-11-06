"""Actual cubic reduction with bounded displacement and unchanged smooth joins."""

from collections.abc import Sequence
from fractions import Fraction
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from panelchemy import curves as c
from panelchemy import outline_simplification as reduction
from panelchemy.outline_simplification import simplify

_PARENT = c.BezierSpan(((0, 0), (2, 1), (6, 7), (8, 8)), False, 0.2)


def split(span: c.BezierSpan, count: int) -> tuple[c.BezierSpan, ...]:
    return tuple(
        c.BezierSpan(
            cast(
                "c.Controls",
                tuple(
                    (float(p[0]), float(p[1]))
                    for p in c._restrict(  # pyright: ignore[reportPrivateUsage]
                        c._controls(span),  # pyright: ignore[reportPrivateUsage]
                        Fraction(i, count),
                        Fraction(i + 1, count),
                    )
                ),
            ),
            False,
            span.certified_error_mm,
        )
        for i in range(count)
    )


def values(span: c.BezierSpan, t: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.asarray(span.control_points)
    q = t[:, None]
    return cast(
        "NDArray[np.float64]",
        (1 - q) ** 3 * p[0]
        + 3 * (1 - q) ** 2 * q * p[1]
        + 3 * (1 - q) * q * q * p[2]
        + q**3 * p[3],
    )


def intervals(count: int) -> tuple[tuple[float, float], ...]:
    return tuple((i / count, (i + 1) / count) for i in range(count))


def exact_direction(a: c.Point2, b: c.Point2) -> tuple[Fraction, Fraction]:
    return Fraction(b[0]) - Fraction(a[0]), Fraction(b[1]) - Fraction(a[1])


def test_subdivided_cubic_becomes_one_actual_cubic_and_inputs_stay_unchanged() -> None:
    originals, times = split(_PARENT, 16), intervals(16)
    snapshot = tuple(span.control_points for span in originals)
    result = simplify(originals, times, 0.01)
    assert result.output_spans == 1
    assert result.accepted_merges == 15
    assert result.spans[0].control_points == _PARENT.control_points
    assert result.intervals == ((0, 1),)
    assert tuple(span.control_points for span in originals) == snapshot


def test_source_acceptance_callback_limits_each_merge_without_changing_the_bound() -> (
    None
):
    seen: list[tuple[float, float]] = []

    def accept(span: c.BezierSpan, start: float, stop: float) -> bool:
        seen.append((start, stop))
        assert span.control_points[0] != span.control_points[1]
        return stop - start <= 0.25

    result = simplify(split(_PARENT, 16), intervals(16), 0.01, accept=accept)
    assert result.output_spans == 4
    assert result.intervals == ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1))
    assert any(stop - start > 0.25 for start, stop in seen)


def test_unexpected_source_callback_errors_are_not_hidden() -> None:
    def fail(_span: c.BezierSpan, _start: float, _stop: float) -> bool:
        raise RuntimeError("source certificate failed unexpectedly")

    with pytest.raises(RuntimeError, match="unexpectedly"):
        _ = simplify(split(_PARENT, 4), intervals(4), 0.1, accept=fail)


def test_approximate_reduction_retains_exact_endpoint_tangent_directions() -> None:
    originals: list[c.BezierSpan] = []
    for span in split(_PARENT, 16):
        a, b, e, d = span.control_points
        originals.append(
            c.BezierSpan(
                (
                    a,
                    (a[0] + (b[0] - a[0]) * 17 / 16, a[1] + (b[1] - a[1]) * 17 / 16),
                    (d[0] + (e[0] - d[0]) * 15 / 16, d[1] + (e[1] - d[1]) * 15 / 16),
                    d,
                ),
                False,
                0.2,
            )
        )
    result = simplify(originals, intervals(16), 0.02)
    assert result.output_spans < len(originals)
    assert result.spans[0].control_points[0] == originals[0].control_points[0]
    assert result.spans[-1].control_points[-1] == originals[-1].control_points[-1]
    for a, b in (
        (
            exact_direction(
                originals[0].control_points[0], originals[0].control_points[1]
            ),
            exact_direction(
                result.spans[0].control_points[0], result.spans[0].control_points[1]
            ),
        ),
        (
            exact_direction(
                originals[-1].control_points[2], originals[-1].control_points[3]
            ),
            exact_direction(
                result.spans[-1].control_points[2], result.spans[-1].control_points[3]
            ),
        ),
    ):
        assert a[0] * b[1] - a[1] * b[0] == 0
        assert a[0] * b[0] + a[1] * b[1] > 0
    for selected, (start, stop) in zip(result.spans, result.intervals, strict=True):
        for old, (a, b) in zip(originals, intervals(16), strict=True):
            if a < start or b > stop:
                continue
            t = np.linspace(0, 1, 129)
            q = (a + (b - a) * t - start) / (stop - start)
            distance = np.linalg.norm(values(selected, q) - values(old, t), axis=1)
            assert float(np.max(distance)) <= 0.02


def test_kinks_and_analytic_lines_are_retained_as_group_boundaries() -> None:
    horizontal = c.BezierSpan(((0, 0), (1, 0), (2, 0), (3, 0)), False)
    vertical = c.BezierSpan(((3, 0), (3, 1), (3, 2), (3, 3)), False)
    line = c.BezierSpan(((3, 3), (4, 3), (5, 3), (6, 3)), True)
    originals = (*split(horizontal, 4), *split(vertical, 4), line)
    result = simplify(originals, intervals(9), 0.1)
    assert result.output_spans == 3
    assert result.spans[0].control_points[-1] == (3, 0)
    assert result.spans[1].control_points[0] == (3, 0)
    assert result.spans[2] is line


def test_a_deviation_between_fit_samples_is_rejected_by_continuous_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = c.BezierSpan(((0, 0), (1 / 3, 0), (2 / 3, 0), (1, 0)), False)
    bad = c.BezierSpan(((0, 0), (1 / 3, -1 / 6), (2 / 3, 1 / 6), (1, 0)), False)
    budget = 0.0475
    samples = np.linspace(0, 1, 7)
    assert (
        float(
            np.max(
                np.linalg.norm(values(bad, samples) - values(original, samples), axis=1)
            )
        )
        < budget
    )
    dense = np.linspace(0, 1, 1001)
    assert (
        float(
            np.max(np.linalg.norm(values(bad, dense) - values(original, dense), axis=1))
        )
        > budget
    )

    def bad_fit(
        _spans: Sequence[c.BezierSpan], _times: Sequence[reduction.Interval]
    ) -> c.BezierSpan:
        return bad

    monkeypatch.setattr(reduction, "_fit", bad_fit)
    originals = split(original, 2)
    result = simplify(originals, intervals(2), budget)
    assert result.spans == originals
    assert result.accepted_merges == 0


def test_merge_budget_is_always_checked_against_original_pieces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    originals = split(_PARENT, 12)
    witness = reduction._within  # pyright: ignore[reportPrivateUsage]
    seen: list[int] = []

    def check(
        candidate: c.BezierSpan,
        group: Sequence[c.BezierSpan],
        times: Sequence[reduction.Interval],
        budget: float,
    ) -> bool:
        assert all(any(span is original for original in originals) for span in group)
        seen.append(len(group))
        return witness(candidate, group, times, budget)

    monkeypatch.setattr(reduction, "_within", check)
    result = simplify(originals, intervals(12), 0.01)
    assert result.output_spans == 1
    assert max(seen) == 12


@pytest.mark.parametrize(
    "controls",
    [
        ((0, 0), (0, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (3, 0), (3, 0)),
        ((0, 0), (4, 0), (-1, 0), (3, 0)),
    ],
)
def test_stationary_or_nonmonotone_proposals_are_rejected_before_source_check(
    monkeypatch: pytest.MonkeyPatch, controls: c.Controls
) -> None:
    original = c.BezierSpan(((0, 0), (1, 0), (2, 0), (3, 0)), False)
    bad = c.BezierSpan(controls, False)

    def bad_fit(
        _spans: Sequence[c.BezierSpan], _times: Sequence[reduction.Interval]
    ) -> c.BezierSpan:
        return bad

    def source_check(_span: c.BezierSpan, _start: float, _stop: float) -> bool:
        pytest.fail("unsafe proposal reached original-source acceptance")

    monkeypatch.setattr(reduction, "_fit", bad_fit)
    originals = split(original, 2)
    result = simplify(originals, intervals(2), 10, accept=source_check)
    assert result.spans == originals


def test_growth_is_bounded_for_large_smooth_chains() -> None:
    result = simplify(split(_PARENT, 128), intervals(128), 0.01)
    assert result.output_spans == 4
    assert result.accepted_merges == 124
    assert result.attempted_merges == 124
    assert result.intervals == ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1))


def test_zero_budget_preserves_every_input_piece() -> None:
    originals = split(_PARENT, 16)
    result = simplify(originals, intervals(16), 0)
    assert result.spans == originals
    assert result.output_spans == 16
    assert result.attempted_merges == 0


@pytest.mark.parametrize("budget", [-0.1, True, float("inf"), float("nan")])
def test_invalid_edit_budget_is_rejected(budget: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        _ = simplify(split(_PARENT, 2), intervals(2), budget)


def test_disconnected_or_collapsed_source_mapping_is_rejected() -> None:
    original = split(_PARENT, 2)
    with pytest.raises(ValueError, match="increasing"):
        _ = simplify(original, ((0, 0.5), (0.5, 0.5)), 0.1)
    with pytest.raises(ValueError, match="join exactly"):
        _ = simplify(original, ((0, 0.4), (0.5, 1)), 0.1)
