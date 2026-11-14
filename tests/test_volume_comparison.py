"""Negligible CAD mass drift is diagnostic; substantial changes still reject."""

import pytest

from panelchemy.volume_comparison import compare_volumes


@pytest.mark.parametrize(
    ("source", "measured", "numerical"),
    [
        (7909.786985817467, 7909.792433673635, 0.0024634114189518975),
        (115968.89972738513, 115972.50455372306, 0.0011596889972738513),
    ],
)
def test_observed_negligible_drift_is_a_recorded_warning(
    source: float, measured: float, numerical: float
) -> None:
    comparison = compare_volumes(source, measured, numerical_allowance_mm3=numerical)
    assert comparison.acceptable
    assert comparison.warning is not None
    assert comparison.diagnostics()["volume_warning"] == comparison.warning
    assert comparison.difference_mm3 == pytest.approx(abs(source - measured))
    assert comparison.numerical_allowance_mm3 == numerical


def test_drift_within_numerical_limit_needs_no_warning() -> None:
    comparison = compare_volumes(100, 100.0000001)
    assert comparison.acceptable
    assert comparison.warning is None


def test_zero_volume_requires_an_explicit_non_solid_context() -> None:
    assert not compare_volumes(0, 0).acceptable
    assert compare_volumes(0, 0, allow_zero=True).acceptable
    assert not compare_volumes(0, 1e-8, allow_zero=True).acceptable
    assert not compare_volumes(1e-8, 0, allow_zero=True).acceptable


@pytest.mark.parametrize("scale", [0.01, 1, 1000])
def test_substantial_relative_volume_changes_still_fail(scale: float) -> None:
    comparison = compare_volumes(100 * scale, 100.02 * scale)
    assert not comparison.acceptable
    assert comparison.warning is None


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("source_invalid", [False, True])
def test_nonpositive_or_nonfinite_volumes_cannot_pass(
    bad: float, source_invalid: bool
) -> None:
    comparison = (
        compare_volumes(bad, 100) if source_invalid else compare_volumes(100, bad)
    )
    assert not comparison.acceptable
