"""Adaptive locations guide retries without approving sampled geometry."""

from typing import Never, cast

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from panelchemy.curves import BezierSpan, Controls
from panelchemy.fitting_feedback import (
    Refinement,
    curvature_locations,
    distance_refinement,
)


def parabola(center: float = 0.37) -> BezierSpan:
    return BezierSpan(
        (
            (0.0, center**2),
            (1 / 3, center**2 - 2 * center / 3),
            (2 / 3, center**2 - 4 * center / 3 + 1 / 3),
            (1.0, (1 - center) ** 2),
        ),
        False,
    )


def line(start: tuple[float, float], stop: tuple[float, float]) -> BezierSpan:
    return BezierSpan((start, start, stop, stop), True)


def test_curvature_peak_between_existing_samples_is_located() -> None:
    locations = curvature_locations(parabola(), 0.46, 1)
    assert any(abs(t - 0.37) < 1e-12 for t in locations)
    assert not curvature_locations(parabola(), 0.46, -1)


def test_search_headroom_includes_peaks_below_final_acceptance_bound() -> None:
    # Peak curvature is 2, so this reaches0.898: above search0.895, below0.9.
    assert curvature_locations(parabola(), 0.449, 1)
    assert not curvature_locations(parabola(), 0.449, 1, fraction=0.9)


def test_all_dangerous_curvature_peaks_are_returned() -> None:
    k = 0.5522847498307936
    curve = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    locations = curvature_locations(curve, 0.895, 1)
    assert len(locations) == 2
    assert 0 < locations[0] < 0.5 < locations[1] < 1


def test_curvature_locations_survive_translation_and_physical_scaling() -> None:
    original = parabola()
    transformed = BezierSpan(
        cast(
            "Controls",
            tuple(
                (1e5 + 23.5 * x, -1e5 + 23.5 * y) for x, y in original.control_points
            ),
        ),
        False,
    )
    expected = curvature_locations(original, 0.46, 1)
    actual = curvature_locations(transformed, 0.46 * 23.5, 1)
    assert np.allclose(actual, expected, rtol=0, atol=1e-8)


def test_nearly_stationary_point_produces_a_speed_probe() -> None:
    curve = BezierSpan(
        ((0.25, 0.0), (-1 / 12, 0.0), (-1 / 12, 0.0), (0.25, 0.0)), False
    )
    assert any(abs(t - 0.5) < 1e-8 for t in curvature_locations(curve, 0.1, 1))


def test_lines_and_zero_offset_need_no_curvature_refinement() -> None:
    assert not curvature_locations(line((0, 0), (1, 0)), 1.0, 1)
    assert not curvature_locations(parabola(), 0, 1)


def test_unavailable_numerical_roots_supply_no_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_self: Polynomial) -> Never:
        raise np.linalg.LinAlgError("root iteration failed")

    monkeypatch.setattr(Polynomial, "roots", unavailable)
    assert not curvature_locations(parabola(), 0.46, 1)


def test_identical_geometry_with_different_parameter_speed_needs_no_distance_probes() -> (
    None
):
    curve = BezierSpan(((0, 0), (0.001, 0), (0.002, 0), (1, 0)), False)
    result = distance_refinement(
        ((0, 0), (1, 0)), np.array([0.0, 1.0]), (curve,), ((0, 1),), 0.2
    )
    assert result == Refinement()


def test_forward_excursion_supplies_curve_parameter() -> None:
    curve = BezierSpan(((0, 0), (1 / 3, 1), (2 / 3, 1), (1, 0)), False)
    result = distance_refinement(
        ((0, 0), (1, 0)), np.array([0.0, 1.0]), (curve,), ((0, 1),), 0.2
    )
    assert 0.5 in result.curve_parameters


def test_reverse_omission_supplies_source_query_and_refines_nearest_proxy_foot() -> (
    None
):
    k = 0.5522847498307936
    quarter_circle = BezierSpan(((1, 0), (1, k), (k, 1), (0, 1)), False)
    source = [(1.0, 0.0), (0.0, 1.0)]
    parameters = np.array([0.0, 1.0])
    original_source = source.copy()
    original_parameters = parameters.copy()
    result = distance_refinement(source, parameters, (quarter_circle,), ((0, 1),), 0.2)
    # Source midpoint lies on the endpoint-only proxy, but far from the arc.
    # Returning both domains prevents that proxy shortcut from hiding it.
    assert 0.5 in result.source_parameters
    assert any(abs(t - 0.5) < 0.02 for t in result.curve_parameters)
    assert source == original_source
    assert np.array_equal(parameters, original_parameters)


def test_many_excursions_return_a_bounded_set_of_locations() -> None:
    curves = tuple(
        BezierSpan(((i, 0), (i + 1 / 3, 1), (i + 2 / 3, 1), (i + 1, 0)), False)
        for i in range(12)
    )
    result = distance_refinement(
        ((0, 0), (12, 0)),
        np.array([0.0, 1.0]),
        curves,
        tuple((i / 12, (i + 1) / 12) for i in range(12)),
        0.2,
    )
    assert len(result.curve_parameters) == 8


def test_curve_locations_map_through_each_original_interval() -> None:
    curve = BezierSpan(((0, 0), (1 / 3, 1), (2 / 3, 1), (1, 0)), False)
    result = distance_refinement(
        ((0, 0), (1, 0)), np.array([0.0, 1.0]), (curve,), ((0.25, 0.75),), 0.2
    )
    assert 0.5 in result.curve_parameters
    assert all(0.25 <= t <= 0.75 for t in result.curve_parameters)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan")])
def test_invalid_source_guidance_inputs_are_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite increasing source data"):
        _ = distance_refinement(
            ((0, 0), (1, 0)),
            np.array([0.0, 1.0]),
            (line((0, 0), (1, 0)),),
            ((0, 1),),
            tolerance,
        )
