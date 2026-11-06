"""Manufacturing-scale corner significance without deleting artwork topology."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from shapely.geometry import Polygon, box

from panelchemy.approximation import boundary_within_tolerance
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours, vectorize
from panelchemy.curves import CurveFitResult
from panelchemy.outline_fitting import (
    _corners,  # pyright: ignore[reportPrivateUsage]
    fit_outlines,
)
from panelchemy.separation import Labels, SeparationConfig, SeparationResult


def _corner_indices(
    points: NDArray[np.float64], profile: ConstructionProfile
) -> list[int]:
    distances = np.concatenate(
        (np.array([0.0]), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    return _corners(
        points,
        distances,
        2 * profile.assumed_line_width_mm,
        profile.tracing_radius_mm,
    )


def _source(polygon: Polygon) -> PanelContours:
    silhouette = box(0, 0, 12.8, 12.8)
    return PanelContours(
        silhouette,
        [],
        [polygon],
        0.1,
        (0, 0, 128, 128),
        4
        + len(polygon.exterior.coords)
        - 1
        + sum(len(ring.coords) - 1 for ring in polygon.interiors),
    )


def test_pixel_sized_elbow_does_not_become_a_sharp_cad_corner() -> None:
    # A two-pixel turn from a 2x enlarged source: the angle alone is 90 degrees,
    # but its excursion is too small for the configured .4-mm nozzle policy.
    points = np.array([(0, 0), (0, 0.177), (0.177, 0.177)], dtype=np.float64)
    distances = np.array([0.0, 0.177, 0.354])
    assert _corners(points, distances, 1, 0) == [1]
    assert _corner_indices(points, ConstructionProfile()) == []


@pytest.mark.parametrize(
    "points",
    [
        np.array([(0, 0), (0, 3), (3, 3)], dtype=np.float64),
        np.array([(0, 0), (4, 1), (0, 2)], dtype=np.float64),
    ],
    ids=["right-angle", "sharp-terminal-tip"],
)
def test_substantial_corners_and_pointed_tips_remain_sharp(
    points: NDArray[np.float64],
) -> None:
    assert _corner_indices(points, ConstructionProfile()) == [1]


def test_the_same_small_elbow_is_significant_for_a_smaller_nozzle() -> None:
    points = np.array([(0, 0), (0, 0.177), (0.177, 0.177)], dtype=np.float64)
    assert _corner_indices(points, ConstructionProfile(nozzle_diameter_mm=0.05)) == [1]
    assert _corner_indices(points, ConstructionProfile(nozzle_diameter_mm=0.4)) == []


def test_corner_decisions_follow_physical_units_and_ignore_translation() -> None:
    points = np.array([(0, 0), (0, 1.2), (1.2, 1.2)], dtype=np.float64)
    profile = ConstructionProfile()
    original = _corner_indices(points, profile)
    assert original == [1]
    scaled = replace(profile, nozzle_diameter_mm=profile.nozzle_diameter_mm * 4)
    assert _corner_indices(points * 4 + (123.375, -45.625), scaled) == original


def test_fitted_substantial_tip_survives_different_nozzle_scales() -> None:
    source = _source(Polygon([(2, 2), (10, 6), (2, 10)]))
    fits: list[CurveFitResult] = []
    for diameter in (0.4, 0.8):
        profile = ConstructionProfile(nozzle_diameter_mm=diameter)
        fitted = fit_outlines(
            source,
            0.2,
            corner_scale_mm=2 * profile.assumed_line_width_mm,
            corner_prominence_mm=profile.tracing_radius_mm,
            fairing_mm=profile.tracing_radius_mm,
        )
        assert len(fitted.proxy.white) == 1
        assert fitted.proxy.white[0].is_valid
        assert fitted.proxy.white[0].area > 0
        assert fitted.proxy.silhouette.equals(source.silhouette)
        assert boundary_within_tolerance(
            source.white[0].boundary,
            fitted.proxy.white[0].boundary,
            0.2 + fitted.proxy_deviation_mm,
        )
        fits.append(fitted)
        selected = fitted.contours.white[0].exterior.spans
        endpoint = min(
            (span.control_points[0] for span in selected),
            key=lambda p: (p[0] - 10) ** 2 + (p[1] - 6) ** 2,
        )
        assert np.linalg.norm(np.asarray(endpoint) - (10, 6)) <= 0.2
        at_tip = next(
            i for i, span in enumerate(selected) if span.control_points[0] == endpoint
        )
        before, after = selected[at_tip - 1], selected[at_tip]
        incoming = np.asarray(endpoint) - np.asarray(
            before.control_points[0 if before.is_line else 2]
        )
        outgoing = (
            np.asarray(after.control_points[3 if after.is_line else 1]) - endpoint
        )
        cosine = float(
            np.sum(incoming * outgoing)
            / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        )
        assert cosine < 0.5  # The fitted tip still turns by more than 60 degrees.
    assert fits[0].diagnostics["physical_corners"] == 3
    assert fits[1].diagnostics["physical_corners"] == 3


def test_upscaled_pixels_do_not_change_nozzle_guided_fitting() -> None:
    rows, columns = np.indices((64, 64))
    labels: Labels = np.where(
        (columns >= 12) & (columns <= 50) & (abs(rows - columns) <= 5), 2, 0
    ).astype(np.uint8)
    profile = ConstructionProfile(nozzle_diameter_mm=0.4)
    fits: list[CurveFitResult] = []
    for raster in (labels, np.repeat(np.repeat(labels, 2, axis=0), 2, axis=1)):
        source = vectorize(
            SeparationResult(
                raster,
                np.ones(raster.shape, dtype=np.bool_),
                SeparationConfig(),
                "explicit-mask",
                [],
                1,
            ),
            6.4,
        )
        fits.append(
            fit_outlines(
                source,
                0.2,
                corner_scale_mm=2 * profile.assumed_line_width_mm,
                corner_prominence_mm=profile.tracing_radius_mm,
                fairing_mm=profile.tracing_radius_mm,
            )
        )
    assert fits[0].contours.white == fits[1].contours.white
    assert (
        fits[0].diagnostics["physical_corners"]
        == fits[1].diagnostics["physical_corners"]
    )
    assert fits[0].proxy.white[0].equals(fits[1].proxy.white[0])


@pytest.mark.parametrize("value", [-0.1, True, float("nan"), float("inf"), "0.25"])
def test_invalid_corner_prominence_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="corner prominence"):
        _ = fit_outlines(
            _source(box(2, 2, 8, 8)),
            0.2,
            corner_prominence_mm=cast("float", value),
        )


def test_resolution_guidance_counts_panel_samples_and_tracks_nozzle() -> None:
    ordinary = ConstructionProfile()
    assert ordinary.assumed_line_width_mm == 0.5
    assert ordinary.tracing_radius_mm == 0.25
    assert ordinary.minimum_source_long_edge_pixels == 2000
    assert ordinary.preferred_source_long_edge_pixels == 4000
    coarse = replace(ordinary, nozzle_diameter_mm=0.8)
    assert coarse.minimum_source_long_edge_pixels == 1000
    assert coarse.preferred_source_long_edge_pixels == 2000
    nonintegral = replace(ordinary, long_edge_mm=250.01)
    assert nonintegral.minimum_source_long_edge_pixels == 2001
    assert nonintegral.preferred_source_long_edge_pixels == 4001


def test_resolution_counts_do_not_overflow_intermediate_float_arithmetic() -> None:
    profile = ConstructionProfile(long_edge_mm=1e308)
    assert profile.minimum_source_long_edge_pixels > 10**308
    assert (
        profile.preferred_source_long_edge_pixels
        == 2 * profile.minimum_source_long_edge_pixels
    )


@pytest.mark.parametrize("diameter", [1.6e308, 1e308, 5e-324])
def test_unrepresentable_derived_tracing_scales_are_rejected(diameter: float) -> None:
    with pytest.raises(ValueError, match="tracing scales"):
        _ = ConstructionProfile(nozzle_diameter_mm=diameter)
