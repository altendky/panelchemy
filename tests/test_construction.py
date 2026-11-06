"""Approved physical defaults and configurable taper/band relationships."""

from math import radians, tan

import pytest

from panelchemy.construction import ConstructionProfile


def test_user_defaults_define_twenty_layers_and_three_bands() -> None:
    profile = ConstructionProfile()
    assert profile.long_edge_mm == 250
    assert profile.nozzle_diameter_mm == 0.4
    assert profile.layer_height_mm == 0.2
    assert (profile.black_top, profile.orange_top, profile.white_top) == (2, 3, 4)
    assert profile.white_top / profile.layer_height_mm == 20
    assert profile.contour_tolerance_mm == 0


def test_taper_is_configurable_from_vertical_including_no_taper() -> None:
    vertical = ConstructionProfile.from_mapping({"taper_angle_deg": 0})
    angled = ConstructionProfile.from_mapping({"taper_angle_deg": 20})
    assert vertical.taper_run_per_height == 0
    assert angled.taper_run_per_height == pytest.approx(tan(radians(20)))
    assert ConstructionProfile.from_mapping(angled.as_dict()) == angled


def test_contour_mode_roundtrips_and_defaults_to_curves() -> None:
    assert ConstructionProfile().contour_mode == "curves"
    lines = ConstructionProfile.from_mapping(
        {"contour_mode": "lines", "contour_tolerance_mm": 0.2}
    )
    assert lines.contour_mode == "lines"
    assert ConstructionProfile.from_mapping(lines.as_dict()) == lines


def test_bounded_corner_softening_is_explicit_and_tracks_the_nozzle_limit() -> None:
    assert ConstructionProfile().corner_softening_mm == 0
    selected = ConstructionProfile.from_mapping({"corner_softening_mm": 0.25})
    assert selected.corner_softening_mm == selected.tracing_radius_mm
    assert ConstructionProfile.from_mapping(selected.as_dict()) == selected
    wider = ConstructionProfile(nozzle_diameter_mm=0.8, corner_softening_mm=0.5)
    assert wider.corner_softening_mm == wider.tracing_radius_mm


def test_bounded_span_simplification_is_explicit_and_roundtrips() -> None:
    assert ConstructionProfile().span_simplification_mm == 0
    selected = ConstructionProfile.from_mapping({"span_simplification_mm": 0.1})
    assert selected.span_simplification_mm == 0.1
    assert ConstructionProfile.from_mapping(selected.as_dict()) == selected


@pytest.mark.parametrize(
    "values",
    [
        {"nozzle_diameter_mm": 0},
        {"layer_height_mm": -1},
        {"long_edge_mm": True},
        {"white_thickness_mm": 1.1},
        {"taper_angle_deg": -1},
        {"taper_angle_deg": 90},
        {"rim_width_mm": 0},
        {"long_edge_mm": float("inf")},
        {"long_edge_mm": 10**1000},
        {"old_name": 2},
        {"contour_mode": "unknown"},
        {"contour_mode": True},
        {"contour_mode": 1},
        {"contour_tolerance_mm": -0.1},
        {"contour_tolerance_mm": True},
        {"contour_tolerance_mm": float("nan")},
        {"corner_softening_mm": -0.1},
        {"corner_softening_mm": True},
        {"corner_softening_mm": float("nan")},
        {"corner_softening_mm": float("inf")},
        {"corner_softening_mm": 0.251},
        {"span_simplification_mm": -0.1},
        {"span_simplification_mm": True},
        {"span_simplification_mm": float("nan")},
        {"span_simplification_mm": float("inf")},
        {"span_simplification_mm": 0.251},
        {"orange_thickness_mm": 1e-12},
        {"orange_thickness_mm": 200000000.02},
        {
            "black_thickness_mm": 1e308,
            "orange_thickness_mm": 1e308,
            "white_thickness_mm": 1e308,
            "layer_height_mm": 1e308,
        },
    ],
)
def test_invalid_or_unaligned_profile_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ = ConstructionProfile.from_mapping(values)
