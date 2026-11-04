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


def test_taper_is_configurable_from_vertical_including_no_taper() -> None:
    vertical = ConstructionProfile.from_mapping({"taper_angle_deg": 0})
    angled = ConstructionProfile.from_mapping({"taper_angle_deg": 20})
    assert vertical.taper_run_per_height == 0
    assert angled.taper_run_per_height == pytest.approx(tan(radians(20)))
    assert ConstructionProfile.from_mapping(angled.as_dict()) == angled


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
