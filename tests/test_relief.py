"""Band support, exact top preservation, and output provenance through OCCT."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf, gp_Vec
from PIL import Image

from panelchemy import occt
from panelchemy import relief as relief_module
from panelchemy.construction import ConstructionProfile
from panelchemy.relief import build_relief, process_relief_file
from panelchemy.separation import SeparationConfig, SeparationResult, separate


def fixture() -> tuple[Image.Image, SeparationResult]:
    labels = np.full((20, 24), 2, dtype=np.uint8)
    labels[1:-1, 1:-1] = 0
    labels[5:9, 5:9] = 2
    labels[5:9, 14] = 2  # One-pixel-wide white stroke.
    labels[12:15, 5:10] = 1
    image = Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )
    return image, separate(image)


def test_white_supports_nest_and_thin_top_is_preserved() -> None:
    _, result = fixture()
    profile = ConstructionProfile(long_edge_mm=2.2)
    model = build_relief(result, profile)
    assert occt.valid(model.solid)
    assert set(model.materials) == {"black", "orange", "white"}
    expected_white_area = np.count_nonzero(result.masks["white"]) * 0.1**2
    assert occt.area(occt.cross_section(model.solid, 4)) == pytest.approx(
        expected_white_area, abs=1e-8
    )
    white_bottom = occt.cross_section(model.materials["white"], 3)
    orange_top = occt.cross_section(model.materials["orange"], 3)
    assert occt.area(occt.cut(white_bottom, orange_top)) == pytest.approx(0, abs=1e-9)
    orange_bottom = occt.cross_section(model.materials["orange"], 2)
    black_top = occt.cross_section(model.materials["black"], 2)
    assert occt.area(occt.cut(orange_bottom, black_top)) == pytest.approx(0, abs=1e-9)
    assert sum(occt.volume(part) for part in model.materials.values()) == pytest.approx(
        occt.volume(model.solid)
    )


def test_nozzle_metadata_does_not_remove_fine_features() -> None:
    _, result = fixture()
    profile = ConstructionProfile(long_edge_mm=2.2, taper_angle_deg=0)
    first = build_relief(result, profile)
    second = build_relief(result, replace(profile, nozzle_diameter_mm=2))
    assert first.contours.white == second.contours.white
    assert occt.volume(first.solid) == pytest.approx(occt.volume(second.solid))
    assert first.diagnostics["roof_methods"] == {"prism": 3}


def test_external_wall_is_vertical_and_higher_sections_are_supported() -> None:
    _, result = fixture()
    model = build_relief(result, ConstructionProfile(long_edge_mm=22))
    assert occt.area(occt.cross_section(model.solid, 0.5)) == pytest.approx(
        model.contours.silhouette.area
    )
    assert occt.area(occt.cross_section(model.solid, 1.5)) == pytest.approx(
        model.contours.silhouette.area
    )
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(0, 0, -1))
    upper = BRepBuilderAPI_Transform(
        occt.cross_section(model.solid, 3.5), transform, True
    ).Shape()
    lower = occt.cross_section(model.solid, 2.5)
    assert occt.area(occt.cut(upper, lower)) == pytest.approx(0, abs=1e-8)


def test_protected_rim_clips_tapers_but_rejects_top_trimming() -> None:
    _, result = fixture()
    profile = ConstructionProfile(long_edge_mm=22, rim_width_mm=3.5, taper_angle_deg=45)
    model = build_relief(result, profile)
    assert cast("float", model.diagnostics["rim_clipped_volume_mm3"]) > 0
    assert any(
        "clipped" in warning
        for warning in cast("list[str]", model.diagnostics["warnings"])
    )
    with pytest.raises(ValueError, match="original feature top"):
        _ = build_relief(result, replace(profile, rim_width_mm=4.1))


def test_closing_counter_is_reported_but_top_counter_survives() -> None:
    image, _ = fixture()
    pixels = np.asarray(image).copy()
    pixels[7, 7] = 0
    result = separate(Image.fromarray(pixels))
    profile = ConstructionProfile(long_edge_mm=2.2, taper_angle_deg=20)
    model = build_relief(result, profile)
    assert (
        cast("dict[str, int]", model.diagnostics["white_source_topology"])["holes"] == 1
    )
    assert (
        cast("dict[str, int]", model.diagnostics["white_band_bottom_topology"])["holes"]
        == 0
    )
    assert model.diagnostics["white_top_missing_mm2"] == pytest.approx(0, abs=1e-9)
    assert model.diagnostics["white_top_extra_mm2"] == pytest.approx(0, abs=1e-9)


def test_resource_limit_rejects_without_simplifying() -> None:
    _, result = fixture()
    with pytest.raises(ValueError, match="no features were removed"):
        _ = build_relief(result, max_vertices=4)


def test_geometry_provenance_and_input_protection(tmp_path: Path) -> None:
    image, _ = fixture()
    source = tmp_path / "input.png"
    image.save(source)
    before = source.read_bytes()
    output = tmp_path / "model"
    manifest = process_relief_file(source, output, ConstructionProfile(long_edge_mm=22))
    assert source.read_bytes() == before
    assert manifest["print_ready"] is False
    assert manifest["source"] == {
        "name": "input.png",
        "sha256": sha256(before).hexdigest(),
    }
    for relative, expected in cast("dict[str, str]", manifest["files"]).items():
        assert sha256((output / relative).read_bytes()).hexdigest() == expected
    assert "Relief colors from above" in (output / "preview.svg").read_text()
    other = tmp_path / "protected"
    other.mkdir()
    (other / "relief.brep").hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_relief_file(source, other)
    assert source.read_bytes() == before


def test_source_snapshot_remains_the_provenance_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, _ = fixture()
    source = tmp_path / "input.png"
    image.save(source)
    original_hash = sha256(source.read_bytes()).hexdigest()
    original = relief_module.build_relief

    def changing_source(
        result: SeparationResult,
        profile: ConstructionProfile | None = None,
        *,
        max_vertices: int = 25_000,
    ) -> relief_module.ReliefModel:
        model = original(result, profile, max_vertices=max_vertices)
        Image.new("RGB", image.size, "black").save(source)
        return model

    monkeypatch.setattr(relief_module, "build_relief", changing_source)
    manifest = process_relief_file(
        source, tmp_path / "model", ConstructionProfile(long_edge_mm=22)
    )
    assert manifest["source"] == {"name": "input.png", "sha256": original_hash}
    assert sha256(source.read_bytes()).hexdigest() != original_hash


def test_optional_approximation_is_recorded_and_comparison_is_protected(
    tmp_path: Path,
) -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[[0, -1], :] = 2
    labels[:, [0, -1]] = 2
    for row in range(8, 22):
        labels[row, 8 + (row - 8) // 2 : 17 + (row - 8) // 2] = 2
    image = Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )
    source = tmp_path / "input.png"
    image.save(source)
    profile = ConstructionProfile(
        long_edge_mm=30, taper_angle_deg=0, contour_tolerance_mm=0.8
    )
    output = tmp_path / "model"
    manifest = process_relief_file(source, output, profile)
    assert manifest["profile"] == profile.as_dict()
    diagnostics = cast("dict[str, object]", manifest["diagnostics"])
    assert diagnostics["contour_approximation"] is not None
    assert "contours.svg" in cast("dict[str, str]", manifest["files"])
    assert (output / "contours.svg").is_file()
    assert (
        sha256((output / "contours.svg").read_bytes()).hexdigest()
        == cast("dict[str, str]", manifest["files"])["contours.svg"]
    )
    # The post-approximation budget applies to kernel input, not raw allocation.
    with pytest.raises(ValueError, match="approximated contours require"):
        _ = build_relief(separate(image), profile, max_vertices=4)
    for invalid in (True, 0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            _ = build_relief(separate(image), profile, max_vertices=invalid)
    other = tmp_path / "protected"
    other.mkdir()
    (other / "contours.svg").hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_relief_file(source, other, profile)
    # Reusing an output directory with exact mode removes the stale comparison.
    exact = process_relief_file(
        source, output, replace(profile, contour_tolerance_mm=0)
    )
    assert not (output / "contours.svg").exists()
    assert (
        cast("dict[str, object]", exact["diagnostics"])["contour_approximation"] is None
    )
