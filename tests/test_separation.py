"""Boundary and material-support behavior on small, exact raster fixtures."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from panelchemy.separation import (
    SeparationConfig,
    load_config,
    load_silhouette,
    process_file,
    separate,
)


def panel() -> tuple[Image.Image, NDArray[np.uint8]]:
    labels = np.full((11, 13), 2, dtype=np.uint8)
    labels[1:-1, 1:-1] = 0
    labels[3:8, 3:10] = 2
    labels[5, 5] = 0  # A black counter must keep its foundation.
    labels[2, 2] = 1
    return Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    ), labels


def test_enclosed_white_counters_and_nested_supports() -> None:
    image, labels = panel()
    result = separate(image)
    masks = result.masks
    assert result.warnings == []
    assert np.count_nonzero(result.silhouette) == 9 * 11
    assert not result.silhouette[0, 0]
    assert result.silhouette[5, 5]
    assert masks["support-black"][5, 5]
    assert not masks["support-orange"][5, 5]
    assert masks["support-orange"][4, 4] and labels[4, 4] == 2
    assert np.array_equal(
        masks["black"].astype(int) + masks["orange"] + masks["white"],
        result.silhouette.astype(int),
    )
    assert not np.any(masks["support-white"] & ~masks["support-orange"])
    assert not np.any(masks["support-orange"] & ~masks["support-black"])


def test_single_pixel_islands_and_strokes_are_unchanged() -> None:
    image, labels = panel()
    labels[1:-1, 1:-1] = 0
    labels[4, 3:10] = 2
    labels[7, 4] = 2
    labels[7, 7] = 1
    image = Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )
    result = separate(image)
    assert np.array_equal(result.labels, labels)
    assert np.count_nonzero(result.masks["white"]) == 8
    assert np.count_nonzero(result.masks["orange"]) == 1


def test_neutral_antialiasing_does_not_become_orange() -> None:
    rgb = np.array(
        [[[0, 0, 0], [128, 128, 128], [255, 255, 255], [245, 166, 35]]], dtype=np.uint8
    )
    result = separate(Image.fromarray(rgb), silhouette=np.ones((1, 4), dtype=bool))
    assert result.labels.tolist() == [[0, 2, 2, 1]]


def test_white_exterior_uses_four_neighbors_and_clipped_corners() -> None:
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[0, :] = labels[-1, :] = 2
    labels[:, 0] = labels[:, -1] = 2
    labels[1, 1] = 2  # Clipped corner opens to gutter.
    labels[2, 2] = 2  # Diagonal-only contact stays enclosed.
    image = Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )
    result = separate(image)
    assert not result.silhouette[1, 1]
    assert result.masks["white"][2, 2]


def test_debris_is_reported_and_preserved() -> None:
    image, _ = panel()
    canvas = Image.new("RGB", (17, 15), "white")
    canvas.paste(image, (2, 2))
    canvas.putpixel((0, 0), (0, 0, 0))
    result = separate(canvas)
    assert result.component_count == 2
    assert result.silhouette[0, 0]
    assert any("disconnected" in warning for warning in result.warnings)


def test_broken_rim_is_inspectable_and_override_restores_boundary() -> None:
    image = Image.new("RGB", (7, 7), "white")
    pixels = np.asarray(image).copy()
    pixels[1:6, 1:6] = 0
    pixels[2:5, 2:5] = 255
    pixels[1, 3] = 255  # Open the top rim.
    image = Image.fromarray(pixels)
    result = separate(image)
    assert any("broken rim" in warning for warning in result.warnings)
    assert not result.silhouette[3, 3]
    override = np.zeros((7, 7), dtype=bool)
    override[1:6, 1:6] = True
    fixed = separate(image, silhouette=override)
    assert fixed.silhouette_method == "explicit-mask"
    assert fixed.masks["white"][3, 3]
    assert np.array_equal(fixed.labels, result.labels)


def test_transparency_composites_over_white() -> None:
    image, _ = panel()
    image = image.convert("RGBA")
    image.putpixel((0, 0), (0, 0, 0, 0))
    result = separate(image)
    assert result.labels[0, 0] == 2
    assert not result.silhouette[0, 0]


def test_empty_panel_and_wrong_override_dimensions_are_errors() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _ = separate(Image.new("RGB", (0, 2)))
    with pytest.raises(ValueError, match="no panel"):
        _ = separate(Image.new("RGB", (5, 5), "white"))
    image, _ = panel()
    with pytest.raises(ValueError, match="same-size"):
        _ = separate(image, silhouette=np.ones((4, 4), dtype=bool))


@pytest.mark.parametrize(
    "data",
    [
        {"unknown": 2},
        {"neutral_tolerance": True},
        {"neutral_tolerance": -1},
        {"neutral_tolerance": float("nan")},
        {"neutral_tolerance": 10**1000},
        {"palette": [[0, 0, 0]]},
        {"palette": [[0, 0, 0], [0, 0, 0], [255, 255, 255]]},
        {"palette": [[False, 0, 0], [245, 166, 35], [255, 255, 255]]},
    ],
)
def test_invalid_configuration_is_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _ = SeparationConfig.from_mapping(data)


def test_cli_provenance_preserves_sources_and_regenerates_exactly(
    tmp_path: Path,
) -> None:
    image, _ = panel()
    source = tmp_path / "input.png"
    image.save(source)
    source_hash = sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "panelchemy",
        "separate",
        str(source),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = cast("dict[str, object]", json.loads(completed.stdout))
    assert manifest["source"] == {"name": "input.png", "sha256": source_hash}
    assert manifest["silhouette_method"] == "edge-connected-white-4"
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    _ = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first == {path.name: path.read_bytes() for path in output.iterdir()}
    assert sha256(source.read_bytes()).hexdigest() == source_hash
    assert np.count_nonzero(load_silhouette(output / "support-orange.png")) > 0
    assert load_config(output / "config.json") == SeparationConfig()


def test_output_cannot_overwrite_source_or_settings(tmp_path: Path) -> None:
    image, _ = panel()
    source = tmp_path / "normalized.png"
    image.save(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_file(source, tmp_path)
    assert source.read_bytes() == before
    config_path = tmp_path / "config.json"
    _ = config_path.write_text("{}")
    source = tmp_path / "input.png"
    image.save(source)
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_file(source, tmp_path, protected_paths=(config_path,))
    assert config_path.read_text() == "{}"


def test_invalid_cli_input_is_actionable_without_traceback(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "panelchemy",
        "separate",
        str(tmp_path / "missing.png"),
        "--output",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["source", "config", "mask"])
def test_hardlinked_outputs_cannot_modify_inputs(tmp_path: Path, kind: str) -> None:
    image, _ = panel()
    source = tmp_path / "input.png"
    image.save(source)
    output = tmp_path / "out"
    output.mkdir()
    protected_path = source
    mask_path = None
    if kind == "config":
        protected_path = tmp_path / "settings.json"
        _ = protected_path.write_text("{}")
    elif kind == "mask":
        protected_path = tmp_path / "boundary.png"
        Image.new("L", image.size, 255).save(protected_path)
        mask_path = protected_path
    before = protected_path.read_bytes()
    (output / "normalized.png").hardlink_to(protected_path)
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_file(
            source,
            output,
            silhouette_path=mask_path,
            protected_paths=(protected_path,) if kind == "config" else (),
        )
    assert protected_path.read_bytes() == before
