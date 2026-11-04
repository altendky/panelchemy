"""Model exports share provenance and fail before replacing visible artifacts."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from PIL import Image, ImageDraw

from panelchemy.construction import ConstructionProfile
from panelchemy.export import ExportConfig, ExportError
from panelchemy.relief import EXPORT_NAMES, process_relief_file


def _source(path: Path) -> None:
    image = Image.new("RGB", (30, 22), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((1, 1, 28, 20), fill="black")
    drawing.rectangle((5, 5, 9, 9), fill=(245, 166, 35))
    drawing.rectangle((16, 7, 20, 11), fill="white")
    image.save(path)


def test_export_manifest_hashes_and_stale_model_removal(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    _source(source)
    before = source.read_bytes()
    output = tmp_path / "model"
    profile = ConstructionProfile(long_edge_mm=28, taper_angle_deg=0)
    settings = ExportConfig()
    manifest = process_relief_file(source, output, profile, export_config=settings)
    assert source.read_bytes() == before
    assert manifest["mesh_validated"] is True
    assert manifest["physical_print_validated"] is False
    assert manifest["print_ready"] is False
    report = cast("dict[str, object]", manifest["export"])
    assert report["config"] == asdict(settings)
    files = cast("dict[str, str]", manifest["files"])
    assert set(EXPORT_NAMES) <= files.keys()
    for relative, expected in files.items():
        assert sha256((output / relative).read_bytes()).hexdigest() == expected
    assert not list(output.rglob("*.3mf"))
    exact = process_relief_file(source, output, profile)
    assert exact["export"] is None
    assert exact["mesh_validated"] is False
    assert all(not (output / name).exists() for name in EXPORT_NAMES)


@pytest.mark.parametrize("name", EXPORT_NAMES)
def test_model_output_cannot_overwrite_an_input_hardlink(
    name: str, tmp_path: Path
) -> None:
    source = tmp_path / "input.png"
    _source(source)
    before = source.read_bytes()
    output = tmp_path / "model"
    output.mkdir()
    (output / name).hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        _ = process_relief_file(source, output, export_config=ExportConfig())
    assert source.read_bytes() == before


def test_failed_mesh_validation_preserves_previous_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    _source(source)
    output = tmp_path / "model"
    output.mkdir()
    previous = {"relief.brep": b"existing CAD", "manifest.json": b"existing manifest"}
    for name, content in previous.items():
        _ = (output / name).write_bytes(content)
    with pytest.raises(ExportError, match="max_triangles"):
        _ = process_relief_file(
            source,
            output,
            ConstructionProfile(long_edge_mm=28, taper_angle_deg=0),
            export_config=ExportConfig(max_triangles=1),
        )
    assert {p.name: p.read_bytes() for p in output.iterdir()} == previous
