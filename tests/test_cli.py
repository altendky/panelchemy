"""Actionable command-line errors for rejected image inputs."""

import json
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from panelchemy.cli import main


@pytest.mark.parametrize("command", ["separate", "relief", "export"])
def test_oversize_image_is_reported_without_traceback(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "oversize.png"
    Image.new("RGB", (3, 3)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    monkeypatch.setattr(
        "sys.argv",
        ["panelchemy", command, str(source), "--output", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "exceeds limit" in error
    assert "Traceback" not in error
    assert not (tmp_path / "out").exists()


def test_step_only_cli_preserves_configured_colors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from PIL import ImageDraw

    source = tmp_path / "input.png"
    colors = [(10, 20, 30), (200, 80, 40), (235, 245, 255)]
    image = Image.new("RGB", (20, 20), colors[0])
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((4, 4, 7, 7), fill=colors[1])
    drawing.rectangle((11, 11, 14, 14), fill=colors[2])
    image.save(source)
    mask = tmp_path / "mask.png"
    Image.new("L", image.size, 255).save(mask)
    config = tmp_path / "config.json"
    _ = config.write_text(json.dumps({"palette": colors}))
    profile = tmp_path / "profile.json"
    _ = profile.write_text(json.dumps({"long_edge_mm": 20, "taper_angle_deg": 0}))
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            "export",
            str(source),
            "--output",
            str(output),
            "--format",
            "step",
            "--config",
            str(config),
            "--profile",
            str(profile),
            "--silhouette-mask",
            str(mask),
        ],
    )
    main()
    manifest = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert manifest["mesh_validated"] is False
    report = cast("dict[str, object]", manifest["export"])
    validation = cast("dict[str, object]", report["validation"])
    multipart = cast("dict[str, object]", validation["multipart_step"])
    parts = cast("dict[str, dict[str, object]]", multipart["materials"])
    assert parts["orange"]["color_rgb"] == list(colors[1])
    assert (output / "relief-multipart.step").is_file()
    assert not list(output.glob("*.stl"))
