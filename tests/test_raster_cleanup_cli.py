"""Physical cleanup previews are independent of CAD and keep explicit inputs."""

import json
import sys
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from panelchemy.cli import main
from panelchemy.separation import SeparationConfig


@pytest.mark.parametrize(
    ("settings", "override", "expected_width"),
    [
        ({}, None, 0.5),
        ({"nozzle_diameter_mm": 0.8}, None, 1.0),
        ({"raster_cleanup_mm": 0.7}, None, 0.7),
        ({"raster_cleanup_mm": 0.7}, "0.3", 0.3),
    ],
)
def test_cleanup_command_uses_physical_profile_and_explicit_width(
    settings: dict[str, float],
    override: str | None,
    expected_width: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from panelchemy import raster_cleanup_artifacts

    source, output = tmp_path / "input.png", tmp_path / "review"
    profile_path, config_path = tmp_path / "profile.json", tmp_path / "colors.json"
    mask_path = tmp_path / "mask.png"
    _ = profile_path.write_text(json.dumps({"long_edge_mm": 100, **settings}))
    _ = config_path.write_text(json.dumps({"neutral_tolerance": 12}))
    seen: list[tuple[Path, Path, SeparationConfig | None, float, float]] = []

    def preview(
        source: Path,
        output: Path,
        config: SeparationConfig | None = None,
        *,
        long_edge_mm: float = 250,
        minimum_width_mm: float = 0.5,
        silhouette_path: Path | None = None,
        protected_paths: tuple[Path, ...] = (),
    ) -> dict[str, object]:
        assert silhouette_path == mask_path
        assert set(protected_paths) == {profile_path, config_path}
        seen.append((source, output, config, long_edge_mm, minimum_width_mm))
        return {"kind": "cleanup-preview"}

    monkeypatch.setattr(raster_cleanup_artifacts, "process_cleanup_file", preview)
    # This command must not need the native relief adapter, even indirectly.
    monkeypatch.setitem(sys.modules, "panelchemy.relief", None)
    arguments = [
        "panelchemy",
        "cleanup",
        str(source),
        "--output",
        str(output),
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
        "--silhouette-mask",
        str(mask_path),
    ]
    if override is not None:
        arguments.extend(("--minimum-width-mm", override))
    monkeypatch.setattr("sys.argv", arguments)
    main()
    assert seen == [
        (source, output, SeparationConfig(neutral_tolerance=12), 100, expected_width)
    ]
    assert json.loads(capsys.readouterr().out) == {"kind": "cleanup-preview"}


def test_cleanup_command_writes_review_without_changing_the_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = tmp_path / "input.png", tmp_path / "review"
    image = Image.new("RGB", (20, 20), "white")
    image.paste("black", (2, 2, 18, 18))
    image.putpixel((8, 8), (255, 255, 255))
    image.save(source)
    snapshot = source.read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["panelchemy", "cleanup", str(source), "--output", str(output)]
    )
    main()
    manifest = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert {p.name for p in output.iterdir()} == {
        "original.png",
        "cleaned.png",
        "changes.png",
        "cleanup.json",
        "review.html",
        "manifest.json",
    }
    assert source.read_bytes() == snapshot


@pytest.mark.parametrize("width", ["0", "-0.1", "nan", "inf"])
def test_invalid_cleanup_width_is_an_actionable_cli_error(
    width: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = tmp_path / "input.png", tmp_path / "review"
    Image.new("RGB", (3, 3), "black").save(source)
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            "cleanup",
            str(source),
            "--output",
            str(output),
            "--minimum-width-mm",
            width,
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "minimum" in error.lower() or "width" in error.lower()
    assert "Traceback" not in error
    assert not output.exists()
