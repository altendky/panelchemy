"""Stroke previews preserve physical settings, explicit references and inputs."""

import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from panelchemy.cli import main
from panelchemy.raster_cleanup import cleanup
from panelchemy.raster_cleanup_artifacts import cleanup_files
from panelchemy.separation import SeparationConfig, separate
from panelchemy.stroke_retention_artifacts import (
    prepare_stroke_preview_artifacts,
    process_stroke_preview_file,
)


def artwork() -> bytes:
    image = Image.new("RGB", (24, 24), "black")
    image.paste("white", (8, 4, 10, 20))
    image.putpixel((4, 4), (255, 255, 255))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("settings", "overrides", "expected"),
    [
        ({}, [], (0.5, 0.25, 0)),
        (
            {"nozzle_diameter_mm": 0.8, "raster_cleanup_mm": 0.6},
            [],
            (1, 0.5, 0.6),
        ),
        (
            {"raster_cleanup_mm": 0.6},
            ["--target-width-mm", "0.75", "--max-growth-mm", "0.15"],
            (0.75, 0.15, 0.6),
        ),
        ({"raster_cleanup_mm": 0.6}, ["--compact-cleanup-mm", "0"], (0.5, 0.25, 0)),
        ({}, ["--max-growth-mm", "0"], (0.5, 0, 0)),
    ],
)
def test_stroke_preview_routes_profile_and_explicit_overrides_without_cad(
    settings: dict[str, float],
    overrides: list[str],
    expected: tuple[float, float, float],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from panelchemy import stroke_retention_artifacts

    source, output = tmp_path / "source.png", tmp_path / "out"
    profile, config, mask = (
        tmp_path / "profile.json",
        tmp_path / "colors.json",
        tmp_path / "mask.png",
    )
    _ = profile.write_text(json.dumps({"long_edge_mm": 100, **settings}))
    _ = config.write_text(json.dumps({"neutral_tolerance": 12}))
    seen: list[tuple[float, float, float]] = []

    def preview(
        source_path: Path,
        output_path: Path,
        selected: SeparationConfig | None = None,
        *,
        long_edge_mm: float = 250,
        target_width_mm: float = 0.5,
        max_growth_mm: float = 0.25,
        compact_cleanup_mm: float = 0,
        silhouette_path: Path | None = None,
        protected_paths: tuple[Path, ...] = (),
    ) -> dict[str, object]:
        assert (source_path, output_path) == (source, output)
        assert selected == SeparationConfig(neutral_tolerance=12)
        assert long_edge_mm == 100
        assert silhouette_path == mask
        assert set(protected_paths) == {profile, config}
        seen.append((target_width_mm, max_growth_mm, compact_cleanup_mm))
        return {"preview_only": True}

    monkeypatch.setattr(
        stroke_retention_artifacts, "process_stroke_preview_file", preview
    )
    monkeypatch.setitem(sys.modules, "panelchemy.relief", None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            "stroke-preview",
            str(source),
            "--output",
            str(output),
            "--profile",
            str(profile),
            "--config",
            str(config),
            "--silhouette-mask",
            str(mask),
            *overrides,
        ],
    )
    main()
    assert seen == [expected]
    assert json.loads(capsys.readouterr().out) == {"preview_only": True}


def test_actual_cli_writes_hashed_proposals_without_changing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "review"
    profile = tmp_path / "profile.json"
    _ = profile.write_text(json.dumps({"long_edge_mm": 24}))
    snapshot = artwork()
    _ = source.write_bytes(snapshot)
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            "stroke-preview",
            str(source),
            "--output",
            str(output),
            "--profile",
            str(profile),
        ],
    )
    main()
    manifest = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert manifest["preview_only"] is True
    assert manifest["source"] == {
        "name": source.name,
        "sha256": sha256(snapshot).hexdigest(),
    }
    assert source.read_bytes() == snapshot
    assert {p.name for p in output.iterdir()} == {
        "original.png",
        "cleaned.png",
        "changes.png",
        "cleanup.json",
        "decisions.json",
        "review.html",
        "manifest.json",
    }
    for name, digest in cast("dict[str, str]", manifest["files"]).items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / "manifest.json").read_text()) == manifest
    html = (output / "review.html").read_text()
    assert "Proposed growth (preview only)" in html
    assert "Model conversion does not use these proposals" in html
    assert "Curve fitting will use the cleaned reference" not in html


def test_compact_cleanup_is_separate_and_published_reference_matches_its_hash() -> None:
    source = artwork()
    first = prepare_stroke_preview_artifacts(
        source,
        "<script>Artwork</script>.png",
        long_edge_mm=24,
        compact_cleanup_mm=1,
    )
    second = prepare_stroke_preview_artifacts(
        source,
        "<script>Artwork</script>.png",
        long_edge_mm=24,
        compact_cleanup_mm=1,
    )
    assert first.files == second.files
    assert first.manifest["reference_stage"] == "compact-cleaned-raster"
    compact = cast("dict[str, object]", first.manifest["compact_cleanup"])
    assert compact["changed_pixels"] == 1
    assert (
        compact["cleaned_labels_sha256"]
        == first.manifest["source_reference_labels_sha256"]
    )
    with Image.open(BytesIO(first.files["original.png"])) as image:
        assert image.getpixel((4, 4)) == (0, 0, 0)
        palette = np.asarray(SeparationConfig().palette, dtype=np.uint8)
        labels = np.argmin(
            np.sum(
                (np.asarray(image, dtype=np.int32)[:, :, None, :] - palette) ** 2,
                axis=3,
            ),
            axis=2,
        ).astype(np.uint8)
    assert (
        sha256(labels.tobytes()).hexdigest()
        == first.manifest["reference_labels_sha256"]
    )
    html = first.files["review.html"].decode()
    assert "&lt;script&gt;Artwork&lt;/script&gt;.png" in html
    assert "<script>Artwork</script>" not in html
    assert "Compact-cleaned reference" in html
    assert "https://" not in html
    assert isinstance(json.loads(first.files["decisions.json"]), list)


@pytest.mark.parametrize("width_mm", [None, 0.75])
def test_shared_preview_renderer_escapes_custom_text_and_supports_width_override(
    width_mm: float | None,
) -> None:
    with Image.open(BytesIO(artwork())) as image:
        reference = cleanup(separate(image), 24, 0)
    _ = reference.diagnostics.pop("minimum_width_mm")
    reference.diagnostics["target_width_mm"] = 0.75
    files = cleanup_files(
        reference,
        "<user-title>",
        description="<description>",
        width_label="</script><img src=x>",
        width_mm=width_mm,
        before_label="<before>",
        after_label="<after>",
    )
    html = files["review.html"].decode()
    for name in ("user-title", "description", "before", "after"):
        assert f"&lt;{name}&gt;" in html
        assert f"<{name}>" not in html
    assert "</script><img src=x>" not in html
    assert '"width_mm": 0.75' in html


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--target-width-mm", "0"),
        ("--target-width-mm", "-1"),
        ("--target-width-mm", "nan"),
        ("--target-width-mm", "inf"),
        ("--max-growth-mm", "-1"),
        ("--max-growth-mm", "nan"),
        ("--compact-cleanup-mm", "-1"),
        ("--compact-cleanup-mm", "nan"),
    ],
)
def test_invalid_preview_settings_fail_before_publishing(
    option: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "out"
    _ = source.write_bytes(artwork())
    monkeypatch.setattr(
        "sys.argv",
        [
            "panelchemy",
            "stroke-preview",
            str(source),
            "--output",
            str(output),
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("alias", ["source", "settings", "mask", "hardlink", "symlink"])
def test_preview_targets_cannot_overwrite_protected_inputs(
    alias: str, tmp_path: Path
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "out"
    _ = source.write_bytes(artwork())
    target = output / "original.png"
    target.parent.mkdir()
    protected_paths: tuple[Path, ...] = ()
    silhouette_path: Path | None = None
    if alias == "source":
        _ = target.write_bytes(source.read_bytes())
        source = target
    elif alias == "settings":
        _ = target.write_text("protected input settings")
        protected_paths = (target,)
    elif alias == "mask":
        Image.new("L", (24, 24), 255).save(target)
        silhouette_path = target
    elif alias == "hardlink":
        target.hardlink_to(source)
    else:
        target.symlink_to(source)
    snapshot = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        _ = process_stroke_preview_file(
            source,
            output,
            long_edge_mm=24,
            target_width_mm=3,
            protected_paths=protected_paths,
            silhouette_path=silhouette_path,
        )
    assert target.read_bytes() == snapshot
    assert {p.name for p in output.iterdir()} == {"original.png"}
