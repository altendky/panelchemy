"""Explicit cleaned references reach CAD and artifacts with unchanged fit bounds."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from panelchemy import relief
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours
from panelchemy.raster_cleanup_artifacts import cleanup_files
from panelchemy.separation import SeparationConfig, separate


def artwork() -> Image.Image:
    labels = np.full((20, 20), 2, dtype=np.uint8)
    labels[1:-1, 1:-1] = 0
    labels[6:12, 6:12] = 2
    labels[3, 3] = 2
    return Image.fromarray(
        np.asarray(SeparationConfig().palette, dtype=np.uint8)[labels]
    )


def profile() -> ConstructionProfile:
    return ConstructionProfile(long_edge_mm=18, taper_angle_deg=0, raster_cleanup_mm=1)


def test_cleanup_precedes_vertex_budget_and_preserves_original_input() -> None:
    original = separate(artwork())
    with pytest.raises(ValueError, match="vertices"):
        _ = relief.build_relief(
            original,
            ConstructionProfile(long_edge_mm=18, taper_angle_deg=0),
            max_vertices=8,
        )
    model = relief.build_relief(original, profile(), max_vertices=8)
    assert model.raster_cleanup is not None
    assert len(model.contours.white) == 1
    assert model.contours.white[0].area == 36
    assert model.raster_cleanup.cleaned.labels[3, 3] == 0
    assert original.labels[3, 3] == 2
    assert model.diagnostics["white_top_missing_mm2"] == 0
    assert model.diagnostics["white_top_extra_mm2"] == 0


def test_curve_fitter_uses_cleaned_reference_and_original_accuracy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capture(contours: PanelContours, tolerance: float, **_kwargs: object) -> None:
        assert len(contours.white) == 1
        assert contours.white[0].area == 36
        assert tolerance == 0.2
        raise RuntimeError("cleaned input captured")

    monkeypatch.setattr(relief, "fit_outlines", capture)
    selected = ConstructionProfile(
        long_edge_mm=18, raster_cleanup_mm=1, contour_tolerance_mm=0.2
    )
    with pytest.raises(RuntimeError, match="cleaned input captured"):
        _ = relief.build_relief(separate(artwork()), selected)


def test_exported_cleanup_artifacts_describe_the_exact_model_input(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "model"
    artwork().save(source)
    snapshot = source.read_bytes()
    manifest = relief.process_relief_file(source, output, profile())
    assert manifest["fitting_reference"] == "cleaned-raster"
    assert manifest["raster_cleanup"] == json.loads(
        (output / "cleanup/cleanup.json").read_text()
    )
    with Image.open(output / "cleanup/cleaned.png") as image:
        assert image.getpixel((3, 3)) == (0, 0, 0)
    with Image.open(output / "separation/normalized.png") as image:
        assert image.getpixel((3, 3)) == (255, 255, 255)
    assert source.read_bytes() == snapshot
    assert manifest["source"] == {
        "name": "source.png",
        "sha256": sha256(snapshot).hexdigest(),
    }
    files = cast("dict[str, str]", manifest["files"])
    assert all(
        "cleanup/" + name in files
        for name in (
            "original.png",
            "cleaned.png",
            "changes.png",
            "cleanup.json",
            "review.html",
        )
    )


def test_comparison_artifacts_match_a_model_cleanup() -> None:
    model = relief.build_relief(separate(artwork()), profile())
    assert model.raster_cleanup is not None
    files = cleanup_files(model.raster_cleanup, "<script>Artwork</script>")
    html = files["review.html"].decode()
    assert "&lt;script&gt;Artwork&lt;/script&gt;" in html
    assert "<script>Artwork</script>" not in html
    assert "data:image/png;base64," in html
    assert "https://" not in html


def test_disabling_cleanup_removes_only_its_known_stale_artifacts(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "model"
    artwork().save(source)
    _ = relief.process_relief_file(source, output, profile())
    unrelated = output / "cleanup/notes.txt"
    _ = unrelated.write_text("Keep this review note")
    manifest = relief.process_relief_file(
        source, output, replace(profile(), raster_cleanup_mm=0)
    )
    assert manifest["raster_cleanup"] is None
    assert manifest["fitting_reference"] == "original-separated-raster"
    assert unrelated.read_text() == "Keep this review note"
    assert {p.name for p in (output / "cleanup").iterdir()} == {"notes.txt"}


@pytest.mark.parametrize("alias", ["source", "settings", "mask", "hardlink", "symlink"])
def test_nested_cleanup_targets_never_overwrite_or_delete_inputs(
    tmp_path: Path,
    alias: str,
) -> None:
    source, output = tmp_path / "source.png", tmp_path / "model"
    artwork().save(source)
    target = output / "cleanup/original.png"
    target.parent.mkdir(parents=True)
    silhouette_path: Path | None = None
    protected_paths: tuple[Path, ...] = ()
    if alias == "source":
        artwork().save(target)
        source = target
    elif alias == "settings":
        _ = target.write_text("source settings")
        protected_paths = (target,)
    elif alias == "mask":
        Image.new("L", (20, 20), 255).save(target)
        silhouette_path = target
    elif alias == "hardlink":
        target.hardlink_to(source)
    else:
        target.symlink_to(source)
    snapshot = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        _ = relief.process_relief_file(
            source,
            output,
            replace(profile(), raster_cleanup_mm=0),
            silhouette_path=silhouette_path,
            protected_paths=protected_paths,
        )
    assert target.read_bytes() == snapshot
