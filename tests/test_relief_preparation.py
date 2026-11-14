"""Shared raster preparation and original RGB evidence reach normal tracing."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from panelchemy import relief
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours
from panelchemy.curves import CurveFitResult
from panelchemy.raster_cleanup_artifacts import prepare_cleanup_artifacts
from panelchemy.separation import SeparationResult, separate
from panelchemy.shade_guides import ShadeGuides


class ReachedTracing(Exception):
    pass


def artwork() -> Image.Image:
    image = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 45, 65, 70), fill="white")
    draw.rectangle((50, 20, 51, 45), fill="white")
    image.putpixel((25, 25), (255, 255, 255))
    return image


@pytest.mark.parametrize("with_rgb", [False, True])
@pytest.mark.parametrize("compact_width", [0, 0.5])
def test_model_and_preview_share_preparation_without_reclassifying_rgb(
    monkeypatch: pytest.MonkeyPatch, with_rgb: bool, compact_width: float
) -> None:
    image = artwork()
    mask = np.ones((100, 100), dtype=np.bool_)
    original = separate(image, silhouette=mask)
    before = original.labels.copy()
    profile = ConstructionProfile(
        long_edge_mm=25,
        contour_tolerance_mm=0.2,
        raster_cleanup_mm=compact_width,
        stroke_body_width_mm=1,
        stroke_support_length_mm=3,
    )
    retained_hashes: list[str] = []
    real_guides = ShadeGuides

    def guides(
        rgb: Image.Image,
        *,
        original: SeparationResult,
        retained: SeparationResult,
        source: PanelContours,
    ) -> ShadeGuides:
        from hashlib import sha256

        assert rgb.tobytes() == image.tobytes()
        assert original.labels[25, 25] == 2
        assert retained.labels[25, 25] == 0
        assert np.array_equal(
            retained.labels[25:40, 50:52], np.full((15, 2), 2, dtype=np.uint8)
        )
        retained_hashes.append(sha256(retained.labels.tobytes()).hexdigest())
        return real_guides(rgb, original=original, retained=retained, source=source)

    def trace(
        source: PanelContours,
        tolerance: float,
        *,
        shade_guides: ShadeGuides | None = None,
        **_kwargs: object,
    ) -> CurveFitResult:
        assert tolerance == 0.2
        assert len(source.white) == 1
        assert (shade_guides is not None) == with_rgb
        raise ReachedTracing

    monkeypatch.setattr(relief, "ShadeGuides", guides)
    monkeypatch.setattr(relief, "fit_outlines", trace)
    with pytest.raises(ReachedTracing):
        _ = relief.build_relief(
            original, profile, rgb_image=image if with_rgb else None
        )
    assert np.array_equal(before, original.labels)
    if with_rgb:
        source_bytes, mask_bytes = BytesIO(), BytesIO()
        image.save(source_bytes, format="PNG")
        Image.new("L", image.size, 255).save(mask_bytes, format="PNG")
        bundle = prepare_cleanup_artifacts(
            source_bytes.getvalue(),
            "synthetic.png",
            long_edge_mm=25,
            minimum_width_mm=compact_width,
            stroke_body_width_mm=1,
            stroke_support_length_mm=3,
            silhouette_bytes=mask_bytes.getvalue(),
        )
        diagnostics = bundle.manifest["cleanup"]
        assert isinstance(diagnostics, dict)
        assert diagnostics["cleaned_labels_sha256"] == retained_hashes[0]


def test_rgb_registration_is_checked_before_preparation() -> None:
    image = artwork()
    original = separate(image, silhouette=np.ones((100, 100), dtype=np.bool_))
    with pytest.raises(ValueError, match="RGB guidance must match"):
        _ = relief.build_relief(original, rgb_image=Image.new("RGB", (1, 1)))


def test_file_adapter_preserves_alpha_before_shade_compositing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 255))
    image.putpixel((5, 5), (0, 0, 0, 0))
    source = tmp_path / "rgba.png"
    mask = tmp_path / "mask.png"
    image.save(source)
    Image.new("L", image.size, 255).save(mask)

    def build(
        result: SeparationResult,
        _profile: ConstructionProfile | None = None,
        *,
        rgb_image: Image.Image | None = None,
        **_kwargs: object,
    ) -> relief.ReliefModel:
        assert rgb_image is not None and rgb_image.mode == "RGBA"
        assert rgb_image.getpixel((5, 5)) == (0, 0, 0, 0)
        assert rgb_image.tobytes() == image.tobytes()
        assert result.labels[5, 5] == 2
        raise ReachedTracing

    monkeypatch.setattr(relief, "build_relief", build)
    with pytest.raises(ReachedTracing):
        _ = relief.process_relief_file(source, tmp_path / "out", silhouette_path=mask)
    assert not (tmp_path / "out").exists()
