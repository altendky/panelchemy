"""Deterministic palette separation and inspectable panel silhouettes."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Self, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy import ndimage

type RGB = tuple[int, int, int]
type Palette = tuple[RGB, RGB, RGB]
type Mask = NDArray[np.bool_]
type Labels = NDArray[np.uint8]


@dataclass(frozen=True)
class SeparationConfig:
    """Palette order is black, orange, white; neutral pixels avoid false orange."""

    palette: Palette = ((0, 0, 0), (245, 166, 35), (255, 255, 255))
    neutral_tolerance: float = 20

    def __post_init__(self) -> None:
        if len(self.palette) != 3 or len(set(self.palette)) != 3:
            raise ValueError("palette must contain three distinct RGB colors")
        for color in self.palette:
            if len(color) != 3 or any(
                type(channel) is not int or not 0 <= channel <= 255 for channel in color
            ):
                raise ValueError(
                    "palette colors must have three integer channels, 0-255"
                )
        if (
            isinstance(self.neutral_tolerance, bool)
            or not 0 <= self.neutral_tolerance <= 255
            or not isfinite(self.neutral_tolerance)
        ):
            raise ValueError("neutral_tolerance must be finite and between 0 and 255")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Self:
        unknown = set(data) - {"palette", "neutral_tolerance"}
        if unknown:
            raise ValueError(
                f"unknown separation settings: {', '.join(sorted(unknown))}"
            )
        palette = cls().palette
        if "palette" in data:
            raw = data["palette"]
            if not isinstance(raw, list):
                raise ValueError("palette must contain three RGB color lists")
            raw_colors = cast("list[object]", raw)
            if len(raw_colors) != 3:
                raise ValueError("palette must contain three RGB color lists")
            colors: list[RGB] = []
            for color in raw_colors:
                if not isinstance(color, list):
                    raise ValueError("each palette color must contain three channels")
                channels = cast("list[object]", color)
                if len(channels) != 3:
                    raise ValueError("each palette color must contain three channels")
                if not all(type(value) is int for value in channels):
                    raise ValueError("palette channels must be integers")
                colors.append(cast("RGB", tuple(channels)))
            palette = (colors[0], colors[1], colors[2])
        tolerance = data.get("neutral_tolerance", cls().neutral_tolerance)
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
            raise ValueError("neutral_tolerance must be a number")
        return cls(palette=palette, neutral_tolerance=tolerance)


@dataclass
class SeparationResult:
    """Full-resolution masks; no feature deletion, gap closing, or island filtering."""

    labels: Labels
    silhouette: Mask
    config: SeparationConfig
    silhouette_method: str
    warnings: list[str]
    component_count: int

    @property
    def masks(self) -> dict[str, Mask]:
        white = cast("Mask", self.silhouette & (self.labels == 2))
        return {
            "silhouette": self.silhouette,
            "exterior": ~self.silhouette,
            "black": self.silhouette & (self.labels == 0),
            "orange": self.silhouette & (self.labels == 1),
            "white": white,
            "support-black": self.silhouette,
            "support-orange": self.silhouette & (self.labels != 0),
            "support-white": white,
        }

    def normalized_image(self) -> Image.Image:
        palette = np.asarray(self.config.palette, dtype=np.uint8)
        return Image.fromarray(palette[self.labels])

    def panel_image(self) -> Image.Image:
        panel = self.normalized_image().convert("RGBA")
        panel.putalpha(Image.fromarray(self.silhouette.astype(np.uint8) * 255))
        return panel


def separate(
    image: Image.Image,
    config: SeparationConfig | None = None,
    *,
    silhouette: Mask | None = None,
) -> SeparationResult:
    """Normalize colors and exclude edge-connected white using 4-connectivity.

    An explicit silhouette overrides only the boundary decision. It never alters
    palette labels or cleans artwork details. Transparency composites over white.
    """
    config = config or SeparationConfig()
    if image.width < 1 or image.height < 1:
        raise ValueError("image dimensions must be positive")
    background = Image.new("RGBA", image.size, (*config.palette[2], 255))
    rgb = np.asarray(
        Image.alpha_composite(background, image.convert("RGBA")).convert("RGB"),
        dtype=np.uint8,
    )
    differences = rgb.astype(np.int32)[:, :, None, :] - np.asarray(
        config.palette, dtype=np.int32
    )
    distances = np.sum(differences * differences, axis=3, dtype=np.int32)
    neutral = (np.max(rgb, axis=2) - np.min(rgb, axis=2)) <= config.neutral_tolerance
    distances[:, :, 1][neutral] = np.iinfo(np.int32).max
    labels = np.argmin(distances, axis=2).astype(np.uint8)
    structure = ndimage.generate_binary_structure(2, 1)
    if silhouette is None:
        white = cast("Mask", labels == 2)
        seeds = np.zeros_like(white)
        seeds[0, :] = white[0, :]
        seeds[-1, :] = white[-1, :]
        seeds[:, 0] = white[:, 0]
        seeds[:, -1] = white[:, -1]
        outside = ndimage.binary_propagation(seeds, structure=structure, mask=white)
        silhouette = np.asarray(~outside, dtype=np.bool_)
        method = "edge-connected-white-4"
    else:
        if silhouette.shape != labels.shape or silhouette.dtype != np.bool_:
            raise ValueError("silhouette override must be a same-size boolean mask")
        silhouette = silhouette.copy()
        method = "explicit-mask"
    if not np.any(silhouette):
        raise ValueError(
            "no panel found; supply artwork with a rim or an explicit mask"
        )
    component_labels = np.empty(silhouette.shape, dtype=np.int32)
    component_count = int(
        ndimage.label(silhouette, structure=structure, output=component_labels)
    )
    warnings: list[str] = []
    if component_count != 1:
        warnings.append(
            f"Silhouette has {component_count} disconnected components; possible debris or multiple panels. All components are preserved; use an explicit mask to select the intended boundary."
        )
    if not np.any(cast("Mask", silhouette & (labels == 2))):
        warnings.append(
            "No enclosed white artwork was found. A broken rim may connect the artwork to the external gutter; inspect or supply an explicit mask."
        )
    if np.any(silhouette[[0, -1], :]) or np.any(silhouette[:, [0, -1]]):
        warnings.append(
            "The silhouette touches the image edge; its rim may be cropped."
        )
    return SeparationResult(
        labels, silhouette, config, method, warnings, component_count
    )


def load_config(path: Path | None) -> SeparationConfig:
    if path is None:
        return SeparationConfig()
    data = cast("object", json.loads(path.read_text()))
    if not isinstance(data, dict):
        raise ValueError("separation configuration must be a JSON object")
    return SeparationConfig.from_mapping(cast("dict[str, object]", data))


def load_silhouette(path: Path) -> Mask:
    """Explicit masks use white inside and black outside, thresholded at 128."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def png_bytes(image: Image.Image) -> bytes:
    """Encode pixels without copying source metadata into derived previews."""
    image = image.copy()
    image.info.clear()
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@dataclass
class ArtifactBundle:
    """The same reproducible artifacts for filesystem and browser adapters."""

    manifest: dict[str, object]
    files: dict[str, bytes]
    source_preview: bytes

    def archive(self) -> bytes:
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            for name, content in self.files.items():
                entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                archive.writestr(entry, content)
        return output.getvalue()


def prepare_artifacts(
    source_bytes: bytes,
    source_name: str,
    config: SeparationConfig | None = None,
    *,
    silhouette_bytes: bytes | None = None,
    silhouette_name: str = "silhouette.png",
) -> ArtifactBundle:
    """Convert in memory; adapters decide how the resulting bytes are retained."""
    silhouette = None
    if silhouette_bytes is not None:
        with Image.open(BytesIO(silhouette_bytes)) as mask_image:
            silhouette = np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128
    with Image.open(BytesIO(source_bytes)) as image:
        result = separate(image, config, silhouette=silhouette)
        source_preview = png_bytes(image.convert("RGBA"))
    masks = result.masks
    files = {
        "normalized.png": png_bytes(result.normalized_image()),
        "panel.png": png_bytes(result.panel_image()),
    }
    for name, mask in masks.items():
        files[f"{name}.png"] = png_bytes(Image.fromarray(mask.astype(np.uint8) * 255))
    resolved_config = {
        "palette": [list(color) for color in result.config.palette],
        "neutral_tolerance": result.config.neutral_tolerance,
    }
    files["config.json"] = (
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
    ).encode()
    filenames = [*files, "manifest.json"]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": {"name": source_name, "sha256": sha256(source_bytes).hexdigest()},
        "dimensions_px": {
            "width": result.labels.shape[1],
            "height": result.labels.shape[0],
        },
        "config": resolved_config,
        "silhouette_method": result.silhouette_method,
        "silhouette_override": (
            {"name": silhouette_name, "sha256": sha256(silhouette_bytes).hexdigest()}
            if silhouette_bytes is not None
            else None
        ),
        "component_count": result.component_count,
        "pixel_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "warnings": result.warnings,
        "versions": {
            name: version(name) for name in ("panelchemy", "numpy", "pillow", "scipy")
        },
        "outputs": filenames,
    }
    files["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    return ArtifactBundle(manifest, files, source_preview)


def process_file(
    source: Path,
    output: Path,
    config: SeparationConfig | None = None,
    *,
    silhouette_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Write reproducible previews and provenance without modifying input files."""
    bundle = prepare_artifacts(
        source.read_bytes(),
        source.name,
        config,
        silhouette_bytes=silhouette_path.read_bytes() if silhouette_path else None,
        silhouette_name=silhouette_path.name if silhouette_path else "silhouette.png",
    )
    protected = {p.resolve() for p in (source, *protected_paths)}
    if silhouette_path is not None:
        protected.add(silhouette_path.resolve())
    if any(
        (output / name).resolve() in protected
        or (
            (output / name).exists()
            and any(p.exists() and (output / name).samefile(p) for p in protected)
        )
        for name in bundle.files
    ):
        raise ValueError(
            "output would overwrite an input file; choose another directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, content in bundle.files.items():
        _ = (output / name).write_bytes(content)
    return bundle.manifest
