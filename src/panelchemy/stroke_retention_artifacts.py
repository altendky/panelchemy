"""Local stroke-growth proposals with explicit reference and decision provenance."""

import json
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from panelchemy.raster_cleanup import RasterCleanupResult, cleanup
from panelchemy.raster_cleanup_artifacts import cleanup_files
from panelchemy.separation import (
    ArtifactBundle,
    Mask,
    SeparationConfig,
    png_bytes,
    separate,
)
from panelchemy.stroke_retention import preview_stroke_growth


def prepare_stroke_preview_artifacts(
    source_bytes: bytes,
    source_name: str,
    config: SeparationConfig | None = None,
    *,
    long_edge_mm: float = 250,
    target_width_mm: float = 0.5,
    max_growth_mm: float = 0.25,
    compact_cleanup_mm: float = 0,
    silhouette_bytes: bytes | None = None,
    silhouette_name: str = "silhouette.png",
) -> ArtifactBundle:
    """Separate once and inspect proposed growth without changing model defaults."""
    compact_value = cast("object", compact_cleanup_mm)
    if isinstance(compact_value, bool) or not isinstance(compact_value, (int, float)):
        raise ValueError("compact_cleanup_mm must be finite and nonnegative")
    try:
        compact_width = float(compact_value)
    except OverflowError as error:
        raise ValueError("compact_cleanup_mm must be finite and nonnegative") from error
    if not isfinite(compact_width) or compact_width < 0:
        raise ValueError("compact_cleanup_mm must be finite and nonnegative")
    mask: Mask | None = None
    if silhouette_bytes is not None:
        with Image.open(BytesIO(silhouette_bytes)) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    with Image.open(BytesIO(source_bytes)) as image:
        original = separate(image, config, silhouette=mask)
        source_preview = png_bytes(image.convert("RGBA"))
    compact = (
        cleanup(original, long_edge_mm, compact_width) if compact_width > 0 else None
    )
    reference = compact.cleaned if compact is not None else original
    preview = preview_stroke_growth(
        reference,
        long_edge_mm=long_edge_mm,
        target_width_mm=target_width_mm,
        max_growth_mm=max_growth_mm,
    )
    diagnostics = preview.diagnostics
    files = cleanup_files(
        RasterCleanupResult(
            preview.reference, preview.proposed, preview.changed, diagnostics
        ),
        source_name,
        description=(
            "Width-aware stroke growth is a proposal for inspection only. "
            "Magenta marks proposed changes. Classification describes geometry, "
            "not artistic importance. See decisions.json for retained features "
            "and growth vetoes. Model conversion does not use these proposals."
        ),
        width_label="target stroke width",
        width_mm=target_width_mm,
        before_label="Compact-cleaned reference"
        if compact is not None
        else "Original separated reference",
        after_label="Proposed growth (preview only)",
    )
    files["decisions.json"] = (
        json.dumps(preview.decisions, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    manifest: dict[str, object] = dict(
        schema_version=1,
        kind="stroke-growth-preview",
        preview_only=True,
        source=dict(name=source_name, sha256=sha256(source_bytes).hexdigest()),
        silhouette_override=(
            dict(name=silhouette_name, sha256=sha256(silhouette_bytes).hexdigest())
            if silhouette_bytes is not None
            else None
        ),
        dimensions_px=dict(
            zip(
                ("height", "width"),
                cast("tuple[int, int]", preview.reference.labels.shape),
                strict=True,
            )
        ),
        source_dimensions_px=dict(
            zip(
                ("height", "width"),
                cast("tuple[int, int]", preview.source_reference.labels.shape),
                strict=True,
            )
        ),
        working_scale_factor=diagnostics["working_scale_factor"],
        long_edge_mm=long_edge_mm,
        target_width_mm=target_width_mm,
        max_growth_mm=max_growth_mm,
        compact_cleanup_mm=compact_width,
        compact_cleanup=compact.diagnostics if compact is not None else None,
        reference_stage="compact-cleaned-raster"
        if compact is not None
        else "original-separated-raster",
        reference_labels_sha256=sha256(preview.reference.labels.tobytes()).hexdigest(),
        source_reference_labels_sha256=sha256(
            preview.source_reference.labels.tobytes()
        ).hexdigest(),
        proposed_labels_sha256=sha256(preview.proposed.labels.tobytes()).hexdigest(),
        silhouette_sha256=sha256(preview.reference.silhouette.tobytes()).hexdigest(),
        palette=[list(color) for color in reference.config.palette],
        neutral_tolerance=reference.config.neutral_tolerance,
        warnings=reference.warnings,
        stroke_growth=diagnostics,
        outputs=[*files, "manifest.json"],
        files={name: sha256(data).hexdigest() for name, data in files.items()},
    )
    files["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return ArtifactBundle(manifest, files, source_preview)


def process_stroke_preview_file(
    source: Path,
    output: Path,
    config: SeparationConfig | None = None,
    *,
    long_edge_mm: float = 250,
    target_width_mm: float = 0.5,
    max_growth_mm: float = 0.25,
    compact_cleanup_mm: float = 0,
    silhouette_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Write inspectable proposals, protecting source/configuration aliases."""
    bundle = prepare_stroke_preview_artifacts(
        source.read_bytes(),
        source.name,
        config,
        long_edge_mm=long_edge_mm,
        target_width_mm=target_width_mm,
        max_growth_mm=max_growth_mm,
        compact_cleanup_mm=compact_cleanup_mm,
        silhouette_bytes=silhouette_path.read_bytes() if silhouette_path else None,
        silhouette_name=silhouette_path.name if silhouette_path else "silhouette.png",
    )
    protected = {p.resolve() for p in (source, *protected_paths)}
    if silhouette_path is not None:
        protected.add(silhouette_path.resolve())
    for name in bundle.files:
        target = output / name
        if target.resolve() in protected or (
            target.exists()
            and any(path.exists() and target.samefile(path) for path in protected)
        ):
            raise ValueError(
                "output would overwrite an input file; choose another directory"
            )
    output.mkdir(parents=True, exist_ok=True)
    for name, content in bundle.files.items():
        _ = (output / name).write_bytes(content)
    return bundle.manifest
