# Color Separation

See [Preparing artwork](artwork-guide.md) for feature-size design targets, flat
colors, boundary options, and a copyable image-generation prompt.

```sh
uv run --locked panelchemy separate /path/to/artwork.png --output local-inputs/preview
```

The reusable `panelchemy.separation.separate` API accepts a Pillow image and
returns full-resolution labels and masks. The file adapter writes normalized
and transparent panel previews, silhouette/exterior masks, visible black/orange/
white masks, cumulative material supports, resolved configuration, and a manifest.
The manifest records input hashes, dependency versions, pixel counts, warnings,
and output names. Regeneration with identical input/configuration is deterministic.

## Palette

The fixed palette order is black, orange, white. Nearest-color normalization uses
integer RGB distances without dithering. Near-neutral pixels choose between
black and white, preventing gray antialiasing from becoming false orange.
Transparency composites over the configured white color.

Use `--config settings.json` to change the palette or neutral guard:

```json
{
  "palette": [[0, 0, 0], [245, 166, 35], [255, 255, 255]],
  "neutral_tolerance": 20
}
```

`neutral_tolerance` is the maximum RGB channel spread considered neutral, in
the range 0–255. Inspect the normalized preview when adjusting it.

## Boundary and Ambiguity

Automatic extraction removes white connected to the image exterior using
four-neighbor connectivity. White enclosed by the black rim remains artwork.
This assumes an enclosing rim; it cannot reconstruct a broken or cropped rim.
Warnings flag multiple silhouette components, missing enclosed white, and a
silhouette touching the canvas. All components remain present.

Supply `--silhouette-mask boundary.png` to override the boundary: same image
dimensions, white inside and black outside, thresholded at 128. This is an
explicit user boundary decision; it does not change palette labels. Inspect
the output masks even if no warning is emitted, since a raster alone cannot
prove the intended boundary.

No gap closing, minimum-feature enforcement, island filtering, or artwork
redesign occurs. One-pixel marks and black counters are retained. Black support
is the complete silhouette, orange support is visible orange union white, and
white support is white artwork. These are image-space supports before tapering.

The source image and optional input settings/mask remain unchanged. An output
directory that would overwrite an input is rejected. Derived previews contain
the supplied artwork and should remain local unless deliberately approved for
distribution. This stage does not produce a physical model.
