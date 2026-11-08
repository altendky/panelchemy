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

The `separate` command applies no gap closing, minimum-feature enforcement,
island filtering, or artwork redesign. One-pixel marks and black counters are
retained. Black support
is the complete silhouette, orange support is visible orange union white, and
white support is white artwork. These are image-space supports before tapering.

The source image and optional input settings/mask remain unchanged. An output
directory that would overwrite an input is rejected. Derived previews contain
the supplied artwork and should remain local unless deliberately approved for
distribution. This stage does not produce a physical model.

## Optional Raster Cleanup

Inspect intentional artwork edits before curve fitting or CAD with:

```sh
uv run --locked panelchemy cleanup /path/to/artwork.png --output local-inputs/cleanup
```

Open `local-inputs/cleanup/review.html` to compare the original separated image,
cleaned image, and highlighted changes. This command does not construct CAD.
It writes `original.png`, `cleaned.png`, `changes.png`, `cleanup.json`,
`review.html`, and `manifest.json`. The report records the selected physical
scale and cleanup decisions separately from later contour approximation.
The original source file, inferred or supplied silhouette, and exterior remain
unchanged.

This first pass handles eligible compact specks, small enclosed counters, and
point-only contacts in the coordinated three-color labels. It is not a general
minimum-body-width filter: long thin strokes, narrow connections, and terminal
points are not automatically removed just because they are below the selected
width. Inspect meaningful lettering, fur, and whiskers in the comparison.
No global blur or independent smoothing of overlapping color masks is applied.

The physical scale uses the panel silhouette's long edge, excluding image
padding. Supply `--profile profile.json` to use its `long_edge_mm` and nozzle
diameter. The cleanup width defaults to a positive `raster_cleanup_mm` from that
profile, otherwise to its assumed printed line width of 1.25 times the nozzle
diameter: 0.5 mm for the default 0.4 mm nozzle. Override the preview cutoff with
`--minimum-width-mm 0.5`. The cutoff must be positive and finite.
The command also accepts the same `--config` and `--silhouette-mask` inputs as
separation. An explicit `cleanup` command requests cleanup even when the profile
leaves it disabled for model conversion.

For `relief` and `export`, raster cleanup is disabled by default. Enable it in
the construction profile, for example:

```json
{
  "long_edge_mm": 250,
  "raster_cleanup_mm": 0.5,
  "contour_tolerance_mm": 0.2
}
```

Zero `raster_cleanup_mm` retains the original separated reference. A positive
value requests the reported artwork edit before tracing. Fitting then certifies
against the cleaned reference with the unchanged configured contour accuracy;
the cleanup edit is not hidden inside or added to the fitting tolerance.
Neither cleanup nor a successful preview guarantees printable features or
successful native geometry.

## Stroke Growth Preview

Inspect width-aware retention and bounded growth proposals separately from CAD:

```sh
uv run --locked panelchemy stroke-preview /path/to/artwork.png \
  --output local-inputs/strokes --profile profile.json
```

Open `local-inputs/strokes/review.html` for aligned reference/proposal images.
`decisions.json` records classifications and reasons why growth was accepted or
vetoed. Classifications describe measured geometry; they do not identify a mark
as an important whisker, letter, or decoration. Uncertain features need review.
This stage is a preview only: `relief` and `export` do not adopt its proposed
growth, and the command does not invoke curve fitting or CAD.

The target width defaults to the profile's assumed line width and the maximum
growth to its tracing radius: 0.5 mm and 0.25 mm for a 0.4 mm nozzle. Override
them with `--target-width-mm` and `--max-growth-mm`. The profile's positive
`raster_cleanup_mm` first applies compact cleanup; override that preliminary
step with `--compact-cleanup-mm`, including zero to skip it. The command also
accepts `--config` and `--silhouette-mask`.

The seven artifacts are `original.png` (the exact growth reference, after any
compact cleanup), `cleaned.png` (proposed growth), `changes.png`, `cleanup.json`
(growth diagnostics), `decisions.json`, `review.html`, and `manifest.json`.
The manifest separately records original-file provenance, optional compact
cleanup, the reference/proposal label hashes, physical settings, and output
hashes. The comparison's highlighted changes show only the growth proposal;
preliminary compact edits are recorded separately in the manifest.

Coarse inputs may use a finer processing grid made by repeating the existing
pixel labels. This preserves their physical cell geometry and adds no source
detail; it permits smaller growth increments. Both comparison images use that
same grid. The manifest records the original and processing dimensions, repeat
factor, and separate source-reference and processing-reference hashes.

The proposal must be inspected before any future adoption as a new artwork
reference. It does not loosen fitting accuracy, guarantee a minimum width
everywhere, or establish printability.

The growth target is the median measured body width of an eligible coherent
stroke. Donor widths and adjacent gaps are checked along fixed source normals;
these are sampled geometric safeguards, not a full boundary certificate.
All accepted edits are checked together, and interacting edits are rolled back
if they violate those safeguards. Short medial-axis spurs may be ignored in
centerline analysis, while their source pixels and donor-tip protection remain.
