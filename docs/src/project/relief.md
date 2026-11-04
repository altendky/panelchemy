# Layered Relief

```sh
uv run --locked panelchemy relief /path/to/artwork.png --output local-inputs/relief
```

The image-space separation workflow feeds exact pixel-cell contours into
OpenCascade. The output directory contains a combined solid and registered
black/orange/white material BReps, a projected color/support SVG preview, a
construction profile, intermediate separation previews, and a manifest with
input hashes, file checksums, diagnostics, and dependency versions. Empty material
bands are omitted. A BRep is inspectable CAD geometry; this stage does not claim
a print-ready mesh or validated physical print.

## Construction Profile

Use `--profile settings.json` with the
[example profile](../../../examples/construction-profile.json). The selected
silhouette's long bounding-box edge becomes 250 mm by default; the external
gutter does not contribute to size. All regions share a lower-left origin and
positive Y points upward. The original source pixel bounds and scale are recorded.

The configurable defaults selected by the user are:

- Nozzle diameter: 0.4 mm; layer height: 0.2 mm.
- Black base: 2 mm; orange band: 1 mm; white band: 1 mm.
- Long edge: 250 mm.

Taper angle is in degrees **from vertical**, with zero producing vertical
internal walls. The initial 10-degree taper and 0.2 mm protected black rim are
adjustable software defaults awaiting print evaluation. Band heights must align
with the declared layer height. Nozzle size is metadata and never triggers
small-feature cleanup.

## Outward Tapers and Support

Each source plateau retains its original top footprint. Its lower sections
expand by horizontal run equal to height below the top multiplied by the tangent
of the taper angle. OpenCascade builds round-join distance roofs, combines them,
protects the outer rim, and divides the result into global material bands.
White therefore has orange beneath it at the actual interface, including where
the artwork has no visible orange.

The black foundation follows the silhouette and keeps its exterior vertical.
If a rim setting would trim an original feature top, conversion fails explicitly.
Lower taper expansion may be clipped at the protected rim; that volume is
reported. Expanding roofs can join across gaps or close counters below their
original tops, and white ramps can cover orange plateau in top view. The
diagnostics and SVG show those interactions. Reduce the taper angle to compare
their effects; the tool does not delete or resize features to make them printable.

Exact duplicate/collinear contour points are removed only after checking that
the pixel-cell region is unchanged. The kernel first attempts an evolved sweep;
if that fails validation, it constructs exact outward edge wedges and corner
sectors. Both paths check original top coverage, including counters. Validity,
connected-solid count, positive volume, and material-volume accounting are checked
before writing geometry. The evolved sweep also checks five distance sections,
including both caps, for expected area, retained source material, and nested
support; a failed check selects the exact construction.

The protected rim uses a conservative polygonal offset with 32 arc segments per
quadrant. Its radius is increased by the corresponding cosine factor so arc
chords cannot reduce the requested rim width. The radius increase is recorded
(about 0.0000603 mm for a 0.2 mm rim). Original top footprints are checked against
that guard and are never silently trimmed.

The taper roofs themselves use native curves. The SVG approximates their arcs
only for display; its display approximation does not alter the BRep.
Its two support diagrams show material footprints at the lower band boundaries.
Source images are read into a fixed snapshot so a later file edit cannot change
the recorded input of an already-running conversion.

## Boundary and Resource Limits

Automatic silhouettes with unresolved boundary warnings require an explicit
mask before relief construction. A silhouette must be one connected, hole-free
panel; black artwork counters remain recesses above a continuous base.

The initial contour budget is 25,000 exact vertices. Use `--max-vertices` to
change the budget. Exceeding it fails before kernel construction without
simplifying artwork. Dense raster contours and many interacting tapers can be
expensive even within that budget; inspect small representative features before
attempting a complete detailed panel. Unverified geometry is never substituted
for a failed kernel result.
