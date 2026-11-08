# Construction Requirements

## Agreed Direction

- Keep original artwork unchanged and make derived results reproducible through
  retained code, configuration, dependency locks, and output manifests.
- The outside edge of the black rim defines the solid panel silhouette. Remove
  the external gutter from the physical object, preserve a protected black rim,
  and keep the outer wall vertical.
- Stack material in global horizontal bands: black, then orange, then white.
  The black foundation is continuous; orange supports every white feature.
- Before tapering, cumulative supports are silhouette for black, visible orange
  union white for orange, and white for white.
- Preserve original feature tops and expand internal tapers outward toward their
  bases. Do not shrink whiskers or other thin features as a consequence of
  tapering. Use real offsets in physical units rather than scaling shapes.
- Higher cross-sections must fit within lower supporting cross-sections,
  including at the orange/white interface. Keep black counters and gaps as
  recessed surfaces with a base underneath, rather than through-holes.
- Expanding lower footprints can meet across gaps or approach the protected rim.
  Make those interactions inspectable and account for meaningful changes.
- Use the [artwork preparation guide](artwork-guide.md) to target 1 mm minimum
  feature bodies, gaps, and recesses while retaining sharp terminal points.
  Bounded point-contact cleanup may join sub-line-width fragments and fill new
  tiny contact counters, with the original source-error bound certified.
  Separate opt-in raster cleanup can remove compact regions, fill small counters,
  and resolve eligible diagonal contacts before tracing. Its intentional changes
  are reported against the original raster, and fitting retains its configured
  accuracy against the cleaned reference. General minimum body-width filtering
  remains future work.
- Optional contour approximation is explicitly selected in millimeters and off
  by default. Retain islands and counters apart from explicit bounded contact
  and raster cleanup, validate the boundary deviation, and provide original/selected
  outlines. Tapering preserves the resulting
  selected tops; approximation is a separate, reported artwork change.
- Use the supplied art samples to evaluate general behavior without reproducing
  their exact letters, animal anatomy, wear, or halftone rules in the converter.

The current three-color artwork is the initial conversion target. A broader
palette or art-generation system remains a later product decision.

## Open Choices

The user selected configurable defaults: 250 mm on the long edge, 0.4 mm nozzle,
0.2 mm layer height, and black/orange/white band thicknesses of 2/1/1 mm (4 mm
total). Taper angle is also configurable, in degrees from vertical. Zero gives
vertical internal walls; larger angles expand the lower footprint while retaining
the original top. Nozzle size guides tracing and bounded contact cleanup; the
broader feature-size targets remain artwork requirements.

The implementation starts with a 10-degree internal taper and a 0.2 mm protected
rim. Those are adjustable software defaults, not physically validated settings.
The actual printer, slicer/material configuration, mesh export settings, and
observed print quality remain to be established. OpenCascade via OCP is the
selected solid-modeling kernel.

Validate geometry before claiming a usable exported model. Evaluate a coupon at
the final artwork scale before claiming physical print quality. The coupon
should exercise dots, strokes, counters, color junctions, and the rim, rather
than shrinking an entire panel.
