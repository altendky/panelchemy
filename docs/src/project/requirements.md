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
- Minimum-feature enforcement, small-island removal, and similar cleanup are
  outside the current scope. Consider them separately if needed later.
- Use the supplied art samples to evaluate general behavior without reproducing
  their exact letters, animal anatomy, wear, or halftone rules in the converter.

The current three-color artwork is the initial conversion target. A broader
palette or art-generation system remains a later product decision.

## Open Choices

Final panel size, printer and nozzle, layer configuration, material-band heights,
taper width/angle, image-processing libraries, export formats, and slicer workflow
remain open. OpenCascade is the selected solid-modeling kernel; its Python binding
and geometry construction strategy will be established during implementation.
If an angle is exposed, document which axis it is measured from.

The handoff proposed black/orange/white heights of 2.0/0.6/1.0 mm. These are
unvalidated examples, not approved dimensions or a default physical profile.

Validate geometry before claiming a usable exported model. Evaluate a coupon at
the final artwork scale before claiming physical print quality. The coupon
should exercise dots, strokes, counters, color junctions, and the rim, rather
than shrinking an entire panel.
