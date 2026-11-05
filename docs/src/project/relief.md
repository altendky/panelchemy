# Layered Relief

```sh
uv run --locked panelchemy relief /path/to/artwork.png --output local-inputs/relief
```

The image-space separation workflow detects boundaries between color regions.
Exact mode retains their pixel-cell outlines; optional approximation fits shared
boundaries with native cubic Bezier curves before OpenCascade construction.
The output directory contains a combined solid and registered
black/orange/white material BReps, a projected color/support SVG preview, a
construction profile, intermediate separation previews, and a manifest with
input hashes, file checksums, diagnostics, and dependency versions. Empty material
bands are omitted. A BRep is inspectable CAD geometry; this stage does not claim
a print-ready mesh or validated physical print.

Use [model export](export.md) for exact STEP and checked combined/material STL
files. Slicer project packaging is separate, deferred work.

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
the pixel-cell region is unchanged. Native boundary curves define a regional
tapered solid, including its counters. Construction does not create a separate
closed wedge or cone solid for each sampled outline segment. Regional solids
are combined with a separately constructed base and then split into material
bands. Validity, top coverage, connected-solid count, positive volume, and
material-volume accounting are checked before writing geometry. Section checks
also detect invalid lower branches when holes close: a native shape being valid
does not by itself prove that it represents the requested outward taper.

Positive fitted tapers use native unjoined draft walls, generated around each
complete exterior and counter boundary. Entirely straight boundaries use exact
planar wall faces. For curved boundaries, the direct `BRepFill_Draft` API preserves
the walls for intersection; its `Modified` transition mode is not exposed by
the higher-level `MakeDraft` wrapper. Convex material corners receive conical
round patches. Concave material corners receive no round patch: adjacent walls
intersect and trim. This classification concerns the corner between edges,
rather than the curvature along an edge. The selected top edges remain intact.

OpenCascade intersects those surfaces inside a bounding slab and constructs
volume cells. Where a curved hole closes, standard native shape repair can add
missing singular UV boundary edges. Repair uses the existing native precision
without increasing native tolerances. Supporting geometry, real edge ranges,
orientations, and existing vertex positions must remain unchanged; only
degenerate UV edges at existing physical vertices may be added.
The same guarded repair applies when a solid Boolean result reconnects UV
wires incorrectly despite valid inputs; invalid planar results and unrelated
fragments remain errors.

Boolean annotations made only of INTERNAL edges may be removed when their
native vertex graph is acyclic. This includes branching trees and disconnected
chains, which enclose no counter. Closed loops, including loops whose native
`Closed` flag is unset, remain. Cleanup must preserve the exact physical boundary
and pass native validity and area checks.

A folded wall may instead be an internal partition between two material cells.
If both the shared invalid face and its neighboring cell's entire bounding box
are provably inside the intended material, a native union may remove that
partition directly. The distance bound covers the complete box, including
native tolerances, rather than sampling a few points. The union must return one
valid solid; unresolved invalid cells still stop conversion. Remaining cells
are classified by their native distance from the filled top face and united.
A surface continuing below a closed hole can therefore disappear as an internal
partition, leaving solid material below closure while preserving the top hole.

For entirely straight fitted boundaries, a failed volume arrangement can retry
after pairwise intersection of wall faces whose bounding boxes overlap. These
local intersections are imprinted before the same native volume construction,
using the same analytical surfaces, selected outlines, and native settings.
The retry uses the same top and section checks, reports the original failure
and intersection counts, and rejects work beyond its resource limits. A failed
retry remains an error; it does not bypass cell repair or geometry validation.

Entirely straight regions with more than 2,048 edges use coarse, overlapping
source pieces before native construction. Distance expansion distributes over
a union, so fusing their roofs preserves the intended taper. The pieces must
cover the original source and boundary within coordinate roundoff; every
positive-area piece is retained. Subdivision has explicit edge, piece, and depth
limits. A failed piece can retry in a local coordinate frame, then must pass the
same checks after translation back. The final union must preserve the original
top and pass the nine section checks. This changes construction size without
simplifying the outline or discarding small features.

Failures stop conversion; this path does not retry a shell-and-cap construction
or replace fitted curves with sampled straight edges. A zero taper still uses a
vertical prism. Top preservation and nine height sections check the requested
distance expansion, retained source material, and downward support. Regression
cases include closing square and curved counters, concave and convex corners,
closing notches, and initially separate features merging below their tops.
Sampled sections do not prove correctness at every height. Full-panel and mesh
acceptance require their own validation after these regional operations.

Face construction reuses native vertices where counter boundaries touch. If a
contact lies in the middle of an edge, the edge is split at that exact contact
without moving its outline. Wire edge counts, native face validity, and source
area must still agree. A checked native planar Boolean remains available for
unsupported arrangements; no counter is removed to obtain a valid face.

The protected rim uses a conservative polygonal offset with 32 arc segments per
quadrant. Its radius is increased by the corresponding cosine factor so arc
chords cannot reduce the requested rim width. The radius increase is recorded
(about 0.0000603 mm for a 0.2 mm rim). Original top footprints are checked against
that guard and are never silently trimmed.

The fitted outlines reach the kernel as native curves. Polygonal proxies are
used for previews and numerical comparisons; they do not replace fitted CAD
edges. Their certified deviation is recorded separately from the artwork-fitting
tolerance and the export mesh deflection.
Its two support diagrams show material footprints at the lower band boundaries.
Source images are read into a fixed snapshot so a later file edit cannot change
the recorded input of an already-running conversion.

## Optional Contour Approximation

Set `contour_tolerance_mm` in the construction profile to a positive physical
distance to request fitted feature outlines before kernel construction. The
default is zero: exact raster-cell contours. Approximation changes the selected
top outlines within the declared distance; tapering then preserves those selected
tops and expands downward as usual. It does not impose a minimum feature size.

The silhouette stays exact, preserving the panel dimensions and vertical outer
wall. Every source island and counter must survive. Shared boundaries are fitted
once and reused by their neighboring regions, with junctions and point contacts
preserved. A cubic fit is checked against the complete source chain, rather than
only its sampled vertices. Refinement adds smooth fit spans while retaining the
declared error bound. An unresolved curve or topology check stops conversion
instead of restoring a pixel staircase.

Set `contour_mode` to `lines` for straight-segment approximation under the same
continuous distance and topology checks. The default, `curves`, also permits
cubic Bezier spans. Native taper operations can reject a fitted spline profile
even when its planar face is valid; line mode is an explicit alternative rather
than an automatic replacement. Both modes preserve source islands and counters.

`contours.svg` compares original and selected outlines. The manifest records the
requested tolerance, source counts, native curve counts, and validation results.
Region and combined native topology counts
make excessive construction complexity visible.
Inspect that comparison before accepting an approximated panel; a tolerance is
an explicit artwork choice, not a print-quality guarantee. With approximation
enabled, source contour allocation permits at least 100,000 vertices; increasing
`--max-vertices` above that also raises the source limit. The normal kernel budget
still applies after fitting, where it counts native outline edges rather than
the denser polylines used to display and validate curves.

## Physical-Scale Outline Fitting

With positive tolerance and `contour_mode: "curves"`, relief construction uses
`panelchemy.outline_fitting.fit_outlines`. It fits shared boundary
chains jointly as cubic B-splines, represented as lines and cubic Bézier spans.
Its `corner_scale_mm` argument defaults to 1 mm: corner detection compares turns
over physical neighborhoods rather than preserving every pixel turn. This is
separate from a minimum feature width and does not delete islands or counters.
Ordinary joins may move within the requested outline tolerance. Smooth joins
share tangent directions; detected corners and actual palette junctions can
remain sharp. The outer silhouette remains exact.

The fitter also penalizes unnecessary bending within smooth spans. Its physical
fairing scale, `fairing_mm`, defaults to 0.25 mm; zero disables that penalty.
The penalty does not cross intentional corner knots. Forward derivative
constraints are part of the fit, so backward-pointing handles are constrained
before final curve checks. Fixed source directions guide that optimization;
the final curves must still pass the existing chord-direction and topology
checks after conversion. Neither the fairing scale nor optimizer convergence
replaces the requested outline error bound or enforces a manufacturing width.

The API also remains usable for 2D outline review without CAD construction.
Exact circular-arc primitives are not implemented by this API. Continuous distance,
per-span monotonicity, native/proxy separation, and palette topology must all
pass before it returns accepted outlines. A candidate can satisfy the distance
bound and still fail a curve or topology check. It reports an `OutlineFitError`
with diagnostic candidate geometry rather than restoring pixel edges. Such
candidates are unaccepted and must not be sent to CAD as validated geometry.

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
