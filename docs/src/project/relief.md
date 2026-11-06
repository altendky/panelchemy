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
still applies after fitting, where it counts fitted polynomial spans rather than
the denser polylines used to display and validate curves.

Tangent-continuous cubic spans are encoded as longer native B-spline edges.
This joins their exact polynomial representations without refitting, removing
knots, or changing the approved outline. Real corners, palette junctions, point
contacts, and analytic lines remain separate. Internal knots retain their full
multiplicity; geometric tangency does not imply parametric C1 continuity.
Experimental consolidated draft wires retain the source seam and add splits at
existing span boundaries to combine cubics only while their derivative
directions fit a quarter turn;
an individual wider-turn cubic remains intact. This
avoids a regression where a closing counter lost its original top opening.
Production taper sweeps continue to use separate input spans: OCCT's composite
draft surfaces can change their top boundary enough to fail preservation checks.
The core `roof(..., consolidate_draft=True)` option allows explicit experiments;
it does not relax those checks. Planar faces and zero-taper extrusions use
consolidation by default. Diagnostics distinguish fitted spans, native top edges,
and generated draft wall faces. Consolidation alone does not guarantee a valid
taper arrangement.

## Physical-Scale Outline Fitting

With positive tolerance and `contour_mode: "curves"`, relief construction uses
`panelchemy.outline_fitting.fit_outlines`. It fits shared boundary
chains jointly as cubic B-splines, represented as lines and cubic Bézier spans.
During relief construction, the configured nozzle diameter defines an assumed
printed line width of 1.25 times that diameter. Corner detection compares turns
over neighborhoods of two line widths, then requires a physical prominence of
half a line width. For the default 0.4 mm nozzle these values are 1 mm and
0.25 mm. On a closed loop shorter than four neighborhoods, the prominence
threshold is limited to a quarter of the available neighborhood; the loop must
remain fit-able before optional local softening. A turn's prominence is its
distance from the chord across that physical neighborhood; a tiny raster tread
does not earn a protected corner solely from
its local angle. These are tracing scales, not circular fillets or minimum
curvature constraints. This is separate from a minimum feature width and does
not delete islands or counters.
Ordinary joins may move within the requested outline tolerance. Smooth joins
share tangent directions; detected corners and actual palette junctions can
remain sharp. The outer silhouette remains exact.

For 2D evaluation, the fitting API can also attempt bounded local softening of
detected corners.
It trims the adjoining fitted curves and inserts a tangent-aligned cubic bridge.
The removed tails and bridge control hull must lie inside a neighborhood of
half an assumed line width around the fitted corner (0.25 mm by default).
This is a bound on the spatial extent of the edit, not a claim that every
bridge has an exact circular radius. The complete result is rechecked against
the original raster distance bound and all shared-boundary, counter, island,
and native/proxy topology checks. A corner remains sharp when softening cannot
pass those gates; the diagnostics report applied and blocked softenings.
Actual graph junctions and the outer silhouette are retained. This experimental
phase is enabled explicitly through the fitting API or the construction profile's
`corner_softening_mm` setting, which defaults to zero. A construction profile
cannot exceed the nozzle-derived tracing radius. Planar acceptance does not establish taper-kernel
compatibility: a previously working native taper regression rejects the local
bridge geometry even though the original planar distance/topology gates pass.

The fitter also penalizes unnecessary bending within smooth spans. Its physical
fairing scale, `fairing_mm`, uses half the assumed line width during relief
construction (0.25 mm by default); zero disables that penalty in the fitting API.
The solver constrains movable spline joins to 90% of the configured outline
tolerance. Four two-sided linear bounds form an inscribed octagon within that
Euclidean target, reusing the existing constrained quadratic solver. This gives
the joins headroom without changing the final acceptance limit. Curve interiors
still undergo continuous source-distance and topology certification at the full
configured tolerance and can require further refinement.
If a coarse basis does not yield a verified constrained solve, the existing
bounded refinement loop adds a knot in the interval identified by the failed
constraint and retries. A failed join cap selects its larger adjacent interval.
Unchanged chain fits are reused while distance and topology checks still run.
No failed solver candidate is accepted as an outline.

For tracing experiments, `fit_outlines(..., span_simplification_mm=...)` can
replace several adjoining tangent-continuous cubic pieces with one cubic.
Unlike CAD edge joining, this reduces the actual polynomial count. Each merge
retains its endpoints and endpoint tangent directions, obeys the requested
continuous displacement bound from the accepted trace, and must still satisfy
the original artwork tolerance. Shared palette boundaries are reduced once and
reused in reverse. Complete topology checks retain every island, counter, and
contact; unsafe reductions are rejected. This phase defaults to zero. The
construction profile's `span_simplification_mm` enables it explicitly and cannot
exceed the nozzle-derived tracing radius.
The penalty does not cross intentional corner knots. Forward derivative
constraints are part of the fit, so backward-pointing handles are constrained
before final curve checks. Fixed source directions guide that optimization;
the final curves must still pass the existing chord-direction and topology
checks after conversion. Neither the fairing scale nor optimizer convergence
replaces the requested outline error bound or enforces a manufacturing width.
The independent fitting API accepts `corner_scale_mm`, `corner_prominence_mm`,
`corner_softening_mm`, and `fairing_mm` directly for tuning; zero prominence
disables the prominence gate. Softening defaults to zero in the independent
API. For nozzle-guided experiments, supply the profile's `tracing_radius_mm`
as `corner_softening_mm` (0.25 mm with the default nozzle). This also enables
full native model/STEP trials explicitly, without changing the default pipeline.
Exact raster mode and explicit straight-line mode retain their existing
behavior. Coarse images continue through fitting and are not rejected merely
for falling below the artwork guide's recommended sampling resolution.

The API also remains usable for 2D outline review without CAD construction.
For fitted-curve relief with a positive taper, a final curvature-refinement pass
uses the height difference across each shared boundary. With 2/1/1 mm bands and
a 10-degree taper, black/orange and orange/white use a 0.176 mm horizontal run;
black/white uses 0.353 mm. The independent fitting API accepts cumulative
`material_taper_runs_mm=(black, orange, white)`; adjacent differences determine
each boundary's run. The fixed outer silhouette remains vertical.

The pass targets only the curvature sign whose offset normals converge. It
tries bounded smooth periodic or endpoint-pinned splines, then canonicalizes
the joins and certifies continuously that curvature times boundary run is at
most 0.9. This provides 10% curvature headroom: a minimum local radius of about
0.196 mm for the one-band example. This is an initial engineering margin, not a
guarantee that native intersections will succeed. The check runs after corner
softening and polynomial reduction so those phases cannot undo it. Shared
boundaries are refined once and reused in reverse. Corners can change within
the original outline tolerance; actual graph junctions remain pinned.

Every proposed change must still pass the original source-error bound and the
complete palette topology checks. No islands or counters are deleted. When the
bounded search or a proof cannot accept a proposal, the already certified trace
is retained and `boundary_curvature.unrefined_chains` reports it. The pass does
not reject legitimate early counter closure. Nor does it replace native CAD
validation or resolve collisions between distant edges. White support currently
extends below the exposed orange/white boundary, so hidden support construction
can still encounter a fold even when the exposed boundary meets its limit.

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
