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

## Parallel Curvature Searches

Use `--workers 4` with `relief` or `export` to run independent initial curvature
repair searches in separate processes. The default is one worker. This execution
setting is separate from the construction profile and applies to approximate
curve fitting; it does not parallelize every conversion stage.

Each child prepares one chain with the existing source and taper checks. The
coordinator applies results in their original order, checks the combined panel,
and performs later retries serially. Search budgets start when a child begins
its work, so queued jobs retain their normal allowance. Worker errors abort the
operation rather than silently substituting a different result.

The `fit_outlines`, `build_relief` and `process_relief_file` APIs accept
`workers=4` as well. Scripts using multiple workers must protect their entry
point with `if __name__ == "__main__":` so spawned processes can import it safely.

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
with the declared layer height. Approximate curve tracing uses nozzle size as
physical guidance, including the bounded point-contact repair described below.
General minimum-feature removal is not implemented.

### Explicit Raster Cleanup

Setting `raster_cleanup_mm` to a positive physical cutoff enables the separate
[raster cleanup stage](separation.md#optional-raster-cleanup) before contour
allocation and curve fitting. Its default is zero, preserving existing inputs.
For a 0.4 mm nozzle, 0.5 mm is the assumed printed line width and a useful
initial compact-detail cutoff. The first pass handles compact isolated regions,
small counters, and eligible same-color diagonal attachments. It preserves
connected terminal points and long narrow regions; it does not enforce a general
minimum body width.

Cleanup deliberately changes the artwork reference. The silhouette, its scale,
and exterior labels remain fixed. The configured curve-fit accuracy applies
unchanged to the cleaned reference. The original separation remains under
`separation/`; `cleanup/` contains the original separated colors, actual cleaned
model input, highlighted changes, a self-contained comparison page, and a report.
The manifest records both label hashes, changed area and color transitions, and
four-connected component/hole counts. These raster counts can differ from the
later geometric polygon counts at diagonal contacts.

The standalone `cleanup` command produces the comparison before any fitting or
CAD is attempted, so an expensive or rejected model does not prevent inspection.
Successful reuse of an output directory with cleanup disabled removes only its
known stale cleanup artifacts; unrelated files and protected inputs are retained.

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

For fitted boundaries, a failed volume arrangement or validation can retry
after pairwise intersection of wall faces whose bounding boxes overlap. These
local intersections are imprinted before the same native volume construction,
using the same analytical surfaces, selected outlines, and native settings.
The retry uses the same top and section checks, reports the original failure
and intersection counts, and rejects work beyond its resource limits. A failed
retry remains an error; it does not bypass cell repair or geometry validation.

If ordinary union of the retry's selected cells fails, a guarded native gluing
operation can reuse their existing partition interfaces. Every input must be a
valid positive solid from that arrangement. Exact shared faces must have opposite
owners and form a connected graph, and every pair must have zero overlapping
volume. Physical BRep data, including input precision, must remain unchanged.
The union still passes the original native, top, and section checks. Cell-pair
work is bounded; memory failures, including wrapped failures, stop retries.

Native draft direction is checked on the generated wall for every input edge.
Projecting the source midpoint onto that wall gives its normal; combining the
normal with the source tangent determines the downward ruling without assuming
surface parameter axes. All measured rulings must agree with outward expansion, or with
inward movement for a counter. A uniformly reversed result gets one fresh draft
with the opposite flag, followed by the same direction check. Missing or
inconsistent evidence fails construction. The source curves, taper angle, and
final geometry checks remain the authority. This avoids relying on the native
draft feature's sparse orientation heuristic for concave loops.

If both native draft flags fail with the known whole-wire array error, the
same source edges can be drafted independently. Each still needs a verified
direction, and the existing ring-level corner patches and intersections remain.
Native analytic lines use the same exact planar walls. Unrelated construction
errors, ambiguous directions, and memory failures do not authorize this retry;
an unresolved individual generator stops conversion.

An individual clamped cubic B-spline can still hit that same native array error.
When it is exactly a nonrational, four-pole Bernstein polynomial on the unit
domain, the adapter can retry it as a native Bézier with identical poles,
endpoint vertices, parameter trim, and orientation. Readback checks preserve
the source geometry and vertex precision; normal draft-direction and wall
validity checks still apply. Unsupported forms or the same array failure retain
the existing quarter-interval trims on the original native curve. This changes
the native representation, without refitting the selected outline.

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
wall. Outside the bounded contact cleanup described below, source islands,
counters and junctions survive. Shared boundaries are fitted once and reused by
their neighboring regions. A cubic fit is checked against the complete source
chain, rather than only its sampled vertices. Refinement adds smooth fit spans
while retaining the declared error bound. An unresolved curve or topology check
stops conversion instead of restoring a pixel staircase.

Set `contour_mode` to `lines` for straight-segment approximation under the same
continuous distance and topology checks. The default, `curves`, also permits
cubic Bezier spans. Native taper operations can reject a fitted spline profile
even when its planar face is valid; line mode is an explicit alternative rather
than an automatic replacement. Both modes preserve source islands and counters,
apart from the explicit attachment repair for tiny point-contact fragments in
curve mode.

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

Source error is geometric distance in both directions: every fitted point must
remain within the configured tolerance of the original boundary, and every
original boundary point must remain within that tolerance of the fitted chain.
Correspondence may slide along the outline instead of matching positions at
fixed source parameters. The reverse direction prevents a fitted shortcut from
leaving source details behind. Exact rational control-hull and segment-capsule
bounds certify the continuous curves; bounded polyline proxies include their
own proven approximation error. Existing parameter-matched certificates remain
a sufficient fast path. An inconclusive or exhausted proof rejects a candidate.
Topology and shared graph junction checks remain separate requirements. Closed
chains retain their exact signed orientation so taper material sides cannot
reverse while tracing the same geometric outline.

Reverse certification starts with a conservative chord proxy whose error is
reserved from the same tolerance. A near-limit source point can need a finer
proxy even when the true curve satisfies the requested bound. Up to three
refinements reduce that approximation error by four each time, sharing one proof
budget. Already certified source intervals remain valid for the unchanged true
curves. An exact necessary-condition check rejects points outside the enlarged
proxy tube. This improves proof precision without enlarging the source limit.

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
Unchanged chain fits and their local distance, orientation, and monotonicity
certificates are reused within one refinement call. Replacing the geometry or
its parameter intervals invalidates that local certificate. Global separation,
proxy, and palette topology checks still run against the combined result.
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

Corner softening first applies this same signed bound to each changed bridge
and trimmed piece. If rounding would violate the margin, its dependency tags
restore the detected corner and its original incident curves through the
existing local rollback. Unrelated safe rounding remains available. This can
leave a real tangent discontinuity for native corner intersection instead of
a tiny rounded connector whose wall folds. It does not label an unsafe smooth
curve as a corner: original incident curves still need their own continuous
certificates. The later curvature fitter may replace the corner with a compliant
smooth candidate if all source and topology checks pass.

Simplification cannot merge across retained corner parameters or introduce a
merged span that fails the boundary's signed curvature limit. Applied softening
corners are removed from that barrier list. With no taper, ordinary softening
and simplification behavior is unchanged. A pre-existing violation that cannot
be repaired still remains unresolved and is rejected by the hard pre-CAD gate.

For a remaining curvature failure, repair first tries a small family of explicit
corner alternatives. It retains active corners and considers additional source
turns only inside failing source intervals, using the taper radius as the
detection scale. The ordinary tracing prominence filter is unchanged. Candidate
families include a few corner-to-corner lines, a jointly fitted cubic spline,
and a mixture that retains curved runs. Original detected-corner provenance is
kept separately from the currently active joins, so a nearby softened corner
can be considered again without restoring unrelated corners elsewhere.

Each alternative must independently pass continuous source distance, signed
curvature, monotonicity and self-separation. Complete palette verification then
checks all selected chains together. A locally valid chord can close a gap
against a different region; only globally rejected chains advance to their next
certified alternative. Unrelated accepted repairs remain unchanged. After the
bounded corner pool is exhausted, the ordinary constrained repair remains
available once, using the remaining boundary budget. Source sampling, point
limits and the final hard radius gate are unchanged.

Constrained local repair can represent those corners with triple spline knots,
using the same corner parameters during native cubic conversion. Numerical
guidance samples both sides of each corner; continuous per-span checks remain
the authority. Ordinary splice endpoints retain their approved positions and
exact inherited tangent rays, and outside curve controls and parameter
intervals remain unchanged. Corner parameters retain their original source
references through wrapped windows. An ordinary free-loop seam stays smooth
unless it is explicitly selected as a corner. Whole-chain alternatives may
rotate graph-pin rays, as the existing circular alternative can, but preserve
pin coordinates and require the complete topology proof.

Chains left unresolved by ordinary repair also get a bounded compact smooth
retry. It replaces a local group with two cubic spans using a C1 spline basis,
allowing curvature to change at a shared tangent. After canonicalization the
joins retain exact geometric tangent continuity. Existing interior corners are
barriers, and outside curves remain unchanged. A straight-chord seed uses
Greville spacing and inherited endpoint directions to avoid encoding every
small source ledge in the initial guess. This retry has an additional four-second
search allowance; only newly accepted candidates enter complete palette checks.
Its numerical guidance can use the full source tolerance, while continuous
source-distance and taper-radius certificates retain their original limits.

The local fitter can also evaluate two cubic sides meeting at one inferred C0
corner. Its source parameter sets the spline break, while the apex coordinates
remain free within the existing fitting constraints. This trial family preserves
splice positions, inherited endpoint rays and all outside curves, and refuses
to replace an existing interior corner. It uses the same continuous source,
curvature and palette proofs. The automatic dispatcher offers it only after
ordinary and compact smooth repair fail, with an additional four-second search
allowance per unresolved chain. The spline break comes from the midpoint of the
failing source intervals, independently of added window padding. Previously
accepted chains remain fixed, and the complete palette is rechecked before any
new inferred-corner repair is accepted. Native construction still validates the
actual taper at the corner.

If the midpoint does not fit, existing curvature witnesses provide additional
source-relative corner positions within the same search allowance. The bounded
two-cubic family can cover a majority of a short pinned chain: fitted span count
alone does not measure the physical extent of a turn. Whole freely periodic
windows still defer to their existing solver, and retained interior corners
remain barriers. Ordinary and compact smooth repair keep their existing window
limits.

After these repairs, an unresolved closed source loop with exactly three or four
convex corners can retain its original straight sides. Exact collinear
subdivisions are removed, while the graph pin remains even if it lies midway
along an edge. This deterministic fallback does not approximate the source or
restore arbitrary raster stairs. Open, concave and more complex loops remain
with their existing repair families. Source-distance, radius, self-separation
and complete palette checks still apply, followed by native taper validation.

If those families leave radius failures, a final bounded trial refits each
complete smooth run bounded by existing active corners or pinned graph
terminals, where boundary ownership changes or branches meet. Its endpoints remain
pinned, while its inside tangent rays may change at those corners. Smooth
splices and freely periodic seams keep their inherited continuity. The run uses
the same source-distance and signed-radius limits and must pass complete palette
checks. Its exact forward
progress test admits a backward middle control-polygon leg only when projected
curve velocity stays strictly positive throughout the interior. Initial tracing
keeps its existing control-polygon restriction. Compact and ordinary run trials
share four seconds per unresolved chain. Successful runs are retained even if
other spans on that chain remain unresolved, subject to complete source and
palette verification. Later contact cleanup can build on those repairs.

After contact cleanup, remaining radius failures get a bounded corner repair.
It first tries intersecting the failed span's endpoint tangent rays. Existing
corners may release their incident tangent directions; smooth outer joins keep
their inherited rays. A two-span window can include a smooth neighbor when its
outer endpoint is already a corner, but cannot erase an interior corner.
Smooth vertices shared by other rings remain pinned too.
Where both window ends are existing corners, a short original source arc can
also propose two or three straight sides, simplified at 90% of the configured
fit tolerance. These are local alternatives, not a return to pixel-cell tracing.

Every candidate must pass continuous forward progress and signed curvature,
the full original-source distance limit, and complete palette topology checks.
Shared spans are excluded. Acceptance must remove at least one radius failure
without adding another, so successful repairs survive later rejected trials.
The hard pre-CAD radius gate and native taper validation still apply.

Source detail is also classified at the assumed extrusion width and twice that
width. Ordinary raster stairs make forward progress along an edge. Fine texture
has repeated small backtracking or transverse excursions, with excess path
length beyond ordinary raster stairs at both scales. A substantial corner has
turn and prominence that persist and grow at the wider scale; ambiguous cases
remain unclassified. These labels describe geometry rather than artistic intent.

A chain whose remaining radius failures overlap fine texture tries the compact
smooth family first. Its windows cannot cross classified structural vertices;
existing corner and topology checks still control the candidate geometry. The
ordinary repair retains its full remaining search budget if that preference
fails. Classification grants no error allowance: every accepted curve must
pass the same configured continuous source-distance and taper-radius checks.
Locally certified proposals rejected by palette checks can be rechecked against
later repaired neighbors without another optimizer search. Diagnostics record
class counts, texture intervals and the selected fit family.

After ordinary fitting and curvature repair, a bounded interpretation step can
attach a group of tiny same-color fragments to one substantial parent. Each
fragment fits within the assumed extrusion width; the whole cluster is bounded
by twice that width. Point and multiple-point contacts are supported. Groups
must remain clear of other colors, unrelated parts and the silhouette.
Only newly enclosed contact counters within one extrusion width may be filled;
every original parent hole must survive unambiguously.

Isolated one-point attachments are attempted when an incident curve fails its
radius check. Chains and repeated contacts also receive cleanup when their
individual curves already meet that limit, because zero-width connections can
destabilize later CAD operations. Bounded ring-arc chords and pointed splices
absorb the fragments while retaining all outside polynomial coefficients.
This explicitly changes selected component connectivity and reports the exact
source members and filled counters.
Both candidate families reject zero-length steps and consecutive segments that
retrace each other within floating-point roundoff. The collinearity guard scales
with coordinate precision, not nozzle size or the source-fit allowance, so
resolved sharp points remain eligible and the search can try another candidate.
The complete analytic-line pair check also rejects an endpoint that nearly
touches another line's interior within eight coordinate ulps, excluding shared
endpoints. Its spatial bounds include that same roundoff guard. This prevents
selection from relying on an arithmetic-sized gap that CAD would collapse;
it does not set a manufacturing minimum gap or relax the source-fit limit.

Each accepted attachment requires the original continuous distance certificate
against all separate raw component rings at the same configured tolerance.
No fictitious segment connects those rings during certification. Complete
curve/proxy separation and palette relationships must match the selected joins,
and no new radius failures are allowed. Interpretive splices can introduce G0
joins; their float-generated rays are not claimed to retain exact G1 tangency.
Exact mode, free islands and unrelated components remain unchanged. The
final independent radius and native construction checks still apply.

Localized failures first try just the contiguous offending spans, then add
neighboring spans when more support is needed. Each independent small region
gets a chance before growth merges nearby windows; one difficult region cannot
starve the other repairs. The existing uniform family gets the first half of a
local window's available search share; corner-guided plans share the remainder.
This prevents earlier soft-budget proofs from starving a viable smooth fit.
Search time is reserved for later growth. A window keeps its approved
endpoint positions and inside tangent directions; all outside curve controls
and parameter intervals remain unchanged. This preserves an existing corner's
angle rather than forcing a smooth join there. Only freely periodic loops may
wrap a repair across their starting point. A window covering at least half the
chain defers to whole-chain fitting, which gives small boundaries more freedom.
Closed loops anchored at a graph junction retain both approved endpoint tangent
rays during whole-chain fitting too. This preserves the branches' order at the
shared point; pinning its position alone can allow a microscopic self-crossing.
After each repair the remaining failures are located again. A partially repaired
chain is never accepted: every span must pass before it can enter shared palette
verification. Local attempts and whole-chain fallback share one search budget.

Each local attempt first tries midpoint-refined knots from the approved piece
intervals, with eight search probes per interval and at most 32 spans. This
places freedom near small bends instead of distributing every knot uniformly
along long stretches. It receives at most half the window's remaining time;
the original uniform search can use the rest. Inherited tangent handles use
finer dyadic steps only when their coordinates remain exactly representable.
Their direction stays exact; the source and curvature checks are unchanged.

A small closed loop anchored at a graph junction can also have a circular
alternative certified before optimization, within the same boundary budget.
It is used only when the preferred searches fail. The shared point remains
exact, but its tangent directions may change. Both directions of source error,
continuous curvature, within-chain separation, and complete palette topology
must still pass. No loop is removed or accepted by size alone.

Curvature searches use nearest-polyline distances in both directions as sampled
guidance, allowing tangential correspondence to move. These searches rebuild
their own candidate splines and do not retain the initial solver's
parameter-matched join caps. Sampled distances do not replace the continuous
source-error proof. A certified candidate can stop the optimizer before
numerical objective convergence; a numerical success flag cannot authorize a
curve without the geometric checks. Final shared topology verification still
runs after combining independent candidate changes.

When continuous checks reject a promising candidate, numerical curvature
extrema or geometric-distance peaks can add constraint locations before the
search increases the span count. Curve parameters and original-source query
parameters are separate: a reverse-source omission also refines its nearest
curve-proxy segment. Each retry retains earlier locations, keeps the same
spline basis, and starts from the rejected control points. At most three
adaptive retries are allowed per seed, with up to 16 new parameters per domain
per retry; all seeds and bases share the existing 12-second search budget.
Numerical hints never authorize a curve. Source distance, curvature,
orientation, exact joins, and shared topology must still pass their existing
checks. Duplicate or unavailable hints do not trigger another retry.

Every proposed change must still pass the original source-error bound and the
complete palette topology checks. No islands or counters are deleted. When the
bounded search or a proof cannot accept a proposal, the original source-distance
and topology-certified trace is retained. Its curvature margin remains
uncertified: `boundary_curvature.margin_status` is `unresolved`, and
`unresolved_boundaries` records the remaining pieces and parameter intervals
after palette rollback, using the selected fit's updated piece intervals.
`unrefined_chains` retains the chain-level summary, including partially repaired
chains; `partially_refined_chains` identifies those retained improvements.
`refined_chains` counts only chains whose selected spans all pass the radius
check. Keeping a successful run does not certify its unresolved neighbors.
A `certified` margin status means all selected boundaries passed the curvature
target. These diagnostic outlines remain available for 2D inspection even
when their margin is unresolved. Relief construction independently recomputes
shared-span ownership from the final selected controls and certifies every
positive exposed boundary run before entering any native geometry adapter.
It does not trust cached fitting status. A violation or an inconclusive proof
raises `CurvatureLimitError`, retaining the selected outlines and failed bounds
for diagnosis; the unresolved trace cannot produce a CAD model. This hard gate
uses the same continuous signed bound and 0.9 margin as fitting, after all
smoothing, simplification, and palette rollback. It does not impose a new
fillet policy at joins or turn the nozzle guidance into a global radius limit.

The pass does not replace native CAD
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
