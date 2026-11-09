# Tracing Performance

## Measurements

The November 2026 investigation used the 250 mm evaluation Lynx, a 0.2 mm
source-fit limit, 0.25 mm corner softening, 0.1 mm span simplification, and a
10 degree taper. These are evaluation settings, not artwork-specific rules.

The baseline run at `9d4ba8f25a612ad60c7da47b0b184352c33cbdbb` took
2,590.67 seconds before stopping at the first native CAD failure. Outline
fitting consumed 2,566.86 seconds; 244 curvature-refit calls consumed
2,240.28 seconds (86.5% of the run). This does not measure a completed model
export or predict the cost of a successful CAD build.

Isolated, cProfile-instrumented refits used the same source and continuous
acceptance checks before and after analytic constraint derivatives and a
single BLAS thread:

| Source | Before, seconds | After, seconds | Acceptance |
| --- | ---: | ---: | --- |
| Small closed counter, 7 points | 0.607 | 0.177 | Accepted in both |
| Large closed counter, 473 points | 12.202 | 9.854 | No accepted candidate |
| Open chain, 5 points | 12.076 | 2.650 | No accepted candidate |

The last two searches previously exhausted their time budgets. Afterward they
finished their bounded searches, but still retained their original geometry.
Faster search does not imply successful refinement.

An independent scan of all 3,837 nonzero-offset spans took 8.72 seconds before
the exact curvature shortcuts and 3.64 seconds afterward. Every result matched.
This saves about five seconds on that scan; it does not explain the much larger
optimizer cost.

A separate cProfile run of initial tracing, without curvature searches, took
513.13 seconds and recorded 683.6 million calls. Exact source certificates
accounted for 218.61 cumulative seconds, including rational subdivision;
native spline solves accounted for 38.72 cumulative seconds. These cumulative
times overlap. Profiler overhead is substantial for Python's many small rational
operations, so this timing must not be compared directly with an unprofiled run.

For 29 captured spans spanning 499 source intervals, reusing consecutive exact
subdivisions reduced ten repeated source-certificate scans from 1.89 to 1.30
seconds (1.46 times faster). Every rational interval supplied to the proof is
unchanged; this removes repeated construction rather than relaxing precision.

The final unprofiled initial-tracing pass took 212.37 seconds. Its generated SVG
and captured curve states matched the earlier profiled pass byte for byte.
This excludes curvature searches and native CAD.

## Full Curvature Pass

A subsequent run timed the entire production curvature stage using per-search
and callback timers. It finished initial tracing and curvature refinement in
1,145.77 seconds (19.1 minutes), without running native CAD or exporting STEP.
Production source hashes remained unchanged during the run.

| Work | Seconds |
| --- | ---: |
| Initial tracing and stage preparation | 183.37 |
| Curvature stage, including shared verification | 962.40 |
| All 244 refits, nested within that stage | 940.15 |
| Shared palette verification, also nested within that stage | 12.91 |

Refits previously consumed 2,240.28 seconds: the observed reduction is
2.38 times. The final 46 accepted replacements all passed shared topology
verification; 198 boundaries retained their earlier source-validated curves.
Fourteen searches exhausted their time budgets, versus 144 previously.
The continuous source-error and curvature limits were unchanged.
The accepted count matched the baseline, but one accepted boundary was
exchanged for another; bounded search outcomes are not geometry-equivalence
claims.

The 46 successful searches took only 4.85 seconds combined. Unsuccessful
searches dominated the remaining time. Of 936.68 seconds inside the optimizer,
914.12 seconds were outside the Python objective and constraint callbacks;
the separate continuous acceptance callbacks took 1.97 seconds. A focused
cProfile of one unsuccessful search attributed 97% of its time to compiled
SLSQP. The largest 32-span searches consumed about 80% of optimizer time.

### Process Parallelism Experiment

Four spawned worker processes completed the first 64 independent refits in
66.54 seconds, including startup, serialization, and shutdown. Their serial
refit timings totaled 200.07 seconds: an observed threefold speedup. All 25
accepted and 39 rejected outcomes, and accepted span counts, matched. Two
rejected searches changed from completed searches to time-budget stops.
Worker BLAS remained single-threaded.

This experiment did not run the final shared topology check or establish
bytewise equality with serial curve controls. It has not been integrated into
production. Shared palette verification must remain the authority after
combining independent results.

These are shared-host timings. Instrumentation can change searches near their
wall-time limits. Callback timing overhead measured about 0.54 microseconds
per call; repeated report writes contribute to stage totals. The bounded
parallel trial overlapped part of the serial run. Timings at different nesting
levels must not be added together.

### Constraint Placement

One four-span candidate passed sampled constraints but had a real continuous
curvature-times-offset peak of 0.90256, above the 0.9 limit. Raw spline and
canonical curve peaks agreed closely, so rounding was not the principal cause.
A temporary same-basis trial with additional extrema constraints passed
continuous curvature, but its source error reached 0.200004092 mm against the
strict 0.2 mm limit and was correctly rejected.

This supports investigating adaptive constraints before adding more spans.
These measurements preceded the geometric-distance changes below and motivated
the later adaptive constraint pass.

## Geometric Distance and Early Acceptance

The source-error authority now checks continuous nearest-outline distance in
both directions. The previous parameter-matched proof remains a sufficient
fast path. Exact closed-loop orientation checks preserve the material side used
by signed-curvature limits. Topology remains a separate final gate.

Curvature search uses indexed nearest-polyline projections for its objective
and sampled constraints, with analytic derivatives and 90% source-distance
headroom. It removes the parameter-matched join caps from this later search;
the initial quadratic fitter retains its conservative join guidance. The
curvature optimizer uses `ftol=1e-6`. It checks up to three promising intermediate
iterates per solve, spaced eight iterations apart, and can stop after continuous
certification. A final candidate can also be certified without a numerical
success flag. No physical tolerance or topology requirement is relaxed.

An isolated six-case solver pilot, still using the prior source checker,
accepted the same three cases at `ftol=1e-11`, `1e-8`, and `1e-6`. Iterations
decreased from 493 to 400, with no clear aggregate wall-time saving. Early
certification was more useful in that pilot: two cases stopped with four spans
instead of eight or 32. These were local checks; shared topology remained
separate.

The first combined seven-case geometric pilot took 21.62 seconds of refit time
versus 18.62 seconds in the earlier serial records. Three local replacements
passed; shared topology accepted two and rejected one. A convex,
parameter-matched objective with the new geometric constraints took 27.52
seconds and produced fewer globally accepted replacements, so it was not
adopted. The observed timing does not establish an overall speedup.

A fresh initial Lynx trace took 262.09 seconds versus the earlier 212.37-second
measurement. Refinements decreased from 624 to 594 and native curve edges from
4,013 to 3,996. The geometric metric provides a more faithful distance test,
but its additional proof work currently outweighs that reduction. This is not
the tenfold improvement being targeted, and no new complete CAD/export timing
is implied by these measurements.

A follow-up timed pass took 238.14 seconds on the shared host. It recorded
6,102 whole-chain validation calls, 44,608 parameter-matched certificates
(65.44 cumulative seconds), and 489 geometric fallbacks (53.34 seconds).
Orientation checks consumed 5.73 seconds. These measurements motivated a local
certificate cache: unchanged geometry and interval lists reuse their result
within the initial refinement loop. New solves replace those lists, while
global topology remains uncached. No coordinate hashing or cross-run cache is
used.

The cached pass took 208.52 seconds, versus 238.14 seconds in the preceding
timed pass. Whole-chain validation calls fell from 6,102 to 2,262 and
parameter-matched checks from 44,608 to 22,670. The complete comparison SVG and
captured curve controls matched byte for byte. This recovers the added tracing
cost on this input; the earlier metric's 212.37-second measurement is close
enough that no robust cross-metric speedup is claimed. The seven-case refit
pilot likewise does not demonstrate a speedup yet.

## Adaptive Constraint Placement

Rejected candidates can now supply additional curve probes and original-source
query locations, then restart from the same control points on the same basis.
Numerical cubic curvature extrema and distance peaks guide this exchange;
the unchanged continuous proofs remain the acceptance authority. Curve and
source parameters are distinct domains, and reverse-source hints also refine
their nearest curve-proxy segment. Three retries per seed and 16 new parameters
per domain per retry share the existing 12-second boundary budget.

A controlled comparison used the first 32 boundaries requiring refinement in
the current Lynx trace. Both modes used the same checkpoint and production
acceptance callbacks; disabling feedback provided the baseline. Production
source hashes stayed unchanged during measurement.

| Work or outcome | Feedback disabled | Adaptive feedback |
| --- | ---: | ---: |
| Refit searches | 144.41 s | 92.77 s |
| Stage including scan and shared verification | 165.08 s | 120.16 s |
| Locally accepted replacements | 18 | 24 |
| Accepted after shared topology verification | 18 | 23 |

Refit time improved by 1.56 times on this subset. Six boundaries gained valid
replacements; one boundary accepted in the baseline instead had its adaptive
candidate rejected by shared topology and retained its original curve. This
does not establish a complete-panel timing or a tenfold speedup.

An earlier isolated pilot illustrates the mechanism: one five-point boundary
previously spent 3.42 seconds without finding a replacement. Adding five
curvature-peak probes and restarting the same four-span basis accepted it in
0.096 seconds. Its shared palette checks also passed. The 473-point boundary
still exhausted its search budget; adaptive sampling does not fix inadequate
whole-loop representations or difficult global optimization.

## First Optimization Pass

- Supply analytic derivatives for the existing SLSQP constraints. The sampled
  constraints, objective, limits, and final continuous acceptance checks remain
  the same. Diagnostics include function, derivative, and iteration counts.
- Limit BLAS to one thread during the small spline solves. The measured host
  loaded two independent 16-thread OpenBLAS pools; thread overhead dominated
  some bounded solves. Scoped limits restore the prior configuration on return
  or exception. A lock prevents overlapping managed scopes from restoring
  process-wide settings out of order. See
  [threadpoolctl's documentation](https://github.com/joblib/threadpoolctl).
- Reject exact endpoint curvature violations early, and accept simple safe
  cases using exact Bernstein coefficient bounds. Unresolved cases retain the
  original rational subdivision proof.
- Retain the unconsumed cubic suffix while checking consecutive source
  intervals, instead of restricting the original cubic twice for each interval.
  This needs no cache and preserves the exact rational proof inputs.

## Local Curvature Repairs

Curvature failures can be local: one failing 21-span evaluation loop has only
two offending spans. Production refinement now tries padded, merged windows
around those failures, preserving approved endpoint positions and inside
tangent rays. Failed windows can grow; accepted repairs trigger a fresh scan
because splicing changes span indices. Outside controls and parameter intervals
remain unchanged. Wrapped windows are allowed only at freely periodic seams,
and windows covering at least half the chain defer to whole-chain fitting.
Local searches and whole-chain fallback share one 12-second budget. A partial
repair cannot escape as an accepted boundary.

An isolated production run repaired both bends in that 473-source-point loop
in 4.99 seconds; whole-loop fitting had exhausted its budget without an accepted
candidate. One window grew from three to five spans and crossed the periodic
seam. The final boundary contained 34 spans, retaining 13 original spans and
their intervals exactly. Original source points and parameters were unchanged.
Continuous source error and curvature checks passed, followed by whole-palette
verification in a separate 18.17 seconds. This is a complete 2D boundary repair,
not a completed CAD model or STEP export.

A separate native probe rebuilt that exact repaired hole, with its original
orientation, inside a simple exterior frame. The unchanged curved-roof builder
produced one valid solid for a 1 mm rise at 10 degrees in 4.73 seconds, using its
surface construction with no repaired arrangement cells. This exercises the
formerly failing boundary in isolation; the full artwork region and panel were
not built by that probe.

A controlled comparison then used the first 32 boundaries needing refinement
in the same current trace, including that large loop. Both modes used identical
checkpoints, adaptive constraints, continuous checks, and shared topology
verification. Production source hashes stayed unchanged throughout the run.

| Work or outcome | Whole-chain search | Local repairs with fallback |
| --- | ---: | ---: |
| Refit searches | 91.42 s | 43.73 s |
| Stage including scan and shared verification | 116.02 s | 70.09 s |
| Locally accepted replacements | 24 | 31 |
| Accepted after shared topology verification | 23 | 30 |

Refit time improved by 2.09 times on this subset; the stage improved by 1.66
times. Seven boundaries gained valid replacements, with no previously accepted
boundary lost. The same independent candidate was rejected by shared topology
in both modes. An earlier local-first trial spent almost its entire budget on
a tiny five-span chain whose three-span window offered little locality; routing
windows covering at least half the chain to whole-chain fitting removed that
regression. These remain shared-host subset timings, not a tenfold improvement
or a complete-panel CAD/export measurement.

## Full Check Before Small-First Repairs

A fresh PNG-to-STEP-only Lynx run exercised the current production working tree
with the same 250 mm, 2/1/1 mm, 10-degree evaluation profile and the tracing
settings above. No fitted curves or native CAD were reused. Source and production
file hashes remained unchanged, and diagnostic capture reported no errors.

| Work or outcome | Full run |
| --- | ---: |
| Initial tracing and related proof work, approximately | 254.9 s |
| Curvature stage, including final palette verification | 1,197.97 s |
| Final palette verification, nested within that stage | 15.18 s |
| Complete outline fitting, including curvature | 1,452.85 s |
| Total until the native construction failure | 1,471.29 s |
| Curvature boundaries attempted | 244 |
| Repairs accepted after shared topology verification | 92 |
| Boundaries retaining their prior certified outlines | 152 |

Four proposed repairs were rolled back by shared topology verification. The
2D result passed its unchanged source and palette checks, but construction of
the first orange tapered region failed native draft-cell repair verification.
No complete model, material partition, or STEP file was produced. The fit and
unvalidated native failure geometry were retained temporarily for inspection.

This full check does not establish an overall speedup. The earlier full
curvature pass took 962.40 seconds and accepted 46 repairs; the current pass
took 1,197.97 seconds and accepted 92. Candidate geometry and distance guidance
changed between those runs, so this is not a controlled local-only ablation.
The 32-boundary gain does not describe the broader workload. Unsuccessful
two-band taper searches remain the principal search-cost target, while the
captured native failure is the next correctness investigation.

The 244 boundary candidates consumed 1,181.12 seconds: 96 successful searches
took 38.55 seconds, while 148 unsuccessful searches took 1,142.58 seconds
(96.7%). Unsuccessful two-band searches accounted for 1,118.5 seconds. Final
verification retained 55 of 61 one-band boundaries and 37 of 183 two-band
boundaries. Most of the expensive failures were open shared boundaries; source
point count and tiny-pocket cleanup alone do not explain the workload. An
unsuccessful bounded search is not proof of geometric infeasibility.

Read-only inspection associated the native invalid walls with an unrepaired
orange exterior boundary, distinct from the repaired loop that passed the
isolated native probe. The captured native top matches the newly fitted curves.
Native wire checks report open boundaries on the affected wall faces; the
combined repair exception alone does not prove physical geometry or precision
was changed. Failure review can resume from the saved fit and BReps without
repeating the complete outline pass.

A subsequent read-only probe confirmed an actual fold in one affected native
wall. Its top curve has a local radius of about 0.09236 mm, while the complete
taper run is 0.17633 mm. The native surface becomes singular at Z=2.47619 mm
between its Z=3 mm top and Z=2 mm base, matching the normal-offset prediction.
Retained face samples below that height reverse orientation. At a nearby trim
junction, edge endpoints coincide in 3D but end on different UV branches of
the folded surface; the wire remains open in the surface parameter domain.
The fold exists before and after the repair attempt. This identifies a real
surface/trim defect rather than merely diagnostic tessellation. It does not
establish that a valid envelope cannot be trimmed from the raw folded support.

## Small-First Repair Scheduling

The failed exterior boundary exposed a scheduling defect. Its first merged
window failed, preventing later independent bends from being tried. Adding
just one adjacent span could also pull a long straight stretch into a small
repair: trial windows covered roughly 49 to 180 mm while the relevant detail
was concentrated near an endpoint. Uniform knots then allocated too little
freedom to that detail.

Local repair now starts with each contiguous failing run, without neighbors,
before growing through one, two, or four neighboring spans. Every independent
group gets a chance at each size. Failed source intervals are deferred within
that pass so index shifts do not repeatedly restart the same failed region.
The shared 12-second budget reserves time for later growth. A partially repaired
boundary still cannot pass as curvature-certified.

The production implementation repaired the complete failed exterior in 3.31
seconds using five accepted windows and 46 final spans. All signed continuous
curvature and original-source checks passed without increasing tolerances.
Complete palette verification retained every other canonical chain exactly,
including all 92 earlier accepted repairs; the targeted patched result has
93 accepted repairs. The previously repaired evaluation loop also passed a
separate regression probe with the new schedule.

Using that globally checked full fit, the exact whole orange region that
previously failed now builds as one valid native solid. Unchanged top-coverage
and section checks passed; no native cell repair was needed. The shared-host
native validation took 30.76 seconds, including 21.57 seconds of section checks.
This was a saved-fit regional CAD retry, not a new full-panel run or STEP export.
The subsequent full workload measurement is recorded below.

Final fitting diagnostics now explicitly distinguish a certified curvature
margin from unresolved boundaries, including their original piece intervals
after rollback. Source-distance/topology certification alone is not described
as curvature certification. Legitimate early counter closure remains permitted.

### Complete Search and Native Retry

The next Lynx run reused the exact initial trace after fresh image separation
and vectorization matched its source WKB and metadata. All 244 curvature
searches and the final palette verification ran again with the small-first
schedule; native CAD was rebuilt from their selected result.

| Work or outcome | Full run |
| --- | ---: |
| Curvature stage, including final palette verification | 1,569.99 s |
| Final palette verification, nested within that stage | 15.56 s |
| Total until the native construction failure, excluding cached initial tracing | 1,614.06 s |
| Locally accepted repairs | 102 |
| Repairs accepted after shared topology verification | 98 |
| Boundaries retaining their earlier outlines | 146 |

All 92 previously accepted repairs remained accepted, with six additional
repairs, including the formerly failing exterior. However, the complete search
was slower than the preceding 1,197.97-second curvature pass. Unsuccessful
searches consumed 1,275.05 seconds; small local guides with three or four source
points accounted for 456.16 seconds of unsuccessful work. Seventeen searches
on similarly small guides succeeded in 0.54 seconds combined, so guide size
alone does not justify skipping them. The subset speedup did not generalize
to a complete-panel speedup.

The first two orange regional roofs passed. The third failed at another
counter whose curvature-qualified proposal had been rolled back by the
unchanged palette topology check. Its selected original curves still have
actual signed-curvature violations. No complete model or STEP was produced.
The completed fit and native failure geometry are retained for regional retries
without repeating the curvature search. Restricting the arrangement to the
slab with a splitter, and separately removing the lower wall extension, both
still failed native validation on this counter; neither prototype was adopted.

The rejected counter proposal actually crossed itself about 1.8 micrometres
from its pinned graph junction. Its two endpoint rays had reversed their
angular order. Whole-chain refits of closed, junction-anchored loops now reuse
the same pinned-position, inherited-tangent, and within-chain separation checks
as local repairs. Freely periodic loops and ordinary open-chain searches retain
their existing routing. An unavailable inherited tangent reports an unresolved
repair rather than aborting conversion.

Regenerating this boundary through production took about 0.10 seconds. Its
source and continuous curvature checks passed, and complete palette verification
preserved the other 588 canonical chains and all 98 prior repairs. The previously
failing third orange region then built with the unchanged native construction,
top-coverage, and section checks in 15.95 seconds, requiring no cell repair.
A synthetic regression independently exercises two touching counters and checks
their exact shared point, endpoint rays, source error, and curvature margin.
These regional results do not establish a completed panel or STEP export.

A native-only full-panel replay then passed the first nine orange regions and
stopped in the tenth after 73.50 seconds, excluding cached tracing and curvature.
Its solid filled an intended 2.94743 square-millimetre opening at the top. The
unchanged top-coverage check rejected it. All seven counters in that region pass
the signed curvature margin; this is a distinct native arrangement failure.
The raw arrangement lacks a separate cell for the filled opening, so changing
cell classification cannot restore it. The same opening fails in isolation in
a simple rectangular frame. Splitter, pairwise preimprint, and adding the exact
source top face did not produce a verified regional solid; none was adopted.

Pairwise preimprint did create the missing cavity, but ordinary union of its
two selected material cells returned invalid UV wiring. A guarded native
`GlueFull` union instead reuses their exact shared partition faces, after proving
opposite face ownership, connectivity, zero pairwise overlap, and unchanged
physical input data. This is restricted to the regional retry; generic Boolean
union is unchanged. The retry is now available to curved profiles as well as
straight profiles, within its existing intersection budgets. Wrapped memory
failures stop both forms of retry.

The production path rebuilt that whole region in 13.33 seconds, including its
initial rejected arrangement, pairwise preimprint, guarded gluing, and unchanged
top and nine-section validation. It produced one valid solid with all seven
openings retained at the top. This is another saved-fit regional measurement;
it excludes tracing and does not establish a complete panel or STEP export.

The next full native replay passed twelve orange regions and stopped in the
thirteenth after 103.21 seconds. Its fitted curves pass the correct outward
curvature bounds, but the generated native walls actually moved inward. The
wrong-side support folds inside the requested height range; this is not a
source-fit curvature violation.

The native orientation heuristic was reproduced with OCCT's own law, bounding,
and angle APIs. Its eight sampled points have a barycentre outside the concave
region, and their angular sum is approximately zero. Floating-point rounding
then selects the wrong sense. See the
[OCCT draft orientation implementation](https://github.com/Open-Cascade-SAS/OCCT/blob/V7_9_0/src/BRepFill/BRepFill_Draft.cxx#L139-L185).

Production now measures the sense of every generated edge wall and corrects a
uniform reversal with one fresh opposite-flag draft. Mixed or ambiguous senses
remain errors. It does not reproduce the heuristic or change source curves.
The same whole region then passed native, top, and nine-section checks in 1.18
seconds, requiring no cell repair. Default and reversed flags were independently
checked on eleven representative rings; all 164 edge probes agreed within each
ring. These are saved-fit regional validations, not a completed STEP export.

## Remaining-Region Inventory and Targeted Repairs

An independent sweep of the seven remaining orange regions and all 187 white
regions completed in 338.27 seconds with production and fitted-input hashes
unchanged. It passed 182 regional roofs and rejected twelve white roofs. Combined
with the fourteen orange roofs already checked in the same-code full replay,
196 of 209 regional roofs passed, with thirteen failures including orange14.
This does not verify the final rim clipping, whole-model union, material
partition, or STEP export. The verified regional BReps and rejected geometry
were retained temporarily for diagnosis.

Four unresolved counter boundaries account for orange14's invalid walls. Two
of its previously rolled-back repairs now pass with inherited endpoint rays.
A third exposes two independent search limitations: uniform knot spacing misses
a short bend beside long stretches, and exact tangent-handle reconstruction
can quantize a requested 0.447 mm handle to 0.918 mm. Finer dyadic steps are
valid when the resulting IEEE coordinates remain exactly representable; they
preserve the tangent direction and all original proof limits.

Local search now tries midpoint-refined approved intervals before the existing
uniform bases, reserving at least half the window's time for uniform fallback.
The difficult 16-span boundary repairs through production in 4.804 seconds,
retaining ten original spans and producing 22 total. Its two successful guided
windows consume 0.054 and 0.044 seconds; failed zero-neighbor attempts account
for most of the remaining time. Direct temporary calls to those successful
windows took approximately 0.11 seconds combined. These nested measurements
must not be added, and no complete-panel speedup is established.

The last tiny counter has a valid circular alternative within the same 0.2 mm
source limit. Its fixed endpoint-angle preference had excluded that option.
The new fallback retains the shared point exactly, passes complete palette
verification, and changes no tolerance or topology requirement. Its production
search, including the preferred attempts, took 5.217 seconds.

Another white region passes every relevant curvature check but encounters a
native array error before wall construction. Drafting its original edges
independently, retaining the existing corner treatment, produces a valid
regional solid with the original top and section checks in 0.493 seconds.
This retry is limited to the identified whole-wire error; unresolved individual
curve failures remain errors.

## Hard Pre-CAD Curvature Gate

The final selected outlines are now independently checked before any native
geometry adapter runs. Shared-span ownership is reconstructed from their exact
controls and winding; the check uses the actual exposed material-height
differences and the same continuous signed curvature bound with 0.9 headroom.
Missing, stale, or optimistic fitting diagnostics cannot authorize geometry.
Unresolved outlines remain available for inspection but raise
`CurvatureLimitError` instead of entering CAD construction.

Rechecking the saved 103-repair Lynx fit took 3.222 seconds and rejected 311
selected spans, including the captured white-region bend. This validates the
gate, not a repair or a completed model. The existing unresolved fitting work
must still be addressed within the original source and topology bounds.

## Retaining Safe Sharp Corners

Softening and simplification now use the same signed boundary curvature limit
as the final fitter. An unsafe rounded bridge restores its detected original
corner through the existing dependency rollback; simplification cannot erase
that corner or introduce another invalid-curvature merge. This adds no new
optimizer or relaxed geometry tolerance.

In the captured Lynx white-region failure, replacing three softened/reduced
pieces with their original sharp incident geometry passes the complete chain's
source, curvature, and separation checks in 0.233 seconds without optimization.
Forty other pieces on that chain are unchanged. Complete palette verification
accepts the edit with all 588 other chains and 238 protected vertices unchanged.
The resulting regional native taper passes the original top and nine section
checks in 4.706 seconds, producing one valid solid with 55 faces. This resolves
that specific regional failure; it is not a complete-panel or STEP result.

## Explicit Corners During Curvature Repair

Curvature repair now tries a bounded family of corner-based alternatives before
nonlinear search. Existing corners remain available, and additional source turns
are considered only inside failing source intervals. Candidates include straight
corner-to-corner spans, fitted cubics, and mixed curves and lines. Constrained
local and whole-chain searches can also use explicit C0 spline joins. The
ordinary smooth fallback remains available even when a window contains a corner.
Every selected alternative still requires the original continuous source,
signed-curvature, separation and complete palette topology checks.

A Lynx replay used the same freshly traced prefix, verified against the original
image, source vectorization and unchanged upstream definitions. Only the current
curvature stage was rerun. The replay took 247.27 seconds, including input checks
and shared verification, but excluding initial tracing, native construction and
export. This is not a complete conversion benchmark.

| Quality check | Before | After |
| --- | ---: | ---: |
| Selected spans failing continuous radius certification | 61 | 33 |
| Canonical boundaries containing those failures | 37 | 16 |
| Regions blocked by radius certification | 9 | 6 |
| Regions passing both radius and native roof checks | 200 | 203 |

All 35 previously accepted boundary repairs remain radius-certified: ten retain
exact coefficients and 25 have certified replacements. Native verification
reuses 195 exact matching accepted roofs from the earlier audit and eight
additional roofs checked freshly after corner repair. Reuse requires identical
region controls, line flags, error bounds, profile, global protected points and
native dependencies, with immutable reports and verified BReps. Six regions
remain blocked and are not constructed. This is regional validation, not a
complete-panel solid, Boolean assembly or STEP export.

A separate eligibility screen found exact source-distance violations for every
whole-chain chord alternative on the remaining 16 boundaries. In 14 cases the
largest vertex-distance witness lies outside the failing intervals: replacing
the whole boundary flattens otherwise useful curved stretches. This motivates
local corner alternatives that preserve those outside curves. It does not prove
that no compliant smooth or cornered fit exists; cubic alternatives and bounded
search failures require separate analysis.

## Local Point-Contact Interpretation

Raster diagonal contacts can turn the end of a larger feature into a separate
tiny square with a protected junction. A local interpretation pass now considers
sub-extrusion-width fragments attached to one larger same-color parent when an
incident fitted curve still fails curvature certification. It splices a pointed
attachment and notch into that parent while retaining all outside coefficients.
Original component boundaries remain separate paths in the exact bidirectional
distance proof; joining them does not grant an additional approximation budget.
Complete palette embedding and the absence of new radius failures are checked
before accepting each edit.

For the isolated evaluation region E, the general pass accepts its second
candidate in 0.444 seconds, leaving the other already-valid attachment alone.
The selected native roof passes unchanged top and section checks in 4.17 seconds.
This is a regional result; it does not establish a complete-panel export.

A full-panel postfit audit against the previously validated 33-failure Lynx
outline accepts three local attachments: one in E and two in F. Remaining failing
spans decrease from 33 to 30, with no new failing curves. Initial tracing and
curvature refinement are cached in this audit. Three other contacts reach their
soft search budgets and retain their original geometry; this is not evidence
that no valid attachment exists. One continuous proof can overrun a soft budget.

Exact geometric-distance certification now indexes finite, outward-rounded
segment boxes. Query boxes retain the previous outward rounding, candidate IDs
retain their order, and the previous bounding-box predicate and exact rational
acceptance proofs still run. Nonfinite boxes use the original linear search.
Tests compare indexed and linear candidate sets and continuous certificate
outcomes. This avoids repeated scans of distant segments without changing the
distance limit; no isolated timing improvement is claimed here.

## Proof Precision and Compact Smooth Repairs

A smooth replacement for the isolated C evaluation bend satisfied the original
0.2 mm source limit but was rejected by the initial reverse proxy certificate.
The proxy reserved 0.0125 mm of that budget, leaving 0.1875 mm around its chords;
an unchanged source feature about 0.196 mm from the true curve could not pass
that conservative check. Refining the proxy to a 0.003125 mm error allowance
certified the same candidate without changing the physical tolerance.

The distance checker now refines only after a coarse reverse certificate fails,
retains already-certified source intervals, and shares one node budget across
all attempts. An enlarged proxy tube gives an exact early rejection when further
precision cannot help. Geometry and the requested distance bound stay fixed.

The compact repair family uses two C1-basis cubics, between the existing C2
smooth basis and explicit C0 corners. A second, tangent-guided chord seed lets
the ordinary optimizer find the C replacement automatically in 0.44 seconds.
Three old pieces become two, with all 15 outside chain pieces unchanged. The
result passes complete palette certification in 10.95 seconds and native roof
checks in 18.03 seconds. These are isolated regional checks, not complete model
or export timings. Original accuracy and taper-radius requirements are retained.

Replaying the current curvature and attachment stages on the previously
certified initial trace reduces the panel's remaining failing spans from 30 to
23, across four regions, without new failing curves. The run takes 318.65 seconds
on the shared host, including 231.35 seconds in curvature repair. Initial
tracing is cached from the earlier fitter, so this is not an uncached conversion
benchmark. C again passes native roof verification in the combined result.

## Remaining Algorithmic Targets

Full-depth regularity is also unnecessary for a counter that legitimately
closes earlier in the taper. A conservative proxy experiment identified four
such cases among 21 unresolved closed counters. That is a useful distinction,
but does not explain most of the runtime. No feature removal or closure-based
search exemption has been introduced by these performance changes.

The optimizer already spends substantial time in compiled SciPy code, so a
Rust rewrite of the search is not the first step. Exact rational subdivision
is a more plausible future native-code target if it remains expensive after
avoiding repeated work. Any replacement must preserve the existing exact
certificate semantics; floating-point sampling is not an equivalent check.

## Native CAD Arrangement and Validation

The current certified Lynx trace has 187 white regions and 22 orange regions,
with no remaining selected-span radius failures. This result replays repair
stages on a saved initial trace; it is not a fresh conversion benchmark.

Profiling its largest white region separates wall construction (about four
seconds) from the much slower native surface arrangement. Oriented bounding
boxes reduce the arrangement work; a bounded four-thread OCCT trial reduces one
arrangement from about 55 to 28 seconds. Thread configuration remains an
experimental runner setting rather than a production default.

Cell selection no longer integrates accurate mass merely to prove that a cell
has positive volume. Native validity, boundedness and a classified interior
point provide that proof. A cheap center-of-mass calculation is only a candidate
point generator and cannot authorize a cell. Final roof and STEP mass checks
retain the original accurate integration. Height and exact-top checks also run
before mass integration, rejecting incomplete arrangements sooner. Removing
open INTERNAL annotation wires retains the exact oriented physical-boundary,
validity and area checks; its previous nonbinding mass comparison is omitted.

The large region's validation outline needs 108,026 exterior proxy points at
the existing 0.00001 mm chord bound. The validation resource ceiling now scales
with input span count, with a finite upper limit, instead of rejecting this
outline at 100,000 points. These reference chords do not construct CAD surfaces
or change fitted geometry. The resulting native roof passes all nine section
checks; the checks themselves take about 224 seconds in the four-thread trial.

One smaller native draft triggers an OCCT array failure for a single B-spline
generator. Retrying that specific failure with four parameter trims of the same
native curve succeeds without changing poles, outline accuracy or taper radius.
The regional roof passes section checks and STEP roundtrip validation. These
regional results do not yet establish a complete-panel STEP export.

A complete native regional replay then passes all 209 roofs in 934 seconds,
including validation and checkpoint overhead. The largest white region accounts
for 623 seconds, including the rejected initial arrangement and the successful
preimprint fallback. Other tracing jobs run concurrently on the shared host, so
this is a progress measurement rather than an isolated timing comparison. The
initial trace remains cached; global assembly and export are separate stages.

## Boundary Clipping and Top Verification

The rim tool now extends vertically beyond both ends of the raised relief. Its
purpose is to clip XY expansion, so this changes no intended intersection. With
coincident horizontal caps, OCCT deleted three tiny, otherwise valid top faces
after a preceding union increased nearby vertex tolerances. Extending the tool
preserves those faces in both serial and parallel trials.

A Boolean section at the highest boundary can independently omit tiny cap faces
when the solid carries those large vertex tolerances. Global top verification
therefore reads the actual physical planar boundary trims. Validation-only
copies retain their exact support planes, pcurves, parameter ranges, sharing and
orientations, using ordinary native construction precision for fresh vertices.
Real trim gaps reject; no healing or source-CAD modification is performed. The
original area, validity and strict missing/extra footprint checks remain.

The corrected combined Lynx STEP passes native and roundtrip checks, including
zero missing/extra top area after STEP re-import. It is about 50 MB, from the
certified cached initial trace described above. A separate material-split audit
still rejects one native edge extending 0.000135 mm beyond its assigned band;
colored multipart export for that unchanged trace is rejected.

A review variant interprets the responsible 0.177 mm contact counter as part of
one continuous boundary. It passes the original source, palette and radius
checks, and the complete strict three-band native audit. Combined and colored
multipart STEP exports pass, including zero missing/extra top area after both
files are re-imported. The multipart file is about 122 MB, with three named color
components at 0–2, 2–3 and 3–4 mm. This is a certified review edit from the cached
trace; automatic cold conversion is a separate validation target.

Contact repair also handles parent interior rings, preserving every outside
polynomial. Reference holes retain source identity when a geometry union
normalizes their ordering; otherwise an unchanged ring-relationship comparison
can reject a valid edit solely because its indices moved. Exact per-ring source
certificates are reused for unchanged coefficients and source paths. Paired
certificates imply the required whole-boundary bound; conservative pairing
failures retain the whole-union fallback within the bounded candidate search.
The search order changes, so identical outcomes under the wall-time limit are
not guaranteed. No accepted result uses a weaker certificate.

An isolated current-code replay repairs two Lion hole attachments, reducing
seven radius failures to five in about 59 seconds. Source preprocessing takes
about 19 seconds and is reported separately from candidate search. Both edits
pass the original 0.2 mm source, complete palette/topology and radius checks;
all other polynomial coefficients remain unchanged. This does not establish
an uncached four-image conversion or a new complete native model.

## Current End-to-End Evaluation

A subsequent cold Lynx run uses the original image and current production
tracing, without injecting reviewed outlines. Its complete selected fit passes
the independent radius gate in about 774 seconds, including checkpoint overhead.
The fit contains 20 orange and 167 white regions. Native construction reuses 175
previously accepted roofs whose exact curves, protected points and native-code
dependencies match, and builds the 12 changed regions afresh. This native replay
is not an uncached CAD performance benchmark.

The complete model passes the connected-solid and strict three-band material
audits. Combined and colored multipart STEP exports also pass their native and
roundtrip checks. Independent comparisons against the exact selected white tops
find zero missing or extra area in both serialized files. The combined file is
about 49 MB and the multipart file about 120 MB; the latter has three named,
colored components at the configured heights. This validates this evaluation
input with the current automatic tracing, rather than a manual review edit.

The separate "Torn once" evaluation remains rejected. It has 55,726 raw contour
vertices and 1,127 white regions. Initial tracing takes about 2,377 seconds,
curvature refinement about 950 seconds, and contact refinement about 4,738
seconds on the shared host. After roughly 2 hours 15 minutes, the final radius
gate still rejects 45 selected boundary spans. The run was allowed to continue
without a time limit; it stops because of the geometric constraint, before CAD
construction or STEP export. Its saved diagnostic trace is not an accepted
printable model. These results do not establish reliable conversion of arbitrary
detailed artwork or a fresh successful four-image batch.

## Contact Preparation Reuse

A later compact-cleaned Torn evaluation took 6,230.73 seconds before the final
radius gate rejected 52 selected spans. Contact refinement accounted for
3,843.34 seconds (64.1 minutes), including 465.75 seconds of per-contact source
preparation. These are fitting-stage measurements, not successful model exports.

Contact refinement now prepares the unchanged source once per call. Source
validation, spatial neighborhoods, small-component groups and individual bridge
eligibility are reused across proposed selections. Conflicts between selected
bridges, cross-color conflicts and combined-parent hole preservation retain
their existing checks. A finalized parent union is reused only for the same
original parent and selected source members within that preparation.

Hole correspondence uses spatial envelope queries before the same exact
interior-intersection predicate. Missing, split or multiply owned holes still
reject; neither the source-distance allowance nor the topology rules change.
Expected polygon and ring relationships are computed lazily once per proposed
reference. Every candidate still independently checks its actual curve/proxy
separation, palette topology, source distance and radius failures.

An isolated replay of initial discovery plus six recorded selection subsets
took 267.44 seconds before preparation reuse and 31.63 seconds afterward.
Selected polygon bytes, source groups, contacts and diagnostics matched in all
cases. These shared-host measurements cover preparation, not the full stage.

The preliminary stage replay identified two additional costs. Whole-graph
assembly repeatedly proves the same curve pairs and rebuilds the same proxies.
A per-refinement cache now retains complete pair results and their subdivision
cuts, flattened curves keyed by the full cuts and tolerance, and exact proxy
hull-separation results. Spatial queries, pair-count limits and candidate
topology checks still run for every assembly; changed controls invalidate their
entries. A temporary whole-assembly comparison took 31.98–35.20 seconds uncached,
37.83 seconds on a cold cache and 5.44 seconds warm. A local handle change took
34.34 seconds uncached and 4.70 seconds cached, with identical complete assembly
outputs and subdivision cuts. These are assembly-only measurements.

The largest parent exterior also exhausted the source-distance proof's shared
100,000-node budget. All 1,611 forward spans passed, but one unresolved source
endpoint triggered progressively finer proxies of the entire outline. Source
certification now builds one complete coarse proxy, then refines nearby curves
only for source segments that need a tighter witness. Every source segment
still requires its own positive certificate, and later segments resume the
complete coarse proxy. The original distance, depth and node limits remain.
The diagnostic replay certifies all 18,758 source segments using 65,848 nodes;
only 140 segments need local refinement. This successful baseline permits the
existing paired-ring proof path instead of repeatedly checking whole-parent
unions. It does not itself establish an accepted panel or STEP export.

Forward source-distance proofs are also reused per original parent ring within
one contact-refinement call. The context owns immutable base paths and the fixed
tolerance. It records only positive proofs of exact controls against those base
paths. Added fragment paths enlarge that source, so unchanged controls retain
their proof; failed proofs and success depending on added paths are not learned.
Every reverse-distance obligation still checks the full current source against
the current candidate. The complete-boundary result cache remains keyed by all
paths and selected coefficients.

In a retained large-boundary replay, the first source-bound check took 18.63
seconds and computed 1,611 forward proofs. A one-span edit took 6.61 seconds and
computed one new forward proof; repeating it took 6.59 seconds with none. All
reverse and proxy checks remained active. These are boundary-only timings, not
complete contact-stage or conversion speedups.

The complete contact-stage replay before forward-proof reuse took 2,391.85
seconds (39.9 minutes), versus the historical 3,843.34 seconds (64.1 minutes).
It made 87 certified attachment repairs rather than 40, and reduced final
radius failures from 52 to 45. Peak RSS was about 503 MiB. This remains a rejected
panel, and bounded search outcomes differ; it is not an equal-output speed
comparison or a complete conversion benchmark. Source checks consumed 1,357
seconds, graph assembly 527 seconds and radius checks 428 seconds. Assembly's
nested timers are included in its total and must not be added again.

With positive forward-proof reuse, the complete replay took 2,078.57 seconds
(34.6 minutes), about 1.85 times faster than the original contact stage. It made
104 certified repairs, retaining all 40 original selections and adding 64;
41 selected spans still fail the final radius gate. Peak RSS was about 537 MiB.
The preceding comparison without forward reuse made 87 repairs in 39.9 minutes;
the additional cache retained those selections and added 17. These are single
shared-host runs with lightweight timers, and parts overlapped test execution
and the comparison run. Different bounded-search outcomes make these useful-work
measurements rather than equal-output benchmarks.

The final stage spent about 847 seconds in source-distance proofs, 634 seconds
in complete graph assembly and 512 seconds in radius checks. It performed 944
paired checks and 72 whole-union checks. Rechecking unchanged parts of complete
boundaries, graph assembly and radius validation remain substantial costs.
This is not the tenfold overall improvement being targeted. Initial tracing,
curvature searches and native CAD were not rerun, and this evaluation panel
remains rejected before CAD. No Rust or additional solver dependency is needed
for the implemented changes.

### Radius and Reverse-Distance Receipts

Contact refinement also reuses exact orientation and signed-curvature results
within one call. Every scan still rebuilds shared ownership and the exposed
material-height differences. Cache hits validate the complete current inputs;
geometry, signed offset, margin and resource-policy changes require the correct
proof. The independent final pre-CAD gate remains uncached by default.

Distance contexts retain positive receipts for complete source segments. Each
receipt names the actual curve controls supporting every accepted subdivision
leaf; all of those owners must survive before reuse. New or changed source
segments and missing owners require new proofs. Failed or incomplete segment
checks never create receipts. Every current source segment remains visited.
The context also retains exact proxy subdivisions keyed by controls and epsilon,
with bounded storage and policy invalidation. Newly added paths are checked
before unchanged base paths to reject invalid attachments sooner. No source,
taper, topology or approximation allowance is increased.

The combined complete contact-stage replay took 1,377.42 seconds (23.0 minutes),
versus 2,078.57 seconds (34.6 minutes) before these receipts. It retained all 104
previous attachment selections and added two, while the independent uncached
radius gate still found 41 failing spans. Peak RSS increased from about 537 to
590 MiB. This is about 34% less stage time, or 2.79 times faster than the original
64.1-minute stage, with different bounded-search outcomes. These are single
shared-host measurements, not equal-output benchmarks.

Radius checks fell from about 512 to 72 seconds and source-distance checks from
847 to 537 seconds. Complete graph assembly took about 681 seconds, including
its nested operations, and is now the largest measured cost. The stage performed
1,064 paired checks and 66 whole-union checks. Initial tracing, curvature
searches and native CAD remain outside this replay; the remaining radius
failures still reject this evaluation panel before model construction.

### Reusing Complete Assembly Artifacts

Positive fixed-source forward proofs now also survive a rejected candidate
temporarily removing their curve. That removal does not invalidate the exact
controls' proof against the immutable base paths. Current-geometry-dependent
proxies and reverse receipts still prune missing owners, and policy changes
and the weighted capacity bound remain active. In a deliberately costly
five-edit diagnostic, this reduced forward computations from 15 to five and
elapsed time from 8.68 to 5.82 seconds; it is not a stage-wide speed estimate.

Assembly retains unchanged curve rings and proxy lines keyed by full ordered
span metadata, traversal direction and actual rounded vertices. Ordered ring
artifacts identify polygons, and exact immutable operands identify cached
validity, silhouette coverage and ordered DE-9IM results. Strong references
keep identity keys valid. Current hole counts, candidate-pair discovery,
subdivision cuts and native/proxy separation checks still run. Relationship
indices are rebuilt from the current component order; a bounding-box query and
the detailed relationship replace a separate intersection predicate on the
cached path. Each map has a bounded entry count; retained geometry sizes vary.

A replay using the actual artwork packet matched complete assembly outputs,
subdivision cuts and unsafe pairs against the preserved implementation,
including a local handle change. Repeated assembly took about 4.03 seconds
before artifact reuse and 3.40 seconds after it. Relationship checks fell from
about 0.69 to 0.02 seconds, but most assembly work remained elsewhere. The first
changed candidate took 3.95 seconds before reuse and 4.46 afterward, so these
single shared-host observations do not establish a universal speedup or a new
complete-stage time. Profiling the warm path identified repeated conservative
proxy-hull bounding-box construction as the next substantial cost.

The follow-up implementation reuses those conservative boxes by the identity of
the immutable flattened hull tuple, retaining the tuple beside its box in a
bounded map. Its exact rational extrema and outward `nextafter` padding are
unchanged. Every assembly still builds the current spatial index and checks
the current candidate pairs with the same limits and separation proofs.

An actual-production replay reduced warm proxy checking from 2.68 to 0.77
seconds. Complete repeated assembly fell from 3.39 to 2.14 seconds; a local
handle change took 3.34 seconds with box reuse disabled and 2.02 enabled.
Complete outputs, cuts and unsafe-pair lists matched, including restoration of
the original geometry. Of 57,634 hulls, the handle change reused 57,628 boxes and
constructed six. First population incurred about 0.9 seconds extra and roughly
22 MiB RSS. These are bounded assembly-only, shared-host measurements; no new
complete contact-stage or conversion time is established by this experiment.

### Process Workers for Independent Curvature Trials

The first curvature-repair pass can now dispatch independent chains to spawned
worker processes. Each job retains the existing corner family, optional texture
candidate, smooth search and remaining-budget calculation. Only the chain,
material-boundary limit, tolerance and failing-span indices are serialized;
callbacks are created inside the child. Queue time does not consume its search
allowance. Results return in source order, and complete panel verification,
selective rollback and subsequent retry families remain in the coordinator.

`relief` and `export` expose `--workers`; the core and file APIs accept the same
positive integer execution setting. One worker is the default and calls the
same trial function directly. Multiple workers use spawn rather than inheriting
initialized numerical-library state. The existing scoped BLAS control remains
active inside each process. Worker exceptions propagate without serial fallback.

An eight-chain, evenly distributed sample from the retained current artwork
checkpoint took 21.12 seconds to generate candidates serially and 14.16 seconds
with four workers, including pool startup and shutdown. All controls and search
statuses matched. Independent source/radius checks and complete palette
verification accepted the same two repairs and rejected the same three of five
proposals. A local repair may retain original unresolved spans; the independent
check required that it introduce no new curvature failure.

Central verification took 40.53 and 40.36 seconds respectively, making the
sample's combined totals 61.64 and 54.51 seconds. One long search limits this
small batch's parallel gain. Individual job payloads were about 0.9–13 KiB.
Peak parent RSS was about 404 MiB; the child high-water report was about 400 MiB
for an individual process, not aggregate worker memory. These are single
shared-host measurements of this initial-pass subset; later fitting families,
initial tracing, the contact stage and CAD were not rerun. This sample neither
establishes a complete-conversion speedup nor accepts the evaluation panel.

## Corner Repair Validation

A later replay addressed the 41 genuine radius failures retained by the Torn
contact-stage checkpoint above. Bounded corner alternatives repaired all 41
without changing the 0.2 mm original-source limit, signed-radius margin, or
palette relationships. Independent checks certified all 13 affected source
rings and the complete palette's curvature. This is a planar tracing result;
the complete contact stage has not been timed again with these repairs. A fresh
replay of this final repair pass took 203.80 seconds, with about 513 MiB peak
RSS. It excludes initial fitting, contact selection, and native construction.

Six affected complete white components then passed stock native roof
construction, top and taper-section checks, and BRep validity after save/reload.
The serial probe took 193.17 seconds, with peak RSS about 789 MiB. Two roofs
used the existing surface-intersection retry. The much larger main white
component, full material assembly, and STEP export are outside this probe, so
it establishes neither a complete model nor a full-conversion timing.

The next full-panel attempt accepted 26 roofs, then stopped on a native planar
face. An earlier attachment candidate left an endpoint about 1e-14 mm from a
nonincident line. Exact checks saw a positive gap, while native vertex precision
treated it as contact. Rejecting line contacts at coordinate-roundoff scale let
the existing candidate pool choose another notch. That alternative passed the
unchanged source, palette, radius, and complete native top-face checks. This
local replay does not establish a fresh full-pipeline fit or completed STEP.

The complete Torn conversion remains rejected in native volume assembly.
The large regional arrangement loses separating walls and can combine intended
material with empty space in one cell. This checkpoint does not resolve that
construction failure or the remaining pixel-sized counters, and does not
establish a completed Torn STEP export.
