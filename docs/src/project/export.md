# Validated Model Export

## STEP Without Meshing

Export exact CAD independently of STL validation:

```sh
uv run --locked panelchemy export /path/to/artwork.png \
  --profile examples/construction-profile.json \
  --format step \
  --output local-inputs/step
```

This writes a fused `relief.step` and a colored `relief-multipart.step`, alongside
the native BReps, previews and provenance. The multipart file has one named
assembly containing black, orange and white material components for present
bands. Disconnected islands remain grouped within their material component.
Component colors use the separation palette, including custom configured colors.

All components retain their original coordinates and millimeter units. With the
default profile, black spans Z=0–2 mm, orange Z=2–3 mm and white Z=3–4 mm; no part
is moved onto the build plate independently. STEP reimport verifies names,
colors, solid counts, native validity, volumes, bounds and component placements.
Source CAD and retained material-partition proofs are checked before export.

STEP readback reports its volume difference and numerical serialization
allowance. The allowance uses the largest of the existing absolute and relative
limits and the source surface area times the kernel's comparison precision.
This scales the mass check for broad, thin panels; it is not a certificate of
whole-surface displacement. Native validity, solid counts, registration, and
material-partition checks remain independent requirements.

Two hole loops can share a vertex in valid OCCT geometry, while STEP requires
separate boundary connectivity. Export first attempts ordinary STEP. If readback
exposes open wires at coincident endpoints, it matches those points to source
hole contacts and partitions only the affected planar faces with exact native
section edges. This preserves their geometric domains and adds internal face
seams. Every correction is followed by complete STEP readback validation.
The preparation report records the partitions and preservation checks; the
caller's CAD remains unchanged.

The reusable `panelchemy.step_export.export_step(model, palette=...)` API returns
both files and validation metadata without constructing a mesh. The manifest
keeps `mesh_validated` false for this mode. CAD colors and assembly grouping are
preserved through OpenCascade reimport; Bambu import and automatic filament
assignment have not been tested. A slicer may expose individual solids as parts.

## STEP and STL

Build relief and validate the serialized model geometry before publishing output:

```sh
uv run --locked panelchemy export /path/to/artwork.png \
  --profile examples/construction-profile.json \
  --output local-inputs/export
```

The default `--format all` output retains the relief BReps, previews,
configuration, source hashes, and manifest. It additionally contains a combined
`relief.stl`, registered `black.stl`/`orange.stl`/`white.stl` for present bands,
and exact combined
`relief.step`. All parts share one origin and scale. STEP
records millimeters; STL has no unit or material metadata, so interpret its
coordinates as millimeters and keep separate bands at their original positions.

## Mesh and Interface Checks

Generated models retain the shared native cells from one horizontal depth split.
Export verifies the source, cell bindings, and physical BRep proofs, then copies
that graph once for meshing. Custom models without a retained partition use their
supplied material solids to construct a checked arrangement. Adjacent material
faces use the same triangulation. Shared internal triangles cancel when building
the combined outer mesh; separate parts retain matching interface caps. Meshing
never changes the source CAD.

Replacing material cells or changing native geometry invalidates a retained
proof. Rebuild the model after changing construction settings. Custom geometry
can use `material_partition=None` and the full material arrangement checks;
export never silently substitutes the old cells for changed inputs.

Validation reads the actual binary STL coordinates back and checks closure,
oriented manifold topology, self-intersections, positive component volumes,
dimensions, band heights, material-volume accounting, and shared interfaces.
Numerical vertex normalization is recorded; failed validation aborts export.
No nozzle-based filtering or minimum-feature cleanup is applied.

Exact-zero numerical seam triangles are replaced by conforming subdivisions
using existing coordinates, preserving positive-area faces and material
interfaces. Native and float32 conformance are reported separately. Coordinate
collapse and unresolved mesh defects still reject export.

After each native or float32 conformance pass, exact coincident facet pairs with
opposite winding within one material cancel as an oriented boundary. Cancellation
leaves retained coordinates and facets unchanged; removed pair counts are
recorded. Same-winding
duplicates and higher multiplicities remain errors. All subsequent topology,
intersection, interface, and serialization checks still apply.

`--mesh-deflection-mm` sets linear tessellation deflection (default 0.02 mm),
`--mesh-angle-deg` sets angular deflection (default about 11.46 degrees), and
`--max-triangles` limits the accepted mesh triangle count (default 2,000,000).
Curved boundary edges use 5% of the configured linear deflection, bounded by
native meshing precision, while face-interior limits and the minimum-edge guard
retain their previous values. This tighter boundary sampling avoids coarse
chords crossing curved trims. Actual meshing parameters are recorded separately
from the requested limits.
The native mesher can allocate memory before that count is available; this is
not a process memory limit. These settings are separate
from `contour_tolerance_mm`, which changes selected artwork outlines before CAD
construction. Actual settings and validation results are recorded in the manifest.

A successful geometry check is not a physical print observation. The current
scope is model geometry; slicer project packaging, printer presets, material-slot
assignment, and slicer-specific metadata are deferred.
