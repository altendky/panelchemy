# Validated Model Export

Build relief and validate the serialized model geometry before publishing output:

```sh
uv run --locked panelchemy export /path/to/artwork.png \
  --profile examples/construction-profile.json \
  --output local-inputs/export
```

The output retains the relief BReps, previews, configuration, source hashes, and
manifest. It additionally contains a combined `relief.stl`, registered
`black.stl`/`orange.stl`/`white.stl` for present bands, and exact combined
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
