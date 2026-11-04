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

The exporter partitions copied CAD bands into a conformal arrangement before
meshing. Adjacent material faces therefore use the same triangulation. Shared
internal triangles cancel when constructing the combined outer mesh; independent
parts retain the matching interface caps. Meshing never changes the source CAD.

Validation reads the actual binary STL coordinates back and checks closure,
oriented manifold topology, self-intersections, positive component volumes,
dimensions, band heights, material-volume accounting, and shared interfaces.
Numerical vertex normalization is recorded; failed validation aborts export.
No nozzle-based filtering or minimum-feature cleanup is applied.

`--mesh-deflection-mm` sets linear tessellation deflection (default 0.02 mm),
`--mesh-angle-deg` sets angular deflection (default about 11.46 degrees), and
`--max-triangles` limits the accepted mesh triangle count (default 2,000,000).
The native mesher can allocate memory before that count is available; this is
not a process memory limit. These settings are separate
from `contour_tolerance_mm`, which changes selected artwork outlines before CAD
construction. Actual settings and validation results are recorded in the manifest.

A successful geometry check is not a physical print observation. The current
scope is model geometry; slicer project packaging, printer presets, material-slot
assignment, and slicer-specific metadata are deferred.
