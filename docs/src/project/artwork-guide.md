# Preparing Artwork

Use this guide when drawing, editing, or asking an image model to create artwork
for Panelchemy. The subject and visual style are up to you.

## Size and Detail

Choose the physical panel size first. For the starting **250 mm long-edge**
design, use these artwork targets:

- **Feature bodies at least 1 mm wide.** This includes strokes, narrow
  connections, and isolated marks. Prefer 1–2 mm or larger for important details.
- **Gaps and enclosed recesses at least 1 mm across.** Fill tiny pinholes and
  remove small slivers instead of reproducing pixel noise.
- **Keep sharp corners and pointed ends.** A substantial stroke may taper to a
  sharp terminal point. This exception is for tips, rather than a long stroke
  that remains below the minimum width. Do not blur or round every outline.
- **Make connections deliberate.** Regions should share a substantial edge or
  have a clear gap. Avoid accidental point-only touches, self-touching outlines,
  and a hole touching its enclosing boundary.

Whiskers and lettering may need thicker bodies while retaining pointed ends.
These are design targets awaiting physical evaluation, not verified print limits.
The current converter does not automatically enforce them or delete small
features. Contour approximation controls outline accuracy; its tolerance is a
separate setting and does not impose a minimum feature size.

Measure at the final physical size. Use the configured nozzle diameter to plan
resolution: assume a printed line width of **1.25 × nozzle diameter**, and aim
for **at least four independent source pixels per line width**, preferably eight.
For a 0.4 mm nozzle, the assumed width is 0.5 mm; a 250 mm panel therefore has
a 2,000-pixel starting target and a preferred 4,000-pixel long edge. Count the
panel itself, excluding padding. These are sampling recommendations, not input
requirements or verified print limits; lower-resolution artwork can still be
traced and inspected.

Count independent samples: doubling an image with nearest-neighbor enlargement
does not add detail. A genuinely higher-resolution rendering or original vector
artwork is more useful than an enlarged bitmap. At 2,000 panel pixels, a 1 mm
feature spans eight pixels; at 4,000 it spans sixteen. More pixels improve
sampling but do not widen physical features. Doubling the physical panel size
doubles every feature width; resizing only the pixel dimensions does not.

The assumed half-line width (0.25 mm with the default nozzle) guides tracing
smoothness and corner significance. It is not a command to draw circular corners
or round every pointed tip. Retain crisp corners on substantial shapes. This
scale does not automatically delete narrow features or guarantee their printing.

## Flat Colors

Use exactly three solid fill colors with the default palette:

| Color | RGB | Hex | Model role |
| --- | --- | --- | --- |
| Black | 0, 0, 0 | `#000000` | Base and visible recessed areas |
| Orange | 245, 166, 35 | `#F5A623` | Middle height |
| White | 255, 255, 255 | `#FFFFFF` | Highest features |

White is raised artwork, rather than empty space. An enclosed black area is a
recess backed by the continuous base, rather than a through-hole.

Use filled regions with clear boundaries. Avoid gradients, shading, shadows,
lighting effects, texture, dithering, halftones, and tiny speckles. Smooth curves
and angular outlines are both welcome. Edge antialiasing is acceptable;
intermediate shades should not be intentional interior colors. Export as PNG
to avoid JPEG compression artifacts. For a custom palette, replace all three
colors consistently and configure the same palette on import.

## Panel Boundary

### Simple Default: Black Border and White Padding

Make one connected panel enclosed by a continuous black outer border. Use a
border at least 2 mm wide as an initial artwork target. Its outside edge defines
the panel shape; it may be curved, angular, or rectangular. Leave plain white
padding around the entire panel so the border never touches the image edge.
Keep all artwork inside the border without breaking it.

The importer treats white connected to the image edge as outside the panel.
White enclosed by the border remains raised artwork. A broken border can let
the background connect to interior white regions and change the inferred
boundary. Inspect the separation preview before constructing the model.

The visible border is part of the artwork. Its width is separate from the
construction profile's protected-rim setting.

### Explicit Boundary Mask

For a boundary that does not fit that convention, supply a separate PNG mask
with exactly the same pixel dimensions and alignment as the artwork:

- Solid white inside the entire panel, including every artwork color and recess.
- Solid black outside the panel.
- One connected, filled panel shape with no internal holes.

Use the silhouette-mask input on import. The mask determines the exterior shape,
not the internal colors or heights. Its boundary should match the intended
outer edge of the artwork.

### Transparency

Transparency alone is **not an explicit boundary input** in the current tool.
Transparent pixels are composited over white before separation. A transparent
exterior can work with the closed-border convention, but its alpha channel does
not override boundary detection. Keep the panel interior opaque; internal
transparency does not create physical holes. A checkerboard painted into the
image is not transparency.

For predictable results today, use opaque white padding or a separate mask.

## Copyable Image-Generation Prompt

Replace the subject and physical-size values as needed. Adjust the pixel/physical
conversion if you change the size.

```text
Create artwork of [SUBJECT / COMPOSITION] for a three-height relief panel.
Choose the visual style freely while meeting these technical requirements:

The finished panel will measure 250 mm on its longest edge. Use exactly three
flat, opaque fill colors: black #000000, orange #F5A623, and white #FFFFFF.
Black represents the base/recesses, orange the middle height, and white the
highest features. Enclosed black regions are recessed details, not cutout holes.

Use clean filled shapes. No gradients, shading, shadows, lighting effects,
texture, dithering, halftones, tiny speckles, or simulated 3D rendering. Smooth
curves, crisp angular corners, and sharp pointed tips are allowed and desired
where appropriate; do not blur or round every contour.

At the finished size, make the bodies of strokes, narrow connections, and
isolated marks at least 1 mm wide; prefer 1–2 mm or larger for important details.
Make gaps and enclosed recesses at least 1 mm across. A substantial feature may
taper to a sharp terminal point, but avoid long hairline strokes. Draw whiskers
and similar fine details as deliberate substantial strokes with pointed ends.
Remove tiny pinholes, slivers, and fussy edge detail. Avoid accidental point-only
contacts, self-touching outlines, and holes touching their enclosing boundary.

Make one connected panel enclosed by an unbroken black outer border at least
2 mm wide. The outside edge of that border defines the intended panel shape.
Leave plain white padding around all sides; do not crop the border or let it
touch the canvas edge. Keep every artwork element inside the border.

Deliver an opaque PNG with no mockup, perspective, scene background, frame
shadow, dimension labels, or color swatches. The planned nozzle diameter is
0.4 mm; assume a 0.5 mm printed line width. Aim for 4,000 genuinely rendered
pixels along the panel itself, excluding padding; 2,000 is a useful starting
target. This gives eight or four independent pixels per printed line width.
Do not meet the resolution target by simply enlarging a coarse bitmap.
Keep intentional corners and terminal points sharp rather than applying
0.25 mm circular rounding to the artwork.
```

Image models may miss exact colors, widths, or boundary requirements. Check the
result and the normalized palette/silhouette previews, and repair the artwork
when needed. A matching prompt does not establish valid CAD or a successful print.
