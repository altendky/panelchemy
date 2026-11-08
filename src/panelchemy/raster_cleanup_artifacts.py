"""Inspectable raster cleanup artifacts, independent of tracing and CAD."""

import base64
import json
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from scipy import ndimage

from panelchemy.raster_cleanup import RasterCleanupResult, cleanup
from panelchemy.separation import (
    ArtifactBundle,
    Mask,
    SeparationConfig,
    png_bytes,
    separate,
)

_DESCRIPTION = (
    "Compact specks and counters, plus eligible diagonal contacts. Connected tips "
    "and long thin strokes are retained. Magenta marks intentionally changed "
    "pixels. Curve fitting will use the cleaned reference."
)


def _review(
    result: RasterCleanupResult,
    images: dict[str, bytes],
    title: str,
    *,
    description: str,
    width_label: str,
    width_mm: float | None,
    before_label: str,
    after_label: str,
) -> bytes:
    height, width = cast("tuple[int, int]", result.changed.shape)
    components, _ = ndimage.label(result.changed, structure=np.ones((3, 3), dtype=bool))
    areas: list[int] = np.bincount(np.asarray(components).ravel()).tolist()
    boxes = ndimage.find_objects(components)
    views: list[dict[str, object]] = []
    for number in sorted(range(1, len(areas)), key=lambda i: (-areas[i], i))[:12]:
        box = cast("tuple[slice, slice]", boxes[number - 1])
        top, bottom = cast("int", box[0].start), cast("int", box[0].stop)
        left, right = cast("int", box[1].start), cast("int", box[1].stop)
        padding = max(12, bottom - top, right - left)
        views.append(
            dict(
                label=f"Change {len(views) + 1} · {areas[number]} pixels",
                bounds=[
                    left - padding,
                    top - padding,
                    right - left + 2 * padding,
                    bottom - top + 2 * padding,
                ],
            )
        )
    payload = dict(
        width=width,
        height=height,
        views=views,
        diagnostics=result.diagnostics,
        width_label=width_label,
        width_mm=width_mm
        if width_mm is not None
        else result.diagnostics.get(
            "minimum_width_mm", result.diagnostics.get("target_width_mm")
        ),
        images={
            name: "data:image/png;base64," + base64.b64encode(data).decode("ascii")
            for name, data in images.items()
        },
    )
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    html = (
        _HTML.replace("__TITLE__", escape(title))
        .replace("__DESCRIPTION__", escape(description))
        .replace("__BEFORE_LABEL__", escape(before_label))
        .replace("__AFTER_LABEL__", escape(after_label))
        .replace("__DATA__", data)
    )
    return html.encode()


_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Raster cleanup · __TITLE__</title>
<style>
body{font:16px system-ui;background:#202020;color:#eee;margin:24px}h1{font-size:24px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:18px 0}
button,select{font:inherit;background:#393939;color:#eee;border:1px solid #666;border-radius:6px;padding:8px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0;min-width:0}
canvas{display:block;width:100%;height:65vh;background:#101010;touch-action:none;cursor:grab}
figcaption{padding:10px 0;font-weight:600}.note{max-width:1000px;color:#ccc;line-height:1.5}
table{border-collapse:collapse;margin:16px 0}td,th{padding:6px 18px 6px 0;text-align:left}
@media(max-width:700px){body{margin:12px}.panels{grid-template-columns:1fr}canvas{height:45vh}}
</style>
<h1>Raster cleanup · __TITLE__</h1><p id="summary"></p>
<p class="note">__DESCRIPTION__</p>
<div class="controls"><select id="view"><option value="-1">Whole image</option></select>
<button id="in">Zoom in</button><button id="out">Zoom out</button><button id="reset">Reset view</button>
<label><input id="overlay" type="checkbox"> Highlight changes</label><label><input id="pixels" type="checkbox" checked> Pixel edges</label>
<span id="scale"></span></div>
<p class="note">Drag to pan · wheel to zoom · both views stay aligned</p>
<div class="panels"><figure><figcaption>__BEFORE_LABEL__</figcaption><canvas id="original"></canvas></figure>
<figure><figcaption>__AFTER_LABEL__</figcaption><canvas id="cleaned"></canvas></figure></div>
<table><thead><tr><th>Color</th><th>Components before → after</th><th>Holes before → after</th></tr></thead><tbody id="counts"></tbody></table>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), d=data.diagnostics;
document.getElementById('summary').textContent=`${data.width_mm} mm ${data.width_label} · ${d.changed_pixels.toLocaleString()} changed pixels (${d.changed_area_mm2.toFixed(3)} mm²) · ${(d.mm_per_pixel).toFixed(4)} mm/pixel`;
for(const color of ['black','orange','white']){const tr=document.createElement('tr');for(const value of [color,`${d.before[color].components} → ${d.after[color].components}`,`${d.before[color].holes} → ${d.after[color].holes}`]){const td=document.createElement('td');td.textContent=value;tr.append(td)}document.getElementById('counts').append(tr)}
data.views.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.label;document.getElementById('view').append(o)});
const canvases=[document.getElementById('original'),document.getElementById('cleaned')], images={};
let view=[0,0,data.width,data.height],drag=null;
function fit(bounds){const [x,y,w,h]=bounds,ratio=canvases[0].clientWidth/canvases[0].clientHeight;let ww=w,hh=h;if(w/h<ratio)ww=h*ratio;else hh=w/ratio;view=[x+(w-ww)/2,y+(h-hh)/2,ww,hh];draw()}
function draw(){canvases.forEach((canvas,index)=>{canvas.width=Math.round(canvas.clientWidth*devicePixelRatio);canvas.height=Math.round(canvas.clientHeight*devicePixelRatio);const ctx=canvas.getContext('2d');ctx.fillStyle='#101010';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=!document.getElementById('pixels').checked;ctx.setTransform(canvas.width/view[2],0,0,canvas.height/view[3],-view[0]*canvas.width/view[2],-view[1]*canvas.height/view[3]);ctx.drawImage(images[index?'cleaned.png':'original.png'],0,0);if(document.getElementById('overlay').checked){ctx.globalAlpha=.85;ctx.drawImage(images['changes.png'],0,0)}});document.getElementById('scale').textContent=`Window ${(view[2]*d.mm_per_pixel).toFixed(2)} x ${(view[3]*d.mm_per_pixel).toFixed(2)} mm`}
function zoom(factor,x=.5,y=.5){view=[view[0]+view[2]*x*(1-factor),view[1]+view[3]*y*(1-factor),view[2]*factor,view[3]*factor];draw()}
function reset(){const index=Number(document.getElementById('view').value);fit(index<0?[0,0,data.width,data.height]:data.views[index].bounds)}
canvases.forEach(canvas=>{canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(e.deltaY*.001),e.offsetX/canvas.clientWidth,e.offsetY/canvas.clientHeight)},{passive:false});canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;view[0]-=(e.clientX-drag[0])*view[2]/canvas.clientWidth;view[1]-=(e.clientY-drag[1])*view[3]/canvas.clientHeight;drag=[e.clientX,e.clientY];draw()});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null)});
document.getElementById('in').onclick=()=>zoom(.5);document.getElementById('out').onclick=()=>zoom(2);
document.getElementById('reset').onclick=reset;document.getElementById('view').onchange=reset;
document.getElementById('overlay').onchange=draw;document.getElementById('pixels').onchange=draw;window.onresize=reset;
Promise.all(Object.entries(data.images).map(([name,url])=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{images[name]=image;resolve()};image.onerror=reject;image.src=url}))).then(reset);
</script></html>"""


def cleanup_files(
    result: RasterCleanupResult,
    title: str = "Artwork",
    *,
    description: str = _DESCRIPTION,
    width_label: str = "compact-detail cutoff",
    width_mm: float | None = None,
    before_label: str = "Original separated colors",
    after_label: str = "Cleaned reference",
) -> dict[str, bytes]:
    """Create reviewable previews from the exact same cleaned model input."""
    pixels = np.zeros((*result.changed.shape, 4), dtype=np.uint8)
    pixels[result.changed] = (255, 0, 180, 255)
    images = {
        "original.png": png_bytes(result.original.normalized_image()),
        "cleaned.png": png_bytes(result.cleaned.normalized_image()),
        "changes.png": png_bytes(Image.fromarray(pixels)),
    }
    return {
        **images,
        "cleanup.json": (
            json.dumps(result.diagnostics, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
        "review.html": _review(
            result,
            images,
            title,
            description=description,
            width_label=width_label,
            width_mm=width_mm,
            before_label=before_label,
            after_label=after_label,
        ),
    }


def prepare_cleanup_artifacts(
    source_bytes: bytes,
    source_name: str,
    config: SeparationConfig | None = None,
    *,
    long_edge_mm: float = 250,
    minimum_width_mm: float = 0.5,
    silhouette_bytes: bytes | None = None,
    silhouette_name: str = "silhouette.png",
) -> ArtifactBundle:
    """Preview intentional edits without invoking vectorization, fitting or CAD."""
    if minimum_width_mm == 0:
        raise ValueError("minimum feature width must be positive for cleanup preview")
    mask: Mask | None = None
    if silhouette_bytes is not None:
        with Image.open(BytesIO(silhouette_bytes)) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    with Image.open(BytesIO(source_bytes)) as image:
        original = separate(image, config, silhouette=mask)
        source_preview = png_bytes(image.convert("RGBA"))
    result = cleanup(original, long_edge_mm, minimum_width_mm)
    files = cleanup_files(result, source_name)
    manifest: dict[str, object] = dict(
        schema_version=1,
        kind="raster-cleanup-preview",
        source=dict(name=source_name, sha256=sha256(source_bytes).hexdigest()),
        silhouette_override=(
            dict(name=silhouette_name, sha256=sha256(silhouette_bytes).hexdigest())
            if silhouette_bytes is not None
            else None
        ),
        dimensions_px=dict(
            zip(
                ("height", "width"),
                cast("tuple[int, int]", original.labels.shape),
                strict=True,
            )
        ),
        long_edge_mm=long_edge_mm,
        minimum_width_mm=minimum_width_mm,
        palette=[list(color) for color in original.config.palette],
        neutral_tolerance=original.config.neutral_tolerance,
        warnings=original.warnings,
        cleanup=result.diagnostics,
        outputs=[*files, "manifest.json"],
        files={name: sha256(data).hexdigest() for name, data in files.items()},
    )
    files["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return ArtifactBundle(manifest, files, source_preview)


def process_cleanup_file(
    source: Path,
    output: Path,
    config: SeparationConfig | None = None,
    *,
    long_edge_mm: float = 250,
    minimum_width_mm: float = 0.5,
    silhouette_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    """Write local comparison artifacts without changing any source file."""
    bundle = prepare_cleanup_artifacts(
        source.read_bytes(),
        source.name,
        config,
        long_edge_mm=long_edge_mm,
        minimum_width_mm=minimum_width_mm,
        silhouette_bytes=silhouette_path.read_bytes() if silhouette_path else None,
        silhouette_name=silhouette_path.name if silhouette_path else "silhouette.png",
    )
    protected = {p.resolve() for p in (source, *protected_paths)}
    if silhouette_path is not None:
        protected.add(silhouette_path.resolve())
    for name in bundle.files:
        target = output / name
        if target.resolve() in protected or (
            target.exists()
            and any(path.exists() and target.samefile(path) for path in protected)
        ):
            raise ValueError(
                "output would overwrite an input file; choose another directory"
            )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in bundle.files.items():
        _ = (output / name).write_bytes(data)
    return bundle.manifest
