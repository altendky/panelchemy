"""Layered OCCT relief from exact raster footprints and expanding distance roofs."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from math import cos, isclose, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import TYPE_CHECKING, cast

import numpy as np
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape
from PIL import Image
from shapely import union_all
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from panelchemy import curved_occt, material_partition, occt
from panelchemy.construction import ConstructionProfile
from panelchemy.contours import PanelContours, vectorize
from panelchemy.curves import CurveContours, fit_contours
from panelchemy.material_partition import MaterialPartition
from panelchemy.outline_fitting import fit_outlines
from panelchemy.separation import (
    Mask,
    SeparationConfig,
    SeparationResult,
    prepare_artifacts,
    separate,
)

if TYPE_CHECKING:
    from panelchemy.export import ExportConfig

EXPORT_NAMES = (
    "relief.stl",
    "black.stl",
    "orange.stl",
    "white.stl",
    "relief.step",
)
STEP_EXPORT_NAMES = ("relief.step", "relief-multipart.step")


def _polygonal(geometry: BaseGeometry) -> Polygon | MultiPolygon:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if geometry.is_empty:
        return Polygon()
    raise ValueError("construction produced a non-polygonal footprint")


def _union(polygons: list[Polygon]) -> Polygon | MultiPolygon:
    return _polygonal(union_all(polygons))


def _topology(geometry: Polygon | MultiPolygon) -> dict[str, int]:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    polygons = [polygon for polygon in polygons if not polygon.is_empty]
    return {
        "components": len(polygons),
        "holes": sum(len(p.interiors) for p in polygons),
    }


@dataclass
class ReliefModel:
    profile: ConstructionProfile
    contours: PanelContours
    solid: TopoDS_Shape
    materials: dict[str, TopoDS_Shape]
    sections: dict[str, Polygon | MultiPolygon]
    diagnostics: dict[str, object]
    contour_comparison_svg: str | None = None
    curves: CurveContours | None = None
    footprint_proxy_tolerance_mm: float = 0.0
    material_partition: MaterialPartition | None = None


def build_relief(
    result: SeparationResult,
    profile: ConstructionProfile | None = None,
    *,
    max_vertices: int = 25_000,
) -> ReliefModel:
    """Preserve original plateaus; enlarge lower sections and split one solid.

    Native operations are never repaired by deleting source features. Resource
    limits reject work explicitly before kernel construction. Taper roofs use
    true arcs; the protected rim uses a reported conservative polygonal guard.
    """
    started = monotonic()
    profile = profile or ConstructionProfile()
    if type(max_vertices) is not int or max_vertices < 1:
        raise ValueError("max_vertices must be a positive integer")
    if result.silhouette_method != "explicit-mask" and result.warnings:
        raise ValueError(
            "Review the boundary warnings and supply an explicit silhouette before building relief"
        )
    # Approximation still starts from exact cells. Bound source allocation
    # separately, then enforce the configured kernel budget on the result.
    source_budget = (
        max(max_vertices, 100_000) if profile.contour_tolerance_mm > 0 else max_vertices
    )
    contours = vectorize(result, profile.long_edge_mm, source_budget)
    approximation_diagnostics: dict[str, object] | None = None
    comparison_svg: str | None = None
    curves: CurveContours | None = None
    proxy_tolerance = 0.0
    native_edge_count = contours.vertex_count
    if profile.contour_tolerance_mm > 0:
        if profile.contour_mode == "curves":
            fitted = fit_outlines(
                contours, profile.contour_tolerance_mm, palette=result.config.palette
            )
        else:
            fitted = fit_contours(
                contours,
                profile.contour_tolerance_mm,
                palette=result.config.palette,
                mode=profile.contour_mode,
            )
        curves = fitted.contours
        contours = fitted.proxy
        approximation_diagnostics = fitted.diagnostics
        comparison_svg = fitted.comparison_svg
        proxy_tolerance = fitted.proxy_deviation_mm
        native_edge_count = sum(
            len(ring.spans)
            for polygon in (curves.silhouette, *curves.orange, *curves.white)
            for ring in (polygon.exterior, *polygon.interiors)
        )
        if native_edge_count > max_vertices:
            raise ValueError(
                f"approximated contours require {native_edge_count:,} native edges, exceeding max_vertices={max_vertices:,}; increase the budget or review another tolerance"
            )
    guard_segments = 32
    guard_radius = profile.rim_width_mm / cos(pi / (4 * guard_segments))
    interior = _polygonal(
        contours.silhouette.buffer(-guard_radius, quad_segs=guard_segments)
    )
    if interior.is_empty:
        raise ValueError("the protected rim consumes the panel; reduce rim_width_mm")
    plateau_union = _union([*contours.orange, *contours.white])
    if plateau_union.difference(interior).area > 1e-9:
        raise ValueError(
            "the protected rim intersects an original feature top; reduce rim_width_mm or revise the boundary"
        )
    slope = profile.taper_run_per_height
    largest_run = slope * (profile.orange_thickness_mm + profile.white_thickness_mm)
    if largest_run > profile.long_edge_mm:
        raise ValueError(
            "taper expansion exceeds the panel scale; reduce taper_angle_deg"
        )
    modes: Counter[str] = Counter()
    roofs: list[TopoDS_Shape] = []
    region_diagnostics: list[dict[str, object]] = []
    expected_white_faces: list[TopoDS_Shape] = []
    guard_tops: dict[float, TopoDS_Shape] = {}
    if curves is not None:
        guard_regions = (
            [interior] if isinstance(interior, Polygon) else list(interior.geoms)
        )
        guard_tops = {
            top: occt.fuse(
                [occt.polygon_face(polygon, top) for polygon in guard_regions]
            )
            for top in (profile.orange_top, profile.white_top)
        }
    for color, polygons, native_regions, top in (
        (
            "orange",
            contours.orange,
            curves.orange if curves is not None else None,
            profile.orange_top,
        ),
        (
            "white",
            contours.white,
            curves.white if curves is not None else None,
            profile.white_top,
        ),
    ):
        for index, polygon in enumerate(polygons):
            region_started = monotonic()
            native_diagnostics: dict[str, object] = {}
            if native_regions is None:
                shape, method = occt.roof(polygon, profile.black_top, top, slope)
                if color == "white":
                    expected_white_faces.append(occt.polygon_face(polygon, top))
            else:
                region = native_regions[index]
                selected_top = curved_occt.polygon_face(region, top)
                if occt.area(occt.cut(selected_top, guard_tops[top])) > 1e-9:
                    raise ValueError(
                        "the protected rim intersects a fitted feature top; reduce rim_width_mm or revise the boundary"
                    )
                shape, method = curved_occt.roof(
                    region,
                    profile.black_top,
                    top,
                    slope,
                    diagnostics=native_diagnostics,
                )
                if color == "white":
                    expected_white_faces.append(selected_top)
            roofs.append(shape)
            modes[method] += 1
            region_diagnostics.append(
                {
                    "color": color,
                    "component": index,
                    "method": method,
                    "elapsed_seconds": monotonic() - region_started,
                    "native_validation": native_diagnostics,
                    **occt.topology_counts(shape),
                }
            )
    black = occt.prism(contours.silhouette, 0, profile.black_top)
    untrimmed = occt.fuse(roofs)
    clipping_solid = occt.prism(interior, profile.black_top, profile.white_top)
    raised = occt.common(untrimmed, clipping_solid)
    solid = occt.fuse([black, raised])
    solids = TopExp_Explorer(solid, TopAbs_SOLID)
    solid_count = 0
    while solids.More():
        solid_count += 1
        solids.Next()
    if solid_count != 1 or not occt.valid(solid):
        raise occt.KernelError("relief must be one valid connected solid")
    material_names = (
        ("black", "orange", "white")
        if contours.white
        else ("black", "orange")
        if contours.orange
        else ("black",)
    )
    partition = material_partition.build_partition(solid, profile, material_names)
    materials = material_partition.materials_for(partition)
    volumes = {name: occt.volume(shape) for name, shape in materials.items()}
    solid_volume = occt.volume(solid)
    if solid_volume <= 0 or not isclose(
        sum(volumes.values()), solid_volume, rel_tol=1e-8, abs_tol=1e-7
    ):
        raise occt.KernelError("material volumes do not account for the combined solid")
    expected_top = occt.fuse(expected_white_faces)
    actual_top = occt.cross_section(solid, profile.white_top)
    top_missing = occt.area(occt.cut(expected_top, actual_top))
    top_extra = occt.area(occt.cut(actual_top, expected_top))
    top_tolerance = max(1e-12, sum(p.area for p in contours.white) * 1e-8)
    if top_missing > top_tolerance or top_extra > top_tolerance:
        raise occt.KernelError(
            "combined relief changed the selected white top footprint"
        )
    orange, white = _union(contours.orange), _union(contours.white)
    # These round-arc polygon approximations are previews, never kernel inputs.
    orange_floor = _polygonal(
        union_all(
            [
                orange.buffer(slope * profile.orange_thickness_mm, quad_segs=16),
                white.buffer(largest_run, quad_segs=16),
            ]
        ).intersection(interior)
    )
    white_floor = _polygonal(
        white.buffer(slope * profile.white_thickness_mm, quad_segs=16).intersection(
            interior
        )
    )
    clipped_volume = max(0.0, occt.volume(untrimmed) - occt.volume(raised))
    orange_occluded = orange.intersection(white_floor).area
    warnings = list(result.warnings)
    if clipped_volume > 1e-7:
        warnings.append(
            "Expanded lower tapers were clipped to protect the black rim; feature tops were preserved."
        )
    if orange_occluded > 1e-9:
        warnings.append(
            "Expanded white ramps cover some orange plateau in top view; inspect the preview or reduce the taper angle."
        )
    if _topology(white) != _topology(white_floor):
        warnings.append(
            "White footprints merge or counters close below their original tops; inspect the band cross-sections."
        )
    source_support = _union([*contours.orange, *contours.white])
    if _topology(source_support) != _topology(orange_floor):
        warnings.append(
            "Colored support footprints merge or counters close within the orange band; selected tops remain unchanged."
        )
    diagnostics: dict[str, object] = {
        "vertex_count": contours.vertex_count,
        "native_outline_edges": native_edge_count,
        "footprint_proxy_tolerance_mm": proxy_tolerance,
        "regions": region_diagnostics,
        "combined_topology": occt.topology_counts(solid),
        "source_components": {
            "orange": len(contours.orange),
            "white": len(contours.white),
        },
        "roof_methods": dict(modes),
        "solid_count": solid_count,
        "solid_volume_mm3": solid_volume,
        "material_volumes_mm3": volumes,
        "native_material_partition": partition.diagnostics,
        "rim_clipped_volume_mm3": clipped_volume,
        "rim_guard_arc_segments_per_quadrant": guard_segments,
        "rim_guard_conservative_radius_mm": guard_radius,
        "rim_guard_radius_excess_mm": guard_radius - profile.rim_width_mm,
        "approximate_orange_plateau_occluded_mm2": orange_occluded,
        "white_source_topology": _topology(white),
        "white_band_bottom_topology": _topology(white_floor),
        "source_support_topology": _topology(source_support),
        "orange_band_bottom_topology": _topology(orange_floor),
        "warnings": warnings,
        "elapsed_seconds": monotonic() - started,
        "max_vertices": max_vertices,
        "source_vertex_budget": source_budget,
        "contour_approximation": approximation_diagnostics,
        "top_footprint_reference": f"fitted native {profile.contour_mode}"
        if curves is not None
        else "exact pixel contours",
        "white_top_missing_mm2": top_missing,
        "white_top_extra_mm2": top_extra,
        "white_top_area_tolerance_mm2": top_tolerance,
        "preview_arc_segments_per_quadrant": 16,
    }
    return ReliefModel(
        profile,
        contours,
        solid,
        materials,
        {
            "orange-floor": orange_floor,
            "white-floor": white_floor,
            "orange-top": orange,
            "white-top": white,
        },
        diagnostics,
        comparison_svg,
        curves,
        proxy_tolerance,
        material_partition=material_partition.seal_partition(solid, partition),
    )


def _path(geometry: Polygon | MultiPolygon, color: str) -> str:
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    paths: list[str] = []
    for polygon in polygons:
        if polygon.is_empty:
            continue
        commands: list[str] = []
        for ring in [polygon.exterior, *polygon.interiors]:
            points = list(ring.coords)
            commands.append(
                "M " + " L ".join(f"{p[0]:.9g},{p[1]:.9g}" for p in points) + " Z"
            )
        paths.append(
            f'<path fill="{color}" fill-rule="evenodd" d="{" ".join(commands)}"/>'
        )
    return "".join(paths)


def preview_svg(model: ReliefModel, palette: tuple[tuple[int, int, int], ...]) -> str:
    """Compare artwork plateaus with projected band colors and support sections."""
    colors = ["#" + "".join(f"{channel:02x}" for channel in color) for color in palette]
    _, _, width, height = model.contours.silhouette.bounds
    gap = width * 0.08
    label_font = min(4.0, width / 18)
    label_height = max(gap, label_font * 2)
    panels = [
        (
            "Selected plateaus"
            if model.contour_comparison_svg is not None
            else "Source plateaus",
            [
                (model.contours.silhouette, colors[0]),
                (model.sections["orange-top"], colors[1]),
                (model.sections["white-top"], colors[2]),
            ],
        ),
        (
            "Relief colors from above",
            [
                (model.contours.silhouette, colors[0]),
                (model.sections["orange-floor"], colors[1]),
                (model.sections["white-floor"], colors[2]),
            ],
        ),
        (
            f"Orange support at Z={model.profile.black_top:g} mm",
            [(model.sections["orange-floor"], colors[1])],
        ),
        (
            f"White support at Z={model.profile.orange_top:g} mm",
            [(model.sections["white-floor"], colors[2])],
        ),
    ]
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {4 * (width + gap) + gap:g} {height + label_height + gap:g}">',
        '<rect width="100%" height="100%" fill="#e8e8e8"/>',
    ]
    for index, (label, shapes) in enumerate(panels):
        offset = gap + index * (width + gap)
        content.append(
            f'<text x="{offset:g}" y="{label_font * 1.5:g}" font-size="{label_font:g}" font-family="sans-serif">{label}</text>'
        )
        content.append(
            f'<g transform="translate({offset:g},{height + label_height:g}) scale(1,-1)">'
        )
        content.extend(_path(shape, color) for shape, color in shapes)
        content.append("</g>")
    return "".join([*content, "</svg>"])


def process_relief_file(
    source: Path,
    output: Path,
    profile: ConstructionProfile | None = None,
    separation_config: SeparationConfig | None = None,
    *,
    silhouette_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
    max_vertices: int = 25_000,
    export_config: "ExportConfig | None" = None,
    step_only: bool = False,
) -> dict[str, object]:
    """Stage checked CAD and optional validated exports with shared provenance."""
    if step_only and export_config is not None:
        raise ValueError("STEP-only export cannot also request mesh export")
    names = [
        "relief.brep",
        "black.brep",
        "orange.brep",
        "white.brep",
        "preview.svg",
        "profile.json",
        "manifest.json",
        "contours.svg",
        *EXPORT_NAMES,
        *STEP_EXPORT_NAMES,
    ]
    protected = {path.resolve() for path in (source, *protected_paths)}
    if silhouette_path:
        protected.add(silhouette_path.resolve())
    for name in names:
        target = output / name
        if target.resolve() in protected or (
            target.exists()
            and any(p.exists() and target.samefile(p) for p in protected)
        ):
            raise ValueError(
                "output would overwrite an input file; choose another directory"
            )
    source_bytes = source.read_bytes()
    silhouette_bytes = silhouette_path.read_bytes() if silhouette_path else None
    silhouette_mask: Mask | None = None
    if silhouette_bytes is not None:
        with Image.open(BytesIO(silhouette_bytes)) as mask_image:
            silhouette_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128
    with Image.open(BytesIO(source_bytes)) as image:
        result = separate(
            image,
            separation_config,
            silhouette=silhouette_mask,
        )
    model = build_relief(result, profile, max_vertices=max_vertices)
    # Staging avoids publishing BRep files if a kernel write or preview fails.
    with TemporaryDirectory(prefix="panelchemy-relief-") as staging:
        stage = Path(staging)
        occt.write_brep(model.solid, stage / "relief.brep")
        for name, shape in model.materials.items():
            occt.write_brep(shape, stage / f"{name}.brep")
        export_report: dict[str, object] | None = None
        if step_only:
            from panelchemy.step_export import export_step

            bundle = export_step(model, palette=result.config.palette)
            for name, content in bundle.files.items():
                if name not in STEP_EXPORT_NAMES:
                    raise ValueError(f"unexpected export artifact: {name}")
                _ = (stage / name).write_bytes(content)
            export_report = {"format": "step", "validation": bundle.validation}
        elif export_config is not None:
            from panelchemy.export import export_model

            bundle = export_model(model, export_config)
            for name, content in bundle.files.items():
                if name not in EXPORT_NAMES:
                    raise ValueError(f"unexpected export artifact: {name}")
                _ = (stage / name).write_bytes(content)
            export_report = {
                "config": asdict(export_config),
                "validation": bundle.validation,
            }
        _ = (stage / "preview.svg").write_text(
            preview_svg(model, result.config.palette)
        )
        if model.contour_comparison_svg is not None:
            _ = (stage / "contours.svg").write_text(model.contour_comparison_svg)
        _ = (stage / "profile.json").write_text(
            json.dumps(model.profile.as_dict(), indent=2, sort_keys=True) + "\n"
        )
        separation_bundle = prepare_artifacts(
            source_bytes,
            source.name,
            result.config,
            silhouette_bytes=silhouette_bytes,
            silhouette_name=silhouette_path.name
            if silhouette_path
            else "silhouette.png",
        )
        (stage / "separation").mkdir()
        for name, content in separation_bundle.files.items():
            _ = (stage / "separation" / name).write_bytes(content)
        separation_manifest = separation_bundle.manifest
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "occt-relief",
            "print_ready": False,
            "mesh_validated": export_config is not None,
            "physical_print_validated": False,
            "export": export_report,
            "source": separation_manifest["source"],
            "separation": separation_manifest,
            "profile": model.profile.as_dict(),
            "taper_angle_reference": "degrees from vertical",
            "mm_per_pixel": model.contours.mm_per_pixel,
            "source_bounds_px": list(model.contours.source_bounds_px),
            "diagnostics": model.diagnostics,
            "versions": {
                name: version(name)
                for name in ("panelchemy", "cadquery-ocp-novtk", "shapely")
            },
            "files": {
                str(path.relative_to(stage)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        # Check every staged target before writing, including nested previews.
        for relative in cast("dict[str, str]", manifest["files"]):
            target = output / relative
            if target.resolve() in protected or (
                target.exists()
                and any(p.exists() and target.samefile(p) for p in protected)
            ):
                raise ValueError(
                    "output would overwrite an input file; choose another directory"
                )
        output.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                target = output / path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                _ = target.write_bytes(path.read_bytes())
        # Remove only stale, known material artifacts when a color is now absent.
        for name in ("orange", "white"):
            if name not in model.materials:
                (output / f"{name}.brep").unlink(missing_ok=True)
        if model.contour_comparison_svg is None:
            (output / "contours.svg").unlink(missing_ok=True)
        for name in (*EXPORT_NAMES, *STEP_EXPORT_NAMES):
            if not (stage / name).exists():
                (output / name).unlink(missing_ok=True)
        _ = (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return manifest
