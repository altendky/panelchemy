"""Local command-line adapter for artwork previews, relief, and model export."""

import json
from argparse import ArgumentParser
from importlib.metadata import version
from math import degrees, radians
from pathlib import Path
from typing import cast

from PIL import Image

from panelchemy.separation import load_config, process_file


def main() -> None:
    """Run the shared conversion core through local commands."""
    parser = ArgumentParser(
        description="Panelchemy: comic-panel artwork to layered relief (in development)."
    )
    _ = parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('panelchemy')}"
    )
    commands = parser.add_subparsers(dest="command")
    separation = commands.add_parser("separate", help="preview palette and panel masks")
    _ = separation.add_argument("input", type=Path)
    _ = separation.add_argument("--output", type=Path, required=True)
    _ = separation.add_argument("--config", type=Path)
    _ = separation.add_argument("--silhouette-mask", type=Path)
    cleanup = commands.add_parser("cleanup", help="preview optional raster cleanup")
    _ = cleanup.add_argument("input", type=Path)
    _ = cleanup.add_argument("--output", type=Path, required=True)
    _ = cleanup.add_argument("--config", type=Path)
    _ = cleanup.add_argument("--profile", type=Path)
    _ = cleanup.add_argument("--silhouette-mask", type=Path)
    _ = cleanup.add_argument(
        "--minimum-width-mm",
        type=float,
        help="cleanup scale; defaults to the profile setting or assumed line width",
    )
    stroke_preview = commands.add_parser(
        "stroke-preview", help="inspect bounded stroke-growth proposals without CAD"
    )
    _ = stroke_preview.add_argument("input", type=Path)
    _ = stroke_preview.add_argument("--output", type=Path, required=True)
    _ = stroke_preview.add_argument("--config", type=Path)
    _ = stroke_preview.add_argument("--profile", type=Path)
    _ = stroke_preview.add_argument("--silhouette-mask", type=Path)
    _ = stroke_preview.add_argument("--target-width-mm", type=float)
    _ = stroke_preview.add_argument("--max-growth-mm", type=float)
    _ = stroke_preview.add_argument("--compact-cleanup-mm", type=float)
    local_server = commands.add_parser("serve", help="open the local browser workflow")
    _ = local_server.add_argument("--port", type=int, default=8765)
    _ = local_server.add_argument("--max-pixels", type=int, default=4_000_000)
    _ = local_server.add_argument("--max-request-mib", type=int, default=32)
    for name, help_text in (
        ("relief", "build inspectable layered OCCT geometry"),
        ("export", "validate meshes and export registered material geometry"),
    ):
        relief = commands.add_parser(name, help=help_text)
        _ = relief.add_argument("input", type=Path)
        _ = relief.add_argument("--output", type=Path, required=True)
        _ = relief.add_argument("--config", type=Path)
        _ = relief.add_argument("--profile", type=Path)
        _ = relief.add_argument("--silhouette-mask", type=Path)
        _ = relief.add_argument("--max-vertices", type=int, default=25_000)
        _ = relief.add_argument(
            "--workers",
            type=int,
            default=1,
            help="opt in to parallel curvature fitting with this many processes (default: 1)",
        )
        if name == "export":
            _ = relief.add_argument(
                "--format",
                choices=("all", "step"),
                default="all",
                help="step writes fused and colored multipart STEP without meshing",
            )
            _ = relief.add_argument("--mesh-deflection-mm", type=float, default=0.02)
            _ = relief.add_argument(
                "--mesh-angle-deg", type=float, default=degrees(0.2)
            )
            _ = relief.add_argument("--max-triangles", type=int, default=2_000_000)
    args = parser.parse_args()
    if cast("str | None", args.command) is None:
        parser.print_help()
        return
    try:
        if cast("str", args.command) == "serve":
            from panelchemy.server import Limits, serve

            serve(
                cast("int", args.port),
                Limits(
                    max_pixels=cast("int", args.max_pixels),
                    max_request_bytes=cast("int", args.max_request_mib) * 1024 * 1024,
                ),
            )
            return
        if cast("str", args.command) == "stroke-preview":
            from panelchemy.construction import load_profile
            from panelchemy.stroke_retention_artifacts import (
                process_stroke_preview_file,
            )

            config_path = cast("Path | None", args.config)
            profile_path = cast("Path | None", args.profile)
            profile = load_profile(profile_path)
            target_width = cast("float | None", args.target_width_mm)
            max_growth = cast("float | None", args.max_growth_mm)
            compact_cleanup = cast("float | None", args.compact_cleanup_mm)
            manifest = process_stroke_preview_file(
                cast("Path", args.input),
                cast("Path", args.output),
                load_config(config_path),
                long_edge_mm=profile.long_edge_mm,
                target_width_mm=profile.assumed_line_width_mm
                if target_width is None
                else target_width,
                max_growth_mm=profile.tracing_radius_mm
                if max_growth is None
                else max_growth,
                compact_cleanup_mm=profile.raster_cleanup_mm
                if compact_cleanup is None
                else compact_cleanup,
                silhouette_path=cast("Path | None", args.silhouette_mask),
                protected_paths=tuple(
                    p for p in (config_path, profile_path) if p is not None
                ),
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        if cast("str", args.command) == "cleanup":
            from panelchemy.construction import load_profile
            from panelchemy.raster_cleanup_artifacts import process_cleanup_file

            config_path = cast("Path | None", args.config)
            profile_path = cast("Path | None", args.profile)
            profile = load_profile(profile_path)
            minimum_width = cast("float | None", args.minimum_width_mm)
            if minimum_width is None:
                minimum_width = (
                    profile.raster_cleanup_mm
                    if profile.raster_cleanup_mm > 0
                    else profile.assumed_line_width_mm
                )
            manifest = process_cleanup_file(
                cast("Path", args.input),
                cast("Path", args.output),
                load_config(config_path),
                long_edge_mm=profile.long_edge_mm,
                minimum_width_mm=minimum_width,
                silhouette_path=cast("Path | None", args.silhouette_mask),
                protected_paths=tuple(
                    p for p in (config_path, profile_path) if p is not None
                ),
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        if cast("str", args.command) in {"relief", "export"}:
            from panelchemy.construction import load_profile
            from panelchemy.relief import process_relief_file

            config_path = cast("Path | None", args.config)
            profile_path = cast("Path | None", args.profile)
            export_config = None
            step_only = (
                cast("str", args.command) == "export"
                and cast("str", args.format) == "step"
            )
            if cast("str", args.command) == "export" and not step_only:
                from panelchemy.export import ExportConfig

                export_config = ExportConfig(
                    linear_deflection_mm=cast("float", args.mesh_deflection_mm),
                    angular_deflection_rad=radians(cast("float", args.mesh_angle_deg)),
                    max_triangles=cast("int", args.max_triangles),
                )
            manifest = process_relief_file(
                cast("Path", args.input),
                cast("Path", args.output),
                load_profile(profile_path),
                load_config(config_path),
                silhouette_path=cast("Path | None", args.silhouette_mask),
                protected_paths=tuple(
                    p for p in (config_path, profile_path) if p is not None
                ),
                max_vertices=cast("int", args.max_vertices),
                workers=cast("int", args.workers),
                export_config=export_config,
                step_only=step_only,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        source = cast("Path", args.input)
        output = cast("Path", args.output)
        config_path = cast("Path | None", args.config)
        mask_path = cast("Path | None", args.silhouette_mask)
        manifest = process_file(
            source,
            output,
            load_config(config_path),
            silhouette_path=mask_path,
            protected_paths=(config_path,) if config_path else (),
        )
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
