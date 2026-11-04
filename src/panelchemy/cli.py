"""Command-line entry point for the local project foundation."""

import json
from argparse import ArgumentParser
from importlib.metadata import version
from pathlib import Path
from typing import cast

from PIL import Image

from panelchemy.separation import load_config, process_file


def main() -> None:
    """Describe the current command surface without claiming conversion support."""
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
    local_server = commands.add_parser("serve", help="open the local browser workflow")
    _ = local_server.add_argument("--port", type=int, default=8765)
    _ = local_server.add_argument("--max-pixels", type=int, default=4_000_000)
    _ = local_server.add_argument("--max-request-mib", type=int, default=32)
    relief = commands.add_parser(
        "relief", help="build inspectable layered OCCT geometry"
    )
    _ = relief.add_argument("input", type=Path)
    _ = relief.add_argument("--output", type=Path, required=True)
    _ = relief.add_argument("--config", type=Path)
    _ = relief.add_argument("--profile", type=Path)
    _ = relief.add_argument("--silhouette-mask", type=Path)
    _ = relief.add_argument("--max-vertices", type=int, default=25_000)
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
        if cast("str", args.command) == "relief":
            from panelchemy.construction import load_profile
            from panelchemy.relief import process_relief_file

            config_path = cast("Path | None", args.config)
            profile_path = cast("Path | None", args.profile)
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
