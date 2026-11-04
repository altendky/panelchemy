"""Command-line entry point for the local project foundation."""

from argparse import ArgumentParser
from importlib.metadata import version


def main() -> None:
    """Describe the current command surface without claiming conversion support."""
    parser = ArgumentParser(
        description="Panelchemy: comic-panel artwork to layered relief (in development)."
    )
    _ = parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('panelchemy')}"
    )
    _ = parser.parse_args()
    parser.print_help()
